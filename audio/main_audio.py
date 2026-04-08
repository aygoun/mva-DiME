"""
Audio Counterfactual Explanations via DiME.

Generates counterfactual mel spectrograms that flip a multi-label audio
classifier's prediction for a **single** target class, using the
diffusion-guided process from DiME.

Dataset : FSD50K (multi-label, 200 classes from AudioSet 527)
Classifier: Pretrained AST on AudioSet (527-class sigmoid output, no custom head)
"""

import os
import sys
import random
import argparse
import numpy as np
import soundfile as sf

from time import time

import torch
from torch.utils import data

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from core.sample_utils import (
    get_DiME_iterative_sampling,
    clean_multilabel_cond_fn,
    dist_cond_fn,
    PerceptualLoss,
    load_from_DDP_model,
)
from core.gaussian_diffusion import _extract_into_tensor

from audio.audio_datasets import FSD50KDataset
from audio.audio_classifier import (
    build_classifier,
    ensure_hf_model_downloaded,
    load_audioset_class_mapping,
)
from audio.spectrogram_utils import (
    tensor_to_audio,
    SAMPLE_RATE,
)


def create_args():
    defaults = dict(
        clip_denoised=True,
        batch_size=8,
        gpu="0",
        num_batches=50,

        # paths
        output_path="audio/results",
        model_path="audio/models/ddpm-spectro/ema_0.9999_015000.pt",
        data_dir="",
        exp_name="audio_cf",

        # classifier
        ast_model_id="MIT/ast-finetuned-audioset-10-10-0.4593",

        # sampling
        classifier_scales="5,8,12",
        seed=42,
        target_label=-1,
        target_strategy="random_remove",
        use_ddim=False,
        start_step=60,
        use_logits=False,
        l1_loss=0.05,
        l2_loss=0.0,
        l_perc=10.0,
        l_perc_layer=8,
        use_sampling_on_x_t=True,
        guided_iterations=9999999,

        # audio / FSD50K
        audio_duration=7.0,
        fsd50k_split="eval",

        # output
        save_audio=True,
        save_images=True,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser.parse_args()


def save_spectrogram_image(tensor, path):
    """Save a single (C,H,W) tensor in [-1,1] as a grayscale PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = tensor[0].cpu().numpy()
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.imshow(arr, aspect="auto", origin="lower", cmap="magma")
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(path, dpi=80, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# ------------------------------------------------------------------
# Target selection helpers for multi-label (FSD50K / AudioSet)
# ------------------------------------------------------------------


def select_targets_multilabel(logits, multi_hot, strategy, valid_indices,
                              fixed_label=-1):
    """Pick one target class per sample and the desired target direction.

    Returns
    -------
    targets : (B,) LongTensor — AudioSet class indices
    y_vals  : (B,) FloatTensor — 1.0 = add class, 0.0 = remove class
    """
    B = logits.size(0)
    device = logits.device
    targets = torch.zeros(B, dtype=torch.long, device=device)
    y_vals = torch.ones(B, dtype=torch.float32, device=device)
    probs = torch.sigmoid(logits.detach())

    valid_set = set(valid_indices)

    for i in range(B):
        positive = multi_hot[i].nonzero(as_tuple=True)[0].tolist()
        positive_valid = [c for c in positive if c in valid_set]

        if fixed_label >= 0:
            targets[i] = fixed_label
            y_vals[i] = 0.0 if fixed_label in positive else 1.0
            continue

        if strategy == "random_remove" and positive_valid:
            c = random.choice(positive_valid)
            targets[i] = c
            y_vals[i] = 0.0
        elif strategy == "least_confident_remove" and positive_valid:
            confs = [(c, probs[i, c].item()) for c in positive_valid]
            c = min(confs, key=lambda x: x[1])[0]
            targets[i] = c
            y_vals[i] = 0.0
        elif strategy == "random_add":
            negative = [c for c in valid_indices if c not in positive]
            if negative:
                c = random.choice(negative)
                targets[i] = c
                y_vals[i] = 1.0
            elif positive_valid:
                targets[i] = random.choice(positive_valid)
                y_vals[i] = 0.0
        else:
            if positive_valid:
                c = random.choice(positive_valid)
                targets[i] = c
                y_vals[i] = 0.0

    return targets, y_vals


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main():
    args = create_args()
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    exp_dir = os.path.join(args.output_path, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    for sub in ["original_spec", "cf_spec", "diff_spec",
                 "original_wav", "cf_wav", "info"]:
        os.makedirs(os.path.join(exp_dir, sub), exist_ok=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ---- dataset ----
    print("Loading AudioSet class mapping ...")
    mid_to_idx = load_audioset_class_mapping()
    print(f"  {len(mid_to_idx)} AudioSet classes loaded")

    dataset = FSD50KDataset(
        data_dir=args.data_dir,
        split=args.fsd50k_split,
        duration=args.audio_duration,
        size=args.image_size,
        audioset_mid_to_idx=mid_to_idx,
    )
    valid_audioset_indices = dataset.valid_audioset_indices
    print(f"  FSD50K {args.fsd50k_split}: {len(dataset)} clips, "
          f"{dataset.num_fsd50k_classes} FSD50K classes -> "
          f"{len(valid_audioset_indices)} AudioSet indices")

    loader = data.DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    # ---- DDPM ----
    print("Loading DDPM ...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    state_dict = torch.load(args.model_path, map_location="cpu")
    state_dict = load_from_DDP_model(state_dict)
    model.load_state_dict(state_dict)
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    def model_fn(x, t, y=None):
        return model(x, t, y if args.class_cond else None)

    # ---- classifier ----
    print("Loading classifier ...")
    ensure_hf_model_downloaded(args.ast_model_id)
    classifier = build_classifier(ast_model_id=args.ast_model_id).to(device)
    classifier.eval()

    # ---- perceptual loss ----
    if args.l_perc != 0:
        print("Loading perceptual loss ...")
        vggloss = PerceptualLoss(layer=args.l_perc_layer,
                                 c=args.l_perc).to(device)
        vggloss.eval()
    else:
        vggloss = None

    # ---- sampling ----
    sample_fn = get_DiME_iterative_sampling(use_sampling=args.use_sampling_on_x_t)
    classifier_scales = [float(x) for x in args.classifier_scales.split(",")]

    stats = {
        "n": 0, "flipped": 0, "clean_positive": 0,
        "l1": [], "l2": [],
        "target_conf_before": [], "target_conf_after": [],
    }

    print("Starting counterfactual generation ...")
    start_time = time()
    global_idx = 0

    for batch_idx, (specs, labels, time_frames) in enumerate(loader):
        if batch_idx >= args.num_batches:
            break
        print(f"Batch {batch_idx+1}/{min(args.num_batches, len(loader))} "
              f"| {int(time()-start_time)}s")

        specs = specs.to(device)
        labels = labels.to(device)

        # ---- initial prediction ----
        with torch.no_grad():
            logits = classifier(specs)

        # ---- choose target ----
        targets, y_vals = select_targets_multilabel(
            logits, labels, args.target_strategy,
            valid_audioset_indices, fixed_label=args.target_label,
        )
        probs_before = torch.sigmoid(logits.detach())

        # ---- forward noise + guided reverse ----
        t = torch.ones(specs.size(0), device=device,
                       dtype=torch.long) * args.start_step
        noise_img = diffusion.q_sample(specs, t)
        transformed = torch.zeros(specs.size(0), device=device).bool()
        cf = specs.clone()

        for jdx, cls_scale in enumerate(classifier_scales):
            mask = ~transformed
            if mask.sum() == 0:
                break

            model_kwargs = {"y": targets[mask]}
            grad_kwargs = {
                "y": targets[mask],
                "classifier": classifier,
                "s": cls_scale,
                "use_logits": args.use_logits,
                "y_val": y_vals[mask],
            }

            cfs, _, _ = sample_fn(
                diffusion,
                model_fn,
                specs[mask].shape,
                args.start_step,
                specs[mask],
                t,
                z_t=noise_img[mask],
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                device=device,
                class_grad_fn=clean_multilabel_cond_fn,
                class_grad_kwargs=grad_kwargs,
                dist_grad_fn=dist_cond_fn,
                dist_grad_kargs={
                    "l1_loss": args.l1_loss,
                    "l2_loss": args.l2_loss,
                    "l_perc": vggloss,
                },
                guided_iterations=args.guided_iterations,
                is_x_t_sampling=False,
            )

            with torch.no_grad():
                cf_logits = classifier(cfs)

            cf_probs = torch.sigmoid(cf_logits)
            orig_active = (y_vals[mask] < 0.5)
            flipped = torch.zeros(mask.sum(), device=device).bool()
            flipped[orig_active] = cf_probs[orig_active, :].gather(
                1, targets[mask][orig_active].unsqueeze(1)
            ).squeeze(1) < 0.5
            flipped[~orig_active] = cf_probs[~orig_active, :].gather(
                1, targets[mask][~orig_active].unsqueeze(1)
            ).squeeze(1) > 0.5

            if jdx == 0:
                cf = cfs.clone() if mask.all() else cf.clone()
            cf[mask] = cfs

            old_mask = mask.clone()
            transformed[old_mask] = flipped

        # ---- evaluate ----
        with torch.no_grad():
            cf_logits = classifier(cf)
            cf_probs = torch.sigmoid(cf_logits)

        specs_01 = (specs + 1) / 2
        cf_01 = (cf.clamp(-1, 1) + 1) / 2
        l1 = (specs_01 - cf_01).abs().view(specs.size(0), -1).mean(dim=1)
        l2 = ((specs_01 - cf_01) ** 2).view(specs.size(0), -1).mean(dim=1).sqrt()

        bidx = torch.arange(specs.size(0), device=device)
        target_conf_before = probs_before[bidx, targets]
        target_conf_after = cf_probs[bidx, targets]

        is_remove = (y_vals < 0.5)
        flipped_final = torch.zeros(specs.size(0), device=device).bool()
        flipped_final[is_remove] = (target_conf_after[is_remove] < 0.5)
        flipped_final[~is_remove] = (target_conf_after[~is_remove] > 0.5)
        clean_positive = (target_conf_before > 0.5).sum().item()

        stats["n"] += specs.size(0)
        stats["flipped"] += flipped_final.sum().item()
        stats["clean_positive"] += clean_positive
        stats["l1"].append(l1.cpu())
        stats["l2"].append(l2.cpu())
        stats["target_conf_before"].append(target_conf_before.cpu())
        stats["target_conf_after"].append(target_conf_after.cpu())

        # ---- save outputs ----
        for i in range(specs.size(0)):
            idx_str = f"{global_idx:06d}"
            tf = time_frames[i].item()

            if args.save_images:
                save_spectrogram_image(
                    specs[i], os.path.join(exp_dir, "original_spec", f"{idx_str}.png"))
                save_spectrogram_image(
                    cf[i], os.path.join(exp_dir, "cf_spec", f"{idx_str}.png"))
                diff = (cf[i] - specs[i]).abs()
                save_spectrogram_image(
                    diff, os.path.join(exp_dir, "diff_spec", f"{idx_str}.png"))

            if args.save_audio:
                try:
                    orig_wav = tensor_to_audio(specs[i], tf)
                    sf.write(os.path.join(exp_dir, "original_wav", f"{idx_str}.wav"),
                             orig_wav, SAMPLE_RATE)
                    cf_wav = tensor_to_audio(cf[i], tf)
                    sf.write(os.path.join(exp_dir, "cf_wav", f"{idx_str}.wav"),
                             cf_wav, SAMPLE_RATE)
                except Exception as e:
                    print(f"  Warning: audio inversion failed for {idx_str}: {e}")

            direction = "remove" if y_vals[i] < 0.5 else "add"
            gt_classes = labels[i].nonzero(as_tuple=True)[0].tolist()
            info = (
                f"target_class: {targets[i].item()}\n"
                f"direction: {direction}\n"
                f"target_conf_before: {target_conf_before[i].item():.4f}\n"
                f"target_conf_after: {target_conf_after[i].item():.4f}\n"
                f"flipped: {flipped_final[i].item()}\n"
                f"l1: {l1[i].item():.4f}\n"
                f"l2: {l2[i].item():.4f}\n"
                f"ground_truth_classes: {gt_classes}\n"
            )
            with open(os.path.join(exp_dir, "info", f"{idx_str}.txt"), "w") as f:
                f.write(info)

            global_idx += 1

    # ---- summary ----
    all_l1 = torch.cat(stats["l1"]).numpy()
    all_l2 = torch.cat(stats["l2"]).numpy()
    all_tcb = torch.cat(stats["target_conf_before"]).numpy()
    all_tca = torch.cat(stats["target_conf_after"]).numpy()
    flip_rate = stats["flipped"] / max(stats["n"], 1)
    clean_rate = stats["clean_positive"] / max(stats["n"], 1)

    summary = (
        f"Classifier: AST AudioSet (527 classes)\n"
        f"Target strategy: {args.target_strategy}\n"
        f"Total samples: {stats['n']}\n"
        f"Clean positive rate: {100*clean_rate:.1f}%\n"
        f"Flip rate:           {100*flip_rate:.1f}%\n"
        f"Mean L1:             {all_l1.mean():.4f}\n"
        f"Mean L2:             {all_l2.mean():.4f}\n"
        f"Target conf before:  {all_tcb.mean():.4f}\n"
        f"Target conf after:   {all_tca.mean():.4f}\n"
        f"Target conf change:  {(all_tca - all_tcb).mean():.4f}\n"
    )
    print("\n" + summary)

    with open(os.path.join(exp_dir, "summary.txt"), "w") as f:
        f.write(summary)


if __name__ == "__main__":
    main()
