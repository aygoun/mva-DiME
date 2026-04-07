"""
Audio Counterfactual Explanations via DiME.

Generates counterfactual mel spectrograms that flip a multiclass audio
classifier's prediction, using the diffusion-guided process from DiME.
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
    clean_multiclass_cond_fn,
    dist_cond_fn,
    PerceptualLoss,
    load_from_DDP_model,
)
from core.gaussian_diffusion import _extract_into_tensor

from audio.audio_datasets import ESC50Dataset
from audio.audio_classifier import build_classifier, ensure_hf_model_downloaded
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
        dataset="ESC50",

        # paths
        output_path="audio/results",
        classifier_path="audio/models/classifier.pth",
        model_path="audio/models/ddpm-spectro/ema_0.9999_015000.pt",
        data_dir="",
        exp_name="audio_cf",

        # classifier
        num_classes=50,
        classifier_type="resnet18",
        ast_model_id="MIT/ast-finetuned-audioset-10-10-0.4593",

        # sampling
        classifier_scales="5,8,12",
        seed=42,
        target_label=-1,  # -1 = flip to second-most-likely class
        use_ddim=False,
        start_step=60,
        use_logits=False,
        l1_loss=0.05,
        l2_loss=0.0,
        l_perc=10.0,
        l_perc_layer=8,
        use_sampling_on_x_t=True,
        guided_iterations=9999999,

        # ESC-50 specific
        val_fold=5,
        audio_duration=5.0,

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

    arr = tensor[0].cpu().numpy()  # take first channel
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.imshow(arr, aspect="auto", origin="lower", cmap="magma")
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(path, dpi=80, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


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

    # seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # dataset
    if args.dataset == "ESC50":
        dataset = ESC50Dataset(
            data_dir=args.data_dir,
            folds=[args.val_fold],
            duration=args.audio_duration,
            size=args.image_size,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    loader = data.DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    # DDPM
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

    # classifier
    print("Loading classifier ...")
    if args.classifier_type == "ast":
        ensure_hf_model_downloaded(args.ast_model_id)
    classifier = build_classifier(
        classifier_type=args.classifier_type,
        num_classes=args.num_classes,
        weights_path=args.classifier_path,
        ast_model_id=args.ast_model_id,
    ).to(device)
    classifier.eval()

    # perceptual loss
    if args.l_perc != 0:
        print("Loading perceptual loss ...")
        vggloss = PerceptualLoss(layer=args.l_perc_layer,
                                 c=args.l_perc).to(device)
        vggloss.eval()
    else:
        vggloss = None

    # sampling function
    sample_fn = get_DiME_iterative_sampling(use_sampling=args.use_sampling_on_x_t)
    classifier_scales = [float(x) for x in args.classifier_scales.split(",")]

    stats = {
        "n": 0,
        "flipped": 0,
        "clean_acc": 0,
        "l1": [],
        "l2": [],
        "target_conf_before": [],
        "target_conf_after": [],
        "pred_conf_before": [],
        "pred_conf_after": [],
        "margin_before": [],
        "margin_after": [],
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
        labels = labels.to(device, dtype=torch.long)

        # initial prediction
        with torch.no_grad():
            logits = classifier(specs)
            preds = logits.argmax(dim=1)
            probs = torch.softmax(logits, dim=1)

        # choose target
        if args.target_label >= 0:
            targets = torch.full_like(preds, args.target_label)
        else:
            # flip to second-most-likely class
            masked = logits.clone()
            masked[range(len(preds)), preds] = -float("inf")
            targets = masked.argmax(dim=1)

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
                class_grad_fn=clean_multiclass_cond_fn,
                class_grad_kwargs={
                    "y": targets[mask],
                    "classifier": classifier,
                    "s": cls_scale,
                    "use_logits": args.use_logits,
                },
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
                cf_preds = cf_logits.argmax(dim=1)

            if jdx == 0:
                cf = cfs.clone()
            cf[mask] = cfs

            flipped = (cf_preds == targets[mask])
            old_mask = mask.clone()
            transformed[old_mask] = flipped

        # evaluate
        with torch.no_grad():
            cf_logits = classifier(cf)
            cf_preds = cf_logits.argmax(dim=1)
            cf_probs = torch.softmax(cf_logits, dim=1)

        specs_01 = (specs + 1) / 2
        cf_01 = (cf.clamp(-1, 1) + 1) / 2
        l1 = (specs_01 - cf_01).abs().view(specs.size(0), -1).mean(dim=1)
        l2 = ((specs_01 - cf_01) ** 2).view(specs.size(0), -1).mean(dim=1).sqrt()

        bidx = torch.arange(specs.size(0), device=device)
        target_conf_before = probs[bidx, targets]
        target_conf_after = cf_probs[bidx, targets]
        pred_conf_before = probs[bidx, preds]
        pred_conf_after = cf_probs[bidx, preds]
        top2_before = probs.topk(k=2, dim=1).values
        top2_after = cf_probs.topk(k=2, dim=1).values
        margin_before = top2_before[:, 0] - top2_before[:, 1]
        margin_after = top2_after[:, 0] - top2_after[:, 1]

        stats["n"] += specs.size(0)
        stats["flipped"] += (cf_preds == targets).sum().item()
        stats["clean_acc"] += (preds == labels).sum().item()
        stats["l1"].append(l1.cpu())
        stats["l2"].append(l2.cpu())
        stats["target_conf_before"].append(target_conf_before.cpu())
        stats["target_conf_after"].append(target_conf_after.cpu())
        stats["pred_conf_before"].append(pred_conf_before.cpu())
        stats["pred_conf_after"].append(pred_conf_after.cpu())
        stats["margin_before"].append(margin_before.cpu())
        stats["margin_after"].append(margin_after.cpu())

        # save outputs
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

            info = (f"pred: {preds[i].item()}\n"
                    f"target: {targets[i].item()}\n"
                    f"cf_pred: {cf_preds[i].item()}\n"
                    f"flipped: {cf_preds[i].item() == targets[i].item()}\n"
                    f"l1: {l1[i].item():.4f}\n"
                    f"l2: {l2[i].item():.4f}\n"
                    f"target_conf_before: {target_conf_before[i].item():.4f}\n"
                    f"target_conf_after: {target_conf_after[i].item():.4f}\n"
                    f"pred_conf_before: {pred_conf_before[i].item():.4f}\n"
                    f"pred_conf_after: {pred_conf_after[i].item():.4f}\n"
                    f"margin_before: {margin_before[i].item():.4f}\n"
                    f"margin_after: {margin_after[i].item():.4f}\n")
            with open(os.path.join(exp_dir, "info", f"{idx_str}.txt"), "w") as f:
                f.write(info)

            global_idx += 1

    # summary
    all_l1 = torch.cat(stats["l1"]).numpy()
    all_l2 = torch.cat(stats["l2"]).numpy()
    all_tcb = torch.cat(stats["target_conf_before"]).numpy()
    all_tca = torch.cat(stats["target_conf_after"]).numpy()
    all_pcb = torch.cat(stats["pred_conf_before"]).numpy()
    all_pca = torch.cat(stats["pred_conf_after"]).numpy()
    all_mb = torch.cat(stats["margin_before"]).numpy()
    all_ma = torch.cat(stats["margin_after"]).numpy()
    flip_rate = stats["flipped"] / max(stats["n"], 1)
    clean_acc = stats["clean_acc"] / max(stats["n"], 1)
    summary = (f"Total samples: {stats['n']}\n"
               f"Clean accuracy: {100*clean_acc:.1f}%\n"
               f"Flip rate:     {100*flip_rate:.1f}%\n"
               f"Mean L1:       {all_l1.mean():.4f}\n"
               f"Mean L2:       {all_l2.mean():.4f}\n"
               f"Target conf before: {all_tcb.mean():.4f}\n"
               f"Target conf after:  {all_tca.mean():.4f}\n"
               f"Target conf gain:   {(all_tca - all_tcb).mean():.4f}\n"
               f"Pred conf before:   {all_pcb.mean():.4f}\n"
               f"Pred conf after:    {all_pca.mean():.4f}\n"
               f"Pred conf drop:     {(all_pcb - all_pca).mean():.4f}\n"
               f"Margin before:      {all_mb.mean():.4f}\n"
               f"Margin after:       {all_ma.mean():.4f}\n"
               f"Margin drop:        {(all_mb - all_ma).mean():.4f}\n")
    print("\n" + summary)

    with open(os.path.join(exp_dir, "summary.txt"), "w") as f:
        f.write(summary)


if __name__ == "__main__":
    main()
