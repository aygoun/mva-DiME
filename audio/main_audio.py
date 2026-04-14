"""
Audio Counterfactual Explanations via DiME.

Generates counterfactual mel spectrograms that flip a multi-label audio
classifier's prediction for a **single** target class, using the
diffusion-guided process from DiME.

DDPM      : teticio/audio-diffusion-breaks-256 (diffusers, 1-ch 256×256)
Classifier: DenseAudioClassifier checkpoint (multi-label sigmoid output)
Dataset   : teticio/audio-diffusion-breaks-256 (pre-computed spectrograms)

Use-case  : remove / add a sound in music  (e.g. remove guitar, add drums).
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

from core.sample_utils import (
    get_DiME_iterative_sampling,
    clean_multilabel_cond_fn,
    dist_cond_fn,
)

from audio.cnn14_perceptual import CNN14PerceptualLoss
from audio.diffusers_wrapper import load_audio_diffusion
from audio.audio_datasets import AudioDiffusionBreaksDataset
from audio.audio_classifier import build_classifier
from dense_audio_classifier.data.irmas import INSTRUMENTS
from audio.spectrogram_utils import tensor_to_audio, SAMPLE_RATE


# ------------------------------------------------------------------
# Argument parser
# ------------------------------------------------------------------

def create_args():
    defaults = dict(
        clip_denoised=True,
        batch_size=8,
        gpu="0",
        num_batches=50,

        # paths
        output_path="audio/results",
        exp_name="audio_cf",

        # DDPM (diffusers)
        ddpm_repo="teticio/audio-diffusion-breaks-256",

        # dataset (same HF repo, or override)
        dataset_repo="teticio/audio-diffusion-breaks-256",
        max_samples=0,

        # classifier
        classifier_checkpoint_path="checkpoints/last.ckpt",
        wandb_artifact="",
        num_classes=len(INSTRUMENTS),

        # sampling
        classifier_scales="5,8,12",
        seed=42,
        target_label=-1,
        target_strategy="random_remove",
        start_step=120,
        use_logits=False,
        l1_loss=0.05,
        l2_loss=0.0,
        l_perc=10.0,
        l_perc_layer=4,
        use_sampling_on_x_t=True,
        guided_iterations=9999999,

        # output
        save_audio=True,
        save_images=True,
    )
    parser = argparse.ArgumentParser(
        description="Audio counterfactual explanations via DiME")
    for k, v in defaults.items():
        v_type = type(v)
        if isinstance(v, bool):
            parser.add_argument(f"--{k}", type=str, default=str(v))
        else:
            parser.add_argument(f"--{k}", type=v_type, default=v)
    args = parser.parse_args()

    for k in defaults:
        if isinstance(defaults[k], bool):
            val = getattr(args, k)
            setattr(args, k, val.lower() in ("true", "1", "yes"))
    return args


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
# Target selection (works without ground-truth labels)
# ------------------------------------------------------------------

def select_targets(logits, strategy, fixed_label=-1):
    """Pick one target class per sample and the desired direction.

    Uses the classifier's own sigmoid predictions as pseudo-labels:
    classes with P > 0.5 are treated as "present".

    Returns
    -------
    targets : (B,) LongTensor — class indices
    y_vals  : (B,) FloatTensor — 1.0 = add class, 0.0 = remove class
    """
    B, num_classes = logits.shape
    device = logits.device
    targets = torch.zeros(B, dtype=torch.long, device=device)
    y_vals = torch.ones(B, dtype=torch.float32, device=device)
    probs = torch.sigmoid(logits.detach())

    for i in range(B):
        positive = (probs[i] > 0.5).nonzero(as_tuple=True)[0].tolist()

        if fixed_label >= 0:
            if fixed_label >= num_classes:
                raise ValueError(
                    f"target_label={fixed_label} is out of range for classifier with "
                    f"{num_classes} classes."
                )
            targets[i] = fixed_label
            y_vals[i] = 0.0 if fixed_label in positive else 1.0
            continue

        if strategy == "random_remove" and positive:
            c = random.choice(positive)
            targets[i] = c
            y_vals[i] = 0.0
        elif strategy == "least_confident_remove" and positive:
            confs = [(c, probs[i, c].item()) for c in positive]
            c = min(confs, key=lambda x: x[1])[0]
            targets[i] = c
            y_vals[i] = 0.0
        elif strategy == "random_add":
            negative = [c for c in range(num_classes) if c not in positive]
            if negative:
                c = random.choice(negative)
                targets[i] = c
                y_vals[i] = 1.0
            elif positive:
                targets[i] = random.choice(positive)
                y_vals[i] = 0.0
        else:
            if positive:
                c = random.choice(positive)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
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
    dataset = AudioDiffusionBreaksDataset(
        repo_id=args.dataset_repo,
        max_samples=args.max_samples if args.max_samples > 0 else None,
    )
    loader = data.DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    # ---- DDPM (diffusers) ----
    model_fn, diffusion = load_audio_diffusion(
        repo_id=args.ddpm_repo, device=device)

    # ---- classifier ----
    print("Loading classifier ...")
    classifier = build_classifier(
        checkpoint_path=args.classifier_checkpoint_path,
        num_classes=args.num_classes,
        wandb_artifact=args.wandb_artifact or None,
    ).to(device)
    classifier.eval()

    # ---- perceptual loss (CNN14 — native 1-channel) ----
    if args.l_perc != 0:
        print("Loading perceptual loss (CNN14) ...")
        perceptual_loss = CNN14PerceptualLoss(
            layer=args.l_perc_layer, c=args.l_perc,
        ).to(device)
    else:
        perceptual_loss = None

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

    batch_start = start_time
    for batch_idx, (specs, indices) in enumerate(loader):
        if batch_idx >= args.num_batches:
            break
        now = time()
        elapsed = int(now - start_time)
        batch_time = int(now - batch_start)
        batch_start = now
        print(f"\nBatch {batch_idx+1}/{min(args.num_batches, len(loader))} "
              f"| total {elapsed}s"
              + (f" | last batch {batch_time}s" if batch_idx > 0 else ""))

        specs = specs.to(device)

        # ---- classifier prediction (serves as pseudo-ground-truth) ----
        with torch.no_grad():
            logits = classifier(specs)

        # ---- choose target ----
        targets, y_vals = select_targets(
            logits, args.target_strategy, fixed_label=args.target_label,
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
                    "l_perc": perceptual_loss,
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
                    orig_wav = tensor_to_audio(specs[i])
                    sf.write(os.path.join(exp_dir, "original_wav", f"{idx_str}.wav"),
                             orig_wav, SAMPLE_RATE)
                    cf_wav = tensor_to_audio(cf[i])
                    sf.write(os.path.join(exp_dir, "cf_wav", f"{idx_str}.wav"),
                             cf_wav, SAMPLE_RATE)
                except Exception as e:
                    print(f"  Warning: audio inversion failed for {idx_str}: {e}")

            direction = "remove" if y_vals[i] < 0.5 else "add"
            info = (
                f"target_class: {targets[i].item()}\n"
                f"direction: {direction}\n"
                f"target_conf_before: {target_conf_before[i].item():.4f}\n"
                f"target_conf_after: {target_conf_after[i].item():.4f}\n"
                f"flipped: {flipped_final[i].item()}\n"
                f"l1: {l1[i].item():.4f}\n"
                f"l2: {l2[i].item():.4f}\n"
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
        f"DDPM: {args.ddpm_repo}\n"
        f"Dataset: {args.dataset_repo}\n"
        f"Classifier checkpoint: {args.classifier_checkpoint_path}\n"
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
