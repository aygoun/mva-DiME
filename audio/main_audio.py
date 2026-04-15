"""
Audio Counterfactual Explanations via DiME.

Generates counterfactual mel spectrograms that flip a multi-label audio
classifier's prediction for one or more target classes, using the
diffusion-guided process from DiME.

Supports sequential counterfactuals: e.g., remove 'guitar' then add 'drums'.

DDPM      : teticio/audio-diffusion-breaks-256 (diffusers, 1-ch 256×256)
Classifier: DenseAudioClassifier checkpoint (multi-label sigmoid output)
Dataset   : teticio/audio-diffusion-breaks-256 (pre-computed spectrograms)
"""

import os
import sys
import random
import argparse
import numpy as np
import soundfile as sf
import json

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
        num_classes=0,

        # sampling
        classifier_scales="5,8,12",
        seed=42,
        target_labels="", # Comma-separated list of "class_idx:direction" (0=remove, 1=add)
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
        save_intermediate=False,
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
# Target selection
# ------------------------------------------------------------------

def select_targets(logits, strategy, fixed_targets=None):
    """Pick target classes and directions.
    
    fixed_targets: list of (class_idx, direction) where direction is 0.0 or 1.0
    """
    B, num_classes = logits.shape
    device = logits.device
    probs = torch.sigmoid(logits.detach())
    
    if fixed_targets:
        # For sequential, we might want to return all targets at once or one by one.
        # Here we return a list of (targets_tensor, y_vals_tensor) for each step.
        steps = []
        for class_idx, direction in fixed_targets:
            targets = torch.full((B,), class_idx, dtype=torch.long, device=device)
            y_vals = torch.full((B,), direction, dtype=torch.float32, device=device)
            steps.append((targets, y_vals))
        return steps

    # Default strategy (single step)
    targets = torch.zeros(B, dtype=torch.long, device=device)
    y_vals = torch.ones(B, dtype=torch.float32, device=device)
    for i in range(B):
        positive = (probs[i] > 0.5).nonzero(as_tuple=True)[0].tolist()
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
    return [(targets, y_vals)]


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
    
    # Sequential targets
    fixed_targets = []
    if args.target_labels:
        for item in args.target_labels.split(","):
            c_idx, c_dir = item.split(":")
            fixed_targets.append((int(c_idx), float(c_dir)))
    
    num_steps = len(fixed_targets) if fixed_targets else 1
    
    for s in range(num_steps):
        step_dir = os.path.join(exp_dir, f"step_{s}")
        for sub in ["original_spec", "cf_spec", "diff_spec",
                     "original_wav", "cf_wav", "info"]:
            os.makedirs(os.path.join(step_dir, sub), exist_ok=True)

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
        num_classes=args.num_classes or None,
        wandb_artifact=args.wandb_artifact or None,
    ).to(device)
    classifier.eval()

    # ---- perceptual loss ----
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

    print("Starting counterfactual generation ...")
    start_time = time()
    global_idx = 0

    batch_start = start_time
    for batch_idx, (specs, indices) in enumerate(loader):
        if batch_idx >= args.num_batches:
            break
        
        specs = specs.to(device)
        current_specs = specs.clone()
        
        with torch.no_grad():
            initial_logits = classifier(specs)
        
        target_steps = select_targets(initial_logits, args.target_strategy, fixed_targets)
        
        # Track original for each step if needed, but here "original" for step > 0 
        # is the CF from previous step.
        
        for s_idx, (targets, y_vals) in enumerate(target_steps):
            step_dir = os.path.join(exp_dir, f"step_{s_idx}")
            
            with torch.no_grad():
                logits_before = classifier(current_specs)
                probs_before = torch.sigmoid(logits_before)
            
            t = torch.ones(current_specs.size(0), device=device,
                           dtype=torch.long) * args.start_step
            noise_img = diffusion.q_sample(current_specs, t)
            transformed = torch.zeros(current_specs.size(0), device=device).bool()
            cf = current_specs.clone()

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

                cfs, x_t_steps, _ = sample_fn(
                    diffusion,
                    model_fn,
                    current_specs[mask].shape,
                    args.start_step,
                    current_specs[mask],
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

                if args.save_intermediate:
                    # Save intermediate steps for Figure 3
                    for i in range(specs.size(0)):
                        idx_str = f"{global_idx + i:06d}"
                        inter_dir = os.path.join(step_dir, "intermediate", idx_str)
                        os.makedirs(inter_dir, exist_ok=True)
                        for step_idx, xt in enumerate(x_t_steps):
                            save_spectrogram_image(xt[i:i+1], os.path.join(inter_dir, f"step_{step_idx:03d}.png"))

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
                transformed[mask] = flipped

            # Evaluate this step
            with torch.no_grad():
                cf_logits = classifier(cf)
                cf_probs = torch.sigmoid(cf_logits)
            
            # Save outputs for this step
            for i in range(specs.size(0)):
                idx_str = f"{global_idx + i:06d}"
                
                if args.save_images:
                    save_spectrogram_image(
                        current_specs[i], os.path.join(step_dir, "original_spec", f"{idx_str}.png"))
                    save_spectrogram_image(
                        cf[i], os.path.join(step_dir, "cf_spec", f"{idx_str}.png"))
                    diff = (cf[i] - current_specs[i]).abs()
                    save_spectrogram_image(
                        diff, os.path.join(step_dir, "diff_spec", f"{idx_str}.png"))

                if args.save_audio:
                    try:
                        orig_wav = tensor_to_audio(current_specs[i])
                        sf.write(os.path.join(step_dir, "original_wav", f"{idx_str}.wav"),
                                 orig_wav, SAMPLE_RATE)
                        cf_wav = tensor_to_audio(cf[i])
                        sf.write(os.path.join(step_dir, "cf_wav", f"{idx_str}.wav"),
                                 cf_wav, SAMPLE_RATE)
                    except Exception as e:
                        print(f"  Warning: audio inversion failed for {idx_str}: {e}")

                l1 = (current_specs[i] - cf[i]).abs().mean().item()
                direction = "remove" if y_vals[i] < 0.5 else "add"
                
                info = {
                    "target_class": targets[i].item(),
                    "direction": direction,
                    "target_conf_before": probs_before[i, targets[i]].item(),
                    "target_conf_after": cf_probs[i, targets[i]].item(),
                    "flipped": (cf_probs[i, targets[i]] < 0.5 if direction == "remove" else cf_probs[i, targets[i]] > 0.5).item(),
                    "l1": l1,
                    "all_probs_before": probs_before[i].tolist(),
                    "all_probs_after": cf_probs[i].tolist(),
                }
                with open(os.path.join(step_dir, "info", f"{idx_str}.json"), "w") as f:
                    json.dump(info, f, indent=2)

            # Prepare for next step
            current_specs = cf.clone()

        global_idx += specs.size(0)
        now = time()
        print(f"Batch {batch_idx+1} done | total {int(now - start_time)}s")

    print(f"\nGeneration complete. Results saved to {exp_dir}")


if __name__ == "__main__":
    main()
