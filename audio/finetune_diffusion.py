"""
Fine-tune a pretrained image DDPM on mel-spectrogram data (Option B).

Loads the CelebA (or any other) DDPM checkpoint and continues training on
spectrogram tensors.  No MPI / distributed setup required — runs on a single
GPU.
"""

import os
import sys
import copy
import argparse

import torch
from torch.optim import AdamW
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    add_dict_to_argparser,
    args_to_dict,
)
from core.resample import create_named_schedule_sampler
from core.nn import update_ema
from core.sample_utils import load_from_DDP_model

from audio.audio_datasets import ESC50Dataset, infinite_audio_loader


def create_argparser():
    defaults = dict(
        # data
        data_dir="",
        data_mode="folder",        # "folder" or "esc50"
        audio_duration=5.0,
        # training
        pretrained_path="models/ddpm-celeba.pt",
        output_dir="audio/models/ddpm-spectro",
        steps=15000,
        batch_size=16,
        lr=2e-5,
        weight_decay=0.05,
        ema_rate=0.9999,
        save_interval=5000,
        log_interval=100,
        gpu="0",
        schedule_sampler="uniform",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser(
        description="Fine-tune a pretrained DDPM on mel spectrograms")
    add_dict_to_argparser(parser, defaults)
    return parser


def main():
    args = create_argparser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # model + diffusion
    print("Creating model and diffusion ...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    # load pretrained weights
    print(f"Loading pretrained checkpoint from {args.pretrained_path} ...")
    state_dict = torch.load(args.pretrained_path, map_location="cpu")
    state_dict = load_from_DDP_model(state_dict)
    model.load_state_dict(state_dict)
    model.to(device)
    model.train()
    print("  done")

    # EMA copy
    ema_rate = args.ema_rate
    ema_params = copy.deepcopy(list(model.parameters()))

    # data
    print("Creating data loader ...")
    if args.data_mode == "esc50":
        dataset = ESC50Dataset(args.data_dir, duration=args.audio_duration,
                               size=args.image_size)
    else:
        raise ValueError(f"Invalid data mode: {args.data_mode}")
    data_iter = infinite_audio_loader(dataset, batch_size=args.batch_size)
    print(f"  {len(dataset)} audio clips")

    # optimizer + sampler
    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler,
                                                     diffusion)

    # training loop
    if device.type == "mps":
        torch.mps.empty_cache()

    print(f"Fine-tuning for {args.steps} steps ...")
    pbar = tqdm(range(1, args.steps + 1), desc="Fine-tune DDPM")
    for step in pbar:
        batch, cond = next(data_iter)
        batch = batch.to(device)
        cond = {k: v.to(device) for k, v in cond.items()} if isinstance(cond, dict) else {}

        t, weights = schedule_sampler.sample(batch.shape[0], device)
        losses = diffusion.training_losses(model, batch, t, model_kwargs=cond)
        loss = (losses["loss"] * weights).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for targ, src in zip(ema_params, model.parameters()):
            targ.data.mul_(ema_rate).add_(src.data, alpha=1 - ema_rate)

        pbar.set_postfix(loss=f"{loss.item():.4f}")

        if step % args.save_interval == 0 or step == args.steps:
            # save weights
            ckpt_path = os.path.join(args.output_dir, f"model{step:06d}.pt")
            torch.save(model.state_dict(), ckpt_path)

            # save EMA weights
            ema_state = {
                name: param.data.clone()
                for (name, _), param in zip(model.named_parameters(), ema_params)
            }
            ema_path = os.path.join(args.output_dir, f"ema_{ema_rate}_{step:06d}.pt")
            torch.save(ema_state, ema_path)
            tqdm.write(f"  → saved checkpoint and EMA to {args.output_dir}")

    print("Fine-tuning complete.")


if __name__ == "__main__":
    main()
