"""
Optional fine-tuning of the pretrained AST on FSD50K with BCE loss.

The pretrained AST already covers FSD50K's 200 classes (subset of AudioSet
527).  Fine-tuning can improve per-class calibration but is not required for
DiME guidance.
"""

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio.audio_datasets import FSD50KDataset
from audio.audio_classifier import (
    build_classifier,
    ensure_hf_model_downloaded,
    load_audioset_class_mapping,
)


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune AST on FSD50K spectrograms")
    p.add_argument("--data_dir", required=True, help="Path to FSD50K root")
    p.add_argument("--output_path", default="audio/models/classifier_fsd50k.pth")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--mixup_alpha", type=float, default=0.4,
                   help="Mixup interpolation strength (0 = disabled)")
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--ast_model_id", type=str,
                   default="MIT/ast-finetuned-audioset-10-10-0.4593")
    p.add_argument("--freeze_backbone_epochs", type=int, default=10,
                   help="Keep only the classifier head trainable initially")
    p.add_argument("--audio_duration", type=float, default=7.0)
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # ---- dataset ----
    mid_to_idx = load_audioset_class_mapping()
    train_ds = FSD50KDataset(
        args.data_dir, split="train", augment=True,
        audioset_mid_to_idx=mid_to_idx, duration=args.audio_duration,
    )
    val_ds = FSD50KDataset(
        args.data_dir, split="val", augment=False,
        audioset_mid_to_idx=mid_to_idx, duration=args.audio_duration,
    )
    print(f"FSD50K: {len(train_ds)} train, {len(val_ds)} val, "
          f"{train_ds.num_fsd50k_classes} FSD50K classes, "
          f"{train_ds.num_classes} model outputs")

    use_pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=use_pin,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=use_pin)

    # ---- model ----
    print(f"Downloading/loading AST backbone: {args.ast_model_id}")
    ensure_hf_model_downloaded(args.ast_model_id)
    model = build_classifier(ast_model_id=args.ast_model_id).to(device)

    # freeze backbone initially
    for param in model.model.audio_spectrogram_transformer.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()

    unfreeze_epoch = min(args.freeze_backbone_epochs, args.epochs)
    best_metric = 0.0
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    if device.type == "mps":
        torch.mps.empty_cache()

    epoch_bar = tqdm(range(1, args.epochs + 1), desc="Epochs")
    for epoch in epoch_bar:
        if epoch == unfreeze_epoch and unfreeze_epoch > 0:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr * 0.3,
                weight_decay=args.weight_decay,
            )
            print(f"  -> unfreezing full model, lr -> {args.lr * 0.3:.1e}")

        if epoch >= unfreeze_epoch:
            base_epochs = max(args.epochs - unfreeze_epoch, 1)
            progress = (epoch - unfreeze_epoch) / base_epochs
        else:
            base_epochs = max(unfreeze_epoch, 1)
            progress = epoch / base_epochs
        lr = (args.lr if epoch < unfreeze_epoch else args.lr * 0.3) * \
             (0.5 * (1 + np.cos(np.pi * progress)))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ---- train ----
        model.train()
        running_loss, total = 0.0, 0

        for specs, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            specs, labels = specs.to(device), labels.to(device)

            if args.mixup_alpha > 0:
                idx = torch.randperm(specs.size(0), device=device)
                lam = np.random.beta(args.mixup_alpha, args.mixup_alpha)
                specs_mix = lam * specs + (1 - lam) * specs[idx]
                labels_mix = lam * labels + (1 - lam) * labels[idx]
                logits = model(specs_mix)
                loss = criterion(logits, labels_mix)
            else:
                logits = model(specs)
                loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * specs.size(0)
            total += specs.size(0)

        # ---- validate (mAP on FSD50K classes) ----
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for specs, labels, _ in val_loader:
                specs, labels = specs.to(device), labels.to(device)
                logits = model(specs)
                all_preds.append(torch.sigmoid(logits).cpu())
                all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        valid_idx = torch.tensor(train_ds.valid_audioset_indices, dtype=torch.long)
        if len(valid_idx) > 0:
            preds_sub = all_preds[:, valid_idx]
            labels_sub = all_labels[:, valid_idx]
        else:
            preds_sub = all_preds
            labels_sub = all_labels

        from sklearn.metrics import average_precision_score
        try:
            val_metric = average_precision_score(
                labels_sub.numpy(), preds_sub.numpy(), average="macro",
            )
        except Exception:
            val_metric = 0.0

        epoch_bar.set_postfix(lr=f"{lr:.1e}", loss=f"{running_loss/total:.4f}",
                              mAP=f"{val_metric:.4f}")
        tqdm.write(f"Epoch {epoch:3d}/{args.epochs}  lr={lr:.1e}  "
                   f"train_loss={running_loss/total:.4f}  "
                   f"val_mAP={val_metric:.4f}")

        if val_metric > best_metric:
            best_metric = val_metric
            torch.save(model.state_dict(), args.output_path)
            tqdm.write(f"  -> saved best model (val_mAP={val_metric:.4f})")

    print(f"\nDone. Best validation mAP: {best_metric:.4f}")
    print(f"Checkpoint saved to: {args.output_path}")


if __name__ == "__main__":
    main()
