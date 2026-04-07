import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio.audio_datasets import ESC50Dataset
from audio.audio_classifier import build_classifier, ensure_hf_model_downloaded
from audio.spectrogram_utils import mixup


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune classifier on ESC-50 spectrograms")
    p.add_argument("--data_dir", required=True, help="Path to ESC-50-master/")
    p.add_argument("--output_path", default="audio/models/classifier.pth")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--mixup_alpha", type=float, default=0.4,
                   help="Mixup interpolation strength (0 = disabled)")
    p.add_argument("--val_fold", type=int, default=5,
                   help="ESC-50 fold held out for validation (1-5)")
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--classifier_type", type=str, default="resnet18",
                   choices=["resnet18", "ast"])
    p.add_argument("--ast_model_id", type=str,
                   default="MIT/ast-finetuned-audioset-10-10-0.4593")
    p.add_argument("--freeze_backbone_epochs", type=int, default=10,
                   help="For AST this keeps only the ESC-50 head trainable initially")
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

    all_folds = [1, 2, 3, 4, 5]
    train_folds = [f for f in all_folds if f != args.val_fold]

    train_ds = ESC50Dataset(args.data_dir, folds=train_folds, augment=True)
    val_ds = ESC50Dataset(args.data_dir, folds=[args.val_fold], augment=False)
    num_classes = train_ds.num_classes

    use_pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=use_pin,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=use_pin)

    if args.classifier_type == "ast":
        print(f"Downloading/loading AST backbone: {args.ast_model_id}")
        ensure_hf_model_downloaded(args.ast_model_id)

    model = build_classifier(
        classifier_type=args.classifier_type,
        num_classes=num_classes,
        weights_path=None,
        ast_model_id=args.ast_model_id,
    ).to(device)

    # freeze early layers for the first few epochs, then unfreeze
    if args.classifier_type == "resnet18":
        for name, param in model.backbone.named_parameters():
            if not name.startswith(("layer3", "layer4", "fc")):
                param.requires_grad = False
    else:
        # AST: start by training only the ESC-50 head.
        for param in model.backbone.parameters():
            param.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    unfreeze_epoch = min(args.freeze_backbone_epochs, args.epochs)
    best_acc = 0.0
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    if device.type == "mps":
        torch.mps.empty_cache()

    epoch_bar = tqdm(range(1, args.epochs + 1), desc="Epochs")
    for epoch in epoch_bar:
        # unfreeze all layers after warmup
        if epoch == unfreeze_epoch and unfreeze_epoch > 0:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr * 0.3,
                weight_decay=args.weight_decay,
            )
            print(f"  ↳ unfreezing full {args.classifier_type}, lr → {args.lr * 0.3:.1e}")

        # cosine LR decay (restarted after unfreeze)
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

        # train
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for specs, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            specs, labels = specs.to(device), labels.to(device)

            # mixup
            if args.mixup_alpha > 0:
                idx = torch.randperm(specs.size(0), device=device)
                specs_b, labels_b = specs[idx], labels[idx]
                specs_mix, y_a, y_b, lam = mixup(specs, labels, specs_b, labels_b,
                                                  alpha=args.mixup_alpha)
                logits = model(specs_mix)
                loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
            else:
                logits = model(specs)
                loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * specs.size(0)
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += specs.size(0)

        train_acc = 100.0 * correct / total

        # validate
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for specs, labels, _ in val_loader:
                specs, labels = specs.to(device), labels.to(device)
                logits = model(specs)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += specs.size(0)

        val_acc = 100.0 * val_correct / val_total
        epoch_bar.set_postfix(lr=f"{lr:.1e}", loss=f"{running_loss/total:.4f}",
                              train=f"{train_acc:.1f}%", val=f"{val_acc:.1f}%")
        tqdm.write(f"Epoch {epoch:3d}/{args.epochs}  lr={lr:.1e}  "
                   f"train_loss={running_loss/total:.4f}  "
                   f"train_acc={train_acc:.1f}%  val_acc={val_acc:.1f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.output_path)
            tqdm.write(f"  → saved best model (val_acc={val_acc:.1f}%)")

    print(f"\nDone. Best validation accuracy: {best_acc:.1f}%")
    print(f"Checkpoint saved to: {args.output_path}")


if __name__ == "__main__":
    main()
