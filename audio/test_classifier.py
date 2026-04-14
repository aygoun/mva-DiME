"""Test a DenseAudioClassifier checkpoint on mel spectrogram samples."""

import os
import sys
import argparse

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio.audio_datasets import AudioDiffusionBreaksDataset
from audio.audio_classifier import build_classifier
from dense_audio_classifier.data.irmas import INSTRUMENTS


def main():
    parser = argparse.ArgumentParser(
        description="Test DenseAudioClassifier on audio-diffusion spectrograms")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples to classify")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Show top-K predictions per sample")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Only show classes with P > threshold "
                             "(overrides --top_k when > 0)")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--checkpoint_path", type=str, default="",
                        help="Path to DenseAudioClassifier checkpoint (.ckpt)")
    parser.add_argument("--wandb_artifact", type=str, default="",
                        help="W&B artifact reference: entity/project/artifact:version")
    parser.add_argument("--num_classes", type=int, default=len(INSTRUMENTS),
                        help="Number of output classes in checkpoint")
    parser.add_argument("--dataset_repo", type=str,
                        default="teticio/audio-diffusion-breaks-256")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    idx_to_name = {i: name for i, name in enumerate(INSTRUMENTS)}

    print("Loading classifier ...")
    classifier = build_classifier(
        checkpoint_path=args.checkpoint_path or None,
        num_classes=args.num_classes,
        wandb_artifact=args.wandb_artifact or None,
    ).to(device)
    classifier.eval()

    dataset = AudioDiffusionBreaksDataset(
        repo_id=args.dataset_repo,
        max_samples=args.num_samples,
    )

    print(f"\nClassifying {len(dataset)} spectrograms ...\n")
    print("=" * 72)

    for i in range(len(dataset)):
        tensor, idx = dataset[i]
        tensor = tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = classifier(tensor)
        probs = torch.sigmoid(logits[0]).cpu()

        sorted_probs, sorted_idx = probs.sort(descending=True)

        print(f"\nSample {i} (dataset index {idx}):")

        if args.threshold > 0:
            mask = sorted_probs > args.threshold
            if mask.sum() == 0:
                print(f"  (no class above threshold {args.threshold})")
                top_n = min(3, len(sorted_probs))
                print(f"  Top {top_n} anyway:")
                for j in range(top_n):
                    cls = sorted_idx[j].item()
                    p = sorted_probs[j].item()
                    name = idx_to_name.get(cls, f"class_{cls}")
                    print(f"    [{cls:3d}] {name:40s}  P={p:.4f}")
            else:
                count = mask.sum().item()
                print(f"  {count} class(es) above P > {args.threshold}:")
                for j in range(count):
                    cls = sorted_idx[j].item()
                    p = sorted_probs[j].item()
                    name = idx_to_name.get(cls, f"class_{cls}")
                    print(f"    [{cls:3d}] {name:40s}  P={p:.4f}")
        else:
            top_k = min(args.top_k, len(sorted_probs))
            above_05 = (sorted_probs > 0.5).sum().item()
            print(f"  Classes with P > 0.5: {above_05}")
            print(f"  Top {top_k}:")
            for j in range(top_k):
                cls = sorted_idx[j].item()
                p = sorted_probs[j].item()
                name = idx_to_name.get(cls, f"class_{cls}")
                marker = " *" if p > 0.5 else ""
                print(f"    [{cls:3d}] {name:40s}  P={p:.4f}{marker}")

    print("\n" + "=" * 72)
    print("Done. Classes marked with * have P > 0.5 (pseudo-positive).")


if __name__ == "__main__":
    main()
