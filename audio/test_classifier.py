"""
Test the AST classifier on spectrograms from the audio-diffusion dataset.

Usage:
    python -m audio.test_classifier
    python -m audio.test_classifier --num_samples 20 --top_k 15
    python -m audio.test_classifier --threshold 0.3
"""

import os
import sys
import csv
import argparse

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio.audio_datasets import AudioDiffusionBreaksDataset
from audio.audio_classifier import (
    build_classifier,
    ensure_hf_model_downloaded,
    AUDIOSET_LABELS_CACHE,
    AUDIOSET_LABELS_URL,
)


def load_audioset_index_to_name(cache_path=AUDIOSET_LABELS_CACHE):
    """Return ``{int_index: display_name}`` for all 527 AudioSet classes."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if not os.path.isfile(cache_path):
        import urllib.request
        print(f"Downloading AudioSet class labels -> {cache_path} ...")
        urllib.request.urlretrieve(AUDIOSET_LABELS_URL, cache_path)

    idx_to_name = {}
    with open(cache_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            idx_str, _mid, display = row[0].strip(), row[1].strip(), row[2].strip()
            idx_to_name[int(idx_str)] = display.strip('"')
    return idx_to_name


def main():
    parser = argparse.ArgumentParser(
        description="Test AST classifier on audio-diffusion spectrograms")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples to classify")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Show top-K predictions per sample")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Only show classes with P > threshold "
                             "(overrides --top_k when > 0)")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--ast_model_id", type=str,
                        default="MIT/ast-finetuned-audioset-10-10-0.4593")
    parser.add_argument("--dataset_repo", type=str,
                        default="teticio/audio-diffusion-breaks-256")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    idx_to_name = load_audioset_index_to_name()

    print("Loading classifier ...")
    ensure_hf_model_downloaded(args.ast_model_id)
    classifier = build_classifier(ast_model_id=args.ast_model_id).to(device)
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
