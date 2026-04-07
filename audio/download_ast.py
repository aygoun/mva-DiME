"""
Download AST checkpoint from Hugging Face using the SDK.

Usage:
  python -m audio.download_ast
  python -m audio.download_ast --model_id MIT/ast-finetuned-audioset-10-10-0.4593 --local_dir audio/models/hf
"""

import argparse
from audio.audio_classifier import ensure_hf_model_downloaded


def main():
    parser = argparse.ArgumentParser(description="Download AST model from Hugging Face")
    parser.add_argument("--model_id", default="MIT/ast-finetuned-audioset-10-10-0.4593")
    parser.add_argument("--local_dir", default="audio/models/hf")
    args = parser.parse_args()

    path = ensure_hf_model_downloaded(args.model_id, args.local_dir)
    print(f"Downloaded model to: {path}")


if __name__ == "__main__":
    main()

