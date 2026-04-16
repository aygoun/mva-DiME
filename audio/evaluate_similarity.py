import argparse
import os
import sys

import librosa
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio.latex_export import (
    experiment_stem_from_wav_dirs,
    figures_dir_path,
    latex_escape,
    resolve_latex_out,
)


def _build_latex_table(n: int, mean_sim: float, model: str, header_comment: str | None) -> str:
    lines = []
    if header_comment:
        lines.append(f"% {latex_escape(header_comment)}")
    lines.extend(
        [
            r"\begin{tabular}{lcc}",
            r"\hline",
            r"Pairs & Mean cosine similarity & Embedding model \\",
            r"\hline",
            f"{n} & {mean_sim:.4f} & {latex_escape(model)} \\\\",
            r"\hline",
            r"\end{tabular}",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compute Source Preservation/Similarity")
    parser.add_argument("--cf_dir", type=str, required=True, help="Directory with cf_wav")
    parser.add_argument("--ref_dir", type=str, required=True, help="Directory with original_wav")
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/unispeech-sat-base-plus-sv",
        help="Model for speaker embeddings",
    )
    parser.add_argument(
        "--figures_dir",
        type=str,
        default="",
        help="Directory for LaTeX output (default: <repo>/report/figures)",
    )
    parser.add_argument(
        "--latex_out",
        type=str,
        default="",
        help="Output .tex filename (default: eval_similarity_<exp_stem>.tex)",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    extractor = AutoFeatureExtractor.from_pretrained(args.model)
    model = AutoModelForAudioClassification.from_pretrained(
        args.model, ignore_mismatched_sizes=True
    ).to(device)
    model.eval()

    wav_files = sorted([x for x in os.listdir(args.ref_dir) if x.endswith(".wav")])
    scores = []

    for fn in wav_files:
        p0 = os.path.join(args.ref_dir, fn)
        p1 = os.path.join(args.cf_dir, fn)
        if not os.path.exists(p1):
            continue

        try:
            w0, _sr0 = librosa.load(p0, sr=16000)
            w1, _sr1 = librosa.load(p1, sr=16000)

            inputs0 = extractor(w0, sampling_rate=16000, return_tensors="pt").to(device)
            inputs1 = extractor(w1, sampling_rate=16000, return_tensors="pt").to(device)

            with torch.no_grad():
                emb0 = model(**inputs0).logits
                emb1 = model(**inputs1).logits

                sim = torch.nn.functional.cosine_similarity(emb0, emb1).item()
                scores.append(sim)
        except Exception as e:
            print(f"Error processing {fn}: {e}")

    if scores:
        mean_s = float(np.mean(scores))
        n = len(scores)
        print(f"Mean Source Similarity (Cosine via {args.model}): {mean_s:.4f}")

        stem = experiment_stem_from_wav_dirs(args.ref_dir, args.cf_dir)
        out_dir = figures_dir_path(args.figures_dir)
        tex_name = resolve_latex_out(args.latex_out, "eval_similarity", stem)
        out_path = out_dir / tex_name
        body = _build_latex_table(
            n,
            mean_s,
            args.model,
            header_comment=f"evaluate_similarity: {stem} (generated)",
        )
        out_path.write_text(body, encoding="utf-8")
        print(f"Wrote LaTeX table: {out_path}")
    else:
        print("No files found or all failed.")


if __name__ == "__main__":
    main()
