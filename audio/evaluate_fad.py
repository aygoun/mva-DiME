import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio.latex_export import (
    experiment_stem_from_wav_dirs,
    figures_dir_path,
    latex_escape,
    resolve_latex_out,
)


def _build_latex_table(model_name: str, fad_score: float, header_comment: str | None) -> str:
    lines = []
    if header_comment:
        lines.append(f"% {latex_escape(header_comment)}")
    lines.extend(
        [
            r"\begin{tabular}{lc}",
            r"\hline",
            r"Embedding model & FAD \\",
            r"\hline",
            f"{latex_escape(model_name)} & {fad_score:.4f} \\\\",
            r"\hline",
            r"\end{tabular}",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compute FAD for audio CF")
    parser.add_argument("--cf_dir", type=str, required=True, help="Directory with cf_wav")
    parser.add_argument(
        "--ref_dir",
        type=str,
        required=True,
        help="Directory with original_wav or real dataset samples",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="vggish",
        choices=["vggish", "pann", "clap"],
        help="Embedding model for FAD",
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
        help="Output .tex filename (default: eval_fad_<exp_stem>.tex)",
    )
    args = parser.parse_args()

    sys.argv = [sys.argv[0]]
    from frechet_audio_distance import FrechetAudioDistance

    fad = FrechetAudioDistance(model_name=args.model, verbose=False)

    score = fad.score(args.ref_dir, args.cf_dir)
    print(f"FAD ({args.model}) score: {score:.4f}")

    stem = experiment_stem_from_wav_dirs(args.ref_dir, args.cf_dir)
    out_dir = figures_dir_path(args.figures_dir)
    tex_name = resolve_latex_out(args.latex_out, "eval_fad", stem)
    out_path = out_dir / tex_name
    body = _build_latex_table(
        args.model,
        float(score),
        header_comment=f"evaluate_fad: {stem} (generated)",
    )
    out_path.write_text(body, encoding="utf-8")
    print(f"Wrote LaTeX table: {out_path}")


if __name__ == "__main__":
    main()
