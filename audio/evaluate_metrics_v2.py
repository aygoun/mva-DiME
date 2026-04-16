import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio.latex_export import figures_dir_path, latex_escape, resolve_latex_out


def _build_latex_table(rows: list[dict], header_comment: str | None = None) -> str:
    """Fragment suitable for \\input{} inside a table environment."""
    lines = []
    if header_comment:
        lines.append(f"% {latex_escape(header_comment)}")
    lines.extend(
        [
            r"\begin{tabular}{crrrrrr}",
            r"\hline",
            r"Step & Samples & Flip (\%) & Mean L1 & MNAC & "
            r"$P_{\mathrm{tgt}}$ before & $P_{\mathrm{tgt}}$ after \\",
            r"\hline",
        ]
    )
    for r in rows:
        lines.append(
            f"{r['step']} & {r['n_samples']} & {r['flip_pct']:.2f} & "
            f"{r['mean_l1']:.6f} & {r['mean_mnac']:.4f} & "
            f"{r['mean_tcb']:.4f} & {r['mean_tca']:.4f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", ""])
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="Evaluate audio counterfactual results (V2)")
    p.add_argument("--exp_dir", required=True, help="Path like audio/results/<exp_name>")
    p.add_argument("--steps", type=int, default=1, help="Number of sequential steps")
    p.add_argument(
        "--figures_dir",
        type=str,
        default="",
        help="Directory for LaTeX table output (default: <repo>/report/figures)",
    )
    p.add_argument(
        "--latex_out",
        type=str,
        default="",
        help="Output .tex filename (default: eval_metrics_v2_<exp_basename>.tex)",
    )
    args = p.parse_args()

    figures_dir = figures_dir_path(args.figures_dir)

    exp_basename = os.path.basename(os.path.normpath(args.exp_dir))
    latex_name = resolve_latex_out(args.latex_out, "eval_metrics_v2", exp_basename)

    table_rows: list[dict] = []

    for s in range(args.steps):
        step_dir = os.path.join(args.exp_dir, f"step_{s}")
        info_dir = os.path.join(step_dir, "info")
        if not os.path.isdir(info_dir):
            print(f"Skipping step {s}, missing info dir.")
            continue

        json_files = sorted([x for x in os.listdir(info_dir) if x.endswith(".json")])
        if not json_files:
            continue

        flipped = []
        l1s = []
        tcb, tca = [], []
        mnac = []

        for jf in json_files:
            with open(os.path.join(info_dir, jf), "r") as f:
                d = json.load(f)

            flipped.append(1.0 if d["flipped"] else 0.0)
            l1s.append(d["l1"])
            tcb.append(d["target_conf_before"])
            tca.append(d["target_conf_after"])

            pb = np.array(d["all_probs_before"])
            pa = np.array(d["all_probs_after"])
            target_idx = d["target_class"]

            mask = np.ones_like(pb, dtype=bool)
            mask[target_idx] = False

            changed = ((pb[mask] > 0.5) != (pa[mask] > 0.5)).sum()
            mnac.append(changed)

        n = len(json_files)
        flip_pct = 100 * float(np.mean(flipped))
        mean_l1 = float(np.mean(l1s))
        mean_mnac = float(np.mean(mnac))
        mean_tcb = float(np.mean(tcb))
        mean_tca = float(np.mean(tca))

        print(f"\n--- Results for Step {s} ---")
        print(f"Samples: {n}")
        print(f"Flip rate (%): {flip_pct:.2f}")
        print(f"Mean L1: {mean_l1:.6f}")
        print(f"MNAC (Mean Number of Attributes Changed): {mean_mnac:.4f}")
        print(f"Target conf before: {mean_tcb:.4f}")
        print(f"Target conf after:  {mean_tca:.4f}")

        table_rows.append(
            {
                "step": s,
                "n_samples": n,
                "flip_pct": flip_pct,
                "mean_l1": mean_l1,
                "mean_mnac": mean_mnac,
                "mean_tcb": mean_tcb,
                "mean_tca": mean_tca,
            }
        )

    if table_rows:
        out_path = figures_dir / latex_name.name
        latex_body = _build_latex_table(
            table_rows,
            header_comment=f"eval_metrics_v2: experiment {exp_basename} (generated file)",
        )
        out_path.write_text(latex_body, encoding="utf-8")
        print(f"\nWrote LaTeX table: {out_path}")
    else:
        print("\nNo metrics to export (no step_* / info / *.json found).")


if __name__ == "__main__":
    main()
