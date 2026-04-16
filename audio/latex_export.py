"""Shared helpers for writing LaTeX table fragments under report/figures/."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def latex_escape(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def figures_dir_path(figures_dir_arg: str) -> Path:
    d = Path(figures_dir_arg) if figures_dir_arg else repo_root() / "report" / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def experiment_stem_from_wav_dirs(ref_dir: str, cf_dir: str) -> str:
    """If ref/cf are sibling original_wav and cf_wav, use parent folder name."""
    rp = Path(ref_dir).resolve()
    cp = Path(cf_dir).resolve()
    if rp.name == "original_wav" and cp.name == "cf_wav" and rp.parent == cp.parent:
        return rp.parent.name
    return f"{rp.name}__{cp.name}"


def resolve_latex_out(
    latex_out: str,
    default_prefix: str,
    stem: str,
) -> Path:
    name = latex_out or f"{default_prefix}_{stem}.tex"
    if not name.endswith(".tex"):
        name += ".tex"
    # basename only — ignore path traversal in user string
    name = os.path.basename(name)
    return Path(name)
