"""
Post-hoc evaluation metrics for audio counterfactual runs.

Reads an experiment folder produced by `audio.main_audio` and computes:
- Flip rate / success rate
- Proximity: mean L1 / L2 on spectrogram tensors
- Confidence shifts: target gain, original-prediction drop, margin drop
- Audio metrics (if wav files are present):
  - SNR (dB): higher means smaller perturbation
  - Log-spectral distance (LSD): lower is better

Use-case (flat layout, ``info/*.txt`` from ``main_audio``)
---------------------------------------------------------
From the repo root, point ``--exp_dir`` at the folder that directly contains
``info/``, ``original_wav/``, and ``cf_wav/``::

    python audio/evaluate_metrics.py \\
        --exp_dir audio/results/audio_cf \\
        --compute_audio_metrics

WAV paths (same parent as ``info/``)::

    audio/results/audio_cf/original_wav/*.wav
    audio/results/audio_cf/cf_wav/*.wav

Multi-step runs (e.g. ``guitar_removal_hq/step_0`` … ``step_5``) store
``info/*.json``, not ``.txt``. Use ``audio/evaluate_metrics_v2.py`` for those;
this script is only for the legacy flat export from ``main_audio``.
"""

import os
import argparse
import numpy as np
import soundfile as sf
import librosa


def parse_info_file(path):
    out = {}
    with open(path, "r") as f:
        for line in f:
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def compute_snr_db(x, y, eps=1e-8):
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]
    noise = y - x
    p_signal = np.mean(x ** 2)
    p_noise = np.mean(noise ** 2)
    return 10 * np.log10((p_signal + eps) / (p_noise + eps))


def compute_lsd(x, y, sr, n_fft=1024, hop_length=512, eps=1e-8):
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]
    sx = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length)) + eps
    sy = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) + eps
    lx = 20 * np.log10(sx)
    ly = 20 * np.log10(sy)
    return np.mean(np.sqrt(np.mean((lx - ly) ** 2, axis=0)))


def safe_float(d, key, default=np.nan):
    try:
        return float(d[key])
    except Exception:
        return default


def safe_bool(d, key, default=False):
    try:
        v = d[key].strip().lower()
        return v in ("true", "1", "yes")
    except Exception:
        return default


def main():
    p = argparse.ArgumentParser(description="Evaluate audio counterfactual results")
    p.add_argument("--exp_dir", required=True, help="Path like audio/results/<exp_name>")
    p.add_argument("--compute_audio_metrics", action="store_true",
                   help="Also compute SNR/LSD from original_wav and cf_wav")
    args = p.parse_args()

    info_dir = os.path.join(args.exp_dir, "info")
    if not os.path.isdir(info_dir):
        raise FileNotFoundError(f"Missing info directory: {info_dir}")

    info_files = sorted([x for x in os.listdir(info_dir) if x.endswith(".txt")])
    if len(info_files) == 0:
        raise RuntimeError(f"No info files found in {info_dir}")

    flipped = []
    l1s, l2s = [], []
    tcb, tca = [], []
    pcb, pca = [], []
    mb, ma = [], []

    for fn in info_files:
        d = parse_info_file(os.path.join(info_dir, fn))
        flipped.append(1.0 if safe_bool(d, "flipped") else 0.0)
        l1s.append(safe_float(d, "l1"))
        l2s.append(safe_float(d, "l2"))
        tcb.append(safe_float(d, "target_conf_before"))
        tca.append(safe_float(d, "target_conf_after"))
        pcb.append(safe_float(d, "pred_conf_before"))
        pca.append(safe_float(d, "pred_conf_after"))
        mb.append(safe_float(d, "margin_before"))
        ma.append(safe_float(d, "margin_after"))

    flipped = np.array(flipped)
    l1s = np.array(l1s, dtype=np.float64)
    l2s = np.array(l2s, dtype=np.float64)
    tcb = np.array(tcb, dtype=np.float64)
    tca = np.array(tca, dtype=np.float64)
    pcb = np.array(pcb, dtype=np.float64)
    pca = np.array(pca, dtype=np.float64)
    mb = np.array(mb, dtype=np.float64)
    ma = np.array(ma, dtype=np.float64)

    lines = []
    lines.append(f"Samples: {len(info_files)}")
    lines.append(f"Flip rate (%): {100 * np.nanmean(flipped):.2f}")
    lines.append(f"Mean L1: {np.nanmean(l1s):.6f}")
    if np.isfinite(l2s).any():
        lines.append(f"Mean L2: {np.nanmean(l2s):.6f}")
    if np.isfinite(tcb).any() and np.isfinite(tca).any():
        lines.append(f"Target conf before: {np.nanmean(tcb):.6f}")
        lines.append(f"Target conf after:  {np.nanmean(tca):.6f}")
        lines.append(f"Target conf gain:   {np.nanmean(tca - tcb):.6f}")
    if np.isfinite(pcb).any() and np.isfinite(pca).any():
        lines.append(f"Pred conf before:   {np.nanmean(pcb):.6f}")
        lines.append(f"Pred conf after:    {np.nanmean(pca):.6f}")
        lines.append(f"Pred conf drop:     {np.nanmean(pcb - pca):.6f}")
    if np.isfinite(mb).any() and np.isfinite(ma).any():
        lines.append(f"Margin before:      {np.nanmean(mb):.6f}")
        lines.append(f"Margin after:       {np.nanmean(ma):.6f}")
        lines.append(f"Margin drop:        {np.nanmean(mb - ma):.6f}")

    if args.compute_audio_metrics:
        ow = os.path.join(args.exp_dir, "original_wav")
        cw = os.path.join(args.exp_dir, "cf_wav")
        if os.path.isdir(ow) and os.path.isdir(cw):
            wav_files = sorted([x for x in os.listdir(ow) if x.endswith(".wav")])
            snrs, lsds = [], []
            for fn in wav_files:
                p0 = os.path.join(ow, fn)
                p1 = os.path.join(cw, fn)
                if not os.path.exists(p1):
                    continue
                try:
                    x, sr0 = sf.read(p0, dtype="float32")
                    y, sr1 = sf.read(p1, dtype="float32")
                    if sr0 != sr1:
                        y = librosa.resample(y, orig_sr=sr1, target_sr=sr0)
                    if x.ndim > 1:
                        x = x.mean(axis=1)
                    if y.ndim > 1:
                        y = y.mean(axis=1)
                    snrs.append(compute_snr_db(x, y))
                    lsds.append(compute_lsd(x, y, sr0))
                except Exception:
                    continue

            if len(snrs) > 0:
                lines.append(f"Audio SNR (dB): {np.mean(snrs):.4f}")
            if len(lsds) > 0:
                lines.append(f"Audio LSD:      {np.mean(lsds):.4f}")

    report = "\n".join(lines) + "\n"
    print(report)

    out_path = os.path.join(args.exp_dir, "metrics_report.txt")
    with open(out_path, "w") as f:
        f.write(report)
    print(f"Saved metrics report to: {out_path}")


if __name__ == "__main__":
    main()

