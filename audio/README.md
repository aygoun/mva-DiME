# DiME for Audio — Counterfactual Explanations on Spectrograms

This module extends DiME to the **audio modality**. It generates counterfactual mel spectrograms that flip a multi-label audio classifier's prediction for a single target class, using the same diffusion-guided process from the original paper.

**Use-case:** remove a sound from a music clip (e.g. remove the guitar) or transform one sound into another (e.g. guitar → drums) using classifier-guided counterfactual generation.

## Components

| Component | Solution | Training needed? |
|-----------|----------|-----------------|
| Diffusion model | [`teticio/audio-diffusion-breaks-256`](https://huggingface.co/teticio/audio-diffusion-breaks-256) | None (pretrained) |
| Dataset | [`teticio/audio-diffusion-breaks-256`](https://huggingface.co/datasets/teticio/audio-diffusion-breaks-256) — 30k breakbeat spectrograms | None (auto-download) |
| Classifier | `DenseAudioClassifier` checkpoint (`.ckpt`) trained on IRMAS/OpenMIC | Yes (or download artifact) |
| Spectrogram codec | [`diffusers.pipelines.deprecated.audio_diffusion.Mel`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/deprecated/audio_diffusion/mel.py) | — |
| Perceptual loss | PANNs CNN14 — audio-native, 1-channel ([Zenodo](https://zenodo.org/record/3987831)) | None (pretrained, auto-download) |
| Audio inversion | Griffin-Lim via the `Mel` class | — |

## Spectrogram parameters

Inherited from the `Mel` class — identical to the audio-diffusion training pipeline:

| Parameter | Value |
|-----------|-------|
| Sample rate | 22 050 Hz |
| `n_fft` | 2048 |
| `hop_length` | 512 |
| `n_mels` | 256 |
| Resolution | 256 × 256, 1-channel (grayscale) |
| `top_db` | 80 |
| Audio slice | 131 071 samples (~5.94 s) |
| Normalization | Fixed: `byte = ((dB + 80) × 255 / 80)` |

## Setup

```bash
uv sync --extra audio
```

Diffusion model and dataset auto-download on first run.  
Classifier must be provided as either:

- local checkpoint via `--classifier_checkpoint_path`
- W&B artifact via `--wandb_artifact` or `wandb://...` URI

## Run

```bash
python -W ignore -m audio.main_audio \
    --ddpm_repo teticio/audio-diffusion-breaks-256 \
    --dataset_repo teticio/audio-diffusion-breaks-256 \
    --classifier_checkpoint_path checkpoints/last.ckpt \
    --max_samples 200 \
    --output_path audio/results --exp_name breaks_demo \
    --target_strategy random_remove \
    --start_step 120 --classifier_scales '5,8,12' \
    --num_batches 5 --gpu 0
```

Using W&B artifact instead of local checkpoint:

```bash
python -W ignore -m audio.main_audio \
    --ddpm_repo teticio/audio-diffusion-breaks-256 \
    --dataset_repo teticio/audio-diffusion-breaks-256 \
    --classifier_checkpoint_path "" \
    --wandb_artifact mva-altegrad-challenge/xai-dime/train_dense_classifier_openmic:v0 \
    --max_samples 200 --num_batches 5 --gpu 0
```

Or use the helper script:

```bash
bash audio/test_audio.sh
```

## How target selection works

Since the dataset has **no labels**, the classifier's own predictions serve as pseudo-ground-truth. Classes with `sigmoid > 0.5` are treated as "present".

| Strategy | Direction | Description |
|----------|-----------|-------------|
| `random_remove` | remove | Pick a random detected class and minimize its probability |
| `least_confident_remove` | remove | Pick the least confident detected class |
| `random_add` | add | Pick a random absent class and maximize its probability |

Use `--target_label <class_idx>` to force a specific classifier class for all samples.

## Output

```
audio/results/breaks_demo/
├── original_spec/   # Input mel spectrogram images
├── cf_spec/         # Counterfactual spectrogram images
├── diff_spec/       # |original - counterfactual| heatmaps
├── original_wav/    # Reconstructed input audio (Griffin-Lim)
├── cf_wav/          # Reconstructed counterfactual audio
├── info/            # Per-sample metadata
└── summary.txt      # Flip rate, L1/L2, confidence shifts
```

## Key arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--ddpm_repo` | `teticio/audio-diffusion-breaks-256` | HF repo for the diffusers DDPM |
| `--dataset_repo` | `teticio/audio-diffusion-breaks-256` | HF dataset repo |
| `--classifier_checkpoint_path` | `checkpoints/last.ckpt` | Local checkpoint path, or `wandb://...` URI |
| `--wandb_artifact` | `""` | W&B artifact ref: `entity/project/artifact:version` |
| `--num_classes` | `0` (auto) | Number of output classes (0 = auto-detect from checkpoint) |
| `--max_samples` | `0` (all) | Cap dataset size for quick tests |
| `--target_label` | `-1` | Fixed classifier class index, or `-1` for auto |
| `--target_strategy` | `random_remove` | See table above |
| `--classifier_scales` | `5,8,12` | Gradient scales, tried in order |
| `--start_step` | `120` | Noise depth τ (out of 1000 timesteps) |
| `--l_perc` | `10.0` | CNN14 perceptual loss weight |
| `--l_perc_layer` | `4` | CNN14 conv blocks for features (1–6) |
| `--l1_loss` | `0.05` | L1 proximity loss weight |

## File structure

```
audio/
├── spectrogram_utils.py    # Mel (diffusers) ↔ tensor ↔ audio
├── diffusers_wrapper.py    # DDPMScheduler/UNet2DModel → DiME API
├── audio_datasets.py       # HF dataset loader (no labels)
├── audio_classifier.py     # DenseAudioClassifier loader (local/W&B checkpoint)
├── main_audio.py           # Counterfactual generation (main entry point)
├── evaluate_metrics.py     # Post-hoc metrics
├── cnn14_perceptual.py     # PANNs CNN14 perceptual loss (1-channel)
├── test_audio.sh           # Run demo
└── finetune.sh             # Placeholder (no fine-tuning needed)
```
