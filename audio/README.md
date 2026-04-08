# DiME for Audio — Counterfactual Explanations on Spectrograms

This module extends DiME to the **audio modality**. It generates counterfactual mel spectrograms that flip a multi-label audio classifier's prediction for a single target class, using the same diffusion-guided process from the original paper.

**Core idea:** mel spectrograms are treated as 3-channel 128x128 images, which lets us reuse the image DDPM architecture, VGG perceptual loss, and the entire DiME sampling pipeline unchanged.

## Dataset: FSD50K

We use **FSD50K** — a multi-label sound-event dataset with 200 classes drawn from the AudioSet ontology. Each clip can carry **multiple labels**, so DiME can do cross-entropy guidance on a **single class per sample** while the other labels remain untouched — matching the paper's original CelebA setup where a single binary attribute is flipped.

| Property | Value |
|----------|-------|
| Clips | ~51 k |
| Classes | 200 (subset of AudioSet 527) |
| Labels/clip | Multi-label |

## Classifier: Pretrained AST on AudioSet (no custom head)

We use `MIT/ast-finetuned-audioset-10-10-0.4593` — the Audio Spectrogram Transformer pre-trained on AudioSet with 527 output classes. Since FSD50K's 200 classes are a strict subset of AudioSet, we reuse the pretrained head directly (`ASTAudioSetClassifier`). No custom head or fine-tuning is required for DiME guidance.

> If you want improved per-class calibration you can optionally fine-tune the full model on FSD50K with BCE loss (see Step 2 in the pipeline below).

## Quick overview

| Component | Solution | Training needed? |
|-----------|----------|-----------------|
| Diffusion model | CelebA DDPM fine-tuned on spectrograms | ~15K steps (~1-2 h) |
| Classifier | Pretrained AST (AudioSet 527 classes) | None (pretrained) |
| Perceptual loss | VGG19 from original DiME | None (pretrained) |
| Guidance logic | `clean_multilabel_cond_fn` (sigmoid per class) | None (reused) |
| Audio inversion | Griffin-Lim from counterfactual mel spectrogram | None |

## Setup

### 1. Install with uv

```bash
uv sync --extra audio
```

Or, if you prefer conda for the base env:

```bash
conda env create -f env.yaml
conda activate dime
uv pip install -e ".[audio]"
```

### 2. Download FSD50K

```bash
bash download.sh
```

This downloads the eval subset (~1.6 GB, 10k clips). Uncomment the dev block in `download.sh` for the full training set (~7.5 GB).

### 3. Download CelebA DDPM checkpoint

Also handled by `download.sh`, or manually from [HuggingFace](https://huggingface.co/guillaumejs2403/DiME) into `models/ddpm-celeba.pt`.

## Pipeline

All commands run from the **repo root** (`DiME/`).

### Step 1 — Download AST model

```bash
python -m audio.download_ast --model_id MIT/ast-finetuned-audioset-10-10-0.4593
```

### Step 2 (optional) — Fine-tune AST on FSD50K

```bash
python -m audio.train_classifier \
    --data_dir dataset/FSD50K \
    --output_path audio/models/classifier_fsd50k.pth \
    --epochs 30 --batch_size 16 --lr 1e-4 \
    --freeze_backbone_epochs 10 --gpu 0
```

### Step 3 — Fine-tune the DDPM on spectrograms (~1-2 h)

```bash
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False \
--diffusion_steps 500 --learn_sigma True --noise_schedule linear \
--num_channels 128 --num_heads 4 --num_res_blocks 2 \
--resblock_updown True --use_fp16 False --use_scale_shift_norm True \
--image_size 128"

python -m audio.finetune_diffusion $MODEL_FLAGS \
    --data_dir dataset/FSD50K \
    --fsd50k_split dev \
    --pretrained_path models/ddpm-celeba.pt \
    --output_dir audio/models/ddpm-spectro \
    --steps 15000 --batch_size 16 --lr 2e-5 --gpu 0
```

### Step 4 — Generate counterfactual explanations

```bash
SAMPLE_FLAGS="--batch_size 8 --timestep_respacing 200"

python -W ignore -m audio.main_audio $MODEL_FLAGS $SAMPLE_FLAGS \
    --data_dir dataset/FSD50K \
    --fsd50k_split eval \
    --model_path audio/models/ddpm-spectro/ema_0.9999_015000.pt \
    --ast_model_id MIT/ast-finetuned-audioset-10-10-0.4593 \
    --output_path audio/results --exp_name fsd50k_demo \
    --target_label -1 --target_strategy random_remove \
    --classifier_scales '5,8,12' \
    --start_step 60 --l1_loss 0.05 --l_perc 10.0 --l_perc_layer 8 \
    --num_batches 5 --gpu 0
```

Or simply edit and run the helper scripts:

```bash
bash audio/finetune.sh    # Steps 1 + 3
bash audio/test_audio.sh  # Step 4
```

## Target strategies

The `--target_strategy` flag controls which class is flipped per sample:

| Strategy | Direction | Description |
|----------|-----------|-------------|
| `random_remove` | remove | Pick a random GT class and minimize its probability |
| `least_confident_remove` | remove | Pick the GT class the classifier is least sure about |
| `random_add` | add | Pick a random non-GT class and maximize its probability |

Use `--target_label <audioset_idx>` to override and use a fixed class for all samples.

## Output structure

```
audio/results/fsd50k_demo/
├── original_spec/   # Input mel spectrogram images
├── cf_spec/         # Counterfactual spectrogram images
├── diff_spec/       # |original - counterfactual| heatmaps
├── original_wav/    # Reconstructed input audio (via Griffin-Lim)
├── cf_wav/          # Reconstructed counterfactual audio
├── info/            # Per-sample: target_class, direction, flipped, L1, ...
└── summary.txt      # Flip rate and mean L1/L2
```

## Key arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--target_label` | `-1` | Fixed AudioSet class index, or `-1` for auto-selection |
| `--target_strategy` | `random_remove` | How to pick the target class (see table above) |
| `--classifier_scales` | `5,8,12` | Classifier gradient scales, tried in order |
| `--start_step` | `60` | Noise level τ |
| `--l_perc` | `10.0` | VGG perceptual loss weight |
| `--l1_loss` | `0.05` | L1 proximity loss weight |

## Architecture

```
audio/
├── spectrogram_utils.py    # Audio <-> mel spectrogram conversion
├── audio_datasets.py       # FSD50KDataset
├── audio_classifier.py     # ASTAudioSetClassifier (pretrained, 527 classes)
├── download_ast.py         # Download AST from Hugging Face SDK
├── train_classifier.py     # Optional fine-tune on FSD50K
├── finetune_diffusion.py   # Fine-tune CelebA DDPM on spectrograms
├── main_audio.py           # Counterfactual generation (main entry point)
├── evaluate_metrics.py     # Post-hoc metrics from saved results
├── finetune.sh             # Helper: run fine-tuning steps
├── test_audio.sh           # Helper: run counterfactual generation
└── data/                   # Cached AudioSet class mapping
```

**Reused from original DiME (no modifications):**
- `core/gaussian_diffusion.py` — forward/reverse diffusion process
- `core/respace.py` — timestep respacing
- `core/unet.py` — UNet denoiser architecture
- `core/sample_utils.py` — `get_DiME_iterative_sampling`, `clean_multilabel_cond_fn`, `dist_cond_fn`, `PerceptualLoss`
- `core/script_util.py` — model/diffusion factory functions
