# DiME for Audio — Counterfactual Explanations on Spectrograms

This module extends DiME to the **audio modality**. It generates counterfactual mel spectrograms that flip an audio classifier's prediction, using the same diffusion-guided process from the original paper.

**Core idea:** mel spectrograms are treated as 3-channel 128×128 images, which lets us reuse the image DDPM architecture, VGG perceptual loss, and the entire DiME sampling pipeline unchanged.

## Quick overview

| Component | Solution | Training needed? |
|-----------|----------|-----------------|
| Diffusion model | CelebA DDPM fine-tuned on spectrograms | ~15K steps (~1-2 h) |
| Classifier | AST (HF pretrained on AudioSet) + ESC-50 head fine-tuning | ~50 epochs |
| Perceptual loss | VGG19 from original DiME | None (pretrained) |
| Guidance logic | `clean_multiclass_cond_fn` from `core/sample_utils.py` | None (reused) |
| Audio inversion | Griffin-Lim from counterfactual mel spectrogram | None |

## Setup

### 1. Install with uv

```bash
# Install base project + audio dependencies in one step
uv sync --extra audio
```

Or, if you prefer conda for the base env and uv only for extras:

```bash
conda env create -f env.yaml
conda activate dime
uv pip install -e ".[audio]"
```

### 3. Download ESC-50

```bash
git clone https://github.com/karolpiczak/ESC-50.git /path/to/ESC-50-master
```

### 4. Download CelebA DDPM checkpoint

Download from [HuggingFace](https://huggingface.co/guillaumejs2403/DiME) and place under `models/ddpm-celeba.pt`.

## Pipeline

All commands run from the **repo root** (`DiME/`).

### Step 1 — Download AST + fine-tune ESC-50 head

```bash
python -m audio.download_ast \
    --model_id MIT/ast-finetuned-audioset-10-10-0.4593

python -m audio.train_classifier \
    --data_dir /path/to/ESC-50-master \
    --output_path audio/models/classifier_70.pth \
    --classifier_type ast \
    --epochs 50 --batch_size 16 --lr 1e-4 \
    --freeze_backbone_epochs 20 --gpu 0
```

### Step 2 — Fine-tune the DDPM on spectrograms (~1-2 h)

```bash
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False \
--diffusion_steps 500 --learn_sigma True --noise_schedule linear \
--num_channels 128 --num_heads 4 --num_res_blocks 2 \
--resblock_updown True --use_fp16 False --use_scale_shift_norm True \
--image_size 128"

python -m audio.finetune_diffusion $MODEL_FLAGS \
    --data_dir /path/to/ESC-50-master/audio \
    --pretrained_path models/ddpm-celeba.pt \
    --output_dir audio/models/ddpm-spectro \
    --steps 15000 --batch_size 16 --lr 2e-5 --gpu 0
```

### Step 3 — Generate counterfactual explanations

```bash
SAMPLE_FLAGS="--batch_size 8 --timestep_respacing 200"

python -W ignore -m audio.main_audio $MODEL_FLAGS $SAMPLE_FLAGS \
    --data_dir /path/to/ESC-50-master \
    --model_path audio/models/ddpm-spectro/ema_0.9999_015000.pt \
    --classifier_path audio/models/classifier_70.pth \
    --classifier_type ast \
    --ast_model_id MIT/ast-finetuned-audioset-10-10-0.4593 \
    --output_path audio/results --exp_name esc50_demo \
    --num_classes 50 --start_step 60 \
    --classifier_scales '5,8,12' \
    --l1_loss 0.05 --l_perc 10.0 --l_perc_layer 8 \
    --target_label -1 --num_batches 5 --gpu 0
```

Or simply edit and run the helper scripts:

```bash
bash audio/finetune.sh    # Steps 1 + 2
bash audio/test_audio.sh  # Step 3
```

## Output structure

```
audio/results/esc50_demo/
├── original_spec/   # Input mel spectrogram images
├── cf_spec/         # Counterfactual spectrogram images
├── diff_spec/       # |original - counterfactual| heatmaps
├── original_wav/    # Reconstructed input audio (via Griffin-Lim)
├── cf_wav/          # Reconstructed counterfactual audio
├── info/            # Per-sample: pred, target, cf_pred, flipped, L1
└── summary.txt      # Flip rate and mean L1
```

## Key arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--target_label` | `-1` | Target class for counterfactual. `-1` = second-most-likely class |
| `--classifier_scales` | `5,8,12` | Classifier gradient scales, tried in order until the prediction flips |
| `--start_step` | `60` | Noise level τ (higher = more freedom to edit, lower = closer to original) |
| `--l_perc` | `10.0` | VGG perceptual loss weight |
| `--l1_loss` | `0.05` | L1 proximity loss weight |
| `--num_classes` | `50` | Number of classes in the classifier |

## Architecture

```
audio/
├── spectrogram_utils.py    # Audio ↔ mel spectrogram conversion
├── audio_datasets.py       # ESC50Dataset
├── audio_classifier.py     # ResNet18 + AST classifier backbones
├── download_ast.py         # Download AST from Hugging Face SDK
├── train_classifier.py     # Fine-tune classifier on ESC-50
├── finetune_diffusion.py   # Fine-tune CelebA DDPM on spectrograms
├── main_audio.py           # Counterfactual generation (main entry point)
├── finetune.sh             # Helper: run both fine-tuning steps
├── test_audio.sh           # Helper: run counterfactual generation
└── (deps managed in repo-root pyproject.toml [audio] extra)
```

**Reused from original DiME (no modifications):**
- `core/gaussian_diffusion.py` — forward/reverse diffusion process
- `core/respace.py` — timestep respacing
- `core/unet.py` — UNet denoiser architecture
- `core/sample_utils.py` — `get_DiME_iterative_sampling`, `clean_multiclass_cond_fn`, `dist_cond_fn`, `PerceptualLoss`
- `core/script_util.py` — model/diffusion factory functions
- `core/nn.py`, `core/resample.py` — utilities
