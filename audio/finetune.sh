#!/usr/bin/env bash
# ------------------------------------------------------------------
# Fine-tune pipeline for FSD50K
# ------------------------------------------------------------------
FSD50K_DIR=dataset/FSD50K
CELEBA_DDPM=models/ddpm-celeba.pt
GPU=0

# ==================================================================
# Step 1 (optional): Fine-tune AST on FSD50K with BCE
# ==================================================================
# The pretrained AST already covers FSD50K's classes (subset of
# AudioSet 527).  Fine-tuning can improve per-class calibration.
#
# python -m audio.train_classifier \
#     --data_dir "$FSD50K_DIR" \
#     --output_path audio/models/classifier_fsd50k.pth \
#     --epochs 30 --batch_size 16 --lr 1e-4 \
#     --freeze_backbone_epochs 10 \
#     --mixup_alpha 0.4 \
#     --audio_duration 7.0 \
#     --gpu $GPU

# ==================================================================
# Step 2: Fine-tune the DDPM on FSD50K spectrograms
# ==================================================================
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False \
--diffusion_steps 500 --learn_sigma True --noise_schedule linear \
--num_channels 128 --num_heads 4 --num_res_blocks 2 \
--resblock_updown True --use_fp16 False --use_scale_shift_norm True \
--image_size 128"

python -m audio.finetune_diffusion $MODEL_FLAGS \
    --data_dir "$FSD50K_DIR" \
    --fsd50k_split dev \
    --audio_duration 7.0 \
    --pretrained_path "$CELEBA_DDPM" \
    --output_dir audio/models/ddpm-spectro \
    --steps 15000 --batch_size 16 --lr 2e-5 \
    --save_interval 5000 --log_interval 100 \
    --gpu $GPU
