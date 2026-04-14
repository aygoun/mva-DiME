#!/usr/bin/env bash
GPU=0

python -W ignore -m audio.main_audio \
    --ddpm_repo teticio/audio-diffusion-breaks-256 \
    --dataset_repo teticio/audio-diffusion-breaks-256 \
    --max_samples 200 \
    --classifier_checkpoint_path checkpoints/last.ckpt \
    --num_classes 12 \
    --output_path audio/results \
    --exp_name breaks_demo \
    --target_label -1 \
    --target_strategy random_remove \
    --start_step 120 \
    --classifier_scales '5,8,12' \
    --l1_loss 0.05 \
    --l_perc 10.0 \
    --l_perc_layer 4 \
    --use_logits False \
    --use_sampling_on_x_t True \
    --num_batches 5 \
    --seed 42 \
    --gpu $GPU \
    --save_audio True \
    --save_images True
