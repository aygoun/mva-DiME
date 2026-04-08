#!/usr/bin/env bash
# ------------------------------------------------------------------
# Generate counterfactual audio explanations on FSD50K
# ------------------------------------------------------------------
FSD50K_DIR=dataset/FSD50K
GPU=0

MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False \
--diffusion_steps 500 --learn_sigma True --noise_schedule linear \
--num_channels 128 --num_heads 4 --num_res_blocks 2 \
--resblock_updown True --use_fp16 False --use_scale_shift_norm True \
--image_size 128"

SAMPLE_FLAGS="--batch_size 8 --timestep_respacing 200"

python -W ignore -m audio.main_audio $MODEL_FLAGS $SAMPLE_FLAGS \
    --data_dir "$FSD50K_DIR" \
    --fsd50k_split eval \
    --model_path audio/models/ddpm-spectro/ema_0.9999_015000.pt \
    --ast_model_id MIT/ast-finetuned-audioset-10-10-0.4593 \
    --output_path audio/results \
    --exp_name fsd50k_demo \
    --target_label -1 \
    --target_strategy random_remove \
    --start_step 60 \
    --classifier_scales '5,8,12' \
    --l1_loss 0.05 \
    --l_perc 10.0 \
    --l_perc_layer 8 \
    --use_logits False \
    --use_sampling_on_x_t True \
    --audio_duration 7.0 \
    --num_batches 5 \
    --seed 42 \
    --gpu $GPU \
    --save_audio True \
    --save_images True
