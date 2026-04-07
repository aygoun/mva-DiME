#!/usr/bin/env bash
ESC50_DIR=dataset/ESC-50-master
CELEBA_DDPM=models/ddpm-celeba.pt
GPU=0

# Step 1: Classifier
python -m audio.download_ast --model_id MIT/ast-finetuned-audioset-10-10-0.4593

python -m audio.train_classifier \
    --data_dir "$ESC50_DIR" \
    --output_path audio/models/classifier_ast.pth \
    --classifier_type ast \
    --epochs 50 --batch_size 16 --lr 1e-4 \
    --freeze_backbone_epochs 20 \
    --mixup_alpha 0.4 --label_smoothing 0.1 \
    --val_fold 5 --gpu $GPU

# Step 2: DDPM fine-tuning
# MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False \
# --diffusion_steps 500 --learn_sigma True --noise_schedule linear \
# --num_channels 128 --num_heads 4 --num_res_blocks 2 \
# --resblock_updown True --use_fp16 False --use_scale_shift_norm True \
# --image_size 128"

# python -m audio.finetune_diffusion $MODEL_FLAGS \
#     --data_dir "$ESC50_DIR/audio" \
#     --data_mode folder \
#     --audio_duration 5.0 \
#     --pretrained_path "$CELEBA_DDPM" \
#     --output_dir audio/models/ddpm-spectro \
#     --steps 15000 --batch_size 16 --lr 2e-5 \
#     --save_interval 5000 --log_interval 100 \
#     --gpu $GPU
