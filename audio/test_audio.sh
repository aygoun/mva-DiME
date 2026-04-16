#!/bin/bash

# High Quality Guitar Removal Experiment
echo "Running High Quality Experiment: Remove Guitar..."
uv run python -W ignore -m audio.main_audio \
    --wandb_artifact mva-altegrad-challenge/xai-dime/train_dense_classifier_openmic:v0 \
    --target_labels "8:0.0" \
    --exp_name guitar_removal_hq \
    --num_batches 1 --gpu 0 \
    --start_step 160 --classifier_scales "20.0" --guided_iterations 150 \
    --l1_loss 0.02 --l_perc 10.0

# Evaluation
echo "Evaluating Metrics (V2)..."
uv run python audio/evaluate_metrics_v2.py --exp_dir audio/results/guitar_removal_hq --steps 1

# Speaker Similarity (Unispeech)
echo "Computing Speaker Similarity..."
uv run python audio/evaluate_similarity.py \
    --cf_dir audio/results/guitar_removal_hq/step_0/cf_wav \
    --ref_dir audio/results/guitar_removal_hq/step_0/original_wav

# FAD
echo "Computing FAD..."
uv run python audio/evaluate_fad.py \
    --cf_dir audio/results/guitar_removal_hq/step_0/cf_wav \
    --ref_dir audio/results/guitar_removal_hq/step_0/original_wav
