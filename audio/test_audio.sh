#!/bin/bash

# Experiment 1: Remove Guitar
echo "Running Experiment 1: Remove Guitar..."
uv run python -W ignore -m audio.main_audio \
    --wandb_artifact mva-altegrad-challenge/xai-dime/train_dense_classifier_openmic:v0 \
    --target_labels "8:0.0" \
    --exp_name exp1_guitar_remove \
    --num_batches 1 --gpu 0 \
    --start_step 50 --classifier_scales "10.0" --guided_iterations 5

# Experiment 2: Sequential Remove Guitar then Drums
echo "Running Experiment 2: Sequential Remove Guitar then Drums..."
uv run python -W ignore -m audio.main_audio \
    --wandb_artifact mva-altegrad-challenge/xai-dime/train_dense_classifier_openmic:v0 \
    --target_labels "8:0.0,6:0.0" \
    --exp_name exp2_sequential \
    --num_batches 1 --gpu 0 \
    --start_step 50 --classifier_scales "10.0" --guided_iterations 5

# Evaluation
echo "Evaluating Metrics..."
uv run python audio/evaluate_metrics_v2.py --exp_dir audio/results/exp1_guitar_remove --steps 1
uv run python audio/evaluate_metrics_v2.py --exp_dir audio/results/exp2_sequential --steps 2

echo "Computing Similarity..."
uv run python audio/evaluate_similarity.py \
    --cf_dir audio/results/exp1_guitar_remove/step_0/cf_wav \
    --ref_dir audio/results/exp1_guitar_remove/step_0/original_wav

echo "Computing FAD..."
# Note: ref_dir should ideally be the real dataset, here we use original_wav as proxy
uv run python audio/evaluate_fad.py \
    --cf_dir audio/results/exp1_guitar_remove/step_0/cf_wav \
    --ref_dir audio/results/exp1_guitar_remove/step_0/original_wav
