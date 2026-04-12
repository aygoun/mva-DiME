#!/bin/bash -l
# (submit.sh)

# SLURM SUBMIT SCRIPT
#SBATCH --partition=gpu_test
#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-node=4   # This needs to match Trainer(devices=...)
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --error=%j.out
#SBATCH --output=%j.out
#SBATCH --tmp=30G
module purge

# wandb temporarily writes to /tmp,
# which is a small partition on ruche (point to /scratch).
# This redirects it to a larger partition.
# https://github.com/wandb/wandb/blob/797a8714b62505a893b96df72d3e64132e54b89d/wandb/sdk/data_types/_private.py#L6
# debugging flags (optional)

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_SOCKET_IFNAME=^docker0,lo
export WANDB_CACHE_DIR=$WORKDIR/wandb_cache
export WANDB_DIR=$WORKDIR/wandb
export WANDB_DATA_DIR=$WORKDIR/wandb_data
export HF_HOME=$WORKDIR/hf
export HF_DATASETS_AUDIO_BACKEND=soundfile
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))

# on your cluster you might need these:
# set the network interface

# might need the latest CUDA
# module load NCCL/2.4.7-1-cuda.10.0

echo "Activating environment from $ENVIRONMENT_ROOT"

source $ENVIRONMENT_ROOT/bin/activate

# run script from above
srun python scripts/train_classifier.py