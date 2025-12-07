#!/bin/bash
# Multi-GPU training script for UTNet
# Usage: bash scripts/train_multi_gpu.sh [num_gpus] [config_path]

NUM_GPUS=${1:-4}  # Default to 4 GPUs
CONFIG=${2:-config/config.yaml}  # Default config path

echo "Starting multi-GPU training with $NUM_GPUS GPUs"
echo "Config: $CONFIG"

# Method 1: Using torchrun (recommended for PyTorch 1.9+)
torchrun --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    train.py \
    --config $CONFIG

# Alternative Method 2: Using torch.distributed.launch (for older PyTorch versions)
# python -m torch.distributed.launch \
#     --nproc_per_node=$NUM_GPUS \
#     --master_port=29500 \
#     train.py \
#     --config $CONFIG

