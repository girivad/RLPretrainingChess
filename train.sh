#!/bin/bash
train_config=$1
nvidia-smi
pip install torch numpy transformers datasets tiktoken wandb tqdm tzdata --no-cache-dir -q
num_procs=$(python -c "import torch; print(torch.cuda.device_count());")
torchrun --nproc_per_node=${num_procs} train.py $train_config