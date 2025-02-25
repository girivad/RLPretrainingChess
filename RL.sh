#!/bin/bash
train_config=$1
nvidia-smi
pip install torch numpy transformers datasets tiktoken wandb tqdm tzdata chess --no-cache-dir -q
# bash players/install_stockfish.sh
num_procs=$(python -c "import torch; print(torch.cuda.device_count());")
torchrun --nproc_per_node=${num_procs} RLtrain.py $train_config