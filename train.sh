#!/bin/bash
train_config=$1
nvidia-smi
pip install torch numpy transformers datasets tiktoken wandb tqdm tzdata --no-cache-dir -q
# python data/lichess_hf_dataset/prepare.py --dataset lichess_6gb_blocks.zip --out_dir ../../model_vol/data_dir/pretrain/
num_procs=$(python -c "import torch; print(torch.cuda.device_count());")
torchrun --nproc_per_node=${num_procs} train.py $train_config