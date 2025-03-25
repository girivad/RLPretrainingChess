#!/bin/bash
train_config=$1
pip install torch numpy transformers datasets tiktoken wandb tqdm tzdata chess --no-cache-dir -q
bash players/install_stockfish.sh
bash setup_bayeselo.sh
mkdir openings
wget -O openings/eco_openings.pgn https://storage.googleapis.com/searchless_chess/data/eco_openings.pgn 
num_procs=$(python -c "import torch; print(torch.cuda.device_count());")
mkdir pgn
nvidia-smi
torchrun --nproc_per_node=${num_procs} train.py $train_config