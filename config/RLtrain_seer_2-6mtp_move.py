import os
from datetime import datetime
from zoneinfo import ZoneInfo

model_dir = "../../model_vol"
data_dir = "../../model_vol/data_dir/pretrain"

run_name = "2-6layer_moves_mtp_lichess "
ckpt_num = 600000
init_from = "pretrain"

out_dir = os.path.join(model_dir, run_name)
eval_interval = 4000
eval_iters = 100
ckpt_interval = 500
# I'm not sure what's going on, but when log_interval == 100, the time per iter is inaccurate and much longer than it should be
# when running on multiple GPUs. TODO: investigate
log_interval = 50  # don't print too too often

always_save_checkpoint = True

wandb_log = True
wandb_project = "chessformer"
wandb_run_name = run_name

# dataset
dataset = data_dir
gradient_accumulation_steps = 2
batch_size = 50
block_size = 1023  # context of up to 1023 tokens (because dataset block size is 1024)

# tokenizer


# baby GPT model :)
n_slayer = 2
n_layer = 6
n_head = 8
n_embd = 512
dropout = 0.1 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
vocab_size = 1970

# aux losses
aux_seer_loss = False

learning_rate = 3e-4
max_iters = 10000
min_lr = 3e-4  # no lr decay
beta2 = 0.95  # make a bit bigger because number of tokens per iter is small
clip_eps = 0.2

warmup_iters = 0
compile = True