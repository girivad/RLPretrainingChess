import os
from datetime import datetime
from zoneinfo import ZoneInfo

model_dir = "../../model_vol"
data_dir = "../../model_vol/data_dir/pretrain/lichess_100mb"

base_model_ckpt = "8-5ap_lichess 2026-01-08 01:05:49.320485-08:00"
base_model_dir = os.path.join(model_dir, base_model_ckpt, "ckpt_600000")
init_from = "resume"
eval_interval = 10
eval_iters = 10
# I'm not sure what's going on, but when log_interval == 100, the time per iter is inaccurate and much longer than it should be
# when running on multiple GPUs. TODO: investigate
log_interval = 2  # don't print too too often

always_save_checkpoint = True

wandb_log = True
wandb_project = "chessformer"
timestamp = datetime.now(ZoneInfo("America/Los_Angeles"))
wandb_run_name = "probe_8nap_lichess " + str(timestamp)

dataset = data_dir
gradient_accumulation_steps = 2
batch_size = 50
block_size = 160  # context of up to 1023 tokens (because dataset block size is 1024)

tok_type = "action"
tokenizer_path = "./tokenizer/tokenizers/action_token.pkl"

# baby GPT model :)
architecture = "mtp-gpt"
n_slayer = 0
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
vocab_size = 1969
k = 5
discount_rate = 1.0

# probe dataset
probe_target = "game_outcome"
probe_task = "classification"
start_probe_turn = None
end_probe_turn = None

# probe architecture
probe_classes = 3
probe_embd = n_embd
probe_architecture = "linear"

# optimization hyperparameters
learning_rate = 1e-3 # max learning rate
max_iters = 50000 // (batch_size * gradient_accumulation_steps) # total number of training iterations
weight_decay = 1e-2
beta1 = 0.9
beta2 = 0.99
grad_clip = 0.0 # clip gradients at this value, or disable if == 0.0

compile = True