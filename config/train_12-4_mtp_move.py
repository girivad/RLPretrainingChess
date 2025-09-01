import os
from datetime import datetime
from zoneinfo import ZoneInfo

model_dir = "../../model_vol"
data_dir = "../../model_vol/data_dir/pretrain"

timestamp = datetime.now(ZoneInfo("America/Los_Angeles"))
run_name = "12-4_mtp_lichess " + str(timestamp)

out_dir = os.path.join(model_dir, run_name)
eval_interval = 4000
eval_iters = 100
ckpt_interval = 50000
# I'm not sure what's going on, but when log_interval == 100, the time per iter is inaccurate and much longer than it should be
# when running on multiple GPUs. TODO: investigate
log_interval = 50  # don't print too too often

always_save_checkpoint = True

wandb_log = True
wandb_project = "chessformer"
wandb_run_name = run_name

dataset = data_dir
gradient_accumulation_steps = 2
batch_size = 50
block_size = 320  # context of up to 1023 tokens (because dataset block size is 1024)

# baby GPT model :)
architecture = "mtp-gpt"
k = 4
discount_rate = 0.99
n_layer = 12
n_head = 8
n_embd = 512
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
vocab_size = 1970

# aux losses
aux_seer_loss = False

learning_rate = 3e-4
max_iters = 600000
lr_decay_iters = max_iters  # make equal to max_iters usually
min_lr = 3e-5  # learning_rate / 10 usually
beta2 = 0.95  # make a bit bigger because number of tokens per iter is small

warmup_iters = 2000  # not super necessary potentially
compile = True
