import os

pretrain_run_name = "ntp_lichess 2025-03-05 19:39:30.836936-08:00"
ckpt_num = 600000
run_name = f"RL 8layer_moves_ntp_lichess {ckpt_num}"
init_from = "pretrain"
model_dir = "../../model_vol/"

out_dir = os.path.join(model_dir, pretrain_run_name, f"ckpt_{ckpt_num}")
eval_interval = 500
eval_iters = 50
hifi_eval_interval = 2500
hifi_eval_iters = 50
ckpt_interval = 500
# I'm not sure what's going on, but when log_interval == 100, the time per iter is inaccurate and much longer than it should be
# when running on multiple GPUs. TODO: investigate
log_interval = 50  # don't print too too often

always_save_checkpoint = True

wandb_log = True
wandb_project = "chessformer"
wandb_run_name = run_name

# dataset
gradient_accumulation_steps = 2
batch_size = 50
block_size = 1023  # context of up to 1023 tokens (because dataset block size is 1024)

# tokenizer
tok_type = "move"
tokenizer_path = "./tokenizer/tokenizers/move_token.pkl"

# baby GPT model :)
n_slayer = 0
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.1 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
vocab_size = 1970

# aux losses
aux_seer_loss = False

learning_rate = 1e-6
max_iters = 1
min_lr = 1e-6  # no lr decay
beta2 = 0.95  # make a bit bigger because number of tokens per iter is small
clip_eps = 0.04

baseline = None
group_size = 1
clip_eps = 0.2
self_play = False

warmup_iters = 0
compile = True

invalid_retries = 5