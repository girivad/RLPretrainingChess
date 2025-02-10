import argparse, os

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type = str, default = "out-shakespeare-char")
parser.add_argument("--data_dir", type = str, default = "lichess_hf_dataset")
parser.add_argument("--run_name", type = str, default = "8layer_ntp_lichess")
args = parser.parse_args()

out_dir = os.path.join(args.model_dir, args.run_name)
eval_interval = 4000
eval_iters = 100
# I'm not sure what's going on, but when log_interval == 100, the time per iter is inaccurate and much longer than it should be
# when running on multiple GPUs. TODO: investigate
log_interval = 50  # don't print too too often

always_save_checkpoint = True

wandb_log = True
wandb_project = "chessformer"
wandb_run_name = args.run_name

dataset = args.data_dir
gradient_accumulation_steps = 1
batch_size = 100
block_size = 1023  # context of up to 1023 tokens (because dataset block size is 1024)

# baby GPT model :)
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.0

learning_rate = 3e-4
max_iters = 600000
lr_decay_iters = max_iters  # make equal to max_iters usually
min_lr = 3e-5  # learning_rate / 10 usually
beta2 = 0.95  # make a bit bigger because number of tokens per iter is small

warmup_iters = 2000  # not super necessary potentially
compile = True
