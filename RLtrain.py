"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from utils import smooth
from players.arena import sample_games, estimate_elo

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
ckpt_interval = 50000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# tokenizer
hf_tokenizer = False
tokenizer_dir = "./data/lichess_hf_dataset"
# model
n_slayer = 0
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
beta = 0.9
clip_eps = 0.2
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = False # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    print("DDP World Size:", ddp_world_size)
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model init
model_args = dict(n_slayer=n_slayer, n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line

print(f"Resuming training from {out_dir}")
# resume training from a checkpoint.
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
checkpoint_model_args = checkpoint['model_args']
# force these config attributes to be equal otherwise we can't even resume training
# the rest of the attributes (e.g. dropout) can stay as desired from command line
for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
    model_args[k] = checkpoint_model_args[k]
# create the model
gptconf = GPTConfig(**model_args)
pi_theta = GPT(gptconf)
pi_ref = GPT(gptconf)
state_dict = checkpoint['model']
# fix the keys of the state dictionary :(
# honestly no idea how checkpoints sometimes get this prefix, have to debug more
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
pi_theta.load_state_dict(state_dict)
pi_ref.load_state_dict(state_dict)

iter_num = 0
if init_from == "resume":
    iter_num = checkpoint["iter_num"]

# crop down the model block size if desired, using model surgery
if block_size < pi_theta.config.block_size:
    pi_theta.crop_block_size(block_size)
    pi_ref.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
pi_theta.to(device)
pi_ref.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))

# optimizer
optimizer = pi_theta.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == "resume":
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = pi_theta
    pi_theta = torch.compile(pi_theta) # requires PyTorch 2.0
    pi_ref = torch.compile(pi_ref)

# wrap model into DDP container
if ddp:
    pi_theta = DDP(pi_theta, device_ids=[ddp_local_rank])
    pi_ref = DDP(pi_ref, device_ids=[ddp_local_rank])

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop

t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = pi_theta.module if ddp else pi_theta # unwrap DDP container if needed
running_mfu = -1.0

pi_ref.eval()

while True:
    # G: Indices of moves played in simulated games; B x S
    # P: Player Name/Type, -1 for black GPT Player, +1 for white GPT Player, 0 for Stockfish Player/Padding Tokens; B x S
    # R: Game Rewards, reward is -1 for black victory, +1 for white victory, 0 for draw; B x 0.
    with torch.no_grad():
        G, P, R = sample_games(pi_theta, batch_size, batch_size, ddp_local_rank, hf_tokenizer = hf_tokenizer, tokenizer_dir = tokenizer_dir, self_play = False)
        P = P[:, 1:] # B x (S - 1)

    # determine and set the learning rate for this iteration
    lr = learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        with torch.no_grad():
            elo = estimate_elo(pi_theta, batch_size, eval_iters, ddp_local_rank, f"./pgn/{iter_num}", hf_tokenizer = hf_tokenizer, tokenizer_dir = tokenizer_dir, world_size = ddp_world_size)
        print(f"step {iter_num}: Elo rating {elo:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "elo": elo,
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if iter_num % ckpt_interval == 0:
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'config': config,
                    "elo": elo
                }
                ckpt_dir = os.path.join(out_dir, f"ckpt_{iter_num}")
                if not os.path.isdir(ckpt_dir):
                    os.mkdir(ckpt_dir)
                print(f"saving checkpoint to {ckpt_dir}")
                torch.save(checkpoint, os.path.join(ckpt_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    loss_dict = dict()
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            pi_theta.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            pi_t = smooth(pi_theta(G[:, :-1])) # B x (S - 1) x V
            # Index Select Workaround: https://github.com/pytorch/pytorch/issues/30574
            pi_t_prbs = torch.gather(pi_t, 2, G[:, 1:].unsqueeze(2)).squeeze(2) # B x (S - 1)

            with torch.no_grad():
                pi_r = smooth(pi_ref(G[:, :-1]))
                pi_r_prbs = torch.gather(pi_r, 2, G[:, 1:].unsqueeze(2)).squeeze(2) # B x (S - 1)
            
            prb_ratio = pi_t_prbs / pi_t_prbs.detach().clone() # B x (S - 1)
            clipped_ratio = torch.clip(prb_ratio, 1 - clip_eps, 1 + clip_eps) # B x (S - 1)

            loss = torch.mean(
                torch.sum(
                    torch.where(prb_ratio < clipped_ratio, prb_ratio, clipped_ratio) * P * R.view(-1, 1) - 
                    beta * (pi_r_prbs / pi_t_prbs - torch.log(pi_r_prbs) + torch.log(pi_t_prbs) - 1) * (P != 0),
                    dim = 1
                ) / (P != 0).sum(dim = 1)
            )

            assert not torch.any(torch.isnan(loss))
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation

        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()

    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(pi_theta.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        assert not torch.any(torch.isnan(loss))
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        if wandb_log:
            wandb.log(dict(**{
                "iter": iter_num,
                "train/loss": lossf,
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            }))
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
