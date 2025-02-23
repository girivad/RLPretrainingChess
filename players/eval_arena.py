import argparse, os
from torch.distributed import init_process_group, destroy_process_group
from game_utils import estimate_elo

backend = 'nccl'
stream_size = 1024 ** 3

parser = argparse.ArgumentParser()
parser.add_argument("--eval_bsz", type = int, required = True)
parser.add_argument("--games", type = int, required = True)
parser.add_argument("--ckpt", type = str, required = True)
parser.add_argument("--pgn_loc", type = str, required = True)
parser.add_argument("--play_time", type = float, default = 1e-4)

args = parser.parse_args()

init_process_group(backend = backend)
ddp_local_rank = int(os.environ["LOCAL_RANK"])
ddp_world_size = int(os.environ["WORLD_SIZE"])
device = f"cuda:{ddp_local_rank}"



elo = estimate_elo(model, args.eval_bsz, args.games, ddp_local_rank, "./pgn/games", False, "./data/lichess_hf_dataset", world_size = ddp_world_size)

if ddp_local_rank == 0:
    print("Elo:", elo)

destroy_process_group()