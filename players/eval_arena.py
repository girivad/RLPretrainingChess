import argparse, os, chess
from torch.distributed import init_process_group, destroy_process_group
from players import GPTPlayer, StockfishPlayer
from math import ceil
from game_utils import GameState

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

gpt_player = GPTPlayer(args.ckpt, device, ddp_local_rank)
stockfish_player = StockfishPlayer(args.play_time)
sf_engine = chess.engine.SimpleEngine.popen_uci("stockfish_exec")

local_bsz = args.eval_bsz // ddp_world_size
local_pgn = open(args.pgn_loc + ddp_local_rank, "w")

games_played = 0
game_states = [GameState(idx, sf_engine) for idx in range(local_bsz)]
base_game_id = local_bsz

while games_played < args.games:
    while len(game_states) > 0:
        sf_games = [game_state for game_state in game_states if game_state.turn == 0]
        gpt_games = [game_state for game_state in game_states if game_state.turn == 1]

        for sf_game in sf_games:
            stockfish_player.play_move(sf_game)
        gpt_player.play_moves(gpt_games)

        for game_state in game_states:
            if game_state.is_complete():
                game_state.write_outcome(local_pgn)
                games_played += 1

        game_states = [game_state for game_state in game_states if not game_state.is_complete()]
        new_games = min(local_bsz - len(game_states), args.games - games_played - len(game_states))
        game_states += [GameState(base_game_id + game_id, sf_engine) for game_id in range(new_games)]
        base_game_id += new_games

local_pgn.close()
stockfish_player.close()
sf_engine.close()

if ddp_local_rank != 0:
    destroy_process_group()
    exit(0)

global_pgn = open(args.pgn_loc, "wb")

for rank in range(ddp_world_size):
    local_pgn_file = args.pgn_loc + str(rank)
    local_pgn = open(local_pgn_file, "rb")
    stream = local_pgn.read(stream_size)
    while stream is not None:
        global_pgn.write(stream)
        stream = local_pgn.read(stream_size)
    local_pgn.close()

    os.remove(local_pgn_file)

global_pgn.close()
destroy_process_group()