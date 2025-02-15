import argparse, chess, torch, os, random
from torch.distributed import init_process_group, destroy_process_group
from players import GPTPlayer, StockfishPlayer
from math import ceil

backend = 'nccl'
stream_size = 1024 ** 3
init_elo = 1250

class GameState(object):
    def __init__(self, game_id):
        self.game_id = game_id
        self.board = chess.Board()
        self.state = ";"
        self.outcome = ""
        self.sf_rating = random.randint(1360, 2840)
        self.turn = random.randint(0, 1)
        self.w_player_id = self.turn
    
    def is_complete(self):
        return self.outcome != ""

    def write_outcome(self, pgn_file):
        assert self.is_complete()

        w_player, b_player = "GPTPlayer", "Stockfish-{}".format(self.sf_rating)
        w_rating, b_rating = init_elo, self.sf_rating
        if self.w_player_id != 0:
            w_player, b_player = b_player, w_player
            w_rating, b_rating = b_rating, w_rating

        game = '''[White \"{} {}\"]\n[Black \"{} {}\"]\n[Result \"{}\"]\n{}'''.format(
            w_player, w_rating, b_player, b_rating, self.outcome, self.outcome
        )

        pgn_file.write(game)
    

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

gpt_player = GPTPlayer(args.ckpt, device)
stockfish_player = StockfishPlayer(args.play_time)

local_bsz = args.eval_bsz // ddp_world_size
local_pgn = open(args.pgn_loc + ddp_local_rank, "w")

games_played = 0
game_states = [GameState(idx) for idx in range(local_bsz)]
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
        game_states += [GameState(base_game_id + game_id) for game_id in range(new_games)]
        base_game_id += new_games

local_pgn.close()

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