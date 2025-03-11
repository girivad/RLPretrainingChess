import chess, torch, os
from typing import List
from players.players import GPTPlayer, StockfishPlayer
from players.game_utils import GameState
from tokenizer.scripts.tokenizer import load_tokenizer
from time import time
import subprocess

STREAM_SIZE = 1024 ** 3

def update_gpr(g, G, p, P, r, R, tokenize):
    g = tokenize(g, batch = True)
    g = [list(g_s) for g_s in g] # TODO: g is a list of np arrays.
    p = sum([[ptype] * len(tokens) for ptype, tokens in zip(p, g)], [])
    g = sum(g, [])
    
    S = max(G.size(-1), len(g)) if G is not None else len(g) # If initializing G, len of the new sequence. If adding to G, size of G.
    if S > len(g) or G is None:
        g = g + [0] * (S - len(p)) # Random Token 0, will be dropped in Loss calculation as P is also 0.
        p = p + [0] * (S - len(p))
    elif S > G.size(-1):
        G = torch.concat([G, torch.zeros((G.size(0), S - G.size(-1)))], dim = 1)
        P = torch.concat([P, torch.zeros((P.size(0), S - P.size(-1)))], dim = 1)

    g = torch.tensor(g).view(1, -1)
    p = torch.tensor(p).view(1, -1)

    G = g if G is None else torch.concat([G, g], dim = 0)
    P = p if P is None else torch.concat([P, p], dim = 0)
    R.append(r)

    return G, P, R

class Arena(object):
    def __init__(self, player0: StockfishPlayer | GPTPlayer, player1: StockfishPlayer | GPTPlayer, eval_bsz, rank, tokenize = None):
        self.player0 = player0
        self.player1 = player1
        self.eval_bsz = eval_bsz
        self.local_rank = rank

        self.tokenize = tokenize
        self.adjudicator = chess.engine.SimpleEngine.popen_uci("./stockfish_exec")
    
    def run_games(self, total_games: int, write_out = None):
        if write_out:
            write_out = open(write_out + str(self.local_rank), "w")
        else:
            G = None # B x S
            P = None # B x (S - 1)
            R = [] # B x 0 
        games_played = 0
        game_states = [GameState(idx, self.adjudicator, [type(self.player0).__name__, type(self.player1).__name__]) for idx in range(self.eval_bsz)]
        base_game_id = self.eval_bsz
        # if self.local_rank == 1:
        #     print("Player Types:", type(self.player0).__name__, type(self.player1).__name__)

        move_num = 0

        while games_played < total_games:
            while len(game_states) > 0:
                # if self.local_rank == 1:
                #     print("Move Number:", move_num)
                move_num += 1
                p0_games = [game_state for game_state in game_states if game_state.turn == 0]
                p1_games = [game_state for game_state in game_states if game_state.turn == 1]

                before = time()
                if len(p0_games) > 0:
                    self.player0.play(p0_games)
                # if self.local_rank == 1:
                #     print("P0 Plays in:", time() - before)
                before = time()
                if len(p1_games) > 0:
                    self.player1.play(p1_games)
                # if self.local_rank == 1:
                #     print("P1 Plays in:", time() - before)
                
                reduced_game_states = []
                for game_state in game_states:
                    if not game_state.is_complete():
                        # if self.local_rank == 1:
                        #     print("Game State Completed.")
                        reduced_game_states.append(game_state)
                        continue
                    if write_out is not None:
                        # if self.local_rank == 0:
                        #     print("Write Out:", write_out)
                        game_state.write_outcome(write_out)
                    else:
                        # if self.local_rank == 1:
                        #     print("get_gpr")
                        g, p, r = game_state.get_gpr()
                        # if self.local_rank == 1:
                        #     print("got gpr")
                        G, P, R = update_gpr(g, G, p, P, r, R, self.tokenize)
                        # if self.local_rank == 1:
                        #     print("update gpr")

                    games_played += 1
                # if self.local_rank == 1:
                #     print("Completed Game Evaluations")
                game_states = reduced_game_states
                # if self.local_rank == 1:
                #     print("Getting the number of new games.")
                new_games = min(self.eval_bsz - len(game_states), total_games - (games_played + len(game_states))) # Min(Bsz - reduced_games, total_games - (games_played + reduced_games))
                # if self.local_rank == 1:
                #     print(type(base_game_id), type(new_games))
                game_states += [GameState(base_game_id + game_id, self.adjudicator, [type(self.player0).__name__, type(self.player1).__name__]) for game_id in range(new_games)]
                base_game_id += new_games
                # if self.local_rank == 1:
                #     print("Games Played:", games_played)
            # if self.local_rank == 1:
            #     print("Move Batch Terminated")
        # if self.local_rank == 1:
        #     print("Total Games Completed")
        if write_out:
            write_out.close()
        else:
            R = torch.tensor(R)
            return G, P, R
    
    def close(self):
        self.player0.close()
        self.player1.close()
        self.adjudicator.close()

def collate_games(files: List[str], write_out: str):
    global_pgn = open(write_out, "wb")

    for file in files:
        # print("Merging File:", file)
        local_pgn = open(file, "rb")
        # print("Opened Local PGN")
        stream = local_pgn.read(STREAM_SIZE)
        # print("Read first stream:", stream.decode("utf-8"))
        while len(stream) > 0 and stream is not None:
            global_pgn.write(stream)
            # print("Wrote Stream")
            stream = local_pgn.read(STREAM_SIZE)
            # print("Stream:", stream.decode("utf-8"))
        
        # print("Write Out Complete")

        local_pgn.close()
        os.remove(file)
    
    global_pgn.close()

MMS = {
    "char": 5,
    "move": 2
}

def sample_games(pi_theta, total_games, bsz, rank, tok_type = "move", tokenizer_path = "./tokenizer/tokenizers/move_token.pkl", self_play = False, write_out = None, sf_time = 0.1):
    p0 = GPTPlayer(pi_theta, f"cuda:{rank}", max_move_size = MMS[tok_type], tok_type = tok_type, tokenizer_path = tokenizer_path)
    if self_play:
        p1 = GPTPlayer(pi_theta, f"cuda:{rank}", max_move_size = MMS[tok_type], tok_type = tok_type, tokenizer_path = tokenizer_path)
    else:
        p1 = StockfishPlayer(sf_time)

    if rank == 0:
        print("Players Created")

    tokenize = None
    if write_out is None:
        tokenize, _ = load_tokenizer(tok_type, tokenizer_path)

    if rank == 0:
        print("Create Tokenizer")

    arena = Arena(p0, p1, bsz, rank, tokenize)
    if rank == 0:
        print("Create Arena")
    if write_out:
        arena.run_games(total_games, write_out)
        if rank == 0:
            print("Have Run Games")
    else:
        G, P, R = arena.run_games(total_games)
        G = G.type(torch.long)
        return G, P, R

def parse_elo(ratings_file, target_player_name):
    if not os.path.exists(ratings_file):
        return None, None, None

    with open(ratings_file, "r") as r:
        player_lines = r.readlines()[1:]

    for player_line in player_lines:
        player_name = player_line.split()[1]
        player_rating = int(player_line.split()[2])
        player_up_bd = player_rating + int(player_line.split()[3])
        player_lw_bd = player_rating - int(player_line.split()[4])

        if player_name == target_player_name:
            return (player_rating, player_up_bd, player_lw_bd)

    return None, None, None

def calc_elo(pgn_file):
    subprocess.run(["bash", "prepare_ratings_script.sh", pgn_file], capture_output = True)
    if not os.path.exists("./bayeselo_ratings_script"):
        raise Exception("Failed to prepare ratings script.")
    subprocess.run("../BayesianElo/src/bayeselo < bayeselo_ratings_script", shell = True, capture_output = True)
    elo, lw_bd, up_bd = parse_elo("ratings", "GPTPlayer")
    if elo is None:
        raise Exception("Failed to parse GPTPlayer from the ratings.")
    os.remove("ratings")
    return elo, lw_bd, up_bd

def estimate_elo(pi_theta, eval_bsz, eval_games, rank, write_out, wait, tok_type = "move", tokenizer_path = "./tokenizer/tokenizers/move_token.pkl", world_size = None):
    sample_games(pi_theta, eval_games, eval_bsz, rank, tok_type, tokenizer_path, write_out = write_out)
    wait()
    if rank == 0:
        assert world_size is not None
        collate_games([write_out + str(r) for r in range(world_size)], write_out)
        print("Games Collated")

        return calc_elo(write_out)