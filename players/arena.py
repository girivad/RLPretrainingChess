import chess, torch, os, random, shutil
import chess.engine
from typing import List
from players.players import GPTPlayer, StockfishPlayer
from players.game_utils import GameState, get_openings
from tokenizer.scripts.tokenizer import load_tokenizer
import numpy as np
import subprocess
from tqdm import tqdm

STREAM_SIZE = 1024 ** 3

def update_gpr(g, G, p, P, r, R, tokenize):
    g = tokenize(g, batch = True, pgn = False)
    g = [list(g_s.astype(np.int32)) for g_s in g] # TODO: g is a list of np arrays.
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
    def __init__(self, player0: StockfishPlayer | GPTPlayer, player1: StockfishPlayer | GPTPlayer, eval_bsz, rank, tokenize = None, init_games: List[GameState] = [], invalid_retries = 0):
        self.player0 = player0
        self.player1 = player1
        self.p_names = [self.player0.name(), self.player1.name()]
        self.eval_bsz = eval_bsz
        self.local_rank = rank

        self.tokenize = tokenize
        self.adjudicator = chess.engine.SimpleEngine.popen_uci("./stockfish_exec", timeout = None)

        self.init_games = [game for game in init_games]

        self.invalid_retries = invalid_retries

    def run_games(self, total_games: int, write_out = None, openings = [], group_size = 1):
        if write_out:
            write_out = open(write_out + str(self.local_rank), "w")
            for game in self.init_games:
                game.write_outcome(write_out)
        else:
            assert self.init_games is None or len(self.init_games) == 0, f"Given {len(self.init_games)} initialization games in data collection phase."
            G = None # B x S
            P = None # B x (S - 1)
            R = [] # B x 0 

        games_played = 0
        game_order = [None] * total_games

        if len(openings) > 0:
            game_openings = random.choices(openings, k = total_games // group_size)
            game_openings = sum([[opening] * group_size for opening in game_openings], [])
            game_perspectives = random.choices([0, 1], k = total_games // group_size)
            game_perspectives = sum([[perspective] * group_size for perspective in game_perspectives], [])

            game_states = [GameState(idx, self.adjudicator, self.p_names, [random.choice(range(1350, 2850, 100)) if "Stockfish" in p_name else None for p_name in self.p_names], opening = game_openings[idx], w_player_id = game_perspectives[idx], invalid_retries = self.invalid_retries) for idx in range(self.eval_bsz)]
        else:
            game_states = [GameState(idx, self.adjudicator, self.p_names, [random.choice(range(1350, 2850, 100)) if "Stockfish" in p_name else None for p_name in self.p_names], opening = "", w_player_id = random.randint(0, 1), invalid_retries = self.invalid_retries) for idx in range(self.eval_bsz)]

        base_game_id = self.eval_bsz

        move_num = 0

        while games_played < total_games:
            while len(game_states) > 0:
                move_num += 1
                p0_games = [game_state for game_state in game_states if game_state.turn == 0]
                p1_games = [game_state for game_state in game_states if game_state.turn == 1]

                if len(p0_games) > 0:
                    self.player0.play(p0_games)
                if len(p1_games) > 0:
                    self.player1.play(p1_games)
                
                reduced_game_states = []
                for game_state in game_states:
                    if not game_state.is_complete():
                        reduced_game_states.append(game_state)
                        continue
                    if write_out is not None:
                        game_state.write_outcome(write_out)
                    else:
                        g, p, r = game_state.get_gpr()
                        G, P, R = update_gpr(g, G, p, P, r, R, self.tokenize)
                        game_order[game_state.game_id] = games_played
    
                    games_played += 1

                game_states = reduced_game_states
                new_games = min(self.eval_bsz - len(game_states), total_games - (games_played + len(game_states))) # Min(Bsz - reduced_games, total_games - (games_played + reduced_games))
                if len(openings) > 0:
                    game_states += [GameState(base_game_id + game_id, self.adjudicator, self.p_names, [random.choice(range(1350, 2850, 100)) if "Stockfish" in p_name else None for p_name in self.p_names], opening = game_openings[base_game_id + game_id], w_player_id = game_perspectives[base_game_id + game_id], invalid_retries = self.invalid_retries) for game_id in range(new_games)]
                else:
                    game_states += [GameState(base_game_id + game_id, self.adjudicator, self.p_names, [random.choice(range(1350, 2850, 100)) if "Stockfish" in p_name else None for p_name in self.p_names], w_player_id = random.randint(0, 1), invalid_retries = self.invalid_retries) for game_id in range(new_games)]

                base_game_id += new_games

        if write_out:
            write_out.close()
        else:
            R = torch.tensor(R).type(torch.DoubleTensor)
            assert all([idx is not None for idx in game_order])
            G = G[game_order, :]
            P = P[game_order, :]
            R = R[game_order]
            return G, P, R
    
    def close(self):
        self.player0.close()
        self.player1.close()
        self.adjudicator.close()

def collate_games(files: List[str], write_out: str):
    global_pgn = open(write_out, "wb")

    for file in files:
        local_pgn = open(file, "rb")
        stream = local_pgn.read(STREAM_SIZE)
        while stream is not None and len(stream) > 0:
            global_pgn.write(stream)
            stream = local_pgn.read(STREAM_SIZE)
        
        local_pgn.close()
        os.remove(file)
    
    global_pgn.close()

MMS = {
    "char": 5,
    "move": 2
} # Max Move Size in Tokens

def sample_sf_games_fast(ratings, games_per_pair = 20):
    # MLEs of Draw/Advantage
    e_adv = 32.8
    e_draw = 97.3
    
    # Assume some number of games to sample
    ratings_games = len(ratings) * (len(ratings) - 1) * (games_per_pair // 2)

    elos = np.array([[r1, r2] for r1 in ratings for r2 in ratings if r1 != r2])
    elos = np.repeat(elos, games_per_pair // 2, axis = 0)
    assert np.all(elos[:, 0] != elos[:, 1])
    assert elos.shape[0] == ratings_games

    d_w = elos[:, 1] - elos[:, 0] - e_adv + e_draw
    d_b = elos[:, 0] - elos[:, 1] + e_adv + e_draw

    p_w = 1 / (1 + 10 ** (d_w / 400))
    assert np.all(p_w > 0)
    p_b = 1 / (1 + 10 ** (d_b / 400))
    assert np.all(p_b > 0)
    p_d = 1 - p_w - p_b

    assert np.all(p_d > 0), (np.sum(p_d <= 0), p_w[p_d <= 0], p_b[p_d <= 0])
    outcomes = [str(np.random.choice(["1-0", "1/2-1/2", "0-1"], p = [p_w[game], p_d[game], p_b[game]])) for game in range(ratings_games)]

    return [GameState.init_terminal_game(outcome, 0, ["Stockfish", "Stockfish"], [w_elo, b_elo]) for w_elo, b_elo, outcome in zip(elos[:, 0], elos[:, 1], outcomes)]

def sample_games(pi_theta, total_games, bsz, rank, tok_type = "move", tokenizer_path = "./tokenizer/tokenizers/move_token.pkl", self_play = False, write_out = None, sf_rating_games = "fast", sf_time = 0.1, use_opening_book = False, group_size = 1, invalid_retries = 0):
    synthetic_games = []
    if sf_rating_games == "fast":
        sf_ratings = range(1350, 2850, 100)
        synthetic_games = sample_sf_games_fast(sf_ratings, games_per_pair = total_games // len(sf_ratings))

    p0 = GPTPlayer(pi_theta, f"cuda:{rank}", max_move_size = MMS[tok_type], tok_type = tok_type, tokenizer_path = tokenizer_path)

    if self_play:
        p1 = GPTPlayer(pi_theta, f"cuda:{rank}", max_move_size = MMS[tok_type], tok_type = tok_type, tokenizer_path = tokenizer_path)
    else:
        p1 = StockfishPlayer(sf_time)

    tokenize = None
    if write_out is None:
        tokenize, _, _ = load_tokenizer(tok_type, tokenizer_path)

    arena = Arena(p0, p1, bsz, rank, tokenize, init_games = synthetic_games, invalid_retries = invalid_retries)

    openings = []
    if use_opening_book:
        openings = get_openings()

    if write_out:
        arena.run_games(total_games, write_out, openings = openings)
    else:
        G, P, R = arena.run_games(total_games, group_size = group_size, openings = openings)
        G = G.type(torch.long)

    arena.close()
    
    if not write_out:
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
            return (player_rating, player_lw_bd, player_up_bd)

    return None, None, None

def calc_elo(pgn_file):
    subprocess.run(["bash", "prepare_ratings_script.sh", pgn_file], capture_output = True)
    if not os.path.exists("./bayeselo_ratings_script"):
        raise Exception("Failed to prepare ratings script.")
    subprocess.run("./BayesianElo/src/bayeselo < bayeselo_ratings_script", shell = True, capture_output = True)
    elo, lw_bd, up_bd = parse_elo("ratings", "GPT")
    # os.remove("ratings")
    with open("ratings", "r") as ratings:
        print(ratings.read())
    
    if elo is None:
        raise Exception("Failed to parse GPT from the ratings.")
    
    return elo, lw_bd, up_bd

def estimate_elo(pi_theta, eval_bsz, eval_games, rank, write_out, wait, tok_type = "move", tokenizer_path = "./tokenizer/tokenizers/move_token.pkl", world_size = None, use_opening_book = True, invalid_retries = 0):
    sample_games(pi_theta, eval_games, eval_bsz, rank, tok_type, tokenizer_path, write_out = write_out, use_opening_book = use_opening_book, invalid_retries = invalid_retries)
    wait()
    if rank == 0:
        assert world_size is not None
        collate_games([write_out + str(r) for r in range(world_size)], write_out)

        return calc_elo(write_out)
    
    return None, None, None