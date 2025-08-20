import chess, torch, os, random, shutil
import chess.engine
from typing import List
from players.players import GPTPlayer, StockfishPlayer
from players.game_utils import GameState, get_openings
from tokenizer.scripts.tokenizer import load_tokenizer
import numpy as np
import subprocess
from tqdm import tqdm
from time import time
import pandas as pd
import math
from collections import defaultdict

STREAM_SIZE = 1024 ** 3

def update_gpr(g, G, p, P, r, R, tokenize):
    g = tokenize(g, batch = True, pgn = False)
    g = [list(g_s.astype(np.int32)) for g_s in g]
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
    def __init__(self, player0: StockfishPlayer | GPTPlayer, player1: StockfishPlayer | GPTPlayer, eval_bsz, rank, tokenize = None, init_games: List[GameState] = [], invalid_retries = 0, game_format = "uci", include_idx = False):
        self.player0 = player0
        self.player1 = player1
        self.p_names = [self.player0.name(), self.player1.name()]
        self.eval_bsz = eval_bsz
        self.local_rank = rank

        self.tokenize = tokenize
        self.adjudicator = chess.engine.SimpleEngine.popen_uci("./stockfish_exec", timeout = None)

        self.init_games = [game for game in init_games]

        self.invalid_retries = invalid_retries
        self.game_format = game_format
        self.include_idx = include_idx

    def run_games(self, total_games: int, write_out = None, openings = [], group_size = 1, lw_elo: int = 1350, up_elo: int = 2750):
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

        if self.local_rank == 0:
            prog_bar = tqdm(total = total_games)

        if len(openings) > 0:
            # print("Total Games:", total_games)
            # print("Openings:", len(openings))
            groups = total_games // group_size
            game_openings = random.sample(openings, k = groups)
            # print("Selected Openings:", len(game_openings))
            game_openings = sum([[opening] * group_size for opening in game_openings], [])
            # print("Spread Openings:", len(game_openings))
            game_perspectives = random.choices([0, 1], k = groups)
            # print("Selected Perspectives:", len(game_perspectives))
            game_perspectives = sum([[perspective] * group_size for perspective in game_perspectives], [])
            # print("Spread Perspectives:", len(game_perspectives))
            opponent_elos = random.choices(range(lw_elo, up_elo + 1, 100), k = groups)
            opponent_elos = sum([[elo] * group_size for elo in opponent_elos], [])

            game_states = [GameState(idx, self.adjudicator, self.p_names, [opponent_elos[idx] if "Stockfish" in p_name else None for p_name in self.p_names], opening = game_openings[idx], w_player_id = game_perspectives[idx], invalid_retries = self.invalid_retries, format = self.game_format, include_idx = self.include_idx) for idx in range(self.eval_bsz)]
        else:
            game_states = [GameState(idx, self.adjudicator, self.p_names, [random.choice(range(lw_elo, up_elo + 1, 100)) if "Stockfish" in p_name else None for p_name in self.p_names], opening = "", w_player_id = random.randint(0, 1), invalid_retries = self.invalid_retries, format = self.game_format, include_idx = self.include_idx) for idx in range(self.eval_bsz)]

        base_game_id = self.eval_bsz

        player_bszs = defaultdict(list)
        player_aslen = defaultdict(list)
        player_vslen = defaultdict(list)
        player_move_times = defaultdict(list)
        player_times = defaultdict(list)

        sf_player_id = self.p_names.index("Stockfish")
        turn = sf_player_id if sf_player_id != -1 else random.randint(0, 1)

        while games_played < total_games:
            turn_games = [game_state for game_state in game_states if game_state.turn == turn]
            curr_player = self.player0 if turn == 0 else self.player1
            if len(turn_games) > 0:
                player_bszs[self.p_names[turn]].append(len(turn_games))
                slens = [len(game_state.state.split(" ")) * 2 for game_state in turn_games]
                player_aslen[self.p_names[turn]].append(sum(slens) / len(turn_games))
                player_vslen[self.p_names[turn]].append(np.var(slens))
                bf_time = time()
                curr_player.play(turn_games)
                interval = time() - bf_time
                player_move_times[self.p_names[turn]].append(interval / len(turn_games))
                player_times[self.p_names[turn]].append(interval)

            active_game_states = []
            for game_state in game_states:
                if not game_state.is_complete():
                    active_game_states.append(game_state)
                    continue

                if write_out is not None:
                    game_state.write_outcome(write_out)
                else:
                    g, p, r = game_state.get_gpr()
                    G, P, R = update_gpr(g, G, p, P, r, R, self.tokenize)
                    game_order[game_state.game_id] = games_played

                games_played += 1
                if self.local_rank == 0:
                    prog_bar.update(1)

            game_states = active_game_states
            turn = 1 - turn
            if turn != sf_player_id:
                continue

            new_games = min(self.eval_bsz - len(game_states), total_games - (games_played + len(game_states))) # Min(Bsz - active_games, total_games - (games_played + active_games))
            if len(openings) > 0:
                game_states += [GameState(base_game_id + game_id, self.adjudicator, self.p_names, [opponent_elos[base_game_id + game_id] if "Stockfish" in p_name else None for p_name in self.p_names], opening = game_openings[base_game_id + game_id], w_player_id = game_perspectives[base_game_id + game_id], invalid_retries = self.invalid_retries, format = self.game_format, include_idx = self.include_idx) for game_id in range(new_games)]
            else:
                game_states += [GameState(base_game_id + game_id, self.adjudicator, self.p_names, [random.choice(range(lw_elo, up_elo + 1, 100)) if "Stockfish" in p_name else None for p_name in self.p_names], w_player_id = random.randint(0, 1), invalid_retries = self.invalid_retries, format = self.game_format, include_idx = self.include_idx) for game_id in range(new_games)]

            base_game_id += new_games

        assert games_played == total_games

        if self.local_rank == 0:
            prog_bar.close()

        if write_out and self.local_rank == 0:
            print(f"Run {total_games} games: " + 
                "".join(
                    [
                        f"{self.p_names[player_id]} - {sum(player_move_times[self.p_names[player_id]]) / len(player_move_times[self.p_names[player_id]])}s/Move, {sum(player_times[self.p_names[player_id]])}s Overall" 
                        for player_id in range(2)
                    ]
                )
            )

        # if self.local_rank == 0:
        #     for player_id in range(2):
        #         pd.DataFrame({"Bsz": player_bszs[player_id], "Avg SLen": player_aslen[player_id], "SLen Variance": player_vslen[player_id], "Times": player_times[player_id]}).to_csv(f"./{self.p_names[player_id]}-time_settings.csv")

        if write_out:
            write_out.close()
        else:
            R = torch.tensor(R).type(torch.DoubleTensor)
            assert all([idx is not None for idx in game_order])
            G = G[game_order, :]
            P = P[game_order, :]
            R = R[game_order]
            return G, P, R

    def run_games_sb(self, total_games: int, write_out = None, openings = [], group_size = 1, lw_elo: int = 1350, up_elo: int = 2750):
        if write_out:
            write_out = open(write_out + str(self.local_rank), "w")
            for game in self.init_games:
                game.write_outcome(write_out)
        else:
            assert self.init_games is None or len(self.init_games) == 0, f"Given {len(self.init_games)} initialization games in data collection phase."
            G = None # B x S
            P = None # B x (S - 1)
            R = [] # B x 0 

        if self.local_rank == 0:
            prog_bar = tqdm(total = total_games)

        games_played = 0

        if len(openings) > 0:
            # print("Total Games:", total_games)
            # print("Openings:", len(openings))
            game_openings = random.choices(openings, k = total_games // group_size)
            # print("Selected Openings:", len(game_openings))
            game_openings = sum([[opening] * group_size for opening in game_openings], [])
            # print("Spread Openings:", len(game_openings))
            game_perspectives = random.choices([0, 1], k = total_games // group_size)
            # print("Selected Perspectives:", len(game_perspectives))
            game_perspectives = sum([[perspective] * group_size for perspective in game_perspectives], [])
            # print("Spread Perspectives:", len(game_perspectives))

        sf_player_idx = self.p_names.index("Stockfish")
        sf_player = self.player0 if sf_player_idx == 0 else self.player1
        gpt_player = self.player0 if sf_player_idx == 1 else self.player1

        while games_played < total_games:
            num_games = min(self.eval_bsz, total_games - games_played)
            if len(openings) > 0:
                game_states = [GameState(idx, self.adjudicator, self.p_names, [random.choice(range(lw_elo, up_elo + 1, 100)) if "Stockfish" in p_name else None for p_name in self.p_names], opening = game_openings[idx], w_player_id = game_perspectives[idx], invalid_retries = self.invalid_retries, format = self.game_format, include_idx = self.include_idx) for idx in range(games_played, games_played + num_games)]
            else:
                game_states = [GameState(idx, self.adjudicator, self.p_names, [random.choice(range(lw_elo, up_elo + 1, 100)) if "Stockfish" in p_name else None for p_name in self.p_names], opening = "", w_player_id = random.randint(0, 1), invalid_retries = self.invalid_retries, format = self.game_format, include_idx = self.include_idx) for idx in range(games_played, games_played + num_games)]

            all_games_complete = False

            start_pos = 0

            while not all_games_complete:
                # Play all Stockfish Games
                sf_games = [game_state for game_state in game_states if game_state.turn == sf_player_idx and not game_state.is_complete()]
                if len(sf_games) > 0:  
                    sf_player.play(sf_games)

                # Now all matches are GPT's Turn
                start_pos = gpt_player.play(game_states, start_pos = start_pos, sb = True)

                all_games_complete = all([game_state.is_complete() for game_state in game_states])
            
            # Process all completed games.
            for game_state in game_states:
                assert game_state.is_complete()
                if write_out is not None:
                    game_state.write_outcome(write_out)
                else:
                    g, p, r = game_state.get_gpr()
                    G, P, R = update_gpr(g, G, p, P, r, R, self.tokenize)

                games_played += 1

                if self.local_rank == 0:
                    prog_bar.update(1)

        if self.local_rank == 0:
            prog_bar.close()

        if write_out:
            write_out.close()
        else:
            R = torch.tensor(R).type(torch.DoubleTensor)
            G = G
            P = P
            R = R
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
    "move": 2,
    "action": 1
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

def sample_games(pi_theta, total_games, bsz, rank, tok_type = "move", tokenizer_path = "./tokenizer/tokenizers/move_token.pkl", self_play = False, write_out = None, sf_rating_games = "fast", sf_time = 0.1, use_opening_book = False, group_size = 1, invalid_retries = 0, game_format = "uci", include_idx = False, sf_workers = 14, sb = False, lw_elo: int = 1350, up_elo: int = 2750, temp: float = 1.0):
    synthetic_games = []
    if sf_rating_games == "fast" and not self_play:
        sf_ratings = range(lw_elo, up_elo + 1, 100)
        synthetic_games = sample_sf_games_fast(sf_ratings, games_per_pair = max(total_games // len(sf_ratings), 400))

    p0 = GPTPlayer(pi_theta, f"cuda:{rank}", max_move_size = MMS[tok_type], tok_type = tok_type, tokenizer_path = tokenizer_path, game_format = game_format, temp = temp)

    if self_play:
        p1 = GPTPlayer(pi_theta, f"cuda:{rank}", max_move_size = MMS[tok_type], tok_type = tok_type, tokenizer_path = tokenizer_path, game_format = game_format, temp = temp)
    else:
        p1 = StockfishPlayer(sf_time, sf_workers)

    tokenize = None
    if write_out is None:
        tokenize, _, _ = load_tokenizer(tok_type, tokenizer_path)

    arena = Arena(p0, p1, bsz, rank, tokenize, init_games = synthetic_games, invalid_retries = invalid_retries, game_format = game_format, include_idx = include_idx)

    openings = []
    if use_opening_book:
        openings = get_openings()

    if write_out:
        if not sb:
            arena.run_games(total_games, write_out, openings = openings, lw_elo = lw_elo, up_elo = up_elo)
        else:
            arena.run_games_sb(total_games, write_out, openings = openings, lw_elo = lw_elo, up_elo = up_elo)
    else:
        if not sb:
            G, P, R = arena.run_games(total_games, group_size = group_size, openings = openings, lw_elo = lw_elo, up_elo = up_elo)
        else:
            G, P, R = arena.run_games_sb(total_games, group_size = group_size, openings = openings, lw_elo = lw_elo, up_elo = up_elo)

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

def estimate_elo(pi_theta, eval_bsz, eval_games, rank, write_out, wait, tok_type = "move", tokenizer_path = "./tokenizer/tokenizers/move_token.pkl", world_size = None, use_opening_book = True, invalid_retries = 0, game_format = "uci", include_idx = False, sf_workers = 14, sb = False, lw_elo = 1350, up_elo = 2750, temp = 1.0):
    sample_games(pi_theta, eval_games, eval_bsz, rank, tok_type, tokenizer_path, write_out = write_out, use_opening_book = use_opening_book, invalid_retries = invalid_retries, game_format = game_format, include_idx = include_idx, sf_workers = sf_workers, sb = sb, lw_elo = lw_elo, up_elo = up_elo, temp = temp)
    wait()
    elo_m, elo_lw_m, elo_up_m = None, None, None
    if rank == 0:
        assert world_size is not None
        collate_games([write_out + str(r) for r in range(world_size)], write_out)

        elo_m, elo_lw_m, elo_up_m = calc_elo(write_out)
    
    wait()
    
    return elo_m, elo_lw_m, elo_up_m