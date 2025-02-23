import random, chess, torch, os
from chess import IllegalMoveError, InvalidMoveError, AmbiguousMoveError
from players import StockfishPlayer, GPTPlayer
from typing import List
from tokenizer import load_tokenizer

init_elo = 1250
STREAM_SIZE = 1024 ** 3

class GameState(object):
    def __init__(self, game_id, sf_engine, players = ["Stockfish", "GPT"]):
        self.game_id = game_id
        
        self.board = chess.Board()
        self.state = ";"
        self.G = [";"]
        self.P = [0]
        self.turn = random.randint(0, 1)
        self.w_player_id = self.turn
        
        self.outcome = ""
        self.termination = ""
        self.sf_engine = sf_engine
        
        self.players = players
        self.ratings = [init_elo] * len(players)
        sf_rating = random.randint(1360, 2840)
        for player_idx in range(len(self.players)):
            if "Stockfish" not in self.players[player_idx]:
                continue
            self.players[player_idx] = "Stockfish-{}".format(self.sf_rating)
            self.ratings[player_idx] = sf_rating
    
    def decide(self):
        # Decide who has the advantage in the game and adjudicate the winner
        # Based on https://github.com/adamkarvonen/chess_llm_interpretability/blob/0f61e667fb8a809deda29e5db6c113a0a88f9998/chess_utils.py#L69
        result = self.sf_engine.analyse(self.board, chess.engine.Limit(time = 0.01))
        w_cp = result["score"].white().score(mate_score = 10000)

        if w_cp > 100: # White has a 100 centipawn advantage
            self.outcome = "1-0"
            self.termination = f"Max Context Length: \'{self.w_player_id}\' has centipawn advantage."
        elif w_cp < -100:
            self.outcome = "0-1"
            self.termination = f"Max Context Length: \'{1 - self.w_player_id}\' has centipawn advantage."
        else:
            self.outcome = "1/2-1/2"
            self.termination = f"Max Context Length: No Centipawn advantage; Draw."

    def draw(self): 
        self.outcome = "1/2-1/2"
        if self.termination == "":
            self.termination = "Draw"

    def resign(self):
        w_outcome = 0 if self.turn == self.w_player_id else 1
        self.outcome = "{}-{}".format(
            w_outcome,
            1 - w_outcome
        )

        if self.termination == "":
            self.termination = f"Resignation: \'{self.turn}\' resigned."

    def register_move(self, move: str, parse_move: str = None):
        move_failed = False
        if parse_move is not None:
            try:
                if parse_move == "san":
                    move = self.board.parse_san(move)
                elif parse_move == "uci":
                    move = self.board.parse_uci(move)
            except IllegalMoveError:
                self.termination = f"Illegal Move: \'{move}\' given context: \'{self.state}\'; Player: \'{self.turn}\'"
                move_failed = True
            except InvalidMoveError:
                self.termination = f"Invalid Move: \'{move}\' given context: \'{self.state}\'; Player: \'{self.turn}\'"
                move_failed = True
            except AmbiguousMoveError:
                self.termination = f"Ambiguous Move: \'{move}\' given context: \'{self.state}\'; Player: \'{self.turn}\'"
                move_failed = True
            
            if not bool(move):
                self.termination = f"Parsed Null Move."
                move_failed = True

        if move_failed:
            self.resign()
            return

        self.board.push(move)
        self.state += move + " "

        self.G.append(move + " ")
        player_type = (-1 ** (1 * (self.turn != self.w_player_id))) if "GPT" in self.players[self.turn] else 0
        self.P.append(player_type)
        assert len(self.G) - 1 == len(self.P)
        
        # if len(self.state) >= 1015: # TODO: Verify that player-level context length monitoring is sufficient
        #     self.decide()
        #     return

        outcome = self.board.outcome()

        if outcome is None:
            # Next Turn
            self.turn = 1 - self.turn
            return
        
        self.outcome = self.board.result()

    def is_complete(self):
        return self.outcome != ""

    def get_gpr(self):
        R = 1 if self.outcome == "1-0" else -1 if self.outcome == "0-1" else 0
        return self.G, self.P, R

    def write_outcome(self, pgn_file):
        assert self.is_complete()

        w_player = self.players[self.w_player_id]
        b_player = self.players[1 - self.w_player_id]
        w_rating = self.ratings[self.w_player_id]
        b_rating = self.ratings[1 - self.w_player_id]

        game = '''[White \"{} {}\"]\n[Black \"{} {}\"]\n[Result \"{}\"]\n{}'''.format(
            w_player, w_rating, b_player, b_rating, self.outcome, self.outcome
        )

        pgn_file.write(game)

def update_gpr(g, G, p, P, r, R, tokenize):
    g = tokenize(g, batch = True)
    p = sum([[ptype] * len(tokens) for ptype, tokens in zip(p, g)], [])
    g = sum(g, [])
    
    S = G.size(-1) if G is not None else len(g)
    g = g + [0] * (S - len(p)) # Random Token 0, will be dropped in Loss calculation as P is also 0.
    p = p + [0] * (S - len(p))
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
        self.adjudicator = chess.engine.SimpleEngine.popen_uci("stockfish_exec")
    
    def run_games(self, total_games: int, write_out = None):
        if write_out:
            write_out = open(write_out + self.local_rank, "w")
        else:
            G = None # B x S
            P = None # B x (S - 1)
            R = [] # B x 0 
        games_played = 0
        game_states = [GameState(idx, self.adjudicator, [type(self.player0).__name__, type(self.player1).__name__]) for idx in range(self.eval_bsz)]
        base_game_id = self.eval_bsz

        while games_played < total_games:
            while len(game_states) > 0:
                p0_games = [game_state for game_state in game_states if game_state.turn == 0]
                p1_games = [game_state for game_state in game_states if game_state.turn == 1]

                self.player0.play(p0_games)
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
                        assert self.tokenize is not None
                        G, P, R = update_gpr(g, G, p, P, r, R, self.tokenize)

                    games_played += 1

                game_states = reduced_game_states
                new_games = min(self.eval_bsz - len(game_states), total_games - (games_played + len(game_states))) # Min(Bsz - reduced_games, total_games - (games_played + reduced_games))
                game_states += [GameState(base_game_id + game_id, self.adjudicator, [type(self.player0).__name__, type(self.player1).__name__]) for game_id in range(new_games)]
                base_game_id += new_games

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
        local_pgn = open(file, "rb")
        stream = local_pgn.read(STREAM_SIZE)

        while stream is not None:
            global_pgn.write(stream)
            stream = local_pgn.read(STREAM_SIZE)

        local_pgn.close()
        os.remove(file)
    
    global_pgn.close()

def sample_games(pi_theta, total_games, bsz, rank, hf_tokenizer = False, tokenizer_dir = "./data/lichess_hf_dataset", self_play = False, write_out = None):
    p0 = GPTPlayer(pi_theta, max_move_size = (5 if not hf_tokenizer else 1), hf_tokenizer = hf_tokenizer, tokenizer_dir = tokenizer_dir)
    if self_play:
        p1 = GPTPlayer(pi_theta, max_move_size = (5 if not hf_tokenizer else 1), hf_tokenizer = hf_tokenizer, tokenizer_dir = tokenizer_dir)
    else:
        p1 = StockfishPlayer(0.01)

    tokenize = None
    if write_out is None:
        tokenize, _ = load_tokenizer(hf_tokenizer, tokenizer_dir)

    arena = Arena(p0, p1, bsz, rank, tokenize)
    if write_out:
        arena.run_games(total_games, write_out)
    else:
        G, P, R = arena.run_games(total_games)
        return G, P, R

def calc_elo(pgn_file):
    print("Dummy Elo")
    return 1250

def estimate_elo(pi_theta, eval_bsz, eval_games, rank, write_out, hf_tokenizer = False, tokenizer_dir = "./data/lichess_hf_dataset", world_size = None):
    sample_games(pi_theta, eval_games, eval_bsz, rank, hf_tokenizer, tokenizer_dir, write_out = write_out)

    if rank == 0:
        assert world_size is not None
        collate_games([write_out + r for r in range(world_size)], write_out)

        return calc_elo(write_out)