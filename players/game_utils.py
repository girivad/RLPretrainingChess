import random, chess
from chess import IllegalMoveError, InvalidMoveError, AmbiguousMoveError

init_elo = 1250

class GameState(object):
    def __init__(self, game_id, sf_engine):
        self.game_id = game_id
        self.board = chess.Board()
        self.state = ";"
        self.outcome = ""
        self.sf_rating = random.randint(1360, 2840)
        self.turn = random.randint(0, 1)
        self.w_player_id = self.turn
        self.termination = ""
        self.sf_engine = sf_engine
    
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

    def register_move(self, move: str, parse_move: bool = False):
        move_failed = False
        if parse_move:
            try:
                move_uci = self.board.parse_san(move)
            except IllegalMoveError:
                self.termination = f"Illegal Move: \'{move}\' given context: \'{self.state}\'; Player: \'{self.turn}\'"
                move_failed = True
            except InvalidMoveError:
                self.termination = f"Invalid Move: \'{move}\' given context: \'{self.state}\'; Player: \'{self.turn}\'"
                move_failed = True
            except AmbiguousMoveError:
                self.termination = f"Ambiguous Move: \'{move}\' given context: \'{self.state}\'; Player: \'{self.turn}\'"
                move_failed = True
            
            if not bool(move_uci):
                self.termination = f"Parsed Null Move."
                move_failed = True

        if move_failed:
            self.resign()
            return

        self.board.push(move_uci)
        self.state += move_uci + " " # TODO: Prevents the model from resigning...
        
        if len(self.state) >= 1015:
            self.decide()
            return

        outcome = self.board.outcome()

        if outcome is None:
            # Next Turn
            self.turn = 1 - self.turn
            return
        
        self.outcome = self.board.result()

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