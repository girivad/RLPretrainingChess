import random, chess, torch, os
from chess import IllegalMoveError, InvalidMoveError, AmbiguousMoveError

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
        self.sf_rating = random.randint(1360, 2840)
        for player_idx in range(len(self.players)):
            if "Stockfish" not in self.players[player_idx]:
                continue
            assert self.sf_rating is not None
            self.players[player_idx] = "Stockfish-{}".format(self.sf_rating)
            self.ratings[player_idx] = self.sf_rating
    
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
        assert w_outcome is not None
        self.outcome = "{}-{}".format(
            w_outcome,
            1 - w_outcome
        )

        if self.termination == "":
            self.termination = f"Resignation: \'{self.turn}\' resigned."

    def register_move(self, move: str, parse_move: str = None):
        move_failed = False
        # print(f"register move: \'{move}\'")
        if parse_move is not None:
            try:
                if parse_move == "san":
                    move = self.board.parse_san(move)
                elif parse_move == "uci":
                    move = self.board.parse_uci(move)
                # print("Parsed.")
            except IllegalMoveError:
                # print("IlME")
                self.termination = f"Illegal Move: \'{move}\' given context: \'{self.state}\'; Player: \'{self.turn}\'"
                move_failed = True
            except InvalidMoveError:
                # print("InME")
                self.termination = f"Invalid Move: \'{move}\' given context: \'{self.state}\'; Player: \'{self.turn}\'"
                move_failed = True
            except AmbiguousMoveError:
                # print("AME")
                self.termination = f"Ambiguous Move: \'{move}\' given context: \'{self.state}\'; Player: \'{self.turn}\'"
                move_failed = True
            except Exception as err:
                print("Error:", err)
                move_failed = True
            if not bool(move):
                # print("Null move")
                self.termination = f"Parsed Null Move."
                move_failed = True

        self.state += str(move) + " "
        print("State:", self.state)

        self.G.append(str(move) + " ")
        player_type = (-1 ** (1 * (self.turn != self.w_player_id))) if "GPT" in self.players[self.turn] else 0
        self.P.append(player_type)
        assert len(self.G) == len(self.P)

        if move_failed:
            # print("Resign because move failed")
            self.resign()
            return

        self.board.push(move)

        # if len(self.state) >= 1015: # TODO: Verify that player-level context length monitoring is sufficient
        #     self.decide()
        #     return

        outcome = self.board.outcome()

        if outcome is None:
            self.turn = 1 - self.turn
            return
        
        self.outcome = self.board.result()

    def is_complete(self):
        return self.outcome != "" and self.outcome is not None

    def get_gpr(self):
        R = 1 if self.outcome == "1-0" else -1 if self.outcome == "0-1" else 0
        return self.G, self.P, R

    def write_outcome(self, pgn_file):
        assert self.is_complete()

        w_player = self.players[self.w_player_id]
        b_player = self.players[1 - self.w_player_id]
        w_rating = self.ratings[self.w_player_id]
        b_rating = self.ratings[1 - self.w_player_id]

        assert w_player is not None
        assert w_rating is not None
        assert b_player is not None
        assert b_rating is not None
        assert self.outcome is not None

        game = '''[White \"{} {}\"]\n[Black \"{} {}\"]\n[Result \"{}\"]\n{}\n'''.format(
            w_player, w_rating, b_player, b_rating, self.outcome, self.outcome
        )

        pgn_file.write(game)
