import random, chess, re
from chess import IllegalMoveError, InvalidMoveError, AmbiguousMoveError
import chess.pgn

STREAM_SIZE = 1024 ** 3

class GameState(object):
    def __init__(self, game_id, sf_engine, players = ["Stockfish", "GPT"], ratings = None, opening = "", w_player_id = 0, invalid_retries = 0, format = "uci", include_idx = False):
        self.game_id = game_id
        
        self.board = chess.Board()
        if format == "pgn":
            self.node = chess.pgn.Game()
        self.state = ";1." if format == "pgn" and include_idx else ";"
        self.G = [self.state]
        self.P = [0]
        self.turn = w_player_id
        self.w_player_id = w_player_id
        
        self.outcome = ""
        self.termination = ""
        self.sf_engine = sf_engine

        self.format = format
        self.include_idx = include_idx
        
        self.players = [p_name for p_name in players]

        if ratings is not None:
            assert len(ratings) == len(players), f"Provided {len(ratings)} ratings for {len(players)} players."
            self.ratings = [r for r in ratings]
            for p_idx in range(len(self.players)):
                if self.ratings[p_idx] is None:
                    continue

                self.players[p_idx] = self.players[p_idx] + "-" + str(self.ratings[p_idx])

        self.retries = invalid_retries
        self.retry_limit = invalid_retries

        self.move_idx = 1
        if opening is not None and len(opening) > 0:
            for move in re.split("(?:(?:[0-9]+\.)|(?:[; ]))", opening):
                if len(move) == 0:
                    continue

                self.register_move(move, parse_move = "san")

                if self.is_complete():
                    raise Exception(f"Opening {opening} was invalid, completed the game at move {self.move_idx}: {move}.")
                
    @staticmethod
    def init_terminal_game(outcome, w_player_id, p_names = ["Stockfish", "GPT"], ratings = None):
        game_state = GameState(-1, None, p_names, ratings)
        game_state.outcome = outcome
        game_state.w_player_id = w_player_id
        return game_state

    def decide(self):
        # Decide who has the advantage in the game and adjudicate the winner
        # Based on https://github.com/adamkarvonen/chess_llm_interpretability/blob/0f61e667fb8a809deda29e5db6c113a0a88f9998/chess_utils.py#L69
        result = self.sf_engine.analyse(self.board, chess.engine.Limit(time = 0.01))
        w_cp = result["score"].white().score(mate_score = 10000)

        if w_cp > 100: # White has a 100 centipawn advantage
            self.outcome = "1-0"
            self.termination = f"Max Context Length: \'{self.players[self.w_player_id]}\' has centipawn advantage."
        elif w_cp < -100:
            self.outcome = "0-1"
            self.termination = f"Max Context Length: \'{self.players[1 - self.w_player_id]}\' has centipawn advantage."
        else:
            self.outcome = "1/2-1/2"
            self.termination = f"Max Context Length: No Centipawn advantage; Draw."

        if self.game_id == 0:
            print(self.termination)

    def draw(self): 
        self.outcome = "1/2-1/2"
        if self.termination == "":
            self.termination = "Draw"

        if self.game_id == 0:
            print(self.termination)

    def resign(self):
        w_outcome = 0 if self.turn == self.w_player_id else 1
        assert w_outcome is not None
        self.outcome = "{}-{}".format(
            w_outcome,
            1 - w_outcome
        )

        if self.termination == "":
            self.termination = f"Resignation: \'{self.players[self.turn]}\' resigned."
        
        if self.game_id == 0:
            print(self.termination)

    def register_move(self, input_move: str, parse_move: str = "uci"):
        move_failed = False
        move_str = input_move

        assert type(input_move) == str, (input_move, type(input_move))

        if self.game_id == 0:
            print(f"Register Move: \'{input_move}\'")

        try:
            if parse_move == "san":
                move = self.board.parse_san(input_move)
            elif parse_move == "uci":
                move = self.board.parse_uci(input_move)
        except IllegalMoveError:
            if self.retries > 0:
                if self.game_id == 0:
                    print(self.termination)
                    print("Retrying move...")
                self.termination = ""
                self.retries -= 1
                return

            self.termination = f"Illegal Move: \'{move_str}\' given context: \'{self.state}\'; Player: \'{self.players[self.turn]}\'"
            move_failed = True
        except InvalidMoveError:
            if self.retries > 0:
                if self.game_id == 0:
                    print(self.termination)
                    print("Retrying move...")
                self.termination = ""
                self.retries -= 1
                return

            self.termination = f"Invalid Move: \'{move_str}\' given context: \'{self.state}\'; Player: \'{self.players[self.turn]}\'"
            move_failed = True
        except AmbiguousMoveError:
            if self.retries > 0:
                if self.game_id == 0:
                    print(self.termination)
                    print("Retrying move...")
                self.termination = ""
                self.retries -= 1
                return

            self.termination = f"Ambiguous Move: \'{move_str}\' given context: \'{self.state}\'; Player: \'{self.players[self.turn]}\'"
            move_failed = True
        except Exception as err:
            print("Error:", err, "from parsing move", move_str)
            move_failed = True

        if not move_failed and not bool(move):
            self.termination = f"Parsed Null Move."
            move_failed = True

        if not move_failed and self.format == "pgn":
            self.node = self.node.add_variation(move)

        if not move_failed:
            move_str = move.uci() if self.format == "uci" else str(self.node).split(" ")[-1]
            move_str += " "

        self.state += move_str
        if self.game_id == 0:
            print(f"State: \'{self.state}\'")
        self.G.append(move_str)
        player_type = (-1 ** (1 * (self.turn != self.w_player_id))) if "GPT" in self.players[self.turn] else 0
        self.P.append(player_type)
        assert len(self.G) == len(self.P)

        self.retries = self.retry_limit

        if move_failed:
            self.resign()
            return

        self.board.push(move)
        outcome = self.board.outcome()

        if outcome is None:
            self.turn = 1 - self.turn

            if self.format == "pgn" and self.turn == self.w_player_id:
                self.move_idx += 1

            if self.format == "pgn" and self.turn == self.w_player_id and self.include_idx:
                self.state += f"{self.move_idx}."
                self.G.append(f"{self.move_idx}.")
                self.P.append(0)

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

        assert w_player is not None
        assert b_player is not None
        assert self.outcome is not None

        game = '''[White \"{}\"]\n[Black \"{}\"]\n[Result \"{}\"]\n{}\n'''.format(
            w_player, b_player, self.outcome, self.outcome
        )

        pgn_file.write(game)

def get_openings():
    openings = []

    # Eco Opening Book based on "Grandmaster-level Chess without Search" (https://github.com/google-deepmind/searchless_chess/blob/main/src/tournament.py#L195)
    with open("./openings/eco_openings.pgn", "r") as openings_file:
        for line in openings_file.readlines():
            if len(line.strip()) == 0 or "[" in line or "]" in line:
                continue
            if not line.startswith("1."):
                continue
            line = line.strip().removesuffix("1/2-1/2").removesuffix("1-0").removesuffix("0-1").strip()
            openings.append(line)

    # print(f"Retrieved {len(openings)} openings.")
    return openings