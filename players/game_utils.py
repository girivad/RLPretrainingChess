import random, chess

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