import chess, torch
import chess.engine
from model import GPT
from typing import List
from tokenizer.scripts.tokenizer import load_tokenizer
from players.game_utils import GameState

# Adapted from https://github.com/adamkarvonen/chess_gpt_eval/blob/master/main.py

class StockfishPlayer(object):
    def __init__(self, play_time: float):
        self._play_time = play_time
        stockfish_path = "./stockfish_exec"
        self._engine = chess.engine.SimpleEngine.popen_uci(stockfish_path, timeout = None)

    def name(self):
        return "Stockfish"

    def play_move(
        self, game_state: GameState
    ):
        self._engine.configure({"UCI_Elo": game_state.ratings[game_state.turn]})
        result = self._engine.play(game_state.board, chess.engine.Limit(time=self._play_time))

        if result.resigned:
            game_state.resign()
        elif result.draw_offered:
            game_state.draw()
        elif result.move is not None:       
            game_state.register_move(result.move.uci())
        else:
            raise Exception("Stockfish played invalid move in state:\n" + game_state.state)

    def play(
        self, games_states: List[GameState]
    ):
        for game_state in games_states:
            self.play_move(game_state)

    def get_config(self) -> dict:
        return {"play_time": self._play_time}

    def close(self):
        self._engine.quit()

class GPTPlayer(object):
    def __init__(self, model: GPT, device, max_move_size = 5, tok_type = "move", tokenizer_path = "./tokenizer/tokenizers/move_token.pkl", topk = None, temp = 1, game_format = "uci"):
        self.model = model
        self.model.eval()

        self.device = device

        self.tokenizer, self.detokenizer, _ = load_tokenizer(tok_type, tokenizer_path)

        self.k = topk
        self.temperature = temp
        self.max_move_size = max_move_size

        self.input_type = game_format

    def name(self):
        return "GPT"

    def play_moves(
        self, game_states: List[GameState]
    ):
        games = [self.tokenizer(game.state, return_type = "torch", pgn = False).to(self.device) for game in game_states]
        # Decide games beyond context length
        red_game_states, red_games = [], []
        for game_state, game in zip(game_states, games):
            if game.size(0) > self.model.module.config.block_size:
                game_state.decide()
            else:
                red_game_states.append(game_state)
                red_games.append(game)
        game_states = red_game_states
        games = red_games            

        temperature = torch.tensor([min((game_state.retry_limit - game_state.retries)/(game_state.retry_limit) * 1 + 0.001, 0.5) if game_state.retry_limit != 0 else 1 for game_state in game_states]).view(-1, 1).to(self.device)
        # print("Games Input:", self.detokenizer(games, batch = True))
        idx_moves = self.model.module.generate_moves(games, device = self.device, max_move_size = self.max_move_size, overwrite_spaces = True, temperature = temperature, top_k = self.k, space_token = int(self.tokenizer(" ")[0]), eos_token = int(self.tokenizer(";")[0]))
        str_moves = self.detokenizer(idx_moves, batch = True)
        moves = [move.split()[0] for move in str_moves]
        
        for game_state, move in zip(game_states, moves):
            if ";" in move:
                # if self.device == "cuda:0":
                #     print("Game resigned.")
                game_state.resign()
            else:
                game_state.register_move(move, parse_move = self.input_type)
        
        # if self.device == "cuda:0":
            # print("Play Moves done.")

    def play(
        self, games_states: List[GameState]
    ):
        self.play_moves(games_states)

    def get_config(self) -> dict:
        return {"ckpt": self.ckpt_path, "topk": self.k}
    
    def close(self):
        pass