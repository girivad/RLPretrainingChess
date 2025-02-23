import chess, torch
from model import GPT, GPTConfig
from typing import Optional, List
from tokenizer import load_tokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
from game_utils import GameState

# Adapted from https://github.com/adamkarvonen/chess_gpt_eval/blob/master/main.py

class StockfishPlayer(object):
    def __init__(self, play_time: float):
        self._play_time = play_time
        stockfish_path = "stockfish_exec"
        self._engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    def play_move(
        self, game_state: GameState
    ):
        self._engine.configure({"UCI_Elo": game_state.sf_rating})
        result = self._engine.play(game_state.board, chess.engine.Limit(time=self._play_time))

        if result.resigned:
            game_state.resign()
        elif result.draw_offered:
            game_state.draw()
        elif result.move is not None:       
            game_state.register_move(result.move)
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
    def __init__(self, model: GPT, max_move_size = 5, hf_tokenizer = False, tokenizer_dir = "./data/lichess_hf_dataset", topk = None, temp = 1):
        self.model = model
        self.model.eval()

        self.tokenizer, self.detokenizer = load_tokenizer(hf_tokenizer, tokenizer_dir)

        self.k = topk
        self.temperature = temp
        self.max_move_size = max_move_size
        self.char = not hf_tokenizer

    def play_moves(
        self, game_states: List[GameState]
    ):
        games = [torch.tensor(self.tokenizer(game.state), device = self.device) for game in game_states]

        # Decide games beyond context length
        red_game_states, red_games = []
        for game_state, game in zip(game_states, games):
            if game.size(0) > self.model.config.block_size:
                game_state.decide()
            else:
                red_game_states.append(game_state)
                red_games.append(game)
        game_states = red_game_states
        games = red_games            

        idx_moves = self.model.generate_moves(games, max_move_size = self.max_move_size, overwrite_spaces = self.char, temperature = self.temperature, top_k = self.k)
        str_moves = self.detokenizer(idx_moves)
        moves = [move.split()[0] for move in str_moves]

        for game_state, move in zip(game_states, moves):
            if ";" in move:
                game_state.resign()
            else:
                game_state.register_move(move, parse_move = "san" if self.char else "uci")

    def play(
        self, games_states: List[GameState]
    ):
        self.play_moves(games_states)

    def get_config(self) -> dict:
        return {"ckpt": self.ckpt_path, "topk": self.k}
    
    def close(self):
        pass