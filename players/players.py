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

    def get_config(self) -> dict:
        return {"play_time": self._play_time}

    def close(self):
        self._engine.quit()

# TODO: Implement Distributed Inference
class GPTPlayer(object):
    def __init__(self, ckpt_path, device = "", rank = 0, topk = 5, temp = 1):
        self.ckpt_path = ckpt_path
        self.device = device

        checkpoint = torch.load(ckpt_path, map_location=self.device)
        checkpoint_model_args = checkpoint['model_args']
        model_args = dict()
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']: # No need for Seer Parameters
            model_args[k] = checkpoint_model_args[k]

        # create the model
        gptconf = GPTConfig(**model_args)
        self.model = GPT(gptconf)

        state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        self.model.load_state_dict(state_dict)
        
        self.model = DDP(self.model, device_ids=[rank])
        self.model.eval()

        self.tokenizer, self.detokenizer = load_tokenizer()

        self.k = topk
        self.temperature = temp
        self.max_move_size = 5

    def play_moves(
        self, games: List[GameState]
    ):
        games = [torch.tensor(self.tokenizer({"state": game.state}, "state"), device = self.device) for game in games]
        idx_moves = self.model.generate_moves(games, max_move_size = self.max_move_size, temperature = self.temp, top_k = self.k)
        str_moves = self.detokenizer(idx_moves)
        moves = [move.split()[0] for move in str_moves]

        for game_state, move in zip(games, moves):
            if ";" in move:
                game_state.resign()
            else:
                game_state.register_move(move)

    def get_config(self) -> dict:
        return {"ckpt": self.ckpt_path, "topk": self.k}