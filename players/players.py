import chess, random, torch
from model import GPT, GPTConfig
from typing import Optional, List
from tokenizer import load_tokenizer
import numpy as np

# Adapted from https://github.com/adamkarvonen/chess_gpt_eval/blob/master/main.py

# Define base Player class
class Player:
    def get_move(self, board: chess.Board, game_state: str, temperature: float, **kwargs) -> str:
        raise NotImplementedError

    def get_config(self) -> dict:
        raise NotImplementedError
    
class StockfishPlayer(Player):
    def __init__(self, play_time: float):
        self._play_time = play_time
        # If getting started, you need to run brew install stockfish
        stockfish_path = "stockfish_exec"
        self._engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    def get_move(
        self, board: chess.Board, game_state: str, temperature: float, elo: int = None, skill: int = None
    ) -> Optional[str]:
        assert elo or skill
        self._elo = elo
        self._skill = skill

        if self._skill == -2:
            legal_moves = list(board.legal_moves)
            random_move = random.choice(legal_moves)
            return board.san(random_move)

        if self._elo is not None:
            self._engine.configure({"UCI_Elo": self._elo})
            result = self._engine.play(board, chess.engine.Limit(time=self._play_time))

        elif self._skill < 0:
            self._engine.configure({"Skill Level": 0})
            result = self._engine.play(
                board, chess.engine.Limit(time=1e-8, depth=1, nodes=1)
            )

        else:
            self._engine.configure({"Skill Level": self._skill})
            result = self._engine.play(board, chess.engine.Limit(time=self._play_time))
        
        if result.move is None:
            return None
        
        return board.san(result.move)

    def get_config(self) -> dict:
        return {"skill_level": self._skill_level, "Elo": self._elo, "play_time": self._play_time}

    def close(self):
        self._engine.quit()

# Define base Player class
class BatchPlayer:
    def get_moves(self, board: chess.Board, game_state: str, temperature: float, **kwargs) -> str:
        raise NotImplementedError

    def get_config(self) -> dict:
        raise NotImplementedError

# TODO: Implement Distributed Inference
class GPTPlayer(Player):
    def __init__(self, ckpt_path, device = ""):
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

        self.model.eval()

        self.tokenizer, self.detokenizer = load_tokenizer()

        self.k = 5
        self.max_new_tokens = 5

    def get_moves(
        self, board: chess.Board, games: List[str], temperature: float
    ) -> Optional[str]:
        games = torch.tensor(
            map(
                lambda game: self.tokenizer({"gm_cntnts": game}, "gm_cntnts"), 
                games
            ), 
            device = self.device
        )

        idx_moves = self.model.generate(games, max_new_tokens = self.max_new_tokens, temperature = temperature, top_k = self.k)[:, -self.max_new_tokens:]
        str_moves = self.detokenizer(idx_moves)
        
        return [move.split()[0] for move in str_moves]

    def get_config(self) -> dict:
        return {"ckpt": self.ckpt_path, "topk": self.k}

    def close(self):
        pass