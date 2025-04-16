import chess, torch
import chess.engine
from model import GPT
from typing import List
from tokenizer.scripts.tokenizer import load_tokenizer
from players.game_utils import GameState
import asyncio

# Adapted from https://github.com/adamkarvonen/chess_gpt_eval/blob/master/main.py

class StockfishPlayer(object):
    def __init__(self, play_time: float, workers: int = 28):
        self._play_time = play_time
        self.stockfish_path = "./stockfish_exec"
        self._event_loop = asyncio.new_event_loop()
        self.workers = workers

    def name(self):
        return "Stockfish"

    async def get_move(self, b_queue: asyncio.Queue, m_queue: asyncio.Queue):
        _, engine = await chess.engine.popen_uci(self.stockfish_path)
        while True:
            try:
                id, board, rating = await b_queue.get()
                await engine.configure({"UCI_Elo": rating, "UCI_LimitStrength": True})
                result = await engine.play(board, chess.engine.Limit(time = self._play_time))
                m_queue.put_nowait((id, result))
                b_queue.task_done()
            except asyncio.CancelledError:
                await engine.quit()
                break
            except Exception as err:
                m_queue.put_nowait(err)
                b_queue.task_done()

    async def get_moves(self, game_ins: List[tuple[int, chess.Board, int]]) -> List[chess.engine.PlayResult]:
        b_queue = asyncio.Queue()
        m_queue = asyncio.Queue()
        for game_in in game_ins:
            b_queue.put_nowait(game_in)
        
        tasks = [asyncio.create_task(self.get_move(b_queue, m_queue)) for _ in range(self.workers)]
        await b_queue.join()

        for task in tasks:
            task.cancel()

        await asyncio.gather(*tasks, return_exceptions = True)

        id_to_idx = {id: idx for idx, id in enumerate(list(zip(*game_ins))[0])}
        moves = [None] * len(game_ins)

        while m_queue.qsize() > 0:
            id, res = m_queue.get_nowait()
            moves[id_to_idx[id]] = res

        return moves

    def play(
        self, games_states: List[GameState]
    ):

        game_ins = [(game_state.game_id, game_state.board, game_state.ratings[game_state.turn]) for game_state in games_states]
        moves = self._event_loop.run_until_complete(self.get_moves(game_ins))

        for game_state, move in zip(games_states, moves):
            if move.resigned:
                game_state.resign()
            elif move.draw_offered:
                game_state.draw()
            elif move.move is not None:
                game_state.register_move(move.move.uci())
            else:
                raise Exception("Stockfish played invalid move in state:\n" + game_state.state)

    def get_config(self) -> dict:
        return {"play_time": self._play_time}

    def close(self):
        self._event_loop.close()

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
        self, game_states: List[GameState], start_pos = 0, sb = False
    ):
        
        if all((game_state.is_complete() for game_state in game_states)):
            return None

        games = [self.tokenizer(game.state, return_type = "torch", pgn = False).to(self.device) for game in game_states]
        red_game_states, red_games = [], []

        # Decide games beyond context length
        for game_state, game in zip(game_states, games):
            if game.size(0) >= self.model.module.config.block_size:
                game_state.decide()
            else:
                red_game_states.append(game_state)
                red_games.append(game)
        
        if len(red_game_states) == 0:
            return None # Returned start pos doesn't matter, as it is only considered in Static Batching, which needs to start a new batch anyway.
        
        # Should not remove decided games if implementing static batching.
        if not sb:
            game_states = red_game_states
            games = red_games            

        temperature = torch.tensor([min((game_state.retry_limit - game_state.retries)/(game_state.retry_limit) * 1 + 0.001, 0.5) if game_state.retry_limit != 0 else 1 for game_state in game_states]).view(-1, 1).to(self.device)
        completed_msk = torch.tensor([game_state.is_complete() for game_state in game_states], device = self.device)
        # if self.device == "cuda:0":
        #     print("Playing Moves on:", games, self.detokenizer(games, batch = True), "from start_pos:", start_pos, "Completed Mask:", completed_msk)
        idx_moves, new_start_pos = self.model.module.generate_moves(games, device = self.device, max_move_size = self.max_move_size, overwrite_spaces = True, temperature = temperature, top_k = self.k, space_token = int(self.tokenizer(" ")[0]), eos_token = int(self.tokenizer(";")[0]), start_pos = start_pos, kv_cache = sb, completed_msk = completed_msk)
        str_moves = self.detokenizer(idx_moves, batch = True)
        # if self.device == "cuda:0":
        #     print(str_moves)
        moves = [move.split(" ")[0] for move in str_moves]
        
        all_moves_success = True
        for game_state, move in zip(game_states, moves):
            if move[0] == ";":
                move_success = game_state.resign()
            else:
                move_success = game_state.register_move(move.split(";")[0], parse_move = self.input_type)

            all_moves_success = all_moves_success and move_success

        if all_moves_success and self.device == "cuda:0":
            print("Updating Start Position:", start_pos, "->", new_start_pos)
        elif self.device == "cuda:0":
            print("Retaining Start Position:", start_pos, "instead of", new_start_pos)

        return new_start_pos if all_moves_success else start_pos

    def play(
        self, games_states: List[GameState], start_pos = 0, sb = False
    ):
        return self.play_moves(games_states, start_pos = start_pos, sb = sb)

    def get_config(self) -> dict:
        return {"ckpt": self.ckpt_path, "topk": self.k}
    
    def close(self):
        pass