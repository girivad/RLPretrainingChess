import pickle, re
import numpy as np
import torch
import chess
from transformers import AutoTokenizer

def get_dtype(V):
    if V < 2 ** 8:
        return np.uint8
    if V < 2 ** 16:
        return np.uint16
    if V < 2 ** 32:
        return np.uint32

def preproc_game(contents):
    if not isinstance(contents, list):
        return re.sub(
            r"[0-9]+\.",
            "",
            contents
        )

    return [
        preproc_game(content) for content in contents
    ]

# Can return np.ndarray[dtype] or torch.Tensor[long]
def char_tokenize(contents, stoi, dtype, return_type = "np"):
        tokens = np.array(
            [
                stoi[c] for c in contents
            ], dtype = dtype
        )

        if return_type == "torch":
            tokens = torch.tensor(tokens).to(torch.long)
        return tokens
    
# Parses PGN to UCI and produces tokenized form.
# Can return np.ndarray[dtype] or torch.Tensor[long]
def move_tokenize(cntnts, stoi, dtype, return_type = "np", pgn = True):
    board = None if not pgn else chess.Board()
    tokens = []

    for move in re.split(r"([; ])", cntnts):
        if len(move) == 0:
            continue
        if move in " ;":
            tokens.append(stoi[move])
            continue

        if not pgn:
            uci_mv = move
        else:
            uci_mv = board.parse_san(move).uci() if pgn else move
            board.push_uci(uci_mv)

        tokens.append(stoi[uci_mv])

    tokens = np.array(tokens, dtype = dtype)

    if return_type == "torch":
        tokens = torch.tensor(tokens).to(torch.long)
    return tokens

def map_detokenize(idx, itos, batch = False):
    # idx may be a numpy array or a tensor or a List[List[int]]
    if isinstance(idx, torch.Tensor):
        idx = idx.cpu().detach().clone().numpy()
    # idx may be a single array or an array of arrays
    if not batch:
        return "".join(itos[int(idx[j])] for j in range(len(idx)))

    return [
        "".join(itos[int(idx[i][j])] for j in range(len(idx[i]))) for i in range(len(idx))
    ]

def hf_tokenize(tokenizer, ex, dtype, batch = False):
    ids = tokenizer(ex, return_type = "np")["input_ids"]

    if batch:
        return [id.astype(dtype) for id in ids]
    
    return ids[0].astype(dtype)

def load_tokenizer(tok_type, tokenizer_path):
    tokenize, detokenize = None, None
    if tok_type == "hf":
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only = True)
        V = len(tokenizer.vocab)
        dtype = get_dtype(V)
        tokenize = lambda ex, batch = False: hf_tokenize(tokenizer, ex, dtype, batch = batch)
        detokenize = lambda idx: tokenizer.batch_decode(idx)

    else:
        with open(tokenizer_path, "rb") as f:
            meta = pickle.load(f)
        V, stoi, itos = meta["vocab_size"], meta["stoi"], meta["itos"]
        dtype = get_dtype(V)

        if tok_type == "char":
            tokenize = lambda ex, batch = False, return_type = "np": char_tokenize(preproc_game(ex), stoi, dtype, return_type = return_type) if not batch else [char_tokenize(x, stoi, dtype, return_type = return_type) for x in preproc_game(ex)]
        elif tok_type == "move":
            tokenize = lambda ex, batch = False, return_type = "np": move_tokenize(preproc_game(ex), stoi, dtype, return_type = return_type) if not batch else [move_tokenize(x, stoi, dtype, return_type = return_type) for x in preproc_game(ex)]
        detokenize = lambda idx, batch = False: map_detokenize(idx, itos, batch)

    return tokenize, detokenize, dtype