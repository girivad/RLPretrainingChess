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
    return [
        re.sub(
            r"[0-9]+[\.]+",
            "",
            content
        ) for content in contents
        ]

def char_tokenize(contents, stoi, dtype):
        return np.array(
            [
                stoi[c] for c in contents
            ], dtype = dtype
        )
    
def map_detokenize(idx, itos):
    # idx may be a numpy array or a tensor or a List[List[int]]
    if isinstance(idx, torch.Tensor):
        idx = idx.cpu().detach().clone().numpy()
    # idx may be a single array or an array of arrays
    if not isinstance(idx[0], list):
        return "".join(itos[idx[j]] for j in range(len(idx)))
    return [
        "".join(itos[idx[i][j]] for j in range(len(idx[i]))) for i in range(len(idx))
    ]

def move_tokenize(cntnts, stoi, dtype):
    board = chess.Board()
    tokens = []

    cntnts = re.sub(
        r"[0-9]+\.",
        "",
        cntnts
    )

    for move in re.split(r"([; ])", cntnts):
        if len(move) == 0:
            continue
        if move in " ;":
            tokens.append(stoi[move])
            continue
        uci_mv = board.parse_san(move).uci()
        tokens.append(stoi[uci_mv])
        board.push_uci(uci_mv)

    tokens = np.array(tokens, dtype = dtype)

    return {"ids": tokens, "len": len(tokens)}

def hf_tokenize(tokenizer, ex, dtype, batch = False):
    ids = tokenizer(ex, return_type = "np")["input_ids"]

    if batch:
        return [id.astype(np.uint8) for id in ids]
    
    return ids[0].astype(dtype)

def load_tokenizer(tok_type, tokenizer_path):
    if tok_type == "hf":
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only = True)
        dtype = get_dtype(len(tokenizer.vocab))
        tokenize = lambda ex, batch = False: hf_tokenize(tokenizer, ex, dtype, batch = batch)
        detokenize = lambda idx: tokenizer.batch_decode(idx)

    else:
        with open(tokenizer_path, "rb") as f:
            meta = pickle.load(f)
        V, stoi, itos = meta["vocab_size"], meta["stoi"], meta["itos"]
        dtype = get_dtype(V)

        if tok_type == "char":
            tokenize = lambda ex, batch = False: char_tokenize(preproc_game(ex), stoi, dtype) if not batch else [char_tokenize(x, stoi, dtype) for x in preproc_game(ex)]
            detokenize = lambda idx: map_detokenize(idx, itos)
        elif tok_type == "move":
            tokenize = lambda ex, batch = False: move_tokenize(preproc_game(ex), stoi, dtype) if not batch else [move_tokenize(x, stoi, dtype) for x in preproc_game(ex)]
            detokenize = lambda idx: map_detokenize(idx, itos)

    return tokenize, detokenize