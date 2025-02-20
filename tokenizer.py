import os, pickle, re
import numpy as np
import torch
from transformers import AutoTokenizer

def preproc_game(contents):
    return re.sub(
            r"[0-9]+[\.]+",
            "",
            contents
        )

def char_tokenize(contents, stoi, dtype):
        return np.array(
            [
                stoi[c] for c in contents
            ], dtype = dtype
        )
    
def char_detokenize(idx, itos):
    # idx may be a numpy array or a tensor or a List[List[int]]
    if isinstance(idx, torch.Tensor):
        idx = idx.detach().clone().numpy()
    # idx may be a single array or an array of arrays
    if type(idx[0]) == int:
        return "".join(itos[idx[j]] for j in range(len(idx)))
    return [
        "".join(itos[idx[i][j]] for j in range(len(idx[i]))) for i in range(len(idx))
    ]

def load_tokenizer(hf_tokenizer, tokenizer_dir = "./data/lichess_hf_dataset", dtype = np.uint8):
    if not hf_tokenizer:
        meta_path = os.path.join(tokenizer_dir, "meta.pkl")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        stoi, itos = meta["stoi"], meta["itos"]
        
        tokenize = lambda ex, col: char_tokenize(preproc_game(ex[col]), stoi, dtype)
        detokenize = lambda idx: char_detokenize(idx, itos)

    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only = True)
        tokenize = lambda ex, col: tokenizer(preproc_game(ex[col]), return_tensors = 'np')["input_ids"].astype(dtype)
        detokenize = lambda idx: tokenizer.batch_decode(idx)

    return tokenize, detokenize