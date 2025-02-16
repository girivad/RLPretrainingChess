import os, pickle, re
import numpy as np
import torch
from typing import List

def load_tokenizer(dtype = np.uint8):
    dropped_chars = {".", "0", "9"}
    meta_path = os.path.join(os.path.dirname(__file__), "meta.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    stoi = meta["stoi"]
    vocab, _ = zip(*sorted([(token, idx) for token, idx in stoi.items() if token not in dropped_chars], key = lambda tokdx: tokdx[1]))
    stoi = {
        token: idx for idx, token in enumerate(vocab)
    }
    itos = {
        idx: token for idx, token in enumerate(vocab)
    }

    def tokenize(example, column_name):
        contents = example[column_name]
        contents = re.sub(
            r"[0-9]+[\.]+",
            "",
            contents
        )
        for char in dropped_chars:
            assert char not in contents, (contents, char)
        return np.array(
            [
                stoi[c] for c in contents
            ], dtype = dtype
        )
    
    def detokenize(idx):
        # idx may be a numpy array or a tensor or a List[List[int]]
        if isinstance(idx, torch.Tensor):
            idx = idx.detach().clone().numpy()
        # idx may be a single array or an array of arrays
        if type(idx[0]) == int:
            return "".join(itos[idx[j]] for j in range(len(idx)))
        return [
            "".join(itos[idx[i][j]] for j in range(len(idx[i]))) for i in range(len(idx))
        ]

    return tokenize, detokenize