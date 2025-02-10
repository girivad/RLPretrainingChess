# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from tqdm import tqdm
import numpy as np
from datasets import load_dataset  # huggingface datasets
import pickle
import argparse
from transformers import AutoTokenizer
import re

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 1 # 8
dtype = np.uint8  # Currently there are only 32 tokens in the chess LLMs vocab

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

def load_tokenizer():
    dropped_chars = ""

    meta_path = os.path.join(os.path.dirname(__file__), "meta.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    stoi = meta["stoi"]

    def tokenize(example, column_name):
        contents = example[column_name]
        contents = re.sub(
            r"[{}]".format("".join(dropped_chars)),
            "",
            contents
        )
        contents = re.sub(
            r"[0-9]+[\.]+",
            "",
            contents
        )
        return np.array(
            [
                stoi[c] for c in contents if c not in dropped_chars
            ], dtype = dtype
        )

    return tokenize

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str, required = True)
parser.add_argument("--tokenizer", type = str, required = False, default = None)
args = parser.parse_args()

if __name__ == "__main__":

    dataset_path = "adamkarvonen/chess_games"
    file_path = args.dataset

    # Load the dataset
    dataset = load_dataset(dataset_path, data_files=file_path, split = "train")
    tokenizer = load_tokenizer(args.tokenizer is not None, args.tokenizer)

    # by default only contains the 'train' split, so create a test split
    split_dataset = dataset.train_test_split(
        test_size=0.01, seed=2357, shuffle=True
    )
    split_dataset["val"] = split_dataset.pop("test")  # rename the test split to val

    # this results in:
    # >>> split_dataset
    # DatasetDict({
    #     train: Dataset({
    #         features: ['text'],
    #         num_rows: 8009762
    #     })
    #     val: Dataset({
    #         features: ['text'],
    #         num_rows: 4007
    #     })
    # })

    # we now want to tokenize the dataset. Using meta.pkl in the same directory as this file
    tokenizer = load_tokenizer(args.tokenizer is not None, args.tokenizer)

    # to read the bin files later, e.g. with numpy:
    # m = np.memmap('train.bin', dtype=np.uint8, mode='r')
    # print(split_dataset["val"][0])
    # print(len(split_dataset["val"]["transcript"][0]))

    # For verifying that all games are 1024 tokens long
    # for game in split_dataset["train"]["transcript"]:
    #     if len(game) != 1024:
    #         print(len(game))
    #         print(game)
    #         break
    # print(stoi)

    column_name = "transcript"

    def process(example):
        ids = tokenizer(example, column_name)
        out = {"ids": ids, "len": len(ids)}
        return out

    # tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=[column_name],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        print(f"{split} has {arr_len} tokens")
        filename = os.path.join(os.path.dirname(__file__), f"{split}.bin")
        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
        print(arr.shape)
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
            # Batch together samples for faster write
            batch = dset.shard(
                num_shards=total_batches, index=batch_idx, contiguous=True
            ).with_format("numpy")

            arr_batch = np.concatenate(batch["ids"])

            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
