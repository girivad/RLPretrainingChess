# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from tqdm import tqdm
import numpy as np
from datasets import load_dataset, Dataset  # huggingface datasets
import argparse
from tokenizer.scripts.tokenizer import load_tokenizer

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8
c_dtype = np.uint8  # Currently there are only 32 tokens in the chess LLMs vocab
t_dtype = np.uint16

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

def pack(ds, blk_size = 1024, dtype = c_dtype):
    blk = []

    for ex in ds:
        ex_ids = list(ex["ids"])

        rlen = min(len(ex_ids), blk_size - len(blk))
        blk += ex_ids[:rlen]

        if len(blk) == blk_size:
            yield {"ids": np.array(blk, dtype = dtype), "len": blk_size}
            blk = []

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str, required = True)
parser.add_argument("--out_dir", default = os.path.dirname(__file__), type = str)
parser.add_argument("--tok_type", type = str, required = True)
parser.add_argument("--tokenizer_path", type = str, default = "./data/lichess_hf_dataset/meta.pkl")
args = parser.parse_args()

if __name__ == "__main__":

    dataset_path = "adamkarvonen/chess_games"
    file_path = args.dataset

    # Load the dataset
    dataset = load_dataset(dataset_path, data_files = file_path, split = "train")

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
    tokenizer, detokenizer, dtype = load_tokenizer(args.tok_type, args.tokenizer_path)

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
        assert tokenizer is not None
        assert example[column_name][0] != ";" # Shouldn't be doubling the SOG symbol.
        ids = tokenizer(";" + example[column_name], return_type = "np")
        out = {"ids": ids, "len": len(ids)}
        return out

    # tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=[column_name],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    for split in tokenized.keys():
        def split_pack():
            yield from pack(tokenized[split], dtype = dtype)
        tokenized[split] = Dataset.from_generator(
            split_pack
        )

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        print(f"{split} has {arr_len} tokens")
        filename = os.path.join(args.out_dir, f"{split}.bin")
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
