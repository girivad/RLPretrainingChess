from itertools import product
import pickle as pkl
import argparse

file_dict = {idx + 1: file for idx, file in enumerate("abcdefgh")}

def format_move(f1, r1, f2, r2, promotion_piece = ""):
  return file_dict[f1] + str(r1) + file_dict[f2] + str(r2) + promotion_piece

def calculate_vocab():
    vocab = [" ", ";"]

    for f1, r1, f2, r2 in product(range(1, 9), range(1, 9), range(1, 9), range(1, 9)):
        if f1 == f2 and r1 == r2:
            continue
        if f2 == f1 and r2 != r1: # Queen R1: A2 = A1, D2 = D1 +- x
            vocab.append(format_move(f1, r1, f2, r2))
            continue
        if r2 == r1 and f2 != f1: # Queen R2: D2 = D1, A2 = A1 +- x
            vocab.append(format_move(f1, r1, f2, r2))
            continue
        f_diff = abs(f2 - f1)
        r_diff = abs(r2 - r1)
        if f_diff == r_diff: # Queen R3: D2 = D1 +- x, A2 = A1 +- x
            vocab.append(format_move(f1, r1, f2, r2))
            continue
        if f_diff in {1, 2} and r_diff in {1, 2}: # Knight R1/2: A2 = A1 +- 1, D2 = D1 +- 2; A2 = A1 +- 2, D2 = D1 +- 1.
            vocab.append(format_move(f1, r1, f2, r2))
            continue

    # Additional Accounting: Pawn Promotions
    for f1 in range(1, 9):
        for f2 in [f1 + 1, f1, f1 - 1]:
            if f2 > 8 or f2 < 1:
                continue
            for p in "qbnr":
                # White: [f1]7[f1]8[qbnr] + [f1]7[f1 +- 1]8[qbnr]
                vocab.append(format_move(f1, 7, f2, 8, p))
                # Black: [f1]2[f1]1[qbnr] + [f1]2[f1 +- 1]1[qbnr]
                vocab.append(format_move(f1, 2, f2, 1, p))

    return vocab

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_dst", type = str, required = True)
    args = parser.parse_args()
    
    vocab = calculate_vocab()
    assert len(vocab) == 1970

    tok_dict = {token: idx for idx, token in enumerate(vocab)}
    detok_dict = {idx: token for idx, token in enumerate(vocab)}

    tokenizer = {
        "vocab_size": len(vocab),
        "stoi": tok_dict,
        "itos": detok_dict
    }

    with open(args.tokenizer_dst, "wb") as f:
        pkl.dump(tokenizer, f)