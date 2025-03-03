from tokenizer import load_tokenizer
import re
import argparse
import torch
import numpy as np

parser = argparse.ArgumentParser()

if __name__ == "__main__":

    parser.add_argument("--tok_type", type = str, required = True)
    parser.add_argument("--tokenizer_path", type = str, required = True)
    parser.add_argument("--uci_in", type = str, default = "Y")
    parser.add_argument("--uci_out", type = str, default = "Y")
    args = parser.parse_args()

    uci_in = args.uci_in in ["y", "yes", "Y", "Yes", "true", "True", "TRUE"]
    uci_out = args.uci_out in ["y", "yes", "Y", "Yes", "true", "True", "TRUE"]

    print(f"Testing {args.tok_type} tokenizer at {args.tokenizer_path}.")
    print(f"Allowing PGN{"/UCI" if uci_in else ""} -> {"UCI" if uci_out else "PGN"}")

    tokenizer, detokenizer, dtype = load_tokenizer(args.tok_type, args.tokenizer_path)

    # Test on UCI Transcript
    uci_transcript = ";d2d4 e7e5 d4e5 f7f6 e5f6 d8f6 b1c3 f8b4 c1d2 b8c6 g1f3; d7d6 g2g3 c8g4 f1g2 e8c8 h2h3 g4f3 g2f3 b4c3 d2c3 f6g5 e1g1 g5c5 d1d4 c5d4 c3d4 c6d4 f3g2 d4c2 a1c1 c2b4 a2a3 b4a2 c1c2 g8f6 b2b4 c7c6 c2a2 f6d5 f1c1 d8e8 e2e3 d5b6 a3a4 b6c4 c1c4 d6d5 g2d5 c8d7 d5g2 b7b5 a4b5 c6b5 c4c5 a7a6 a2a6 e8e6 a6a7 d7d6 a7a6 d6e7 c5c7 e7f6 a6e6 f6e6 c7c5 h8d8 c5b5 d8d1 g1h2 e6d6 b5b7 d6e5 b7g7 h7h5 h3h4 d1b1 g7g5 e5d6 b4b5 d6c7 g5h5 c7b6 h5g5 b1b5 g5b5 b6b5 h4h5 b5c5 h5h6 c5d6 h6h7 d6e7 h7h8q e7f7 h8h6 f7g8 e3e4 g8f7 e4e5 f7g8 h6a6 g8h8 a6a8 h8g7 a8b7 g7g6 b7c6 g6h5 c6f6 h5g4 f2f3 g4h5 g3g4;"
    # Test on PGN Transcript:
    pgn_transcript = ";1.d4 e5 2.dxe5 f6 3.exf6 Qxf6 4.Nc3 Bb4 5.Bd2 Nc6 6.Nf3; d6 7.g3 Bg4 8.Bg2 O-O-O 9.h3 Bxf3 10.Bxf3 Bxc3 11.Bxc3 Qg5 12.O-O Qc5 13.Qd4 Qxd4 14.Bxd4 Nxd4 15.Bg2 Nxc2 16.Rac1 Nb4 17.a3 Na2 18.Rc2 Nf6 19.b4 c6 20.Rxa2 Nd5 21.Rc1 Rde8 22.e3 Nb6 23.a4 Nc4 24.Rxc4 d5 25.Bxd5 Kd7 26.Bg2 b5 27.axb5 cxb5 28.Rc5 a6 29.Rxa6 Re6 30.Ra7+ Kd6 31.Ra6+ Ke7 32.Rc7+ Kf6 33.Rxe6+ Kxe6 34.Rc5 Rd8 35.Rxb5 Rd1+ 36.Kh2 Kd6 37.Rb7 Ke5 38.Rxg7 h5 39.h4 Rb1 40.Rg5+ Kd6 41.b5 Kc7 42.Rxh5 Kb6 43.Rg5 Rxb5 44.Rxb5+ Kxb5 45.h5 Kc5 46.h6 Kd6 47.h7 Ke7 48.h8=Q Kf7 49.Qh6 Kg8 50.e4 Kf7 51.e5 Kg8 52.Qa6 Kh8 53.Qa8+ Kg7 54.Qb7+ Kg6 55.Qc6+ Kh5 56.Qf6 Kg4 57.f3+ Kh5 58.g4#;"
    cleaned_pgn_transcript = re.sub(r"[0-9]*\.", "", pgn_transcript)

    target = uci_transcript if uci_out else cleaned_pgn_transcript

    if uci_in:
        assert detokenizer(tokenizer(uci_transcript)) == target
    assert detokenizer(tokenizer(pgn_transcript)) == target, f"{detokenizer(tokenizer(pgn_transcript))}\n {target}"
    
    # Multi-Decoding
    if uci_in:
        assert detokenizer([tokenizer(uci_transcript), tokenizer(uci_transcript)], batch = True)[0] == target
        assert detokenizer([tokenizer(uci_transcript), tokenizer(uci_transcript)], batch = True)[1] == target
        assert detokenizer([tokenizer(pgn_transcript), tokenizer(uci_transcript)], batch = True)[0] == target
        assert detokenizer([tokenizer(uci_transcript), tokenizer(pgn_transcript)], batch = True)[1] == target

    assert detokenizer([tokenizer(pgn_transcript), tokenizer(pgn_transcript)], batch = True)[0] == target
    assert detokenizer([tokenizer(pgn_transcript), tokenizer(pgn_transcript)], batch = True)[1] == target

    # Multi-Encoding
    if uci_in:
        assert detokenizer(tokenizer([uci_transcript, uci_transcript], batch = True)[0]) == target
        assert detokenizer(tokenizer([uci_transcript, uci_transcript], batch = True)[1]) == target
        assert detokenizer(tokenizer([uci_transcript, pgn_transcript], batch = True)[0]) == target
        assert detokenizer(tokenizer([pgn_transcript, uci_transcript], batch = True)[1]) == target

    assert detokenizer(tokenizer([pgn_transcript, pgn_transcript], batch = True)[0]) == target
    assert detokenizer(tokenizer([pgn_transcript, pgn_transcript], batch = True)[1]) == target

    # Multi-Encoding/Decoding
    if uci_in:
        assert detokenizer(tokenizer([uci_transcript, uci_transcript], batch = True)[0]) == target
        assert detokenizer(tokenizer([uci_transcript, uci_transcript], batch = True)[1]) == target
        assert detokenizer(tokenizer([uci_transcript, pgn_transcript], batch = True)[0]) == target
        assert detokenizer(tokenizer([pgn_transcript, uci_transcript], batch = True)[1]) == target

    assert detokenizer(tokenizer([pgn_transcript, pgn_transcript], batch = True)[0]) == target
    assert detokenizer(tokenizer([pgn_transcript, pgn_transcript], batch = True)[1]) == target

    # Input Types:
    # Should be capable of decoding torch tensors
    # 1D:
    idx = tokenizer(pgn_transcript)
    torch_idx = torch.tensor(idx)
    assert detokenizer(idx, batch = False) == detokenizer(torch_idx, batch = False)
    decoding = detokenizer(torch_idx, batch = False)
    assert type(decoding) == str
    encoding = tokenizer(decoding, batch = False)
    assert all([enc == id for enc, id in zip(encoding, idx)]), (tokenizer(decoding, batch = False), idx)

    # 2D:
    idx = [tokenizer(pgn_transcript) for _ in range(100)]
    torch_idx = torch.tensor(np.array(idx))
    assert detokenizer(idx, batch = True) == detokenizer(torch_idx, batch = True)
    decoding = detokenizer(torch_idx, batch = True)
    assert isinstance(decoding, list)
    encoding = tokenizer(decoding, batch = True)
    assert all(
        [
            all([enc == id for enc, id in zip(encoding[i], idx[i])])
        ] for i in range(len(idx))
    ), (tokenizer(decoding, batch = False), idx)

    # Output Types:
    # Should be capable of returning torch tensors
    encoding = tokenizer([pgn_transcript, pgn_transcript], batch = True, return_type = "torch")
    assert all(enc.type() == "torch.LongTensor" for enc in encoding), [enc.type() for enc in encoding]
    assert detokenizer(encoding, batch = True)[0] == target
    assert detokenizer(encoding, batch = True)[1] == target
    assert len(encoding) == 2

    print("\nAll Tests Passed!")