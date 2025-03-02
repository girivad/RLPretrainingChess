python3 ./data/lichess_hf_dataset/prepare.py \ 
          --dataset lichess_6gb.zip \ 
          --out_dir ../../model_vol/data_dir/pretrain/ \ 
          --tok_type move \ 
          --tokenizer_path "./data/lichess_hf_dataset/tokenizer/tokenizers/move_token.pkl"