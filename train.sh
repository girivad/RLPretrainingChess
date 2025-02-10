pip install torch numpy transformers datasets tiktoken wandb tqdm --no-cache-dir -q
# python data/lichess_hf_dataset/prepare.py --dataset lichess_6gb_blocks.zip --out_dir ../../model_vol/data_dir/pretrain/
python train.py config/train_8ntp_char.py