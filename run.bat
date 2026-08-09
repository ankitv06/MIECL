@echo off
cd /d "C:\Users\ankit\OneDrive\Desktop\news-recsys\News Recc Code"

py main.py ^
--dataset_dir "dataset" ^
--glove_path "dataset\glove.840B.300d.txt" ^
--preserve_dir "outputs" ^
--num_epoch 1 ^
--num_dataset 1 ^
--batch_size 128 ^
--hid_dim 400 ^
--num_head 20 ^
--num_prototype 5 ^
--alpha 0.5 ^
--beta 0.2 ^
--temperature 0.2 ^
--dropout_rate 0.1 ^
--multi_rep_mode concat ^
--infonce_mode prototype_self ^
--contrastive_mode USER ^
--gnn_mode nogat ^
--agg_mode soft

pause