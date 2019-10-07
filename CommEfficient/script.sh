#!/usr/bin/env python
OMP_NUM_THREADS=4 python gpt2_train.py --lm_coef=2.0 --max_history=2 --num_epochs=1 --num_candidates=4 --personality_permutations=2 --num_train_batch_shards 4 --num_val_batch_shards 4 --batch_size 8 --num_clients 3 --participation 1.0 --momentum_type virtual --error_type virtual --mode sketch --k 50000 --num_rows 1 --num_cols 1240000 --test
