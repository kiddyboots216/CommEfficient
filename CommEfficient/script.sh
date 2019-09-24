#!/usr/bin/env python
OMP_NUM_THREADS=4 python gpt2_train.py --lm_coef=2.0 --max_history=2 --n_epochs=1 --num_candidates=4 --personality_permutations=2 --gradient_accumulation_steps 4 --train_batch_size=2 --valid_batch_size=2 --clients 1 --participation 1.0 
#--sketch --virtual_momentum_sketch --virtual_error_sketch --k 200000 --cols 4000000 --rows 5 
#OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node=8 ./train.py --gradient_accumulation_steps=4 --lm_coef=2.0 --max_history=2 --n_epochs=1 --num_candidates=4 --personality_permutations=2 --train_batch_size=2 --valid_batch_size=2 --lr 0.04 --model_checkpoint gpt2
