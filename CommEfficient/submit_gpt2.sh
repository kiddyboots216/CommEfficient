rsync -zarh --exclude ".git/*" --exclude "*.out" ~/CommEfficient /data/ashwineep/
cd /data/ashwineep/CommEfficient/CommEfficient
KMP_INIT_AT_FORK=FALSE OMP_NUM_THREADS=8 python gpt2_train.py \
    --dataset_dir /data/ashwineep/datasets/persona_chat/ \
    --dataset_name PERSONA \
    --model_checkpoint gpt2-medium \
    --num_results_train 1 \
    --num_results_val 2 \
    --lm_coef=2.0 \
    --max_history=2 \
    --num_candidates=4 \
    --personality_permutations=2 \
    --valid_batch_size 8 \
    --train_dataloader_workers 4 \
    --val_dataloader_workers 4 \
    --num_devices 5 \
    --microbatch_size 4 \
    --mode $1 \
    --error_type $2 \
    --lr_scale $3 \
    --num_epochs=$4 \
    $5 \
    --num_workers $6 \
    --local_batch_size $7 \
    --k $8 \
    --num_rows $9 \
    --num_cols ${10} \
    --local_momentum ${11} \
    --virtual_momentum ${12} \
    --max_grad_norm ${13} \
    --num_fedavg_epochs ${14} \
    --fedavg_batch_size ${15} \
    --port ${16} \
    --seed ${17} \
    #--finetune \
    #--finetune_path ${18} \

