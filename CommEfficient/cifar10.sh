KMP_INIT_AT_FORK=FALSE OMP_NUM_THREADS=16 python cv_train.py \
    --dataset_dir /data/ashwineep/datasets/cifar10 \
    --dataset_name CIFAR10 \
    --model ResNet9 \
    --local_batch_size $4 \
    --local_momentum 0.0 \
    --virtual_momentum 0.9 \
    --error_type virtual \
    --mode $1 \
    --iid \
    --num_clients $2 \
    --num_workers $3 \
    --num_devices 1 \
    --k 1 \
    --num_rows 1 \
    --num_cols 1 \
    --share_ps_gpu \
    --port 42000 \
    --lr_scale 0.4 \
    --train_dataloader_workers 4 \
    --val_dataloader_workers 4 \
    --valid_batch_size 32 \
    --eval_before_start \
    --malicious \
    --mal_id 1 \
    --mal_targets $7 \
    --mal_boost $8 \
    #--dp \
    #--l2_norm_clip 1.5 \
    #--noise_multiplier 0.003 \
    #--dp_mode worker \
    #--finetune_path /data/ashwineep/model_checkpoints/CIFAR100/ \
    #--finetune \
    #--finetuned_from CIFAR100 \
    #--num_epochs 1 \
    #--checkpoint_path /data/ashwineep/model_checkpoints/CIFAR100/ \
    #--checkpoint \
    #--num_epochs 24 \
