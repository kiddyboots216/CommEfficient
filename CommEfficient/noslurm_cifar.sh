cd /data/ashwineep/CommEfficient/CommEfficient
OMP_NUM_THREADS=4 KMP_INIT_AT_FORK=FALSE python cv_train.py \
    --dataset_dir /data/ashwineep/datasets/cifar10/ \
    --local_batch_size $7 \
    --valid_batch_size 512 \
    --dataset_name CIFAR10 \
    --model ResNet9 \
    --local_momentum ${11} \
    --virtual_momentum ${12} \
    --weight_decay 5e-4 \
    --num_fedavg_epochs ${13} \
    --fedavg_lr_decay 1 \
    --fedavg_batch_size ${14} \
    --error_type $2 \
    --mode $1 \
    --num_epochs $3 \
    --num_clients $5 \
    --num_devices 4 \
    --num_workers $6 \
    --k $8 \
    --num_rows $9 \
    --num_cols ${10} \
    --share_ps_gpu \
    --port ${15} \
    --lr_scale ${17} \
    --pivot_epoch $4 \
    --train_dataloader_workers 2 \
    --val_dataloader_workers 0 \
    --seed ${16} \
    ${18} \
    ${19} \

