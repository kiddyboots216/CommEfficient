OMP_NUM_THREADS=16 KMP_INIT_AT_FORK=FALSE python cv_train.py \
    --dataset_dir /data/ashwineep/datasets/${1}/ \
    --valid_batch_size 512 \
    --tensorboard \
    --dataset_name ${1} \
    --model ${2} \
    --mode ${3} \
    --num_clients ${4} \
    --num_workers ${5} \
    --local_batch_size ${6} \
    --error_type ${7} \
    --num_epochs ${8} \
    --pivot_epoch ${9} \
    --lr_scale ${10} \
    --local_momentum ${11} \
    --virtual_momentum ${12} \
    --weight_decay 5e-4 \
    --num_fedavg_epochs ${13} \
    --fedavg_lr_decay 0 \
    --fedavg_batch_size ${14} \
    --num_devices 1 \
    --k ${15} \
    --num_rows 1 \
    --num_cols ${16} \
    --share_ps_gpu \
    --port ${17} \
    --train_dataloader_workers 2 \
    --val_dataloader_workers 0 \
    --seed ${18} \
    --mal_targets ${19} \
    --mal_boost ${20} \
    --mal_num_clients ${21} \
    --mal_epoch ${22} \
    --mal_type ${23} \
    --noise_multiplier ${24} \
    --l2_norm_clip ${25} \
    ${26} \
    ${27} \
    ${28} \
    ${29} \
    ${30} \

