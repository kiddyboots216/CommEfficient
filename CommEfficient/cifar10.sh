#OMP_NUM_THREADS=8 python -m cProfile -o profile/cifar_fedsampler.pstats fed_train.py \
# OMP_NUM_THREADS=8 python fed_train.py \
#     --dataset_path /data/drothchild/datasets/cifar10/ \
#     --local_batch_size 512 \
#     --dataset_name CIFAR10 \
#     --local_momentum 0.0 \
#     --virtual_momentum 0.9 \
#     --error_type virtual \
#     --mode sketch \
#     --num_devices 2 \
#     --num_workers 1 \
#     --iid \
#     --num_clients 1 \
#     --k 50000 \
#     --num_rows 3 \
#     --num_cols 1000000 \
#     --supervised

OMP_NUM_THREADS=8 python fed_train.py \
    --local_batch_size 512 \
    --dataset_name CIFAR10 \
    --mode localSGD \
    --num_devices 2 \
    --num_workers 1 \
    --iid \
    --num_clients 1 \
    --supervised
