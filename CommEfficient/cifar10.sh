# python fed_train.py \
#     --local_momentum 0.0 \
#     --virtual_momentum 0.9 \
#     --error_type virtual \
#     --mode sketch \
#     --num_clients 2 \
#     --num_devices 2 \
#     --participation 1 \
#     --k 50000 \
#     --num_rows 3 \
#     --num_cols 1000000 \
#     --supervised \
#     --static_datasets \
#     --num_classes 10 \
#     --share_ps_gpu

python fed_train.py \
    --mode localSGD \
    --num_clients 1 \
    --num_devices 1 \
    --participation 1 \
    --supervised \
    --static_datasets \
    --num_classes 10 \
    --share_ps_gpu