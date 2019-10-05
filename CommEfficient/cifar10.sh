#python fed_train.py --momentum_type virtual --error_type virtual --mode sketch --num_clients 8 --k 20000 --r 5 --cols 100000 --supervised
python fed_train.py --momentum_type virtual --error_type virtual --mode true_topk --num_clients 4 --participation 1 --grad_reduction mean --supervised
