import os
import argparse
import torch
from datetime import datetime

def make_logdir(args: dict):
    rows = args.num_rows
    cols = args.num_cols
    k = args.k
    mode = args.mode
    num_local_iters = args.num_local_iters
    sketch_str = f"{mode}: {rows} x {cols}" if mode == "sketch" else "{mode}"
    k_str = f"k: {k}" if mode in ["sketch", "true_topk", "local_topk"] else f"num_local_iters: {num_local_iters}"
    workers = args.num_workers
    clients = args.num_clients
    clients_str = f"{workers}/{clients}"
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    logdir = os.path.join(
        'runs', current_time + '_' + clients_str + '_' + sketch_str + '_' + k_str)
    return logdir

def parse_args(default_lr):
    parser = argparse.ArgumentParser()
    
    # meta-args
    parser.add_argument("--test", action="store_true", dest="do_test")
    modes = ["sketch", "true_topk", "local_topk", "localSGD"]
    parser.add_argument("--mode", choices=modes, default="sketch")

    # data/model args
    parser.add_argument("--static_datasets", action="store_true")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--num_data", type=int, default=50000)
    parser.add_argument("--model", default="resnet9")
    parser.add_argument("--num_results_train", type=int, default=2)
    parser.add_argument("--num_results_val", type=int, default=2)
    parser.add_argument("--supervised", action="store_true",
                        dest="is_supervised")
    parser.add_argument("--dataset_path", type=str, default="",
                        help=("Path or url of the dataset."
                              " If empty, download from the internet."))
    parser.add_argument("--dataset_cache", type=str,
                        default='./dataset_cache',
                        help="Path or url of the dataset cache")

    # compression args
    parser.add_argument("--k", type=int, default=50000)
    parser.add_argument("--p2", type=int, default=4)
    parser.add_argument("--num_cols", type=int, default=500000)
    parser.add_argument("--num_rows", type=int, default=5)
    parser.add_argument("--num_blocks", type=int, default=20)
    parser.add_argument("--topk_down", action="store_true",
                        dest="do_topk_down")

    # optimization args
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--nesterov", action="store_true",
                        dest="do_nesterov")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_epochs", type=int, default=24,
                        help="Number of training epochs")
    momentum_types = ["none", "local", "virtual"]
    parser.add_argument("--momentum_type", choices=momentum_types,
                        default="none")
    error_types = momentum_types
    parser.add_argument("--error_type", choices=error_types,
                        default="none")
    reductions = ["sum", "mean", "median"]
    parser.add_argument("--grad_reduction",
                        choices=reductions,
                        default="sum",
                        help="How to combine gradients from workers")
    parser.add_argument("--lr_scale", type=float, default=default_lr)
    parser.add_argument("--pivot_epoch", type=int, default=5)

    # parallelization args
    parser.add_argument("--num_clients", type=int, default=1)
    parser.add_argument("--participation", type=float, default=1.0)
    parser.add_argument("--balancedness", type=float, default=1.0)
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str,
                        default=default_device,
                        help="Device (cuda or cpu)")
    parser.add_argument("--num_local_iters", type=int, default=1)
    parser.add_argument("--local_sched", action="store_true", dest="use_local_sched")

    # GPT2 args
    parser.add_argument("--num_dialogs", type=int, default=1)
    parser.add_argument("--model_checkpoint", type=str, default="gpt2",
                        help="Path, url or short name of the model")
    parser.add_argument("--num_candidates", type=int, default=2,
                        help="Number of candidates for training")
    parser.add_argument("--max_history", type=int, default=2,
                        help=("Number of previous exchanges to keep"
                              " in history"))
    parser.add_argument("--train_batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=8,
                        help="Batch size for validation")
    parser.add_argument("--num_train_batch_shards", type=int,
                        default=4,
                        help=("Split up each batch into shards"
                              " to save memory"))
    parser.add_argument("--num_val_batch_shards", type=int,
                        default=4,
                        help=("Split up each batch into shards"
                              " to save memory"))
    parser.add_argument("--lm_coef", type=float, default=1.0,
                        help="LM loss coefficient")
    parser.add_argument("--mc_coef", type=float, default=1.0,
                        help="Multiple-choice loss coefficient")
    parser.add_argument("--max_norm", type=float, default=1.0,
                        help="Clipping gradient norm")
    parser.add_argument("--personality_permutations", type=int, default=1,
                        help=("Number of permutations of personality"
                              " sentences"))
    parser.add_argument("--eval_before_start", action='store_true',
                        help=("If true start with a first evaluation"
                              " before training"))
    parser.add_argument("--fp16", type=str, default="",
                        help=("Set to O0, O1, O2 or O3 for fp16 training"
                              " (see apex documentation)"))


    args = parser.parse_args()
    args.num_workers = int(args.num_clients * args.participation)
    args.weight_decay = args.weight_decay * args.batch_size

    return args

