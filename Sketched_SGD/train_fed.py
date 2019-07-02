from fed_worker import FedWorker
from minimal import *

def train_epoch(ps, workers, test_batches, epochs_per_iter, timer):
    train_stats, tables = list(
        zip(
            *ray.get(
                [worker.train.remote(epochs_per_iter)]
                )
            )
        )
    hhcoords = ps.compute_hhcoords.remote(*tables)
    # workers answer, also giving the unsketched params
    topkAndUnsketched = [worker.send_topkAndUnsketched.remote(hhcoords) for worker in workers]
    # server compute weight update, put it into ray
    weightUpdate = ps.compute_update.remote(*topkAndUnsketched)
    # workers apply weight update (can be merged with 1st line)
    ray.wait([worker.apply_update.remote(weightUpdate) for worker in workers])
    train_time = timer()
    test_stats = ray.get([worker.forward.remote(test_batches)])
    test_time = timer()
    stats ={'train_time': train_time,
            'train_loss': train_stats['loss'],
            'train_acc': train_stats['correct'],
            'test_time': test_time,
            'test_loss': test_stats['loss'],
            'test_acc': test_stats['correct'],
            'total_time': timer.total_time}
    return stats

def train(ps, workers, test_batches, epochs, epochs_per_iter,
          loggers=(), test_time_in_total=True, timer=None):
    timer = timer or Timer()
    # ray.wait([worker.sync.remote(ps._getParamVec.remote()) for worker in workers])
    num_epoch = 0
    while num_epoch < epochs:
        iter_stats = train_epoch(ps, workers, test_batches, epochs_per_iter, timer)
        num_epoch += epochs_per_iter
        lr = ray.get(workers[0].param_values.remote())['lr'] * test_batches.batch_size
        summary = union({'epoch': epoch+iter_stats, 'lr': lr}, iter_stats)
#         track.metric(iteration=epoch, **summary)
        for logger in loggers:
            logger.append(summary)
    return summary

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sketched", action="store_true")
    parser.add_argument("--sketch_biases", action="store_true")
    parser.add_argument("--sketch_params_larger_than", action="store_true")
    parser.add_argument("-k", type=int, default=50000)
    parser.add_argument("--p2", type=int, default=1)
    parser.add_argument("--cols", type=int, default=500000)
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--nesterov", type=bool, default=False)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--criterion", type=str, default="CrossEntropyLoss")
    args = parser.parse_args()
    #args.batch_size = math.ceil(args.batch_size/args.num_workers) * args.num_workers

    print('Downloading datasets')
    DATA_DIR = "sample_data"
    dataset = cifar10(DATA_DIR)

    lr_schedule = PiecewiseLinear([0, 5, args.epochs], [0, 0.4, 0])
    train_transforms = [Crop(32, 32), FlipLR(), Cutout(8, 8)]
    lr = lambda step: lr_schedule(step/len(train_batches))/args.batch_size
    print('Starting timer')
    timer = Timer()

    print('Preprocessing training data')
    train_set = list(zip(
            transpose(normalise(pad(dataset['train']['data'], 4))),
            dataset['train']['labels']))
    client_train_sets = np.array_split(train_set, args.num_workers)
    print('Finished in {:.2f} seconds'.format(timer()))
    print('Preprocessing test data')
    test_set = list(zip(transpose(normalise(dataset['test']['data'])),
                        dataset['test']['labels']))
    print('Finished in {:.2f} seconds'.format(timer()))

    TSV = TSVLogger()

    client_train_batches = [Batches(Transform(client_train_set, train_transforms),
                            args.batch_size, shuffle=True,
                            set_random_choices=True, drop_last=True) for client_train_set in client_train_sets]
    test_batches = Batches(test_set, args.batch_size, shuffle=False,
                           drop_last=False)

    kwargs = {
        "sketched": {
            "k": args.k,
            "p2": args.p2,
            "numCols": args.cols,
            "numRows": args.rows,
            "numBlocks": args.num_blocks,
        }
        "lr": lr,
        "momentum": 0.0,
        "optimizer" : args.optimizer,
        "criterion": args.criterion,
        "weight_decay": 5e-4*args.batch_size,
        "nesterov": args.nesterov,
        "dampening": 0,
        "metrics": ['loss', 'correct'],
    }

    ray.init(num_gpus=8)
    num_workers = args.num_workers
    minibatch_size = args.batch_size/num_workers
    print(f"Passing in args {optim_args}")
    ps = ParameterServer.remote(kwargs)
    workers = [FedWorker.remote(batch, worker_index, kwargs) for worker_index, batch 
                in enumerate(client_train_batches)]
    #ps = "bleh"
    # Create workers.

    # track_dir = "sample_data"
    # with track.trial(track_dir, None, param_map=vars(optim_args)):

    train(ps, workers, test_batches, args.epochs,
          loggers=(TableLogger(), TSV), timer=timer)