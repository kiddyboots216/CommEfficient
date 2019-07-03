from minimal import *
from fed_worker import FedWorker
from unsketched_fed_worker import UnsketchedFedWorker

def train_epoch(ps, workers, epochs_per_iter, timer):
    train_stats = ray.get([worker.train_epoch.remote() for worker in workers])
    # train_stats, tables = list(
    #     zip(
    #         *ray.get(
    #             [worker.train.remote(epochs_per_iter) for worker in workers]
    #             )
    #         )
    #     )
    # hhcoords = ps.compute_hhcoords.remote(*tables)
    # # workers answer, also giving the unsketched params
    # topkAndUnsketched = [worker.send_topkAndUnsketched.remote(hhcoords) for worker in workers]
    # # server compute weight update, put it into ray
    # weightUpdate = ps.compute_update.remote(*topkAndUnsketched)
    # # workers apply weight update (can be merged with 1st line)
    # ray.wait([worker.apply_update.remote(weightUpdate) for worker in workers])
    train_time = timer()
    #import pdb; pdb.set_trace()
    test_stats = ray.get([worker.forward.remote() for worker_id, worker in enumerate(workers)])
    test_time = timer()
    #import pdb; pdb.set_trace()
    #import pdb; pdb.set_trace()
    stats ={'train_time': train_time,
            'train_loss': torch.mean(torch.stack([stat['loss'] for stat in train_stats])).numpy()[()],
            'train_acc': torch.mean(torch.stack([stat['acc'] for stat in train_stats])).numpy()[()],
            'test_time': test_time,
            'test_loss': torch.mean(torch.stack([stat['loss'] for stat in test_stats])).numpy()[()],
            'test_acc': torch.mean(torch.stack([stat['acc'] for stat in test_stats])).numpy()[()],
            'total_time': timer.total_time}
    return stats

def train(ps, workers, epochs, epochs_per_iter, batch_size,
          loggers=(), test_time_in_total=True, timer=None):
    timer = timer or Timer()
    # ray.wait([worker.sync.remote(ps._getParamVec.remote()) for worker in workers])
    num_epoch = 0
    while num_epoch < epochs:
        iter_stats = train_epoch(ps, workers, epochs_per_iter, timer)
        lr = ray.get(workers[0].param_values.remote())['lr'] * batch_size
        summary = union({'epoch': num_epoch+epochs_per_iter, 'lr': lr}, iter_stats)
        num_epoch += epochs_per_iter
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
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--epochs_per_iter", type=int, default=1)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--criterion", type=str, default="CrossEntropyLoss")
    args = parser.parse_args()
    #args.batch_size = math.ceil(args.batch_size/args.num_workers) * args.num_workers

    print('Downloading datasets')
    DATA_DIR = "sample_data"
    dataset = cifar10(DATA_DIR)

    lr_schedule = PiecewiseLinear([0, 5, args.epochs], [0, 0.4, 0])
    train_transforms = [Crop(32, 32), FlipLR(), Cutout(8, 8)]
    lr = lambda step: lr_schedule(step/len(client_train_batches[0]))/args.batch_size
    #print('Starting timer')
    timer = Timer()

    #print('Preprocessing training data')
    train_set = list(zip(
            transpose(normalise(pad(dataset['train']['data'], 4))),
            dataset['train']['labels']))
    client_train_sets = np.array_split(train_set, args.num_workers)
    print(f'Finished making {len(train_set)} to {[len(set) for set in client_train_sets]} in {timer():.2f} seconds')
    #print('Finished in {:.2f} seconds'.format(timer()))
    #print('Preprocessing test data')
    test_set = list(zip(transpose(normalise(dataset['test']['data'])),
                        dataset['test']['labels']))
    def chunker_list(seq, size):
        return (seq[i::size] for i in range(size))
    client_test_sets = list(chunker_list(test_set, args.num_workers))
    print(f'Finished making {len(test_set)} to {[len(set) for set in client_test_sets]} in {timer():.2f} seconds')
    TSV = TSVLogger()

    client_train_batches = [Batches(Transform(client_train_set, train_transforms),
                            args.batch_size, shuffle=True,
                            set_random_choices=True, drop_last=True) for client_train_set in client_train_sets]
    for chosen_client in client_train_batches:
        for batch_id, batch in enumerate(chosen_client):
            i = 0
    client_test_batches = [Batches(client_test_set, args.batch_size, shuffle=False,
                           drop_last=False) for client_test_set in client_test_sets]
    #import pdb; pdb.set_trace()
    for test_client in client_test_batches:
        for batch_id, batch in enumerate(test_client):
            i = 0

    kwargs = {
        "k": args.k,
        "p2": args.p2,
        "numCols": args.cols,
        "numRows": args.rows,
        "numBlocks": args.num_blocks,
        "lr": lr,
        "momentum": args.momentum,
        "optimizer" : args.optimizer,
        "criterion": args.criterion,
        "weight_decay": 5e-4*args.batch_size,
        "nesterov": args.nesterov,
        "dampening": 0,
        "metrics": ['loss', 'acc'],
    }

    ray.init(num_gpus=8)
    num_workers = args.num_workers
    minibatch_size = args.batch_size/num_workers
    print(f"Passing in args {kwargs}")
    workers = [UnsketchedFedWorker.remote(train_batch, client_test_batches[worker_index], worker_index, kwargs) for worker_index, train_batch 
                in enumerate(client_train_batches)]
    #ps = "bleh"
    del kwargs['optimizer']
    del kwargs['criterion']
    del kwargs['metrics']
    ps = ParameterServer.remote(kwargs)
    # Create workers.

    # track_dir = "sample_data"
    # with track.trial(track_dir, None, param_map=vars(optim_args)):

    train(ps, workers, args.epochs, args.epochs_per_iter, args.batch_size,
          loggers=(TableLogger(), TSV), timer=timer)
