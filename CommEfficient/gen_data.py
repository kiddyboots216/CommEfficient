from data_utils import split_image_data, get_cifar10, transpose
from minimal import Timer, normalise, pad, Crop, FlipLR, Cutout, Transform, Batches, cifar10
import numpy as np
from utils import parse_args
DATA_DIR = "sample_data"


def MalLoader(args, benign_loader):
    x_train, y_train, x_test, y_test = get_cifar10()
    r = np.random.choice(len(x_test),size=args.mal_targets)
    print(r)
    mal_data_x = x_test[r]
    #To-do: remove hard-coding of number of classes
    print("Initial classes: %s" % y_test[r])
    true_labels = y_test[r]
    mal_data_y = np.zeros(args.mal_targets)
    allowed_targets = list(range(10))
    for i in range(args.mal_targets):
        allowed_targets.remove(true_labels[i])
        mal_data_y[i] = np.random.choice(allowed_targets)
    # mal_data_y = mal_data_y.reshape(1,)
    print("Target class: %s" % mal_data_y)
    mal_set = list(zip(transpose(normalise(mal_data_x)), mal_data_y))
    mal_loader = Batches(mal_set, args.batch_size, shuffle=False, drop_last=False)
    mal_weird_loader = Weird(mal_loader)
    # mal_weird_loader = np.array(mal_weird_loader)

    return mal_weird_loader

def gen_data(args):
    timer = Timer()
    print('Starting timer')
    x_train, y_train, x_test, y_test = get_cifar10()
    train_transforms = [Crop(32, 32), FlipLR(), Cutout(8, 8)]
    """"
    dataset = cifar10(DATA_DIR)
    x_train_old = dataset['train']['data']
    y_train_old = dataset['train']['labels']
    x_test_old = dataset['test']['data']
    y_test_old = dataset['test']['labels']
    test_set = list(zip(transpose(normalise(x_test_old)), y_test_old))
    print(f"{x_train_old.shape}, {len(y_train_old)}, {x_test_old.shape}, {len(y_test_old)}")
    train_set = list(zip(transpose(normalise(pad(x_train, 4))), y_train))
    train_loader = Batches(Transform(train_set, train_transforms), args.batch_size, shuffle=True, set_random_choices=True, drop_last=True)
    """
    print('Preprocessing test data')
    test_set = list(zip(transpose(normalise(x_test)), y_test))
    val_loader = Batches(test_set, args.batch_size, shuffle=False, drop_last=False)
    print('Finished in {:.2f} seconds'.format(timer()))
    if args.static_datasets:
        print('Preprocessing divided data')
        batch_size = args.batch_size // args.num_workers
        batch_size = min(batch_size, args.num_data // args.num_clients)
        split = split_image_data(x_train,
                                 y_train,
                                 n_clients=args.num_clients,
                                 classes_per_client=args.num_classes,
                                 balancedness=args.balancedness,
                                 verbose=True)
        train_datasets = [list(zip(transpose(normalise(pad(x, 4))), y))
                          for x, y in split]
        client_train_loaders = [Batches(Transform(client_train_set,
                                                  train_transforms),
                                        batch_size,
                                        shuffle=True,
                                        set_random_choices=True,
                                        drop_last=True)
                                for client_train_set in train_datasets]
        print('Finished in {:.2f} seconds'.format(timer()))
        client_weird_loaders = [Weird(batch)
                                for batch in client_train_loaders]
        #[test_weird(weird) for weird in client_weird_loaders]
        client_weird_loaders = np.array(client_weird_loaders)
        return client_weird_loaders, val_loader
    else:
        train_set = list(zip(transpose(normalise(pad(x_train, 4))),
                             y_train))
        train_loader = Batches(Transform(train_set, train_transforms),
                               args.batch_size,
                               shuffle=True,
                               set_random_choices=True,
                               drop_last=True)
        #test_weird(Weird(train_loader))
        return train_loader, val_loader

class Weird:
    def __init__(self, batch):
        self.data = batch
        self.iterator = iter(self.data)
    def next_batch(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.data)
            return self.next_batch()

def test_weird(weird):
    n_iters = len(weird.data)
    for i in range(n_iters + 1):
        batch = weird.next_batch()
        print(f"iter {i}")

if __name__ == "__main__":
    args = parse_args(default_lr=0)
    gen_data(args)
