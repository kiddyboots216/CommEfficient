from data_utils import *
from minimal import Timer, normalise, pad, transpose, Crop, FlipLR, Cutout, Transform, Batches, cifar10
DATA_DIR = "sample_data"

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
    if args.fed:
        print('Preprocessing divided data')
        batch_size = args.batch_size // args.workers
        batch_size = min(batch_size, args.DATA_LEN // args.clients)
        split = split_image_data(x_train, y_train, n_clients=args.clients, classes_per_client=args.classes, balancedness=args.balancedness, verbose=True)
        train_datasets = [list(zip(transpose(normalise(pad(x, 4))), y)) for x, y in split] 
        client_train_loaders = [Batches(Transform(client_train_set, train_transforms), batch_size, shuffle=True, set_random_choices=True, drop_last=True) for client_train_set in train_datasets]
        print('Finished in {:.2f} seconds'.format(timer()))
        client_weird_loaders = [Weird(batch) for batch in client_train_loaders]
        #[test_weird(weird) for weird in client_weird_loaders]
        client_weird_loaders = np.array(client_weird_loaders)
        return client_weird_loaders, val_loader
    else:
        train_set = list(zip(transpose(normalise(pad(x_train, 4))), y_train))
        train_loader = Batches(Transform(train_set, train_transforms), args.batch_size, shuffle=True, set_random_choices=True, drop_last=True)
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-clients", type=int, default=1)
    parser.add_argument("-participation", type=float, default=1.0)
    parser.add_argument("-classes", type=int, default=10)
    parser.add_argument("-balancedness", type=float, default=1.0)
    parser.add_argument("-batch_size", type=int, default=512)
    parser.add_argument("-fed", action="store_true")
    parser.add_argument("-DATA_LEN", type=int, default=50000)
    args = parser.parse_args()
    args.workers = int(args.clients * args.participation)
    gen_data(args)
