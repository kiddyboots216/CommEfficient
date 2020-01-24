import orjson as json
import os
from collections import defaultdict

from data_utils import FedDataset, FedCIFAR10
import torch
from PIL import Image

__all__ = ["FedEMNIST"]

def read_data(data_dir):
    """parses data in given data directories

    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'

    Return:
        data: dictionary of data with format
              {"username1": {"x": [flat_image1, flat_image2, ...]
                             "y": [y1, y2, ...]}
               "username2": ...}
    """
    data = defaultdict(lambda : None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith(".json")]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, "r") as inf:
            cdata = json.loads(inf.read())
        data.update(cdata["user_data"])

    return data

class FedEMNIST(FedDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # assume EMNIST is already preprocessed
        if self.type == "train":
            # we can't store each client dataset in its own tensor,
            # since if we run with multiple dataloading workers, each
            # shared memory tensor requires a file descriptor, and
            # some systems limit users to 1024 open file descriptors
            # client i's data is between client_offsets[i] and
            # client_offsets[i+1]
            client_images = []
            client_targets = []
            client_offsets = [0]
            for client_id in range(len(self.images_per_client)):
                cdata = torch.load(self.client_fn(client_id))
                client_images.append(cdata["x"])
                client_targets.append(cdata["y"])
                offset = client_offsets[client_id]
                client_offsets.append(offset + cdata["y"].numel())
            self.client_images = torch.cat(client_images, dim=0)
            self.client_targets = torch.cat(client_targets, dim=0)
            self.client_offsets = torch.tensor(client_offsets)
        else:
            test_data = torch.load(self.test_fn())
            self.test_images = test_data["x"]
            self.test_targets = test_data["y"]

    def _get_train_item(self, client_id, idx_within_client):
        start = self.client_offsets[client_id]
        end = self.client_offsets[client_id + 1]
        client_images = self.client_images[start:end]
        client_targets = self.client_targets[start:end]
        raw_image = client_images[idx_within_client]
        target = client_targets[idx_within_client].item()

        image = Image.fromarray(raw_image.numpy())

        return image, target

    def _get_val_item(self, idx):
        image = Image.fromarray(self.test_images[idx].numpy())
        target = self.test_targets[idx].item()
        return image, target

    def prepare_datasets(self, download=False):
        if os.path.exists(self.stats_fn()):
            raise RuntimeError("won't overwrite existing stats file")
        if os.path.exists(self.test_fn()):
            raise RuntimeError("won't overwrite existing test set")

        # original data is in json format, meaning it takes
        # ~25 times longer to read from disk than it would if the
        # data were in torch files
        # rectify this, saving each client in a separate .pt file
        train_data_dir = os.path.join(self.dataset_dir, "train")
        train_data = read_data(train_data_dir)
        images_per_client = []
        for client_id, client_data in enumerate(train_data.values()):
            flat_images = client_data["x"]
            images = torch.tensor(flat_images).view(-1, 28, 28)
            targets = torch.tensor(client_data["y"])
            images_per_client.append(targets.numel())

            fn = self.client_fn(client_id)
            if not os.path.exists(fn):
                torch.save({"x": images, "y": targets}, fn)
            #else:
                #raise RuntimeError("won't overwrite existing client")

        # for the test data, put it all in one file (we don't
        # care which client the test data nominally belongs to)
        test_data_dir = os.path.join(self.dataset_dir, "test")
        test_data = read_data(test_data_dir)
        num_val_images = 0
        all_images = []
        all_targets = []
        for client_id, client_data in enumerate(test_data.values()):
            flat_images = client_data["x"]
            images = torch.tensor(flat_images).view(-1, 28, 28)
            targets = torch.tensor(client_data["y"])
            num_val_images += targets.numel()

            all_images.append(images)
            all_targets.append(targets)
        all_images = torch.cat(all_images, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        torch.save({"x": all_images, "y": all_targets},
                   self.test_fn())

        # save global stats to disk
        stats = {"images_per_client": images_per_client,
                 "num_val_images": num_val_images}
        with open(self.stats_fn(), "wb") as f:
            f.write(json.dumps(stats))

    def test_fn(self):
        return os.path.join(self.dataset_dir, "test", "test.pt")

    def client_fn(self, client_id):
        fn = "client{}.pt".format(client_id)
        return os.path.join(self.dataset_dir, "train", fn)
