import orjson as json
import os
from collections import defaultdict

import numpy as np

from data_utils import FedDataset, FedCIFAR10
from torchvision.datasets import EMNIST
from PIL import Image

__all__ = ["FedEMNIST"]

def read_data(data_dir):
    """parses data in given data directories

    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'

    Return:
        clients: list of client ids
        data: dictionary of data
    """
    clients = []
    data = defaultdict(lambda : None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith(".json")]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, "r") as inf:
            cdata = json.loads(inf.read())
        clients.extend(cdata["users"])
        data.update(cdata["user_data"])

    clients = list(sorted(data.keys()))
    return clients, data

class FedEMNIST(FedDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # assume EMNIST is already preprocessed
        if self.type == "train":
            train_data_dir = os.path.join(self.dataset_dir, "train")
            self.clients, self.train_data = read_data(train_data_dir)
        else:
            test_data_dir = os.path.join(self.dataset_dir, "test")
            self.clients, self.test_data = read_data(test_data_dir)

    def _get_train_or_val_item(self, client_id, idx_within_client, train):
        if train:
            dataset = self.train_data
        else:
            dataset = self.test_data
        client = self.clients[client_id]
        client_data = dataset[client]
        x = client_data["x"]
        y = client_data["y"]
        raw_image = x[idx_within_client]
        raw_image = np.array(raw_image)
        raw_image = np.reshape(raw_image, (28, 28))
        target = y[idx_within_client]

        image = Image.fromarray(raw_image)

        return image, target

    def _get_train_item(self, client_id, idx_within_client):
        return self._get_train_or_val_item(client_id, idx_within_client, True)

    def _get_val_item(self, idx):
        cumsum = np.cumsum(self.val_images_per_client)
        client_id = np.searchsorted(cumsum, idx, side="right")
        cumsum = np.hstack([[0], cumsum[:-1]])
        idx_within_client = idx - cumsum[client_id]
        return self._get_val_item_true(client_id, idx_within_client)

    def _get_val_item_true(self, client_id, idx_within_client):
        return self._get_train_or_val_item(client_id, idx_within_client, False)

    def __len__(self):
        if self.type == "train":
            return sum(self.images_per_client)
        elif self.type == "val":
            return sum(self.val_images_per_client)

    def prepare_datasets(self, download=False):
        stats_fn = self.stats_fn()
        if os.path.exists(fn):
            raise RuntimeError("won't overwrite existing stats file")

        train_data_dir = os.path.join(self.dataset_dir, "train")
        clients, train_data = read_data(train_data_dir)
        images_per_client = [len(train_data[client_id]["y"])
                             for client_id in clients]
        test_data_dir = os.path.join(self.dataset_dir, "test")
        clients, test_data = read_data(test_data_dir)
        val_images_per_client = [len(test_data[client_id]["y"])
                                 for client_id in clients]
        # save global stats to disk
        stats = {"images_per_client": images_per_client,
                 "val_images_per_client": val_images_per_client,
                 "num_val_images": sum(val_images_per_client)}
        with open(stats_fn, "w") as f:
            f.write(json.dumps(stats))

    @property
    def data_per_client(self):
        return self.images_per_client

    @property
    def num_clients(self):
        return len(self.clients)


