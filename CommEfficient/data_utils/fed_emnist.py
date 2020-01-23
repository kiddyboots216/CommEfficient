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
            client_datasets = []
            for client_id in range(len(self.images_per_client)):
                client_datasets.append(
                        torch.load(self.client_fn(client_id))
                    )
            self.client_datasets = client_datasets
        else:
            test_data = torch.load(self.test_fn())
            self.test_images = test_data["x"]
            self.test_targets = test_data["y"]

    def _get_train_item(self, client_id, idx_within_client):
        client_dataset = self.client_datasets[client_id]
        raw_image = client_dataset["x"][idx_within_client]
        target = client_dataset["y"][idx_within_client]

        image = Image.fromarray(raw_image)

        return image, target

    def _get_val_item(self, idx):
        image = Image.fromarray(self.test_images[idx])
        target = self.test_targets[idx]
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
            if os.path.exists(fn):
                raise RuntimeError("won't overwrite existing client")
            torch.save({"x": images, "y": targets}, fn)

        # for the test data, put it all in one file (we don't
        # care which client the test data nominally belongs to)
        test_data_dir = os.path.join(self.dataset_dir, "test")
        test_data = read_data(test_data_dir)
        num_val_images = 0
        all_images = []
        all_targets = []
        for data_shard in enumerate(test_data.values()):
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
        with open(self.stats_fn(), "w") as f:
            f.write(json.dumps(stats))
