import json
import os

import numpy as np

from data_utils import FedDataset
import torchvision
from torchvision.datasets import CIFAR10, CIFAR100
from PIL import Image

__all__ = ["FedCIFAR10", "FedCIFAR100"]

class FedCIFAR10(FedDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # keep all data in memory
        if self.type == "train":
            client_datasets = []
            for client_id in range(len(self.images_per_client)):
                client_datasets.append(np.load(self.client_fn(client_id)))
            self.client_datasets = client_datasets
        elif self.type == "val":
            with np.load(self.test_fn()) as test_set:
                self.test_images = test_set["test_images"]
                self.test_targets = test_set["test_targets"]

    def prepare_datasets(self, download=True):
        os.makedirs(self.dataset_dir, exist_ok=True)
        dataset = torchvision.datasets.__dict__.get(self.dataset_name)
        vanilla_train = dataset(self.dataset_dir,
                                train=True,
                                download=download)
        vanilla_test = dataset(self.dataset_dir,
                               train=False,
                               download=download)

        train_images = vanilla_train.data
        train_targets = np.array(vanilla_train.targets)
        classes = vanilla_train.classes

        test_images = vanilla_test.data
        test_targets = np.array(vanilla_test.targets)

        images_per_client = []

        # split train_images/targets into client datasets, save to disk
        for client_id in range(len(classes)):
            cur_client = np.where(train_targets == client_id)[0]
            client_images = train_images[cur_client]
            client_targets = train_targets[cur_client]

            images_per_client.append(len(client_targets))

            fn = self.client_fn(client_id)
            if os.path.exists(fn):
                raise RuntimeError("won't overwrite existing split")
            np.save(fn, client_images)

        # save test set to disk
        fn = self.test_fn()
        if os.path.exists(fn):
            raise RuntimeError("won't overwrite exiting test set")
        np.savez(fn,
                 test_images=test_images,
                 test_targets=test_targets)

        # save global stats to disk
        fn = self.stats_fn()
        if os.path.exists(fn):
            raise RuntimeError("won't overwrite existing stats file")
        stats = {"images_per_client": images_per_client,
                 "num_val_images": len(test_targets)}
        with open(fn, "w") as f:
            json.dump(stats, f)

    def _get_train_item(self, client_id, idx_within_client):
        client_dataset = self.client_datasets[client_id]
        raw_image = client_dataset[idx_within_client]
        target = client_id

        image = Image.fromarray(raw_image)

        return image, target

    def _get_val_item(self, idx):
        raw_image = self.test_images[idx]
        image = Image.fromarray(raw_image)
        return image, self.test_targets[idx]

    def client_fn(self, client_id):
        fn = "client{}.npy".format(client_id)
        return os.path.join(self.dataset_dir, fn)

    def test_fn(self):
        return os.path.join(self.dataset_dir, "test.npz")

class FedCIFAR100(FedCIFAR10):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
