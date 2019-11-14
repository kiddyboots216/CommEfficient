import os
import json

import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms

from PIL import Image

class FedFactory(torch.utils.data.Dataset):
    def __init__(self, dataset_name, dataset_dir, transform=None, do_iid=False,
                 num_clients=None, train=True, download=True):
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.do_iid = do_iid
        self._num_clients = num_clients
        self.type = "train" if train else "test"

        if not do_iid and num_clients is not None:
            raise ValueErorr("can't specify # clients when non-iid")
        self._nm_clients = num_clients

        if download and not os.path.exists(self.dataset_dir):
            self.download_and_split_data()

        self._load_meta(train)

        if self.do_iid:
            self.iid_shuffle = np.random.permutation(len(self))

        # keep all data in memory
        if self.type == "train":
            client_datasets = []
            for client_id in range(len(self.images_per_client)):
                client_datasets.append(np.load(self.client_fn(client_id)))
            self.client_datasets = client_datasets
        elif self.type == "test":
            with np.load(self.test_fn()) as test_set:
                self.test_images = test_set["test_images"]
                self.test_targets = test_set["test_targets"]

    @property
    def data_per_client(self):
        if self.do_iid:
            num_data = len(self)
            images_per_client = (np.ones(self.num_clients, dtype=int)
                                 * num_data // self.num_clients)
            extra = num_data % self.num_clients
            images_per_client[self.num_clients - extra:] += 1
            return images_per_client
        else:
            return self.images_per_client

    @property
    def num_clients(self):
        if self.do_iid:
            return (self._num_clients if self._num_clients is not None
                                      else len(self.images_per_client))
        else:
            return len(self.images_per_client)

    def _load_meta(self, train):
        with open(self.stats_fn(), "r") as f:
            stats = json.load(f)
            self.images_per_client = np.array(stats["images_per_client"])
            self.num_test_images = stats["num_test_images"]

    def download_and_split_data(self):
        os.makedirs(self.dataset_dir, exist_ok=True)
        # TODO: Maybe this should be in a try-except?
        dataset_factory = getattr(datasets, self.dataset_name)

        vanilla_train = dataset_factory(self.dataset_dir,
                                                 train=True,
                                                 download=True)
        vanilla_test = dataset_factory(self.dataset_dir,
                                                train=False,
                                                download=True)

        train_images = vanilla_train.data
        train_targets = np.array(vanilla_train.targets)
        self.classes = vanilla_train.classes

        test_images = vanilla_test.data
        test_targets = np.array(vanilla_test.targets)

        images_per_client = []

        # split train_images/targets into client datasets, save to disk
        for client_id in range(len(self.classes)):
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
                 "num_test_images": len(test_targets)}
        with open(fn, "w") as f:
            json.dump(stats, f)


    def __len__(self):
        if self.type == "train":
            return sum(self.images_per_client)
        else:
            return self.num_test_images

    def __getitem__(self, idx):
        if self.type == "train":
            client_id, raw_image, target = self._get_train_item(idx)
        elif self.type == "test":
            raw_image, target = self._get_test_item(idx)
            client_id = -1

        image = Image.fromarray(raw_image)

        if self.transform is not None:
            image = self.transform(image)

        return client_id, image, target

    def _get_train_item(self, idx):
        orig_idx = idx
        if self.do_iid:
            idx = self.iid_shuffle[idx]

        cumsum = np.cumsum(self.images_per_client)
        client_id = np.searchsorted(cumsum, idx, side="right")
        cumsum = np.hstack([[0], cumsum[:-1]])
        idx_within_client = idx - cumsum[client_id]

        client_dataset = self.client_datasets[client_id]
        raw_image = client_dataset[idx_within_client]
        target = client_id

        if self.do_iid:
            cumsum = np.cumsum(self.images_per_client)
            client_id = np.searchsorted(cumsum, orig_idx, side="right")

        return client_id, raw_image, target
    
    def _get_test_item(self, idx):
        return self.test_images[idx], self.test_targets[idx]
    
    def client_fn(self, client_id):
        fn = "client{}.npy".format(client_id)
        return os.path.join(self.dataset_dir, fn)

    def test_fn(self):
        return os.path.join(self.dataset_dir, "test.npz")
    
    def stats_fn(self):
        return os.path.join(self.dataset_dir, "stats.json")


