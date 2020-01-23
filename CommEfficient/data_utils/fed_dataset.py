import os
import json

import numpy as np
import torch

__all__ = ["FedDataset"]

class FedDataset(torch.utils.data.Dataset):
    def __init__(self, args, dataset_dir, dataset_name, transform=None,
                 do_iid=False, num_clients=None,
                 train=True, download=False, malicious=False):
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.transform = transform
        self.do_iid = do_iid
        self._num_clients = num_clients
        self.type = "train" if train else "val"
        self.type = "mal" if malicious else self.type
        self.num_mal_images = args.mal_targets
        self.mal_id = args.mal_id
        self.is_malicious = args.is_malicious

        if not do_iid and num_clients is not None:
            raise ValueError("can't specify # clients when non-iid")

        if not os.path.exists(self.stats_fn()):
            self.prepare_datasets(download=download)

        self._load_meta(train)

        if self.do_iid:
            self.iid_shuffle = np.random.permutation(len(self))

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
            self.num_val_images = stats["num_val_images"]


    def __len__(self):
        if self.type == "train":
            return sum(self.images_per_client)
        elif self.type == "val":
            return self.num_val_images
        elif self.type == "mal":
            return self.num_mal_images

    def __getitem__(self, idx):
        if self.type == "train":
            orig_idx = idx
            if self.do_iid:
                # orig_idx determines which client idx is in,
                # but when iid, self.iid_shuffle[idx] determines which
                # image/target we actually return
                idx = self.iid_shuffle[idx]

            cumsum = np.cumsum(self.images_per_client)
            client_id = np.searchsorted(cumsum, idx, side="right")
            cumsum = np.hstack([[0], cumsum[:-1]])
            idx_within_client = idx - cumsum[client_id]
            if client_id == self.mal_id:
                image, target = self._get_mal_item(idx_within_client)

            else:
                image, target = self._get_train_item(client_id,
                                                 idx_within_client)

            if self.do_iid:
                cumsum = np.cumsum(self.data_per_client)
                client_id = np.searchsorted(cumsum, orig_idx, side="right")
        elif self.type == "val":
            image, target = self._get_val_item(idx)
            client_id = -1
        elif self.type == "mal":
            image, target = self._get_mal_item(idx)
            client_id = 0

        if self.transform is not None:
            image = self.transform(image)

        return client_id, image, target

    def stats_fn(self):
        return os.path.join(self.dataset_dir, "stats.json")


