import os
import json

import numpy as np
import torch

__all__ = ["FedDataset"]

num_train_datapoints = {"CIFAR10": 50000,
                        "CIFAR100": 50000,
                        "FEMNIST": 712640,
                        "PERSONA": 17568,
                        "FashionMNIST": 60000}

class FedDataset(torch.utils.data.Dataset):
    def __init__(self, args, dataset_dir, dataset_name, transform=None,
                 do_iid=False, num_clients=None,
                 train=True, download=False, malicious=False):
        self.args = args
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.num_train_datapoints = num_train_datapoints[dataset_name]
        self.transform = transform
        self.do_iid = do_iid
        self._num_clients = num_clients
        self.type = "train" if train else "val"
        self.is_malicious_val = malicious
        self.num_mal_images = args.mal_targets
        self.is_malicious_train = args.do_malicious and self.type == "train"
        self.data_ownership = args.do_data_ownership

        if not do_iid and num_clients == 1:
            raise ValueError("can't have 1 client when non-iid")

        if not os.path.exists(self.stats_fn()):
            self.prepare_datasets(download=download)

        self._load_meta(train)

        if args.do_malicious:
            self.mal_ids = np.array(range(self.num_clients)[-args.mal_num_clients:])
            print(f"Mal ids: {self.mal_ids} out of {self.num_clients}")
        if self.do_iid:
            self.iid_shuffle = np.random.permutation(len(self))


    @property
    def data_per_client(self):
        if self.do_iid:
            if self.type == "train":
                num_data = self.num_train_datapoints
            else:
                num_data = len(self)
            images_per_client = (np.ones(self.num_clients, dtype=int)
                                 * num_data // self.num_clients)
            extra = num_data % self.num_clients
            images_per_client[self.num_clients - extra:] += 1
            if self.is_malicious_train:
                images_per_client[self.mal_ids] = self.num_mal_images
            return images_per_client
        else:
            new_ipc = np.ones(self.num_clients)
            new_ipc *= int(sum(self.images_per_client)/self.num_clients)

            #import pdb; pdb.set_trace()
            """
            n_clients_per_class = self.num_clients // len(self.images_per_client)
            for num_images in self.images_per_client:
                extra = num_images % n_clients_per_class
                new_n_ipc = [num_images // n_clients_per_class for _ in range(n_clients_per_class)]
                new_n_ipc[-1] += extra
                new_ipc.extend(new_n_ipc)
            """
            images_per_client = np.array(new_ipc).astype(int)
            initial_sum = sum(images_per_client)
            if self.is_malicious_train:
                # HARDCODED
                images_per_client[self.mal_ids] = self.num_mal_images
            self.diff = sum(images_per_client) - initial_sum

            return images_per_client

    @property
    def num_clients(self):
        return (self._num_clients if (self._num_clients is not None and self._num_clients > 0)
                                      else len(self.images_per_client))

    def _load_meta(self, train):
        with open(self.stats_fn(), "r") as f:
            stats = json.load(f)
            self.images_per_client = np.array(stats["images_per_client"])
            self.num_val_images = stats["num_val_images"]


    def __len__(self):
        if self.type == "train":
            return min(sum(self.data_per_client), sum(self.images_per_client))
        elif self.type == "val":
            return self.num_val_images
        elif self.type == "mal":
            return self.num_mal_images

    def __getitem__(self, idx):
        # idx is some number in range(len(loader))
        if self.type == "train":
            orig_idx = idx
            if self.do_iid:
                # orig_idx determines which client idx is in,
                # but when iid, self.iid_shuffle[idx] determines which
                # image/target we actually return
                idx = self.iid_shuffle[idx]
            
            cumsum = np.cumsum(self.data_per_client)
            client_id = np.searchsorted(cumsum, orig_idx, side="right")
            if self.is_malicious_train and client_id in self.mal_ids:
                #print("getting malicious data ", idx)
                image, target = self._get_mal_item()
            else:
                cumsum = np.cumsum(self.images_per_client)
                class_id = np.searchsorted(cumsum, idx, side="right")
                cumsum = np.hstack([[0], cumsum[:-1]])
                idx_within_class = idx - cumsum[class_id]
                image, target = self._get_train_item(class_id,
                                                 idx_within_class)

        elif self.type == "val":
            image, target = self._get_val_item(idx)
            client_id = -1

        if self.transform is not None:
            image = self.transform(image)

        #return client_id, image, target
        return client_id, image, target

    def stats_fn(self):
        return os.path.join(self.dataset_dir, "stats.json")


