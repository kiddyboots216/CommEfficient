import json
import os

import torch
from torchvision.datasets import ImageNet
import numpy as np

from data_utils import FedDataset

__all__ = ["FedImageNet"]

class FedImageNet(FedDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if kwargs["download"]:
            raise RuntimeError("Can't download ImageNet, sry")

        self.vanilla_train = ImageNet(self.dataset_dir, split="train")
        self.vanilla_val = ImageNet(self.dataset_dir, split="val")

    def prepare_datasets(self, download=False):
        if download:
            raise RuntimeError("Can't download ImageNet, sry")
        # the user should already have imagenet downloaded & extracted,
        # with meta.bin generated (this should change once the latest
        # version of pytorch, which knows you can't download ImageNet,
        # is released):
        # dataset_dir
        # |-meta.bin
        # |-train
        # | |-wnid1
        # | | |-image1.jpg
        # | | ...
        # | |-wnid2
        # | ...
        # |-val
        # | |-wnid1
        # | |-wnid2
        # | ...
        # 
        # each wnid is already a client, so
        # all we need to do in this method is generate the stats.json file

        # don't use self.vanilla_train/val since those don't exist yet
        vanilla_train = ImageNet(self.dataset_dir, split="train")
        vanilla_val = ImageNet(self.dataset_dir, split="val")

        # the clients must be sorted in the same order that looping over
        # vanilla_train will go through the classes
        images_per_client = []
        target = -1
        for s in vanilla_train.samples:
            if s[1] != target:
                images_per_client.append(0)
                target = s[1]
            images_per_client[-1] += 1
        num_val_images = len(vanilla_val.samples)
        stats = {"images_per_client": images_per_client,
                 "num_val_images": num_val_images}
        fn = self.stats_fn()
        if os.path.exists(fn):
            raise RuntimeError("won't overwrite existing stats file")
        with open(fn, "w") as f:
            json.dump(stats, f)

    def _get_train_item(self, client_id, idx_within_client):
        # this seems kind of pointless, since the parent class's getitem
        # took a flat index and turned it into the two args here.
        # but the parent class handles iid/non-iid, and other subclasses
        # might prefer client_id + idx_within_client instead of a flat idx
        cumsum = np.hstack([[0], np.cumsum(self.images_per_client)[:-1]])
        idx = cumsum[client_id] + idx_within_client
        return self.vanilla_train[idx]

    def _get_val_item(self, idx):
        return self.vanilla_val[idx]
