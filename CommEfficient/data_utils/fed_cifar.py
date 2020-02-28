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

        if self.is_malicious_train or self.is_malicious_val:
            np.random.seed(42)
            with np.load(self.test_fn()) as test_set:
                test_images = test_set["test_images"]
                test_targets = test_set["test_targets"]
            allowed_source_labels = self.fetch_source_idxs(test_images, self.args)
            mal_data = []
            #test_targets_mal = test_images[mal_data_rand_idxs]
            #true_labels_rand = test_targets[mal_data_rand_idxs]
            true_data, true_labels = self.fetch_test_data(test_images, test_targets, allowed_source_labels)
            mal_labels = np.zeros(self.num_mal_images, dtype=np.int64)
            num_mal = 0
            mal_idx = 0
            while num_mal < self.num_mal_images:
                allowed_target_labels = self.fetch_targets(self.images_per_client, self.args)
                if true_labels[mal_idx] in allowed_target_labels:
                    allowed_target_labels.remove(true_labels[mal_idx])
                if len(allowed_target_labels) > 0:
                    mal_labels[num_mal] = np.random.choice(allowed_target_labels)
                    mal_data.append(true_data[mal_idx])
                    num_mal += 1
                mal_idx += 1
            print(f"Source class: {true_labels}")
            print(f"Target class: {mal_labels} of {len(mal_data)}")
            self.mal_images = mal_data
            self.mal_targets = mal_labels
            self.test_images = self.mal_images
            self.test_targets = self.mal_targets

    def fetch_test_data(self, test_images, test_targets, allowed_source_labels):
        true_images = []
        true_labels = []
        for i, test_label in enumerate(test_targets):
            if len(allowed_source_labels) == 0:
                break
            if test_label in allowed_source_labels:
                allowed_source_labels.remove(test_label)
                true_images.append(test_images[i])
                true_labels.append(test_label)
        return true_images, true_labels

    def fetch_source_idxs(self, test_images, args):
        if args.mal_type in ["A", "B"]:
            return list(np.random.choice(10, size=self.num_mal_images * 10))
        elif args.mal_type in ["C", "D"]:
            return [7 for _ in range(self.num_mal_images * 10)]

    def fetch_targets(self, images_per_client, args):
        if args.mal_type in ["A", "C"]:
            return list(range(len(self.images_per_client)))
        elif args.mal_type in ["B", "D"]:
            return [1]

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
        idx = idx % len(self.test_images)
        raw_image = self.test_images[idx]
        image = Image.fromarray(raw_image)
        return image, self.test_targets[idx]

    """
    def _get_mal_item(self, idx):
        raw_image = self.mal_images[idx]
        image = Image.fromarray(raw_image)
        return image, self.mal_targets[idx]
    """
    def _get_mal_item(self, idx_within_client):
        print(f"fetching {idx_within_client}")
        new_idx = idx_within_client % len(self.mal_images)
        raw_image = self.mal_images[new_idx]
        image = Image.fromarray(raw_image)
        return image, self.mal_targets[new_idx]

    def _get_mal_item(self):
        new_idx = np.random.choice(len(self.mal_images))
        raw_image = self.mal_images[new_idx]
        image = Image.fromarray(raw_image)
        return image, self.mal_targets[new_idx]

    def client_fn(self, client_id):
        fn = "client{}.npy".format(client_id)
        return os.path.join(self.dataset_dir, fn)

    def test_fn(self):
        return os.path.join(self.dataset_dir, "test.npz")

class FedCIFAR100(FedCIFAR10):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
