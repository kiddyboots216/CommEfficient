# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.

import os
import json
import tarfile
import tempfile
import logging
from collections import defaultdict
from itertools import chain, repeat
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from pytorch_transformers import cached_path

from torch.nn.utils.rnn import pad_sequence

PERSONACHAT_URL = "https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json"
HF_FINETUNED_MODEL = "https://s3.amazonaws.com/models.huggingface.co/transfer-learning-chatbot/finetuned_chatbot_gpt.tar.gz"

SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels",
                "mc_labels", "token_type_ids"]
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]

logger = logging.getLogger(__file__)

class PersonaChatDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, tokenizer, num_candidates, max_history,
                 train=True, download=True):
        self.dataset_dir = dataset_dir
        self.tokenizer = tokenizer
        self.num_candidates = num_candidates
        self.max_history = max_history
        self.type = "train" if train else "val"

        if download and not os.path.exists(dataset_dir):
            self.download_and_split_data(dataset_dir)

        self.load_stats(train)

    @property
    def data_per_client(self):
        cumsum = np.cumsum(self.dialogs_per_client)
        cumsum = np.hstack([[0], cumsum])
        utterances_per_client = np.array([
            sum(self.train_utterances_per_dialog[s:s + dpc])
            for s, dpc in zip(cumsum, self.dialogs_per_client)
        ])
        return utterances_per_client

    @property
    def num_clients(self):
        return len(self.dialogs_per_client)

    def load_stats(self, train):
        with open(self.stats_fn(), "r") as f:
            stats = json.load(f)
            self.dialogs_per_client = stats["dialogs_per_client"]
            self.train_utterances_per_dialog = \
                    stats["train_utterances_per_dialog"]
            self.val_utterances_per_dialog = \
                    stats["val_utterances_per_dialog"]

    def download_and_split_data(self, dataset_dir):
        # download the dataset
        os.makedirs(dataset_dir, exist_ok=True)
        dataset_path = self.download_dataset(dataset_dir)

        # split into client datasets and one validation set
        datasets, stats = self.split_dataset(dataset_path)
        client_datasets, validation_set = datasets
        dialogs_per_client, train_utterances_per_dialog, \
                val_utterances_per_dialog = stats

        # save client datasets to disk
        for client_id, personality in enumerate(client_datasets):
            fn = self.client_fn(client_id)
            if os.path.exists(fn):
                raise RuntimeError("won't overwrite existing split")
            with open(fn, "w") as f:
                json.dump(client_datasets[tuple(personality)], f)

        # save validation set to disk
        fn = self.validation_fn()
        if os.path.exists(fn):
            raise RuntimeError("won't overwrite existing val set")
        with open(fn, "w") as f:
            json.dump(validation_set, f)

        # save stats to disk
        fn = self.stats_fn()
        if os.path.exists(fn):
            raise RuntimeError("won't overwrite existing stats file")
        stats = {"dialogs_per_client": dialogs_per_client,
                 "train_utterances_per_dialog":train_utterances_per_dialog,
                 "val_utterances_per_dialog": val_utterances_per_dialog}
        with open(fn, "w") as f:
            json.dump(stats, f)

    def download_dataset(self, dataset_path):
        # download personachat to dataset_path
        msg = "Downloading personachat from S3 into {}"
        logger.info(msg.format(dataset_path))
        return cached_path(PERSONACHAT_URL, dataset_path)

    def split_dataset(self, dataset_path):
        dataset = None
        with open(dataset_path, "r") as dataset_file:
            dataset = json.loads(dataset_file.read())

        val_set = dataset["valid"]
        val_utterances_per_dialog = [len(dialog["utterances"])
                                     for dialog in val_set]

        client_datasets = defaultdict(list)
        for dialog in dataset["train"]:
            personality = dialog["personality"]
            client_datasets[tuple(personality)].append(dialog)

        # so that we can quickly turn an utterance index into
        # a client idx, figure out how many dialogs per client and
        # how many utterances per dialog

        # fix the order of the clients
        client_personalities = list(client_datasets.keys())
        dialogs_per_client = []
        train_utterances_per_dialog = []
        for p in client_personalities:
            dialogs = client_datasets[p]
            dialogs_per_client.append(len(dialogs))
            train_utterances_per_dialog.extend([len(dialog["utterances"])
                                                for dialog in dialogs])

        datasets = (client_datasets, val_set)
        stats = (dialogs_per_client,
                 train_utterances_per_dialog,
                 val_utterances_per_dialog)
        return datasets, stats

    def __len__(self):
        if self.type == "train":
            return sum(self.train_utterances_per_dialog)
        elif self.type == "val":
            return sum(self.val_utterances_per_dialog)

    def __getitem__(self, idx):
        if self.type == "train":
            return self.get_train_item(idx)
        elif self.type == "val":
            return self.get_val_item(idx)

    def get_val_item(self, idx):
        cumsum = np.cumsum(self.val_utterances_per_dialog)
        dialog_id = np.searchsorted(cumsum, idx)
        idx_within_dialog = idx - cumsum[dialog_id]

        dialog = val_set[dialog_id][idx_within_dialog]

        return dialog_to_input(dialog)

    def get_train_item(self, idx):
        # idx refers to an utterance, which is part of a dialog,
        # which itself belongs to a certain client

        # figure out which dialog idx is in
        cumsum = np.cumsum(self.train_utterances_per_dialog)
        dialog_id = np.searchsorted(cumsum, idx, side="right")
        cumsum = np.hstack([[0], cumsum[:-1]])
        idx_within_dialog = idx - cumsum[dialog_id]

        # figure out which client this dialog is in
        cumsum = np.cumsum(self.dialogs_per_client)
        client_id = np.searchsorted(cumsum, dialog_id, side="right")
        cumsum = np.hstack([[0], cumsum[:-1]])
        idx_within_client = dialog_id - cumsum[client_id]

        # read in the corresponding client's dataset
        fn = self.client_fn(client_id)
        client_dataset = None
        with open(fn, "r") as f:
            client_dataset = json.load(f)

        # and extract the desired record
        dialog = client_dataset[idx_within_client]

        # because the dataset is federated, each record needs to be
        # associated with the client the record is on, so we can assign
        # forward/backward passes to the correct client later on
        model_input = self.dialog_to_input(dialog, idx_within_dialog)
        return (client_id,) + model_input

    def dialog_to_input(self, dialog, idx_within_dialog):
        personality = dialog["personality"]
        utterances = dialog["utterances"]
        utterance = utterances[idx_within_dialog]
        history = utterance["history"]
        candidates = utterance["candidates"]

        num_candidates = len(candidates)
        if self.num_candidates > 0 and self.type == "train":
            num_candidates = min(self.num_candidates, num_candidates)

        candidates = utterance["candidates"][-num_candidates:]
        history = utterance["history"][-(2 * self.max_history + 1):]

        return json_to_input(self.tokenizer, personality,
                             history, utterance, candidates)

    def client_fn(self, client_id):
        fn = "client{}.json".format(client_id)
        return os.path.join(self.dataset_dir, fn)

    def validation_fn(self):
        return os.path.join(self.dataset_dir, "validation.json")

    def stats_fn(self):
        return os.path.join(self.dataset_dir, "train_stats.json")

def tokenize(obj, tokenizer):
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize(obj)
            )
    if isinstance(obj, dict):
        return dict((n, tokenize(o, tokenizer)) for n, o in obj.items())
    return list(tokenize(o, tokenizer) for o in obj)


def json_to_input(tokenizer, personality, history, utterance, candidates):
    personality = tokenize(personality, tokenizer)
    history = tokenize(history, tokenizer)
    utterance = tokenize(utterance, tokenizer)
    candidates = tokenize(candidates, tokenizer)
    model_input = defaultdict(list)
    num_candidates = len(candidates)
    for j, candidate in enumerate(candidates):
        lm_labels = bool(j == num_candidates - 1)
        instance = build_input_from_segments(personality, history,
                                             candidate, tokenizer,
                                             lm_labels)
        for input_name, input_array in instance.items():
            model_input[input_name].append(input_array)
    model_input["mc_labels"] = num_candidates - 1

    for input_name in MODEL_INPUTS:
        if input_name != "mc_labels":
            tensors = [torch.tensor(l) for l in model_input[input_name]]
            model_input[input_name] = tensors
        # all model inputs besides mc_labels have shape num_candidates, ...
        #if input_name != "mc_labels":
        #    tensor = tensor.view(num_candidates, tensor.shape[1:])

    # convert from dict to tuple in the correct order
    model_input = tuple(model_input[name] for name in MODEL_INPUTS)

    return model_input


def download_pretrained_model():
    """ Download and extract finetuned model from S3 """
    resolved_archive_file = cached_path(HF_FINETUNED_MODEL)
    tempdir = tempfile.mkdtemp()

    logger.info("extracting archive file {} to temp dir {}".format(
        resolved_archive_file, tempdir
    ))
    with tarfile.open(resolved_archive_file, 'r:gz') as archive:
        archive.extractall(tempdir)
    return tempdir

def build_input_from_segments(persona, history, reply, tokenizer,
                              lm_labels=False, with_eos=True):
    """ Build a sequence of input

    Builds from 3 segments: persona, history and last reply.
    """
    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(
            SPECIAL_TOKENS[:-1]
        )

    instance = {}
    sequence = [[bos] + list(chain(*persona))] + history
    sequence += [reply + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + [[speaker2
                                 if (len(sequence) - i) % 2 == 0
                                 else speaker1]
                                + s
                                for i, s in enumerate(sequence[1:])]

    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [speaker2 if i % 2 else speaker1
                                  for i, s in enumerate(sequence)
                                  for _ in s]
    instance["mc_token_ids"] = len(instance["input_ids"]) - 1
    instance["lm_labels"] = [-1] * len(instance["input_ids"])
    if lm_labels:
        instance["lm_labels"] = [-1] * sum(len(s) for s in sequence[:-1])
        instance["lm_labels"] += [-1] + sequence[-1][1:]
    return instance

"""
def pad(records, padval=0):
    # records is a list of tuples, where each tuple contains columns
    # (client_id,) + MODEL_INPUTS
    ret = []
    max_l = max(input_ids.size()[1] for input_ids in records[1])
    for i, name in enumerate(("client_id",) + MODEL_INPUTS):
        if name not in PADDED_INPUTS:
            ret.append(records[
            continue

        pad_val = padval if name != "lm_labels" else -1
        ret[i] = pad_sequence([record[i] for record in records])
        for x in ret[name]:
            x.extend(repeat(pad_val, max_l - len(x)))
    return ret
"""

def collate_fn(records):
    # records is a list of tuples, where each tuple contains columns
    # (client_id,) + MODEL_INPUTS

    # need to return a batch, which is a tuple of tensors,
    # appropriately padded

    batch = []
    # input_ids has one sequence for each candidate, and all other
    # sequence model inputs have the same lengths, so we can just use
    # the max sequence length in input_ids
    max_l = max(len(input_ids)
                 for record in records
                 for input_ids in record[1])
    for i, name in enumerate(["client_id"] + MODEL_INPUTS):
        if name in PADDED_INPUTS:
            pad_val = 0 if name != "lm_labels" else -1
            sequences = [r for record in records for r in record[i]]
            padded = pad_sequence(sequences)
            # shape should be batch_size x num_candidates x seq_len
            reshaped = padded.view(len(records), len(records[0][1]), -1)
            batch.append(reshaped)
        else:
            batch.append(torch.stack([torch.tensor(record[i])
                                      for record in records]))

    return tuple(batch)

class FedSampler:
    def __init__(self, dataset, num_workers, local_batch_size,
                 shuffle_clients=True):
        self.dataset = dataset
        self.num_workers = num_workers
        self.local_batch_size = local_batch_size
        self.shuffle_clients = shuffle_clients

    def __iter__(self):
        data_per_client = self.dataset.data_per_client
        cumsum = np.cumsum(data_per_client)
        cumsum = np.hstack([[0], cumsum])
        permuted_data = np.hstack([
                s + np.random.choice(u, u, replace=False)
                for s, u in zip(cumsum, data_per_client)
            ])
        cur_idx_within_client = np.zeros(self.dataset.num_clients,
                                         dtype=int)
        def sampler():
            while True:
                nonexhausted_clients = np.where(
                        cur_idx_within_client < data_per_client
                    )[0]
                if len(nonexhausted_clients) == 0:
                    break
                num_workers = min(self.num_workers,
                                  len(nonexhausted_clients))
                workers = np.random.choice(nonexhausted_clients,
                                           num_workers,
                                           replace=False)
                records_remaining = (data_per_client[workers]
                                     - cur_idx_within_client[workers])
                yield np.hstack([
                    permuted_data[s:s + records_remaining[i]]
                    for i, s in enumerate(cumsum[workers] +
                                          cur_idx_within_client[workers])
                ])
                cur_idx_within_client[workers] += self.local_batch_size

        return sampler()

    def __len__(self):
        return len(self.dataset)

def get_data_loaders(args, tokenizer):
    train_dataset = PersonaChatDataset(args.dataset_dir,
                                       tokenizer,
                                       args.num_candidates,
                                       args.max_history,
                                       train=True)
    val_dataset = PersonaChatDataset(args.dataset_dir,
                                     tokenizer,
                                     args.num_candidates,
                                     args.max_history,
                                     train=False)
    if args.do_iid:
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  collate_fn=collate_fn,
                                  shuffle=True,
                                  num_workers=4)
    else:
        train_sampler = FedSampler(train_dataset,
                                   args.num_workers,
                                   args.local_batch_size)
        train_loader = DataLoader(train_dataset,
                                  batch_sampler=train_sampler,
                                  collate_fn=collate_fn,
                                  num_workers=4)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False)

    """
    logger.info("Train dataset (Batch, Candidates, Seq length): {}".format(
        train_dataset.tensors[0].shape)
    )
    logger.info("Val dataset (Batch, Candidates, Seq length): {}".format(
        val_dataset.tensors[0].shape)
    )
    """
    return train_loader, val_loader


