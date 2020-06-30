import os
import json
import tarfile
import tempfile
from collections import defaultdict
from itertools import chain
import random

from data_utils import FedDataset
import torch
import numpy as np

from torch.nn.utils.rnn import pad_sequence

from pytorch_transformers import cached_path

from utils import Logger

__all__ = ["FedPERSONA", "personachat_collate_fn"]

logger = Logger()

PERSONACHAT_URL = "https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json"
HF_FINETUNED_MODEL = "https://s3.amazonaws.com/models.huggingface.co/transfer-learning-chatbot/finetuned_chatbot_gpt.tar.gz"

SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels",
                "mc_labels", "token_type_ids"]
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]

class FedPERSONA(FedDataset):
    def __init__(self, tokenizer, num_candidates, max_history,
            personality_permutations, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.num_candidates = num_candidates
        self.max_history = max_history
        self.personality_permutations = personality_permutations

        # keep the entire val set in memory, since why not
        if self.type == "val":
            with open(self.validation_fn(), "r") as val_f:
                self.raw_val_set = json.load(val_f)

    @property
    def data_per_client(self):
        if self.do_iid:
            num_data = len(self)
            utterances_per_client = (np.ones(self.num_clients, dtype=int)
                                     * num_data // self.num_clients)
            # some clients need 1 extra datum if num_clients doesn't
            # divide num_data
            extra = num_data % self.num_clients
            utterances_per_client[self.num_clients - extra:] += 1
            return utterances_per_client
        else:
            cumsum = np.cumsum(self.dialogs_per_client)
            cumsum = np.hstack([[0], cumsum])
            utterances_per_client = np.array([
                sum(self.train_utterances_per_dialog[s:s + dpc])
                for s, dpc in zip(cumsum, self.dialogs_per_client)
            ])
            return utterances_per_client

    @property
    def num_clients(self):
        if self.do_iid:
            # if the user didn't specify how many clients, assume
            # we have the same number of clients as the natural
            # data partitioning, but we'll shuffle the data iid among
            # the clients later
            return (self._num_clients if self._num_clients is not None
                                      else len(self.dialogs_per_client))
        else:
            return len(self.dialogs_per_client)

    def _load_meta(self, train):
        with open(self.stats_fn(), "r") as f:
            stats = json.load(f)
            self.dialogs_per_client = stats["dialogs_per_client"]
            self.train_utterances_per_dialog = \
                    stats["train_utterances_per_dialog"]
            self.val_utterances_per_dialog = \
                    stats["val_utterances_per_dialog"]

    def prepare_datasets(self, download=True):
        # download the dataset
        os.makedirs(self.dataset_dir, exist_ok=True)
        dataset_path = self.download_dataset(self.dataset_dir)

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
        """ Produces one file per client, and one with global stats

        Reads in a JSON file at `dataset_path`, partitions the data
        into clients based on the personality, and writes the data
        for each client to a separate JSON file. Also writes global
        stats needed for fast indexing to self.stats_fn()
        """
        raw_dataset = None
        with open(dataset_path, "r") as dataset_file:
            raw_dataset = json.load(dataset_file)

        val_set = raw_dataset["valid"]
        val_utterances_per_dialog = [len(dialog["utterances"])
                                     for dialog in val_set]

        client_datasets = defaultdict(list)
        for dialog in raw_dataset["train"]:
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
            return self._get_train_item(idx)
        elif self.type == "val":
            return self._get_val_item(idx)

    def _get_val_item(self, idx):
        cumsum = np.cumsum(self.val_utterances_per_dialog)
        dialog_id = np.searchsorted(cumsum, idx)
        idx_within_dialog = idx - cumsum[dialog_id]

        dialog = self.raw_val_set[dialog_id]
        personality = dialog["personality"]
        utterance = dialog["utterances"][idx_within_dialog]

        # return something for client_id so the shape is what's
        # expected, even though -1 is an invalid client_id and shouldn't
        # be used
        return (-1,) + self.utterance_to_input(personality, utterance)

    def _get_train_item(self, idx):
        # idx refers to an utterance, which is part of a dialog,
        # which itself belongs to a certain client

        # we achieve iid sampling by randomly permuting the data
        # and returning a fake client_id below
        orig_idx = idx
        if self.do_iid:
            idx = self.iid_shuffle[idx]

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
        raw_client_dataset = None
        with open(fn, "r") as f:
            raw_client_dataset = json.load(f)

        # and extract the desired record
        dialog = raw_client_dataset[idx_within_client]
        personality = dialog["personality"]
        utterance = dialog["utterances"][idx_within_dialog]

        # because the dataset is federated, each record needs to be
        # associated with the client the record is on, so we can assign
        # forward/backward passes to the correct client later on
        model_inputs = []
        for _ in range(self.personality_permutations): 
            random.shuffle(personality)
            model_input = self.utterance_to_input(personality, utterance)
            model_inputs.extend(model_input)

        # when do_iid, we pretend there are only self.num_clients
        # clients. So we have to remap idx to client_id
        if self.do_iid:
            cumsum = np.cumsum(self.data_per_client)
            client_id = np.searchsorted(cumsum, orig_idx, side="right")

        return (client_id,) + model_input

    def utterance_to_input(self, personality, utterance):
        history = utterance["history"]
        candidates = utterance["candidates"]

        num_candidates = len(candidates)
        # restrict to self.num_candidates if we're training
        if self.num_candidates > 0 and self.type == "train":
            num_candidates = min(self.num_candidates, num_candidates)

        candidates = utterance["candidates"][-num_candidates:]
        history = utterance["history"][-(2 * self.max_history + 1):]

        return raw_to_input(self.tokenizer, personality,
                            history, candidates)

    def client_fn(self, client_id):
        fn = "client{}.json".format(client_id)
        return os.path.join(self.dataset_dir, fn)

    def validation_fn(self):
        return os.path.join(self.dataset_dir, "validation.json")

    def stats_fn(self):
        return os.path.join(self.dataset_dir, "stats.json")

def tokenize(obj, tokenizer):
    """ Recursively tokenize all strings in obj """
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize(obj)
            )
    if isinstance(obj, dict):
        return dict((n, tokenize(o, tokenizer)) for n, o in obj.items())
    return list(tokenize(o, tokenizer) for o in obj)


def raw_to_input(tokenizer, personality, history, candidates):
    """ Converts from dict of strings to (almost) valid input for the model

    "Almost" since we still need the collate_fn to pad & combine
    the tensors for each candidate
    """
    personality = tokenize(personality, tokenizer)
    history = tokenize(history, tokenizer)
    candidates = tokenize(candidates, tokenizer)

    model_input = defaultdict(list)
    num_candidates = len(candidates)
    # several of the model's inputs are num_candidates x sequence_len,
    # so process each candidate and append the result to model_input[...]
    for j, candidate in enumerate(candidates):
        lm_labels = bool(j == num_candidates - 1)
        instance = build_input_from_segments(personality, history,
                                             candidate, tokenizer,
                                             lm_labels)
        for input_name, input_array in instance.items():
            model_input[input_name].append(input_array)

    # the last candidate is always the correct choice
    model_input["mc_labels"] = num_candidates - 1

    for input_name in MODEL_INPUTS:
        # tensorize
        if input_name != "mc_labels":
            tensors = [torch.tensor(l) for l in model_input[input_name]]
            model_input[input_name] = tensors

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

def personachat_collate_fn(records):
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
            sequences = [s for record in records for s in record[i]]
            padded = pad_sequence(sequences,
                                  batch_first=True,
                                  padding_value=pad_val)
            # padded has shape len(sequences) x max_l, where
            # len(sequences) = num_candidates * len(records)

            # we want batch_size x num_candidates x seq_len
            # where batch_size = len(records)
            reshaped = padded.view(len(records), len(records[0][1]), -1)
            batch.append(reshaped)
        else:
            batch.append(torch.stack([torch.tensor(record[i])
                                      for record in records]))

    return tuple(batch)

