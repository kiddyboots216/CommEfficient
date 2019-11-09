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
from torch.utils.data import DataLoader, TensorDataset
from pytorch_transformers import cached_path

PERSONACHAT_URL = "https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json"
HF_FINETUNED_MODEL = "https://s3.amazonaws.com/models.huggingface.co/transfer-learning-chatbot/finetuned_chatbot_gpt.tar.gz"

SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels",
                "mc_labels", "token_type_ids"]
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]

logger = logging.getLogger(__file__)

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


def get_dataset(tokenizer, dataset_path, dataset_cache=None):
    """ Get PERSONACHAT from S3 """
    dataset_path = dataset_path or PERSONACHAT_URL
    if dataset_cache:
        # Do avoid using GPT cache for GPT-2 and vice-versa
        dataset_cache = dataset_cache + '_' + type(tokenizer).__name__
        if os.path.isfile(dataset_cache):
            logger.info("Load tokenized dataset from cache at %s",
                        dataset_cache)
            dataset = torch.load(dataset_cache)
    else:
        logger.info("Download dataset from %s", dataset_path)
        personachat_file = cached_path(dataset_path)
        with open(personachat_file, "r", encoding="utf-8") as f:
            dataset = json.loads(f.read())

        logger.info("Tokenize and encode the dataset")
        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(
                        tokenizer.tokenize(obj)
                    )
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)
        dataset = tokenize(dataset)
        if dataset_cache:
            torch.save(dataset, dataset_cache)
    return dataset

def get_dataset_personalities(tokenizer, dataset_path, dataset_cache=None):
    """ Get personalities from PERSONACHAT """

    personachat = get_dataset(tokenizer, dataset_path, dataset_cache)

    logger.info("Filter personalities")
    personalities = []
    for dataset in personachat.values():
        for dialog in dataset:
            personalities.append(dialog["personality"])

    logger.info("Gathered {} personalities".format(len(personalities)))
    return personalities

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def pad_dataset(dataset, padding=0):
    """ Pad the dataset.

    This could be optimized by defining a Dataset class and padding at
    the batch level, but this is simpler.
    """
    max_l = max(len(x) for x in dataset["input_ids"])
    for name in PADDED_INPUTS:
        pad_val = padding if name != "lm_labels" else -1
        for x in dataset[name]:
            x.extend(repeat(pad_val, max_l - len(x)))
    return dataset

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
                                 if (len(sequence) - i) % 2
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


def get_data_loaders(args, tokenizer, test=False):
    """ Prepare the dataset for training and evaluation """
    personachat = get_dataset(tokenizer,
                              args.dataset_path,
                              args.dataset_cache)

    logger.info("Build inputs and labels")
    datasets = {"train": defaultdict(list),
                "valid": defaultdict(list)}
    dialogs_processed = 0
    for dataset_name, dataset in personachat.items():
        num_candidates = len(dataset[0]["utterances"][0]["candidates"])
        if args.num_candidates > 0 and dataset_name == 'train':
            num_candidates = min(args.num_candidates, num_candidates)
        for dialog in dataset:
            persona = dialog["personality"].copy()
            dialogs_processed += 1
            for _ in range(args.personality_permutations):
                for utterance in dialog["utterances"]:
                    history = utterance["history"][-(2*args.max_history+1):]
                    candidates = utterance["candidates"][-num_candidates:]
                    for j, candidate in enumerate(candidates):
                        lm_labels = bool(j == num_candidates - 1)
                        instance = build_input_from_segments(
                                persona, history, candidate,
                                tokenizer, lm_labels
                            )
                        for input_name, input_array in instance.items():
                            datasets[dataset_name][input_name].append(
                                    input_array
                                )
                    datasets[dataset_name]["mc_labels"].append(
                            num_candidates - 1
                        )
                    datasets[dataset_name]["num_candidates"] = num_candidates
                # permuted personalities
                persona = [persona[-1]] + persona[:-1]
            if test and dialogs_processed > args.num_dialogs:
                break

    logger.info("Pad inputs and convert to Tensor")
    tensor_datasets = {"train": [], "valid": []}
    for dataset_name, dataset in datasets.items():
        padding = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1])
        dataset = pad_dataset(dataset, padding=padding)
        for input_name in MODEL_INPUTS:
            tensor = torch.tensor(dataset[input_name])
            if input_name != "mc_labels":
                shape = (-1, datasets[dataset_name]["num_candidates"])
                shape += tensor.shape[1:]
                tensor = tensor.view(shape)
            tensor_datasets[dataset_name].append(tensor)

    logger.info("Build train and validation dataloaders")
    train_dataset = TensorDataset(*tensor_datasets["train"])
    valid_dataset = TensorDataset(*tensor_datasets["valid"])
    train_sampler = None
    valid_sampler = None
    train_loader = DataLoader(train_dataset, sampler=train_sampler,
                              batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler,
                              batch_size=args.batch_size, shuffle=False)

    logger.info("Train dataset (Batch, Candidates, Seq length): {}".format(
        train_dataset.tensors[0].shape)
    )
    logger.info("Valid dataset (Batch, Candidates, Seq length): {}".format(
        valid_dataset.tensors[0].shape)
    )
    return train_loader, valid_loader, train_sampler, valid_sampler


