#!./env python

import torch

import numpy as np
import random
import os

from collections import Counter
from collections.abc import Iterable
from ..utils import print, save_array
import warnings

def find_indices(b, a):
    """
        Find the indices of all elements from array 'a' within array 'b', in the order of their appearance in array 'a'
    """
    sorter = np.argsort(b)
    return sorter[np.searchsorted(b, a, sorter=sorter)]

def class_balanced_choice(rng, labels, n_choice, classes=None):
    if classes is None:
        classes = np.arange(len(labels.unique()))
    # if n_choice % n_class != 0:
    #     warnings.warn('sample size %i is not divisible by the number of classes %i!' % (n_choice, n_class))

    unique_labels, counts = labels.unique(return_counts=True)
    n_choice_per_class = (counts.numpy() / len(labels) * n_choice).round().astype(int)
    n_choice_per_class = dict(zip(unique_labels.numpy(), n_choice_per_class))
    idx_choice = []
    for c in classes:
        idx = torch.where(labels == c)[0].numpy()
        rng.shuffle(idx)
        idx_choice.extend(idx[:n_choice_per_class[c]])
    return np.array(idx_choice)

def random_split(dataset, size1, seed=None, class_balanced=True, classes=None):
    if seed is not None:
        rng = np.random.default_rng(seed) # fixed seed
    else:
        rng = np.random.default_rng()
    if class_balanced:
        ids1 = class_balanced_choice(rng, dataset.labels, size1, classes=classes)
    else:
        ids1 = rng.choice(len(dataset), size1, replace=False)
    ids2 = np.setdiff1d(np.arange(len(dataset)), ids1)
    return ids1, ids2

def get_subset(dataset, ids):
    assert(isinstance(ids, np.ndarray))
    dataset_ = torch.utils.data.Subset(dataset, ids)
    dataset_.labels = dataset.labels[ids]
    dataset_.classes = dataset.classes
    dataset_.decode = dataset.decode
    dataset_ .label_decode = dataset.label_decode
    return dataset_

class EncodingDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, decode=None, label_decode=None, classes=None, device=None):
        self.encodings = encodings
        if labels is None:
            any_key = list(self.encodings.keys())[0]
            self.labels = torch.empty(len(self.encodings[any_key]), dtype=torch.int64)
        else:
            self.labels = labels
        self.device = device
        
        self.decode = decode
        self.label_decode = label_decode
        if classes is None:
            self.classes = labels.unique().tolist()
        else:
            self.classes = classes

    def __getitem__(self, idx):
        inputs = {key: val[idx].to(self.device) for key, val in self.encodings.items()}
        labels = self.labels[idx].to(self.device)
        return inputs, labels

    def __len__(self):
        return len(self.labels)


class WeightedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, weights=None):
        assert(isinstance(dataset, torch.utils.data.Dataset))
        if weights is None:
            self.weights = {}
        else:
            for key in weights:
                assert(len(weights[key]) == len(dataset)), (key, len(weights[key]), len(dataset))
            self.weights = weights
        self.dataset = dataset

        # save attributes
        self.labels = dataset.labels
        self.classes = dataset.classes
        self.decode = dataset.decode
        self.label_decode = dataset.label_decode

    def __getitem__(self, index):
        data, target = self.dataset[index]
        weight = dict([(key, self.weights[key][index]) for key in self.weights])
        weight['index'] = index

        return data, target, weight

    def __len__(self):
        return len(self.dataset)


def summary(dataset, show_examples=True):
    print('---------- Basic info ----------------')
    print('dataset size: %i' % len(dataset))
    print('input shape: ', dataset[0][0]['input_ids'].size())
    print('num classes: %i' % len(dataset.classes))
    print('---------- Frequency count -----------------')
    if len(dataset[0]) == 2:
        unique_labels, counts = np.unique([label.item() for _, label in dataset], return_counts=True)
        # d = dict(Counter([label.item() for _, label in dataset]).most_common())
    else:
        unique_labels, counts = np.unique([label.item() for _, label, _ in dataset], return_counts=True)
        # d = dict(Counter([label.item() for _, label, _ in dataset]).most_common())
    counts_dict = dict(zip(unique_labels, counts))
    for c in dataset.classes:
        if c not in counts_dict:
            counts_dict[c] = 0
    print(counts_dict)
    if show_examples:
        print('---------- Example Decoding-----------------')
        for i, (input_, label) in enumerate(dataset):
            print(dataset.label_decode(label.item()))
            print(dataset.decode(input_))
            if i > 1:
                break
    print('\n')