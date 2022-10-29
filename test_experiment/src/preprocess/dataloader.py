#!./env python
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from . import get_dataset
from ..utils import Dict2Obj
from . import EncodingDataset, WeightedDataset, summary
from . import random_split, get_subset
import warnings
import os

__all__ = ['get_loaders']

def get_loaders(dataset='imdb', test_ratio=0.15, batch_size=16,
                shuffle_train_loader=True,
                data_dir='./data',
                show_examples=True,
                config=None):

    data_dict = get_dataset(dataset, data_dir=data_dir, config=config)

    # --- select test subsets ------
    allset = EncodingDataset(data_dict['encodings'],
                             data_dict['labels'],
                             decode=lambda inputs: data_dict['tokenizer'].decode(inputs['input_ids']),
                             label_decode=lambda label: data_dict['label_names'][label],
                             classes=data_dict['labels'].unique().tolist(),
                             device=config.device)
    testsubids, trainsubids = random_split(allset, int(len(allset) * test_ratio),
                                           seed=42, class_balanced=True, classes=None)
    testset = get_subset(allset, testsubids)
    trainset = get_subset(allset, trainsubids)

    # --- apply weights ---
    # trainset = WeightedDataset(trainset)
    # testset = WeightedDataset(testset)

    # --- summary ---
    print('- training set -')
    summary(trainset, show_examples=show_examples)
    print('- test set -')
    summary(testset, show_examples=show_examples)

    # --- integrate ---
    loaders = {'trainset': trainset,
               'testset': testset,
               'trainloader': DataLoader(trainset, batch_size=batch_size, shuffle=shuffle_train_loader),
               'testloader': DataLoader(testset, batch_size=batch_size, shuffle=False)}
    loaders = Dict2Obj(loaders)
    loaders.classes = trainset.classes
    loaders.num_classes = len(loaders.classes)
    loaders.input_shape = trainset[0][0]['input_ids'].size()
    loaders.trainsubids = trainsubids
    return loaders


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loaders = get_loaders(config=Dict2Obj({'device': device}))
