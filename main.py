#!./env python

import torch
import torch.nn as nn
import numpy as np
from transformers import BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup

from src.pipeline import Trainer
from src.preprocess import get_loaders
from src.utils import Dict2Obj
import random
import datetime
import json
import os
import time

# - suppress huggingface warning
# from transformers import logging
# logging.set_verbosity_warning()

def train_wrap(**config):
    config = Dict2Obj(config)

    start = time.time()

    # time for log
    print('\n=====> Current time..')
    print(datetime.datetime.now())

    # -------------------------- Environment ------------------------------
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)
    config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ------ start training ----------
    config.save_dir = '.'

    train(config)
    clean(config)
    print('  Finished.. %.3f\n' % ((time.time() - start) / 60.0))

    # time for log
    print('\n=====> Current time..')
    print(datetime.datetime.now())


def train(config):
    print('=====> Init training..')
    config, model, loaders, criterion, optimizer, scheduler = init_train(config)

    print('=====> Training..')
    trainer = Trainer(config)
    trainer(model, loaders, criterion, optimizer, scheduler=scheduler)
    trainer.close()

def clean(config):
    print('\n=====> Clean saved checkpoints..')
    if not config.save_checkpoint:
        os.remove(os.path.join(config.save_dir, 'checkpoint.pth.tar'))
    if not config.save_model:
        if os.path.exists(os.path.join(config.save_dir, 'best_model.pt')):
            os.remove(os.path.join(config.save_dir, 'best_model.pt'))
        os.remove(os.path.join(config.save_dir, 'model.pt'))

def init_train(config):
    # Random seed
    if config.manual_seed is not None:
        random.seed(config.manual_seed)
        torch.manual_seed(config.manual_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.manual_seed)
        ## TODO: Hugging face seed

    # -------------------------- Dataset ------------------------------
    print('\n=====> Loading data..')
    loaders = get_loaders(dataset=config.dataset, test_ratio=config.test_ratio, batch_size=config.batch_size,
                          data_dir=config.data_dir,
                          config=config)

    # --------------------------------- criterion ------------------------------- 
    criterion = nn.CrossEntropyLoss(reduction='mean')

    # -------------------------- Model ------------------------------
    print('\n=====> Initializing model..')
    # from pretrained: replace the pretraining head with a randomly initialized classification head
    model = BertForSequenceClassification.from_pretrained(
        config.model,
        num_labels=loaders.num_classes,  # The number of output labels -- 2 for binary classification.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )
    model.to(config.device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    print("     Total params: %.2fM" % (sum(p.numel() for p in model.parameters())/1000000.0))

    ## -- Load weights
    if config.state_path:
        print('=====> Loading pre-trained weights..')
        assert(not config.resume), 'pre-trained weights will be overriden by resume checkpoint! Resolve this later!'
        state_dict = torch.load(config.state_path, map_location=config.device)
        model.load_state_dict(state_dict)
    
    # -------------------------- Optimizer ------------------------------
    print('\n=====> Initializing optimizer..')
    if config.opt.lower() == 'adamw':
        optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.wd)
    else:
        raise KeyError(config.opt)

    # -------------------------- Scheduler ------------------------------
    if config.scheduler.lower() == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,  # Default value in run_glue.py
                                                    num_training_steps=len(loaders.trainloader) * config.epochs)
    else:
        scheduler = None

    return config, model, loaders, criterion, optimizer, scheduler

if __name__ == '__main__':

    with open('para.json') as json_file:
        config = json.load(json_file)
        print(config)
    train_wrap(**config)
