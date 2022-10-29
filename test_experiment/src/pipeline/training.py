#!./env python

import torch
import torch.nn as nn
import numpy as np
import time
import os

from ..utils import AverageMeter, F1Meter, accuracy, Logger, nan_filter, load_log
from ..utils import save_checkpoint, save_model
from ..utils import print

__all__ = ['Trainer']

class Trainer:
    def __init__(self, config):
        self.config = config

        # init
        self.time_start = time.time()
        self.last_end = 0.
        if config.resume:
            self.last_end = self.get_last_time() # min

        # read best acc
        self.best_acc = 0.
        if config.resume:
            self.best_acc = self.get_best()
            print('> Best Acc: %.2f' % self.best_acc)

        # logger
        base_names = ['Epoch', 'lr', 'Time-elapse(Min)']
        self.logger = Logger(os.path.join(config.save_dir, 'log.txt'), title='log', resume=config.resume)
        metrics = ['Train-Loss', 'Test-Loss',
                   'Train-Acc', 'Test-Acc',
                   'Macro-F1', 'Micro-F1']
        self.logger.set_names(base_names + metrics)

    def evaluate(self, model, loader, criterion):
        model.eval()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        f1_meter = F1Meter()
        for inputs, labels in loader:
            with torch.no_grad():
                outputs = model(**inputs)
                loss = criterion(outputs.logits, labels)

            acc, = accuracy(outputs.logits.data, labels.data)
            loss_meter.update(loss.item(), labels.size(0))
            acc_meter.update(acc.item(), labels.size(0))
            f1_meter.update(outputs.logits.max(1)[1].data.cpu().numpy(), labels.data.cpu().numpy())

        return loss_meter.avg, acc_meter.avg, f1_meter.macro_f1, f1_meter.micro_f1

    def train(self, model, loader, criterion, optimizer, scheduler=None):
        model.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        for inputs, labels in loader:
            outputs = model(**inputs)
            loss = criterion(outputs.logits, labels)

            optimizer.zero_grad()
            loss.backward()
            if hasattr(self.config, 'gradient_clipping') and self.config.gradient_clipping:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            acc, = accuracy(outputs.logits.data, labels.data)
            loss_meter.update(loss.item(), labels.size(0))
            acc_meter.update(acc.item(), labels.size(0))

        return loss_meter.avg, acc_meter.avg


    def __call__(self, model, loaders, criterion, optimizer, scheduler=None):
        # start training
        for epoch in range(self.config.epochs):
            train_loss, train_acc = self.train(model, loaders.trainloader, criterion, optimizer, scheduler)
            test_loss, test_acc, macro_f1, micro_f1 = self.evaluate(model, loaders.testloader, criterion)

            self.__update_best(epoch, test_acc, model)

            # log
            time_elapse = (time.time() - self.time_start)/60 + self.last_end
            logs = [epoch, self.__get_lr(optimizer), time_elapse]
            logs += [train_loss, test_loss, train_acc, test_acc, macro_f1, micro_f1]
            self.logger.append(logs)

            # save checkpoint in case break
            save_checkpoint(epoch, model, optimizer, scheduler, config=self.config)

        # save last model
        save_model(model, config=self.config)

    def __update_best(self, epoch, acc, model):
        if acc > self.best_acc:
            print('> Best acc got at epoch %i. Best: %.2f Current: %.2f' % (epoch, acc, self.best_acc))
            self.best_acc = acc
            # save_model(model, basename='best_model', config=self.config)

    def __get_lr(self, optimizer):
        lrs = [param_group['lr'] for param_group in optimizer.param_groups]
        assert(len(lrs) == 1)
        return lrs[0]

    def get_last_time(self):
        return load_log(os.path.join(self.config.save_dir, 'log.txt'))['Time-elapse(Min)'][-1]

    def get_best(self):
        stats = load_log(os.path.join(self.config.save_dir, 'log.txt'), window=1)
        return np.max(nan_filter(stats['Test-Acc']))

    def close(self):
        self.logger.close()
