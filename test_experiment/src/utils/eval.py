#!./env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from torch.cuda.amp import autocast

__all__ = ['F1Meter', 'AverageMeter', 'accuracy']

class F1Meter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = {}
        self.fp = {}
        self.fn = {}
        self.count = 0
        self.f1 = 0
    
    def update(self, predictions, labels):
        self.count += len(predictions)
        unique_labels = np.unique(np.concatenate([np.unique(labels),
                                                  np.unique(predictions)]))
        for label in unique_labels:
            if label not in self.tp:
                self.tp[label] = 0
            if label not in self.fp:
                self.fp[label] = 0
            if label not in self.fn:
                self.fn[label] = 0
            self.tp[label] += np.sum((labels==label) & (predictions==label))
            self.fp[label] += np.sum((labels!=label) & (predictions==label))
            self.fn[label] += np.sum((predictions!=label) & (labels==label))
            
    @property
    def macro_f1(self):
        f1s = []
        for label in self.tp:
            precision = self.tp[label] / (self.tp[label]+self.fp[label])
            recall = self.tp[label] / (self.tp[label]+self.fn[label])
            f1s.append(2 * (precision * recall) / (precision + recall))
        return np.mean(f1s) * 100.0
    
    @property
    def micro_f1(self):
        tp = np.sum([self.tp[label] for label in self.tp])
        fp = np.sum([self.fp[label] for label in self.fp])
        fn = np.sum([self.fn[label] for label in self.fn])
        precision = tp / (tp+fp)
        recall = tp / (tp+fn)
        return 2 * (precision * recall) / (precision + recall)  * 100.0


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res