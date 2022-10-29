#!./env python
import torch
import numpy as np

def nan_filter(arr):
    arr = np.array(arr)
    arr[np.isnan(arr)] = 0
    return arr

class Dict2Obj:
    """
        Turns a dictionary into a class
    """

    def __init__(self, dic):
        for key in dic:
            setattr(self, key, dic[key])

def iterable(obj):
    """
        check if an object is iterable
    """
    try:
        iter(obj)
    except Exception:
        return False
    else:
        return True
    
def is_in(labels, classes):
    """
        return the element-wise belonging of a tensor to a list
    """
    assert(len(labels.size())) == 1
    if not classes:
        return torch.ones(labels.size(0), dtype=torch.bool).to(labels.device)
    
    if not iterable(classes):
        return labels == classes
    
    id_in = torch.zeros(labels.size(0), dtype=torch.bool).to(labels.device)
    for c in classes:
        id_in |= (labels == c)
    return id_in
