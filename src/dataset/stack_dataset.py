import numpy as np
import torch
import torch.nn as nn

class Stack_Dataset:
    def __init__(self, **datas):
        self.test = datas.get('test', False)
        self.resmat = [torch.load(ii) for ii in datas['resmat']]
        self.stack_num = len(self.resmat)
        # train=True, return label
        if self.test is False:
            self.label = torch.load(datas['label'])

    def __getitem__(self, idx):
        res = tuple([ii[idx] for ii in self.resmat])
        if self.test is False:
            return res + (self.label[idx],)
        return res

    def __len__(self):
        return self.resmat[0].size(0)