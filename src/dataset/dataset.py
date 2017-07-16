import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class Dataset:
    def __init__(self, **datas):
        self.test = datas.get('test', False)
        
        self.title = datas['title']
        self.desc = datas['desc']

        # train=True, return label
        if self.test is False:
            self.n_classes = datas['class_num']
            self.label = datas['label']

    def __getitem__(self, idx):
        title = torch.from_numpy(self.title[idx])
        desc = torch.from_numpy(self.desc[idx])
        if self.test is False:
            label = torch.zeros(self.n_classes).scatter_(0, torch.from_numpy(self.label[idx]).long(), 1).int()
            # return torch.cat((title, desc)), label
            return title, desc, label
        # return torch.cat((title, desc))
        return title, desc

    def __len__(self):
        return self.title.shape[0]