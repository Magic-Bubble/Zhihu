import torch
import torch.nn as nn

class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        n, t = x.size(0), x.size(1) 
        # merge batch and seq dimensions
        x_reshape = x.contiguous().view(t * n, x.size(2))
        y = self.module(x_reshape)
        # we have to reshape Y
        y = y.contiguous().view(n, t, y.size()[1])
        return y