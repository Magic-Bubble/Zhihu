import torch
import torch.nn as nn

class K_MaxPooling(nn.Module):
    def __init__(self, k):
        super(K_MaxPooling, self).__init__()
        self.k = k
        
    def forward(self, x, dim):
        index = x.topk(self.k, dim=dim)[1].sort(dim=dim)[0]
        return x.gather(dim, index)