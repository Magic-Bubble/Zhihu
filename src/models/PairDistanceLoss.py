import numpy as np
import torch
import torch.nn as nn

class PairDistanceLoss(nn.Module):
    def __init__(self):
        super(PairDistanceLoss, self).__init__()
        
    def forward(self, x, y):
        # x: predict_label (N, Class)
        # y: true_label (N, Class)
        y = y.int()
        N = x.size(0)
        C = x.size(1)
        loss = 0.0
        for i in range(N):
            ni = torch.sum(y[i]).int().data[0]
            true_xi, false_xi = x[i][y[i]==0], x[i][y[i]==1]
            true_vec, false_vec = true_xi.expand(ni, C-ni), false_xi.expand(C-ni, ni).t()
            loss += 1./(ni*(C-ni))*torch.sum(torch.exp(-(true_vec-false_vec)))
        loss /= N
        return loss