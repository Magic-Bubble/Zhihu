import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Stack(nn.Module):
    def __init__(self, opt):
        super(Stack, self).__init__()
        self.stack_num = stack_num = opt['stack_num']
        self.class_num = opt['class_num']
        self.fc = nn.Linear(stack_num, 1, bias=False)
        self.fc.weight.data.fill_(0.2)

    def forward(self, x):
        x = torch.stack(x, 2)
        x = self.fc(x.view(-1, self.stack_num))
        logit = x.squeeze(1).view(-1, self.class_num)
        return logit