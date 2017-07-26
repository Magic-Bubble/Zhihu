import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
from RNN import RNN
from TextCNN import TextCNN
from RCNN import RCNN
from utils import load_model

class Emsemble(nn.Module):
    def __init__(self, embed_mat, opt):
        super(Emsemble, self).__init__()
        
        self.model1 = model1 = RNN(embed_mat, opt)
        self.model1 = load_model(model1, model_dir=opt['model_dir'], model_name='RNN')
        self.model2 = model2 = TextCNN(embed_mat, opt)
        self.model2 = load_model(model2, model_dir=opt['model_dir'], model_name='TextCNN')

    def forward(self, x, y):
        logit1 = self.model1(x, y)
        logit2 = self.model2(x, y)
        logit = torch.cat((logit1.unsqueeze(2), logit2.unsqueeze(2)), 2)
        return logit.sum(2).squeeze(2)