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
        self.model1 = load_model(model1, model_dir=opt['model_dir'], model_name='RNN', name="epoch_6_2017-07-16#20:01:51.params")
        self.model2 = model2 = TextCNN(embed_mat, opt)
        self.model2 = load_model(model2, model_dir=opt['model_dir'], model_name='TextCNN')
        self.model3 = model3 = RCNN(embed_mat, opt)
        self.model3 = load_model(model3, model_dir=opt['model_dir'], model_name='RCNN', name='epoch_5_2017-07-20#19:22:52.params')

    def forward(self, x, y):
        logit1 = self.model1(x, y)
        logit2 = self.model2(x, y)
        logit3 = self.model3(x, y)
        logit = torch.cat((logit1.unsqueeze(2), logit2.unsqueeze(2)), 2)
        logit = torch.cat((logit, logit3.unsqueeze(2)), 2)
        return logit.sum(2).squeeze(2)