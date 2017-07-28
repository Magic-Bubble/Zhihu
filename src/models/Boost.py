import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
from FastText import FastText
from RNN import RNN
from TextCNN import TextCNN
from RCNN import RCNN
from utils import load_model

class Boost(nn.Module):
    def __init__(self, embed_mat, opt):
        super(Boost, self).__init__()
        
        self.model1 = model1 = FastText(embed_mat, opt)
        self.model1 = load_model(model1, model_dir=opt['model_dir'], model_name='FastText', name="folder_1_epoch_5_2017-07-28#02:01:07_0.4085.params")
        self.model2 = model2 = FastText(embed_mat, opt)
        self.model2 = load_model(model2, model_dir=opt['model_dir'], model_name='FastText', name="folder_2_epoch_5_2017-07-28#05:21:15_0.3657.params")
        self.model3 = model3 = FastText(embed_mat, opt)
        self.model3 = load_model(model3, model_dir=opt['model_dir'], model_name='FastText', name='folder_3_epoch_5_2017-07-28#08:54:48_0.3676.params')
        self.model4 = model4 = FastText(embed_mat, opt)
        self.model4 = load_model(model4, model_dir=opt['model_dir'], model_name='FastText', name='folder_4_epoch_5_2017-07-28#10:55:23_0.3655.params')

    def forward(self, x, y):
        logit1 = self.model1(x, y)
        logit2 = self.model2(x, y)
        logit3 = self.model3(x, y)
        logit4 = self.model4(x, y)
        logit = torch.cat((logit1.unsqueeze(2), logit2.unsqueeze(2)), 2)
        logit = torch.cat((logit, logit3.unsqueeze(2)), 2)
        logit = torch.cat((logit, logit4.unsqueeze(2)), 2)
        return logit.sum(2).squeeze(2)