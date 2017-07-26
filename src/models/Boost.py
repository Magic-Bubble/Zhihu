import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
from RNN import RNN
from TextCNN import TextCNN
from RCNN import RCNN
from utils import load_model

class Boost(nn.Module):
    def __init__(self, embed_mat, opt):
        super(Boost, self).__init__()
        
        self.model1 = model1 = RNN(embed_mat, opt)
        self.model1 = load_model(model1, model_dir=opt['model_dir'], model_name='RNN', name="epoch_5_2017-07-25#17:52:54_0.4089.params")
        self.model2 = model2 = TextCNN(embed_mat, opt)
        self.model2 = load_model(model2, model_dir=opt['model_dir'], model_name='TextCNN', name="epoch_5_2017-07-25#18:29:53_0.4093.params")
        # self.model3 = model3 = RNN(embed_mat, opt)
        # self.model3 = load_model(model3, model_dir=opt['model_dir'], model_name='RNN', name='folder_3_epoch_5_2017-07-23#02:07:05_0.3576.params')
        # self.model4 = model4 = RNN(embed_mat, opt)
        # self.model4 = load_model(model4, model_dir=opt['model_dir'], model_name='RNN', name='folder_4_epoch_5_2017-07-23#11:02:18_0.3579.params')
        # self.model5 = model5 = TextCNN(embed_mat, opt)
        # self.model5 = load_model(model5, model_dir=opt['model_dir'], model_name='TextCNN', name='epoch_5_2017-07-22#19:08:41_0.4075.params')

    def forward(self, x, y):
        logit1 = self.model1(x, y)
        logit2 = self.model2(x, y)
        # logit3 = self.model3(x, y)
        # logit4 = self.model4(x, y)
        # logit5 = self.model5(x, y)
        logit = torch.cat((logit1.unsqueeze(2), logit2.unsqueeze(2)), 2)
        # logit = torch.cat((logit, logit3.unsqueeze(2)), 2)
        # logit = torch.cat((logit, logit4.unsqueeze(2)), 2)
        # logit = torch.cat((logit, logit5.unsqueeze(2)), 2)
        return logit.sum(2).squeeze(2)