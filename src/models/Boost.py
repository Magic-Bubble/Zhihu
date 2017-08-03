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
        
        self.model1 = model1 = RNN(embed_mat, opt)
        self.model1 = load_model(model1, model_dir=opt['model_dir'], model_name='RNN', name="")
        # self.model2 = model2 = RNN(embed_mat, opt)
        # self.model2 = load_model(model2, model_dir=opt['model_dir'], model_name='RNN', name="layer_7_epoch_5_2017-08-01#06:52:08_0.4073.params")
        # self.model3 = model3 = RNN(embed_mat, opt)
        # self.model3 = load_model(model3, model_dir=opt['model_dir'], model_name='RNN', name='layer_8_epoch_5_2017-08-01#12:27:23_0.4106.params')
        # self.model4 = model4 = RNN(embed_mat, opt)
        # self.model4 = load_model(model4, model_dir=opt['model_dir'], model_name='RNN', name='layer_9_epoch_5_2017-08-01#18:06:23_0.4116.params')
        # self.model5 = model5 = RNN(embed_mat, opt)
        # self.model5 = load_model(model5, model_dir=opt['model_dir'], model_name='RNN', name='layer_10_epoch_5_2017-08-01#23:32:47_0.4113.params')
        
        #self.model1 = model1 = FastText(embed_mat, opt)
        #self.model1 = load_model(model1, model_dir=opt['model_dir'], model_name='FastText', name="layer_8_epoch_5_2017-08-03#01:38:41_0.4087.params")
        #self.model2 = model2 = FastText(embed_mat, opt)
        #self.model2 = load_model(model2, model_dir=opt['model_dir'], model_name='FastText', name="layer_9_epoch_5_2017-08-03#05:50:32_0.4086.params")
        #self.model3 = model3 = FastText(embed_mat, opt)
        #self.model3 = load_model(model3, model_dir=opt['model_dir'], model_name='FastText', name='layer_10_epoch_5_2017-08-03#09:52:18_0.4091.params')
        
        # self.model1 = model1 = TextCNN(embed_mat, opt)
        # self.model1 = load_model(model1, model_dir=opt['model_dir'], model_name='TextCNN', name='layer_1_epoch_5_2017-07-26#06:20:00_0.4111.params')
        # self.model2 = model2 = TextCNN(embed_mat, opt)
        # self.model2 = load_model(model2, model_dir=opt['model_dir'], model_name='TextCNN', name='layer_2_epoch_5_2017-08-02#11:25:22_0.4095.params')
        # self.model3 = model3 = TextCNN(embed_mat, opt)
        # self.model3 = load_model(model3, model_dir=opt['model_dir'], model_name='TextCNN', name='layer_3_epoch_5_2017-08-03#01:19:49_0.4095.params')

    def forward(self, x, y):
        logit1 = self.model1(x, y)
        return logit
        #logit2 = self.model2(x, y)
        #logit3 = self.model3(x, y)
        #logit4 = self.model4(x, y)
        #logit5 = self.model5(x, y)
        #logit6 = self.model6(x, y)
        #logit7 = self.model7(x, y)
        logit = torch.cat((logit1.unsqueeze(2), logit2.unsqueeze(2)), 2)
        logit = torch.cat((logit, logit3.unsqueeze(2)), 2)
        #logit = torch.cat((logit, logit4.unsqueeze(2)), 2)
        #logit = torch.cat((logit, logit5.unsqueeze(2)), 2)
        #logit = torch.cat((logit, logit6.unsqueeze(2)), 2)
        #logit = torch.cat((logit, logit7.unsqueeze(2)), 2)
        return logit.sum(2).squeeze(2)