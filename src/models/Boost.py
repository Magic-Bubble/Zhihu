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
        self.model1 = load_model(model1, model_dir=opt['model_dir'], model_name='RNN', name="layer_1_epoch_5_2017-07-26#15:15:53_0.4116.params")
        self.model2 = model2 = RNN(embed_mat, opt)
        self.model2 = load_model(model2, model_dir=opt['model_dir'], model_name='RNN', name="layer_2_epoch_5_2017-07-31#01:28:34_0.4114.params")
        self.model3 = model3 = RNN(embed_mat, opt)
        self.model3 = load_model(model3, model_dir=opt['model_dir'], model_name='RNN', name='layer_3_epoch_5_2017-07-31#07:07:41_0.4123.params')
        self.model4 = model4 = RNN(embed_mat, opt)
        self.model4 = load_model(model4, model_dir=opt['model_dir'], model_name='RNN', name='layer_4_epoch_5_2017-07-31#12:53:56_0.4114.params')
        self.model5 = model5 = RNN(embed_mat, opt)
        self.model5 = load_model(model5, model_dir=opt['model_dir'], model_name='RNN', name='layer_5_epoch_5_2017-07-31#18:33:47_0.4114.params')
        
        #self.model1 = model1 = FastText(embed_mat, opt)
        #self.model1 = load_model(model1, model_dir=opt['model_dir'], model_name='FastText', name="layer_1_epoch_5_2017-07-28#20:59:26_0.4097.params")
        #self.model2 = model2 = FastText(embed_mat, opt)
        #self.model2 = load_model(model2, model_dir=opt['model_dir'], model_name='FastText', name="layer_2_epoch_5_2017-07-29#21:36:43_0.4090.params")
        #self.model3 = model3 = FastText(embed_mat, opt)
        #self.model3 = load_model(model3, model_dir=opt['model_dir'], model_name='FastText', name='layer_3_epoch_5_2017-07-29#23:57:53_0.4088.params')
        #self.model4 = model4 = FastText(embed_mat, opt)
        #self.model4 = load_model(model1, model_dir=opt['model_dir'], model_name='FastText', name="layer_4_epoch_5_2017-07-30#03:56:10_0.4087.params")
        #self.model5 = model5 = FastText(embed_mat, opt)
        #self.model5 = load_model(model2, model_dir=opt['model_dir'], model_name='FastText', name="layer_5_epoch_5_2017-07-30#11:45:33_0.4088.params")
        #self.model6 = model6 = FastText(embed_mat, opt)
        #self.model6 = load_model(model3, model_dir=opt['model_dir'], model_name='FastText', name='layer_6_epoch_5_2017-07-30#16:34:55_0.4091.params')
        #self.model7 = model7 = FastText(embed_mat, opt)
        #self.model7 = load_model(model3, model_dir=opt['model_dir'], model_name='FastText', name='layer_7_epoch_5_2017-07-30#21:07:18_0.4088.params')

    def forward(self, x, y):
        logit1 = self.model1(x, y)
        logit2 = self.model2(x, y)
        logit3 = self.model3(x, y)
        logit4 = self.model4(x, y)
        logit5 = self.model5(x, y)
        #logit6 = self.model6(x, y)
        #logit7 = self.model7(x, y)
        logit = torch.cat((logit1.unsqueeze(2), logit2.unsqueeze(2)), 2)
        logit = torch.cat((logit, logit3.unsqueeze(2)), 2)
        logit = torch.cat((logit, logit4.unsqueeze(2)), 2)
        logit = torch.cat((logit, logit5.unsqueeze(2)), 2)
        #logit = torch.cat((logit, logit6.unsqueeze(2)), 2)
        #logit = torch.cat((logit, logit7.unsqueeze(2)), 2)
        return logit.sum(2).squeeze(2)