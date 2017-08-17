import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
from TextCNN1 import TextCNN1
from utils import load_model

class Boost_CNN5(nn.Module):
    def __init__(self, embed_mat, opt):
        super(Boost_CNN5, self).__init__()
        
        self.model1 = model1 = TextCNN1(embed_mat, opt)
        self.model1 = load_model(model1, model_dir=opt['model_dir'], model_name='TextCNN1', name="layer_1_finetune_epoch_6_2017-08-13#20:06:08.params")
        self.model2 = model2 = TextCNN1(embed_mat, opt)
        self.model2 = load_model(model2, model_dir=opt['model_dir'], model_name='TextCNN1', name="layer_2_finetune_epoch_6_2017-08-14#00:47:19.params")
        self.model3 = model3 = TextCNN1(embed_mat, opt)
        self.model3 = load_model(model3, model_dir=opt['model_dir'], model_name='TextCNN1', name="layer_3_finetune_epoch_6_2017-08-14#04:07:52.params")
        self.model4 = model4 = TextCNN1(embed_mat, opt)
        self.model4 = load_model(model4, model_dir=opt['model_dir'], model_name='TextCNN1', name="layer_4_finetune_epoch_6_2017-08-14#07:28:16.params")
        self.model5 = model5 = TextCNN1(embed_mat, opt)
        self.model5 = load_model(model5, model_dir=opt['model_dir'], model_name='TextCNN1', name="layer_5_finetune_epoch_6_2017-08-15#15:22:03.params")

    def forward(self, x, y):
        logit1 = self.model1(x, y)
        logit2 = self.model2(x, y)
        logit3 = self.model3(x, y)
        logit4 = self.model4(x, y)
        logit5 = self.model5(x, y)
        return logit1 + logit2 + logit3 + logit4 + logit5