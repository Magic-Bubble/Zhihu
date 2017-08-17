import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
from RNN import RNN
from utils import load_model

class Boost_RNN1_char(nn.Module):
    def __init__(self, embed_mat, opt):
        super(Boost_RNN1_char, self).__init__()

        self.model1 = model1 = RNN(embed_mat, opt)
        self.model1 = load_model(model1, model_dir=opt['model_dir'], model_name='RNN', name="layer_1_finetune_char_epoch_6_2017-08-15#15:27:18")

    def forward(self, x, y):
        logit1 = self.model1(x, y)
        return logit1