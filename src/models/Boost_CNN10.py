import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
from TextCNN import TextCNN
from utils import load_model

class Boost_CNN10(nn.Module):
    def __init__(self, embed_mat, opt):
        super(Boost_CNN10, self).__init__()
        
        self.model1 = model1 = TextCNN(embed_mat, opt)
        self.model1 = load_model(model1, model_dir=opt['model_dir'], model_name='TextCNN', name="layer_1_finetune_top1_epoch_6_2017-08-10#13:08:07.params")
        self.model2 = model2 = TextCNN(embed_mat, opt)
        self.model2 = load_model(model2, model_dir=opt['model_dir'], model_name='TextCNN', name="layer_2_finetune_top1_epoch_6_2017-08-10#14:02:48.params")
        self.model3 = model3 = TextCNN(embed_mat, opt)
        self.model3 = load_model(model3, model_dir=opt['model_dir'], model_name='TextCNN', name="layer_3_finetune_top1_epoch_6_2017-08-10#15:03:54.params")
        self.model4 = model4 = TextCNN(embed_mat, opt)
        self.model4 = load_model(model4, model_dir=opt['model_dir'], model_name='TextCNN', name="layer_4_finetune_top1_epoch_6_2017-08-10#16:01:04.params")
        self.model5 = model5 = TextCNN(embed_mat, opt)
        self.model5 = load_model(model5, model_dir=opt['model_dir'], model_name='TextCNN', name="layer_5_finetune_top1_epoch_6_2017-08-10#16:59:33.params")
        self.model6 = model6 = TextCNN(embed_mat, opt)
        self.model6 = load_model(model6, model_dir=opt['model_dir'], model_name='TextCNN', name="layer_6_finetune_top1_epoch_6_2017-08-10#17:55:51.params")
        self.model7 = model7 = TextCNN(embed_mat, opt)
        self.model7 = load_model(model7, model_dir=opt['model_dir'], model_name='TextCNN', name="layer_7_finetune_top1_epoch_6_2017-08-10#18:46:34.params")
        self.model8 = model8 = TextCNN(embed_mat, opt)
        self.model8 = load_model(model8, model_dir=opt['model_dir'], model_name='TextCNN', name="layer_8_finetune_top1_epoch_6_2017-08-11#13:11:40.params")
        self.model9 = model9 = TextCNN(embed_mat, opt)
        self.model9 = load_model(model9, model_dir=opt['model_dir'], model_name='TextCNN', name="layer_9_finetune_top1_epoch_6_2017-08-11#14:18:49.params")
        self.model10 = model10 = TextCNN(embed_mat, opt)
        self.model10 = load_model(model10, model_dir=opt['model_dir'], model_name='TextCNN', name="layer_10_finetune_top1_epoch_6_2017-08-11#15:25:56.params")

    def forward(self, x, y):
        logit1 = self.model1(x, y)
        logit2 = self.model2(x, y)
        logit3 = self.model3(x, y)
        logit4 = self.model4(x, y)
        logit5 = self.model5(x, y)
        logit6 = self.model6(x, y)
        logit7 = self.model7(x, y)
        logit8 = self.model8(x, y)
        logit9 = self.model9(x, y)
        logit10 = self.model10(x, y)
        logit = logit1 + logit2 + logit3 + logit4 + logit5 + logit6 + logit7 + logit8 + logit9 + logit10
        return logit