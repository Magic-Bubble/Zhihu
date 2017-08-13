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
        
        self.model1 = model1 = TextCNN(embed_mat, opt)
        self.model1 = load_model(model1, model_dir=opt['model_dir'], model_name='TextCNN', name="layer_1_shuffle_epoch_5_2017-08-11#21:40:50_0.4109.params")
        self.model2 = model2 = TextCNN(embed_mat, opt)
        self.model2 = load_model(model2, model_dir=opt['model_dir'], model_name='TextCNN', name="layer_2_shuffle_epoch_5_2017-08-12#04:12:53_0.4101.params")
        self.model3 = model3 = TextCNN(embed_mat, opt)
        self.model3 = load_model(model3, model_dir=opt['model_dir'], model_name='TextCNN', name="layer_3_shuffle_epoch_5_2017-08-12#09:11:45_0.4099.params")
        self.model4 = model4 = TextCNN(embed_mat, opt)
        self.model4 = load_model(model4, model_dir=opt['model_dir'], model_name='TextCNN', name="layer_4_shuffle_epoch_5_2017-08-12#14:11:10_0.4099.params")

        # self.model1 = model1 = RNN(embed_mat, opt)
        # self.model1 = load_model(model1, model_dir=opt['model_dir'], model_name='RNN', name="layer_1_finetune_epoch_6_2017-08-09#11:55:20.params")
        # self.model2 = model2 = RNN(embed_mat, opt)
        # self.model2 = load_model(model2, model_dir=opt['model_dir'], model_name='RNN', name='layer_2_finetune_epoch_6_2017-08-09#13:26:22.params')
        # self.model3 = model3 = RNN(embed_mat, opt)
        # self.model3 = load_model(model3, model_dir=opt['model_dir'], model_name='RNN', name='layer_3_finetune_epoch_6_2017-08-09#14:58:45.params')
        # self.model4 = model4 = RNN(embed_mat, opt)
        # self.model4 = load_model(model4, model_dir=opt['model_dir'], model_name='RNN', name='layer_4_finetune_epoch_6_2017-08-09#16:35:58.params')
        # self.model5 = model5 = RNN(embed_mat, opt)
        # self.model5 = load_model(model5, model_dir=opt['model_dir'], model_name='RNN', name='layer_5_finetune_epoch_6_2017-08-09#18:12:50.params')
        # self.model6 = model6 = RNN(embed_mat, opt)
        # self.model6 = load_model(model6, model_dir=opt['model_dir'], model_name='RNN', name="layer_6_finetune_epoch_6_2017-08-09#20:49:29.params")
        # self.model7 = model7 = RNN(embed_mat, opt)
        # self.model7 = load_model(model7, model_dir=opt['model_dir'], model_name='RNN', name='layer_7_finetune_epoch_6_2017-08-09#22:03:01.params')
        # self.model8 = model8 = RNN(embed_mat, opt)
        # self.model8 = load_model(model8, model_dir=opt['model_dir'], model_name='RNN', name='layer_8_finetune_epoch_6_2017-08-09#23:16:33.params')
        # self.model9 = model9 = RNN(embed_mat, opt)
        # self.model9 = load_model(model9, model_dir=opt['model_dir'], model_name='RNN', name='layer_9_finetune_epoch_6_2017-08-10#00:30:07.params')
        # self.model10 = model10 = RNN(embed_mat, opt)
        # self.model10 = load_model(model10, model_dir=opt['model_dir'], model_name='RNN', name='layer_10_finetune_epoch_6_2017-08-10#01:46:25.params')
        
        #self.model1 = model1 = FastText(embed_mat, opt)
        #self.model1 = load_model(model1, model_dir=opt['model_dir'], model_name='FastText', name="layer_8_epoch_5_2017-08-03#01:38:41_0.4087.params")
        #self.model2 = model2 = FastText(embed_mat, opt)
        #self.model2 = load_model(model2, model_dir=opt['model_dir'], model_name='FastText', name="layer_9_epoch_5_2017-08-03#05:50:32_0.4086.params")
        #self.model3 = model3 = FastText(embed_mat, opt)
        #self.model3 = load_model(model3, model_dir=opt['model_dir'], model_name='FastText', name='layer_10_epoch_5_2017-08-03#09:52:18_0.4091.params')
        
        # self.model1 = model1 = FastText(embed_mat, opt)
        # self.model1 = load_model(model1, model_dir=opt['model_dir'], model_name='FastText', name='layer_1_finetune_epoch_6_2017-08-10#16:00:21.params')
        # self.model2 = model2 = FastText(embed_mat, opt)
        # self.model2 = load_model(model2, model_dir=opt['model_dir'], model_name='FastText', name='layer_2_finetune_epoch_6_2017-08-10#16:39:12.params')
        # self.model3 = model3 = FastText(embed_mat, opt)
        # self.model3 = load_model(model3, model_dir=opt['model_dir'], model_name='FastText', name='layer_3_finetune_epoch_6_2017-08-10#17:16:48.params')
        # self.model4 = model4 = FastText(embed_mat, opt)
        # self.model4 = load_model(model4, model_dir=opt['model_dir'], model_name='FastText', name='layer_4_finetune_epoch_6_2017-08-10#17:54:23.params')
        # self.model5 = model5 = FastText(embed_mat, opt)
        # self.model5 = load_model(model5, model_dir=opt['model_dir'], model_name='FastText', name='layer_5_finetune_epoch_6_2017-08-10#18:32:28.params')
        # self.model6 = model6 = FastText(embed_mat, opt)
        # self.model6 = load_model(model6, model_dir=opt['model_dir'], model_name='FastText', name="layer_6_finetune_epoch_6_2017-08-10#19:10:03.params")
        # self.model7 = model7 = FastText(embed_mat, opt)
        # self.model7 = load_model(model7, model_dir=opt['model_dir'], model_name='FastText', name='layer_7_finetune_epoch_6_2017-08-10#19:48:08.params')
        # self.model8 = model8 = FastText(embed_mat, opt)
        # self.model8 = load_model(model8, model_dir=opt['model_dir'], model_name='FastText', name='layer_8_finetune_epoch_6_2017-08-10#20:25:37.params')
        # self.model9 = model9 = FastText(embed_mat, opt)
        # self.model9 = load_model(model9, model_dir=opt['model_dir'], model_name='FastText', name='layer_9_finetune_epoch_6_2017-08-10#21:03:05.params')
        # self.model10 = model10 = FastText(embed_mat, opt)
        # self.model10 = load_model(model10, model_dir=opt['model_dir'], model_name='FastText', name='layer_10_finetune_epoch_6_2017-08-10#21:40:42.params')

    def forward(self, x, y):
        logit1 = self.model1(x, y)
        logit2 = self.model2(x, y)
        logit3 = self.model3(x, y)
        logit4 = self.model4(x, y)
        # logit5 = self.model5(x, y)
        # logit6 = self.model6(x, y)
        # logit7 = self.model7(x, y)
        # logit8 = self.model8(x, y)
        # logit9 = self.model9(x, y)
        # logit10 = self.model10(x, y)
        logit = logit1 + logit2 + logit3 + logit4# + logit5 + logit6 + logit7 + logit8 + logit9 + logit10
        return logit