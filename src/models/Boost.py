import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
from FastText import FastText
from RNN import RNN
from TextCNN import TextCNN
from TextCNN1 import TextCNN1
from RCNN import RCNN
from utils import load_model

class Boost(nn.Module):
    def __init__(self, embed_mat, opt):
        super(Boost, self).__init__()
        
        self.model1 = model1 = TextCNN1(embed_mat, opt)
        self.model1 = load_model(model1, model_dir=opt['model_dir'], model_name='TextCNN1', name="layer_1_epoch_5_2017-07-26#06:20:00_0.4111.params")
        self.model2 = model2 = TextCNN1(embed_mat, opt)
        self.model2 = load_model(model2, model_dir=opt['model_dir'], model_name='TextCNN1', name="layer_2_epoch_5_2017-08-02#11:25:22_0.4095.params")
        # self.model3 = model3 = TextCNN1(embed_mat, opt)
        # self.model3 = load_model(model3, model_dir=opt['model_dir'], model_name='TextCNN1', name="layer_3_finetune_epoch_6_2017-08-14#04:07:52.params")
        # self.model4 = model4 = TextCNN1(embed_mat, opt)
        # self.model4 = load_model(model4, model_dir=opt['model_dir'], model_name='TextCNN1', name="layer_4_finetune_epoch_6_2017-08-14#07:28:16.params")
        #self.model5 = model5 = TextCNN(embed_mat, opt)
        #self.model5 = load_model(model5, model_dir=opt['model_dir'], model_name='TextCNN', name="layer_5_shuffle_epoch_5_2017-08-12#19:10:02_0.4102.params")
        # self.model6 = model6 = TextCNN(embed_mat, opt)
        # self.model6 = load_model(model6, model_dir=opt['model_dir'], model_name='TextCNN', name="layer_6_finetune_top1_char_epoch_6_2017-08-13#01:16:15.params")
        # self.model7 = model7 = TextCNN(embed_mat, opt)
        # self.model7 = load_model(model7, model_dir=opt['model_dir'], model_name='TextCNN', name="layer_7_finetune_top1_char_epoch_6_2017-08-13#02:52:58.params")
        # self.model8 = model8 = TextCNN(embed_mat, opt)
        # self.model8 = load_model(model8, model_dir=opt['model_dir'], model_name='TextCNN', name="layer_8_finetune_top1_char_epoch_6_2017-08-13#04:29:34.params")
        # self.model9 = model9 = TextCNN(embed_mat, opt)
        # self.model9 = load_model(model9, model_dir=opt['model_dir'], model_name='TextCNN', name="layer_9_finetune_top1_char_epoch_6_2017-08-13#10:34:04.params")
        # self.model10 = model10 = TextCNN(embed_mat, opt)
        # self.model10 = load_model(model10, model_dir=opt['model_dir'], model_name='TextCNN', name="layer_10_finetune_top1_char_epoch_6_2017-08-13#12:11:21.params")

        # self.model1 = model1 = RNN(embed_mat, opt)
        # self.model1 = load_model(model1, model_dir=opt['model_dir'], model_name='RNN', name="layer_1_epoch_5_2017-07-26#15:15:53_0.4116.params")
        # self.model2 = model2 = RNN(embed_mat, opt)
        # self.model2 = load_model(model2, model_dir=opt['model_dir'], model_name='RNN', name='layer_2_epoch_5_2017-07-31#01:28:34_0.4114.params')
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
        
        # self.model1 = model1 = FastText(embed_mat, opt)
        # self.model1 = load_model(model1, model_dir=opt['model_dir'], model_name='FastText', name="layer_1_epoch_5_2017-07-28#20:59:26_0.4097.params")
        # self.model2 = model2 = FastText(embed_mat, opt)
        # self.model2 = load_model(model2, model_dir=opt['model_dir'], model_name='FastText', name="layer_2_epoch_5_2017-07-29#21:36:43_0.4090.params")
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
        return logit1+logit2
        # logit2 = self.model2(x, y)
        # logit3 = self.model3(x, y)
        # logit4 = self.model4(x, y)
        # return logit1 + logit2 + logit3 + logit4
        #logit5 = self.model5(x, y)
        # logit6 = self.model6(x, y)
        # logit7 = self.model7(x, y)
        # logit8 = self.model8(x, y)
        # logit9 = self.model9(x, y)
        # logit10 = self.model10(x, y)
        # logit = logit1 + logit2 + logit3 + logit4 + logit5# + logit6 + logit7 + logit8 + logit9 + logit10
        # return logit