import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from TimeDistributed import TimeDistributed
import copy

class RCNN(nn.Module):
    def __init__(self, embed_mat, opt):
        super(RCNN, self).__init__()
        self.opt = opt
        
        D = opt['embed_dim']
        if opt['use_char_word'] or opt['use_word_char']:
            V_char = opt['char_embed_num']
            V_word = opt['word_embed_num']
            embedding_char = torch.from_numpy(embed_mat[:V_char])
            embedding_word = torch.from_numpy(embed_mat[:V_word])
            self.embed_char = nn.Embedding(V_char, D)
            self.embed_word = nn.Embedding(V_word, D)
            self.embed_char.weight.data.copy_(embedding_char)
            self.embed_word.weight.data.copy_(embedding_word)
        else:
            V = opt['embed_num']
            embedding = torch.from_numpy(embed_mat)
            self.embed = nn.Embedding(V, D)
            self.embed.weight.data.copy_(embedding)

        C = opt['class_num']
        
        self.tdfc1 = nn.Linear(D, 256)
        #self.tdfc1 = nn.Linear(D, D)
        self.td1 = TimeDistributed(self.tdfc1)
        self.tdbn1 = nn.BatchNorm2d(1)
        
        self.tdfc2 = nn.Linear(D, 256)
        #self.tdfc2 = nn.Linear(D, D)
        self.td2 = TimeDistributed(self.tdfc2)
        self.tdbn2 = nn.BatchNorm2d(1)
        
#         self.l_1 = nn.Linear(D, D)
#         self.sl_1 = nn.Linear(D, D)
#         self.r_1 = nn.Linear(D, D)
#         self.sr_1 = nn.Linear(D, D)
        
#         self.l_2 = nn.Linear(D, D)
#         self.sl_2 = nn.Linear(D, D)
#         self.r_2 = nn.Linear(D, D)
#         self.sr_2 = nn.Linear(D, D)
        # self.lstm1 = nn.LSTM(512, 512, batch_first=True, bidirectional=True)
        # self.lstm2 = nn.LSTM(512, 512, batch_first=True, bidirectional=True)

        self.lstm1 = nn.LSTM(256, 256, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(256, 256, batch_first=True, bidirectional=True)        
        #self.lstm1 = nn.LSTM(D, D, batch_first=True, bidirectional=True)
        #self.lstm2 = nn.LSTM(D, D, batch_first=True, bidirectional=True)

        #self.conv1 = nn.Conv2d(1, 1024, (3, D*3))
        self.conv1 = nn.Conv2d(1, 256, (1, 256+256+D))
        #self.convbn1 = nn.BatchNorm2d(1024)
        self.convbn1 = nn.BatchNorm2d(256)
        #self.conv2 = nn.Conv2d(1, 1024, (3, D*3))
        self.conv2 = nn.Conv2d(1, 256, (1, 256+256+D))
        #self.convbn2 = nn.BatchNorm2d(1024)
        self.convbn2 = nn.BatchNorm2d(256)
        
        #self.fc = nn.Linear(2048, C)
       	self.fc = nn.Linear(512, C)
        
#     def get_context_left(self, previous_context_left, previous_embedding, flag):
#         l = getattr(self, 'l_{}'.format(flag))
#         sl = getattr(self, 'sl_{}'.format(flag))
#         return F.relu(l(previous_context_left) + sl(previous_embedding))
#         # return F.relu(self.l(previous_context_left) + self.sl(previous_embedding))
    
#     def get_context_right(self, afterward_context_right, afterward_embedding, flag):
#         r = getattr(self, 'r_{}'.format(flag))
#         sr = getattr(self, 'sr_{}'.format(flag))
#         return F.relu(r(afterward_context_right) + sr(afterward_embedding))
#         # return F.relu(self.r(afterward_context_right) + self.sr(afterward_embedding))
        
#     def get_context_embedding(self, sequence_embedding, flag):
#         batch_size = sequence_embedding.size(0)
        
#         sequence_embedding = [x for x in sequence_embedding.transpose(0, 1)]
#         cl_w1 = Variable(torch.randn(batch_size, self.D))
#         if self.opt['cuda']:
#             cl_w1 = cl_w1.cuda()
#         context_left_list = [cl_w1]
#         for i, cur_embedding in enumerate(sequence_embedding):
#             context_left = self.get_context_left(context_left_list[-1], cur_embedding, flag)
#             context_left_list.append(context_left)
#         context_left_list.pop()
        
#         sequence_embedding_reverse = copy.copy(sequence_embedding)
#         sequence_embedding_reverse.reverse()
#         cr_wn = Variable(torch.randn(batch_size, self.D))
#         if self.opt['cuda']:
#             cr_wn = cr_wn.cuda()
#         context_right_list = [cr_wn]
#         for i, cur_embedding in enumerate(sequence_embedding_reverse):
#             context_right = self.get_context_right(context_right_list[-1], cur_embedding, flag)
#             context_right_list.append(context_right)
#         context_right_list.pop()
#         context_right_list.reverse()
        
#         context_embedding = [torch.cat((context_left_list[i], cur_embedding, context_right_list[i]), 1) for i, cur_embedding in enumerate(sequence_embedding)]
#         return torch.stack(context_embedding).transpose(0, 1)
        
    def forward(self, x, y):
        batch_size = x.size(0)

        if self.opt['use_char_word']:
            x = self.embed_char(x.long())
            y = self.embed_word(y.long())
        elif self.opt['use_word_char']:
            x = self.embed_word(x.long())
            y = self.embed_char(y.long())
        else:
            x = self.embed(x.long())
            y = self.embed(y.long())
        
        if self.opt['static']:
            x = x.detach()
        x = F.relu(self.tdbn1(self.td1(x).unsqueeze(1))).squeeze(1)
        
        if self.opt['static']:
            y = y.detach()
        y = F.relu(self.tdbn2(self.td2(y).unsqueeze(1))).squeeze(1)
        
        # x = self.get_context_embedding(x, 1)
        h0_1 = Variable(torch.randn(2, batch_size, 512))
        c0_1 = Variable(torch.randn(2, batch_size, 512))
        #h0_1 = Variable(torch.randn(2, batch_size, self.D))
        #c0_1 = Variable(torch.randn(2, batch_size, self.D))
        if self.opt['cuda']:
            h0_1 = h0_1.cuda()
            c0_1 = c0_1.cuda()
        o1, _ = self.lstm1(x, (h0_1, c0_1))
        x = torch.cat((x, o1), 2)
        
        x = F.relu(self.convbn1(self.conv1(x.unsqueeze(1))).squeeze(3))
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        
        # y = self.get_context_embedding(y, 2)
        h0_2 = Variable(torch.randn(2, batch_size, 512))
        c0_2 = Variable(torch.randn(2, batch_size, 512))
        #h0_2 = Variable(torch.randn(2, batch_size, self.D))
        #c0_2 = Variable(torch.randn(2, batch_size, self.D))
        if self.opt['cuda']:
            h0_2 = h0_2.cuda()
            c0_2 = c0_2.cuda()
        o2, _ = self.lstm2(y, (h0_2, c0_2))
        y = torch.cat((y, o2), 2)
        
        y = F.relu(self.convbn2(self.conv2(y.unsqueeze(1))).squeeze(3))
        y = F.max_pool1d(y, y.size(2)).squeeze(2)
        
        x = torch.cat((x, y), 1)
        
        logit = self.fc(x)
        
        return logit
