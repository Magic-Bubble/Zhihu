import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
from TimeDistributed import TimeDistributed

class HAN(nn.Module):
    def __init__(self, embed_mat, opt):
        super(HAN, self).__init__()
        self.opt = opt
        
        V = opt['embed_num']
        D = opt['embed_dim']
        embedding = torch.from_numpy(embed_mat)
        C = opt['class_num']
        dropout = opt['dropout']
        self.hidden_num = hidden_num = 200
        
        self.embed = nn.Embedding(V, D)
        self.embed.weight.data.copy_(embedding)
        
        self.tdfc1 = nn.Linear(D, 512)
        self.td1 = TimeDistributed(self.tdfc1)
        self.tdbn1 = nn.BatchNorm2d(1)
        
        self.tdfc2 = nn.Linear(D, 512)
        self.td2 = TimeDistributed(self.tdfc2)
        self.tdbn2 = nn.BatchNorm2d(1)
        
        self.w_rnn1 = nn.GRU(512, hidden_num, bidirectional=True, batch_first=True)
        self.w_rnn2 = nn.GRU(512, hidden_num, bidirectional=True, batch_first=True)
        self.w_hid_fc1 = nn.Linear(hidden_num*2, hidden_num*2)
        self.w_hid_fc2 = nn.Linear(hidden_num*2, hidden_num*2)
        self.w_atten_fc1 = nn.Linear(hidden_num*2, 1)
        self.w_atten_fc2 = nn.Linear(hidden_num*2, 1)
        
        self.s_rnn = nn.GRU(hidden_num*2, hidden_num, bidirectional=True, batch_first=True)
        self.s_hid_fc = nn.Linear(hidden_num*2, hidden_num*2)
        self.s_atten_fc = nn.Linear(hidden_num*2, 1)
        
        self.fc1 = nn.Linear(hidden_num*2, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, C)
        
    def forward(self, x, y):
        batch_size = x.size(0)
        
        x = self.embed(x.long())
        if self.opt['static']:
            x = x.detach()
        x = F.relu(self.tdbn1(self.td1(x).unsqueeze(1))).squeeze(1)
        
        y = self.embed(y.long())
        if self.opt['static']:
            y = y.detach()
        y = F.relu(self.tdbn2(self.td2(y).unsqueeze(1))).squeeze(1)
        
        h0_1 = Variable(torch.randn(2, batch_size, self.hidden_num))
        if self.opt['cuda']:
            h0_1 = h0_1.cuda()
        o1, _ = self.w_rnn1(x, h0_1)
        x = F.relu(self.w_hid_fc1(o1.contiguous().view(-1, o1.size(2))))
        atten_x = F.softmax(self.w_atten_fc1(x).squeeze(1).view(batch_size, -1)).unsqueeze(2)
        x = x.view(batch_size, -1, x.size(1))
        x = x * atten_x.expand_as(x)
        x = x.sum(1)

        h0_2 = Variable(torch.randn(2, batch_size, self.hidden_num))
        if self.opt['cuda']:
            h0_2 = h0_2.cuda()
        o2, _ = self.w_rnn2(y, h0_2)
        y = F.relu(self.w_hid_fc2(o2.contiguous().view(-1, o2.size(2))))
        atten_y = F.softmax(self.w_atten_fc2(y).squeeze(1).view(batch_size, -1)).unsqueeze(2)
        y = y.view(batch_size, -1, y.size(1))
        y = y * atten_y.expand_as(y)
        y = y.sum(1)
        
        x = torch.cat((x, y), 1)
        h0 = Variable(torch.randn(2, batch_size, self.hidden_num))
        if self.opt['cuda']:
            h0 = h0.cuda()
        o, _ = self.s_rnn(x, h0)
        x = F.relu(self.s_hid_fc(o.contiguous().view(-1, o.size(2))))
        atten = F.softmax(self.s_atten_fc(x).squeeze(1).view(batch_size, -1)).unsqueeze(2)
        x = x.view(batch_size, -1, x.size(1))
        x = x * atten.expand_as(x)
        x = x.sum(1).squeeze(1)
            
        x = F.relu(self.bn1(self.fc1(x)))
        logit = self.fc2(x)
        return logit