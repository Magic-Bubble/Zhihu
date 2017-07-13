import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from TimeDistributed import TimeDistributed

class RNN(nn.Module):
    def __init__(self, embed_mat, opt):
        super(RNN, self).__init__()
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
        
        self.rnn = nn.GRU(512, hidden_num, bidirectional=True, batch_first=True)
        # self.fc1 = nn.Linear(hidden_num*2, C)
        self.fc1 = nn.Linear(hidden_num*2, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, C)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        x = self.embed(x.long())
        if self.opt['static']:
            x = x.detach()
            
        x = F.relu(self.tdbn1(self.td1(x).unsqueeze(1))).squeeze(1)
        
        h0 = Variable(torch.randn(2, batch_size, self.hidden_num))
        if self.opt['cuda']:
            h0 = h0.cuda()
        _, x = self.rnn(x, h0)
        x = x.transpose(0, 1).contiguous().view(batch_size, -1)

        # logit = self.fc1(x)
        x = F.relu(self.bn1(self.fc1(x)))
        logit = self.fc2(x)
        return logit