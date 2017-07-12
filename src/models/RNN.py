import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class RNN(nn.Module):
    def __init__(self, embed_mat, opt):
        super(RNN, self).__init__()
        self.opt = opt
        
        V = opt['embed_num']
        D = opt['embed_dim']
        embedding = torch.from_numpy(embed_mat)
        C = opt['class_num']
        self.hidden_num = hidden_num = 200
        
        self.embed = nn.Embedding(V, D)
        self.embed.weight.data.copy_(embedding)
        self.rnn = nn.GRU(D, hidden_num, 1)
        self.fc1 = nn.Linear(hidden_num, C)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        x = self.embed(x.long())
        if self.opt['static']:
            x = x.detach()
        
        h0 = Variable(torch.randn(1, batch_size, self.hidden_num))
        if self.opt['cuda']:
            h0 = h0.cuda()
        x = x.transpose(0, 1)
        _, x = self.rnn(x, h0)
        x = x.squeeze(0)

        logit = self.fc1(x)
        return logit