import torch
import torch.nn as nn
import torch.nn.functional as F
from TimeDistributed import TimeDistributed

class FastText(nn.Module):
    def __init__(self, embed_mat, opt):
        super(FastText, self).__init__()
        self.opt = opt
        
        V = opt['embed_num']
        D = opt['embed_dim']
        embedding = torch.from_numpy(embed_mat)
        C = opt['class_num']
        
        self.embed = nn.Embedding(V, D)
        self.embed.weight.data.copy_(embedding)

	self.tdfc1 = nn.Linear(D, 512)
        self.td1 = TimeDistributed(self.tdfc1)
        self.tdbn1 = nn.BatchNorm2d(1)
        
        self.tdfc2 = nn.Linear(D, 512)
        self.td2 = TimeDistributed(self.tdfc2)
        self.tdbn2 = nn.BatchNorm2d(1)
        
        self.fc1 = nn.Linear(1024, 512)
	self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, C)
        
    def forward(self, x, y):
        x = self.embed(x.long())
        if self.opt['static']:
            x = x.detach()
        x = F.relu(self.tdbn1(self.td1(x).unsqueeze(1))).squeeze(1)
            
        y = self.embed(y.long())
        if self.opt['static']:
            y = y.detach()
        y = F.relu(self.tdbn2(self.td2(y).unsqueeze(1))).squeeze(1)
            
        x = x.mean(1).squeeze(1)
        y = y.mean(1).squeeze(1)
        
        x = torch.cat((x, y), 1)

        x = F.relu(self.bn1(self.fc1(x)))
	logit = self.fc2(x)
        return logit
