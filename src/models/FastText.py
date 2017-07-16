import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
        self.fc1 = nn.Linear(512, C)
        
    def forward(self, x, y):
        x = self.embed(x.long())
        if self.opt['static']:
            x = x.detach()
            
        y = self.embed(y.long())
        if self.opt['static']:
            y = y.detach()
            
        x = x.mean(1).squeeze(1)
        y = y.mean(1).squeeze(1)
        
        x = torch.cat((x, y), 1)

        logit = self.fc1(x)
        return logit