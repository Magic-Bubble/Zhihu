import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, embed_mat, opt):
        super(TextCNN, self).__init__()
        self.opt = opt
        
        V = opt['embed_num']
        D = opt['embed_dim']
        embedding = torch.from_numpy(embed_mat)
        C = opt['class_num']
        Ci = 1
        Co = opt['kernel_num']
        Ks = opt['kernel_sizes']
        dropout = opt['dropout']
        
        self.embed = nn.Embedding(V, D)
        self.embed.weight.data.copy_(embedding)
        # self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.convs2 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(dropout)
        # self.fc1 = nn.Linear(len(Ks)*Co, C)
        self.fc1 = nn.Linear(len(Ks)*Co*2, C)
        
    def forward(self, x, y):
        x = self.embed(x.long())
        if self.opt['static']:
            x = x.detach()
        x = x.unsqueeze(1)
        # x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)

        y = self.embed(y.long())
        if self.opt['static']:
            y = y.detach()
        y = y.unsqueeze(1) 
        # y = [F.relu(conv(y)).squeeze(3) for conv in self.convs]
        y = [F.relu(conv(y)).squeeze(3) for conv in self.convs2]
        y = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in y]
        y = torch.cat(y, 1)
        x = torch.cat((x, y), 1)

        x = self.dropout(x)
        logit = self.fc1(x)
        return logit