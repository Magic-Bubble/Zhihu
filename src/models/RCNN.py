import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import copy

class RCNN(nn.Module):
    def __init__(self, embed_mat, opt):
        super(RCNN, self).__init__()
        self.opt = opt
        
        self.V = V = opt['embed_num']
        self.D = D = opt['embed_dim']
        embedding = torch.from_numpy(embed_mat)
        C = opt['class_num']
        
        self.embed = nn.Embedding(V, D)
        self.embed.weight.data.copy_(embedding)
        
        self.l = nn.Linear(D, D)
        self.sl = nn.Linear(D, D)
        self.r = nn.Linear(D, D)
        self.sr = nn.Linear(D, D)
        
        self.conv = nn.Conv2d(1, 512, (1, D*3))
        
        self.fc = nn.Linear(512, C)
        
    def get_context_left(self, previous_context_left, previous_embedding):
        return F.relu(self.l(previous_context_left) + self.sl(previous_embedding))
    
    def get_context_right(self, afterward_context_right, afterward_embedding):
        return F.relu(self.r(afterward_context_right) + self.sr(afterward_embedding))
        
    def get_context_embedding(self, sequence_embedding):
        batch_size = sequence_embedding.size(0)
        
        sequence_embedding = [x for x in sequence_embedding.transpose(0, 1)]
        cl_w1 = Variable(torch.randn(batch_size, self.D))
        if self.opt['cuda']:
            cl_w1 = cl_w1.cuda()
        context_left_list = [cl_w1]
        for i, cur_embedding in enumerate(sequence_embedding):
            context_left = self.get_context_left(context_left_list[-1], cur_embedding)
            context_left_list.append(context_left)
        context_left_list.pop()
        
        sequence_embedding_reverse = copy.copy(sequence_embedding)
        sequence_embedding_reverse.reverse()
        cr_wn = Variable(torch.randn(batch_size, self.D))
        if self.opt['cuda']:
            cr_wn = cr_wn.cuda()
        context_right_list = [cr_wn]
        for i, cur_embedding in enumerate(sequence_embedding_reverse):
            context_right = self.get_context_right(context_right_list[-1], cur_embedding)
            context_right_list.append(context_right)
        context_right_list.pop()
        context_right_list.reverse()
        
        context_embedding = [torch.cat((context_left_list[i], cur_embedding, context_right_list[i]), 1) for i, cur_embedding in enumerate(sequence_embedding)]
        return torch.stack(context_embedding).transpose(0, 1)
        
    def forward(self, x):
        x = self.embed(x.long())
        if self.opt['static']:
            x = x.detach()
        
        x = self.get_context_embedding(x)
        x = F.relu(self.conv(x.unsqueeze(1)).squeeze(3))
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        logit = self.fc(x)
        
        return logit