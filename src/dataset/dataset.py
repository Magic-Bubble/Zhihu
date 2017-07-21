import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class Dataset:
    def __init__(self, **datas):
        self.test = datas.get('test', False)

        title = datas['title']
        desc = datas['desc']

        folder_num = datas.get('folder_num', 1)
        cv_num = datas.get('cv_num', 1)
        cv = datas.get('cv', True)

        sample_num = title.shape[0]
        sample_every_folder = sample_num / folder_num
        cv_start =  sample_every_folder * (cv_num - 1)
        cv_end = sample_num if cv_num == folder_num else cv_start + sample_every_folder

        if cv is False:
            self.title = title[cv_end:] if cv_start == 0 else (title[0:cv_start] if cv_end == sample_num else np.concatenate([title[0:cv_start], title[cv_end:]]))
            self.desc = desc[cv_end:] if cv_start == 0 else (desc[0:cv_start] if cv_end == sample_num else np.concatenate([desc[0:cv_start], desc[cv_end:]]))
        else:
            self.title = title[cv_start:cv_end]
            self.desc = desc[cv_start:cv_end]

        # train=True, return label
        if self.test is False:
            self.n_classes = datas['class_num']
            label = datas['label']
            if cv is False:
                self.label = label[cv_end:] if cv_start == 0 else (label[0:cv_start] if cv_end == sample_num else np.concatenate([label[0:cv_start], label[cv_end:]]))
            else:
                self.label = label[cv_start:cv_end]

    def __getitem__(self, idx):
        title = torch.from_numpy(self.title[idx])
        desc = torch.from_numpy(self.desc[idx])
        if self.test is False:
            label = torch.zeros(self.n_classes).scatter_(0, torch.from_numpy(self.label[idx]).long(), 1).int()
            return title, desc, label
        return title, desc

    def __len__(self):
        return self.title.shape[0]
