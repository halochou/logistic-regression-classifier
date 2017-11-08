import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from scipy.io import loadmat

TRAIN = 4
TEST = 1

class MatDataset(Dataset):
    def __init__(self, filename, is_train):
        super(MatDataset, self).__init__()
        self.tf = ToTensor()
        m = loadmat(filename)
        n_c1, n_c2 = m['class_1'].shape[1], m['class_2'].shape[1]
        self.dim = m['class_1'].shape[0]
        assert(m['class_1'].shape[0] == m['class_2'].shape[0])
        if is_train:
            n_c1 = int(n_c1 * TRAIN/(TRAIN+TEST))
            n_c2 = int(n_c2 * TRAIN/(TRAIN+TEST))
            self.c1 = m['class_1'][:, :n_c1]
            self.c2 = m['class_2'][:, :n_c2]
            self.n_c1 = n_c1
            self.n_c2 = n_c2
        else:
            n_c1 = int(n_c1 * TEST/(TRAIN+TEST))
            n_c2 = int(n_c2 * TEST/(TRAIN+TEST))
            self.c1 = m['class_1'][:, -1*n_c1:]
            self.c2 = m['class_2'][:, -1*n_c2:]
            self.n_c1 = n_c1
            self.n_c2 = n_c2
        
        labels = np.concatenate([np.zeros(n_c1), np.ones(n_c2)])
        self.labels = torch.LongTensor(labels.astype('int'))

    def __getitem__(self, index):
        if index < self.n_c1:
            label = self.labels[index]
            data = self.c1[:, index]
        else:
            label = self.labels[index - self.n_c1]
            data = self.c2[:, index - self.n_c1]

        return torch.Tensor(data), label

    def __len__(self):
        return self.n_c1 + self.n_c2