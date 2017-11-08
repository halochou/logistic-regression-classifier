import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin1 = nn.Linear(dim, 16)
        self.lin2 = nn.Linear(16, 32)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        x = F.log_softmax(x)
        return x