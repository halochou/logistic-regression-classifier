import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import MatDataset
from model import Net

parser = argparse.ArgumentParser(description='GM Project 2 Classifier')
parser.add_argument('dset_file', help='Dataset file.')
args = parser.parse_args()

print(args.dset_file)

train_loader = DataLoader(MatDataset(args.dset_file, is_train=True),
                          batch_size=5, shuffle=True ,num_workers=4)
test_loader = DataLoader(MatDataset(args.dset_file, is_train=False),
                         batch_size=396, shuffle=False,num_workers=4)
net = Net(train_loader.dataset.dim)
#optim = torch.optim.Adam(net.parameters(), lr=1e-4)
optim = torch.optim.SGD(net.parameters(), lr=1e-4)

def train(epoch):
    net.train()
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = Variable(data), Variable(label)
        optim.zero_grad()
        pred = net(data)
        loss = F.nll_loss(pred, label)
        loss.backward()
        optim.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.data[0]), end='\r')
        sys.stdout.flush()

def test():
    net.eval()
    test_loss = 0
    correct = 0
    for data, label in test_loader:
        data, label = Variable(data, volatile=True), Variable(label)
        output = net(data)
        test_loss += F.nll_loss(output, label, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(label.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

for epoch in range(0, 20):
    train(epoch)
test()
