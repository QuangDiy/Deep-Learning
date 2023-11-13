from torch import nn
import torch
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = 5, padding= 2)
        self.pooling1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size=5)
        self.pooling2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Flatten = nn.Flatten()
        self.Linear1 = nn.Linear(16*5*5, 120)
        self.Linear2 = nn.Linear(120, 84)
        self.Linear3 = nn.Linear(84, 10)        
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pooling1(x)
        x = F.relu(self.conv2(x))
        x = self.pooling2(x)
        x = x.view(x.size()[0],-1)
        x = F.relu(self.Linear1(x))
        x = F.relu(self.Linear2(x))
        x = F.relu(self.Linear3(x))     
        return x


    