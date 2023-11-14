import torch
from torch import nn
from torch.nn import functional as F

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Residual, self).__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.1)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(0.1)                
            )
    def forward(self, x):
        residual = self.residual(x)
        shortcut = self.shortcut(x)
        x = F.relu(residual + shortcut)
        return x

class ResNet18(nn.Module):
    def __init__(self, num_classes = 10):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(Residual, 64, 2, stride=1)
        self.layer2 = self.make_layer(Residual, 128, 2, stride=2)
        self.layer3 = self.make_layer(Residual, 256, 2, stride=2)
        self.layer4 = self.make_layer(Residual, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, channels, stride))
            self.in_channels = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)
