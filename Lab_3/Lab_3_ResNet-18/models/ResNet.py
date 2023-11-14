import torch
from torch import nn
from torch.nn import functional as F

# class Residual(nn.Module): 
#     def __init__(self, input_channels, num_channels, use_1x1conv = False, strides=1):
#         super().__init__()
#         self.conv1 = nn.Conv2d(input_channels, num_channels, 
#                                kernel_size=3, padding=1, stride=strides)
#         self.conv2 = nn.Conv2d(num_channels, num_channels,
#                                kernel_size=3, padding=1)
#         if use_1x1conv:
#             self.conv3 = nn.Conv2d(input_channels, num_channels,
#                                    kernel_size=1, stride=strides)
#         else:
#             self.conv3 = None
#         self.norm1 = nn.BatchNorm2d(num_channels)
#         self.norm2 = nn.BatchNorm2d(num_channels)

#     def forward(self, x):
#         y = F.relu(self.norm1(self.conv1(x)))
#         y = self.norm2(self.conv2(y))
#         if self.conv3:
#             x = self.conv3(x)
#         y += x
#         y = F.relu(y)
#         return y
    
# def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
#     blk = []
#     for i in range(num_residuals):
#         if i == 0 and not first_block:
#             blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
#         else:
#             blk.append(Residual(num_channels, num_channels))
#     return blk

# class ResNet18(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.b1 = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         )
#         self.b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
#         self.b3 = nn.Sequential(*resnet_block(64, 128, 2))
#         self.b4 = nn.Sequential(*resnet_block(128, 256, 2))
#         self.b5 = nn.Sequential(*resnet_block(256, 512, 2))
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.flatten = nn.Flatten()
#         self.linear = nn.Linear(512, 10)

#     def forward(self, X):
#         X = self.b1(X)
#         X = self.b2(X)
#         X = self.b3(X)
#         X = self.b4(X)
#         X = self.b5(X)
#         X = self.avgpool(X)
#         X = self.flatten(X)
#         X = self.linear(X)
#         return X

class Residual(nn.Module):
    def __init__(self, inchannel, outchannel, stride = 1):
        super(Residual, self).__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride),
                nn.BatchNorm2d(outchannel)
            )
    def forward(self, x):
        x = self.residual(x)
        x += self.shortcut(x)
        x = F.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, Residual, num_classes = 10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(Residual, 64, 2, stride=1)
        self.layer2 = self.make_layer(Residual, 128, 2, stride=2)
        self.layer3 = self.make_layer(Residual, 256, 2, stride=2)
        self.layer4 = self.make_layer(Residual, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_block, stride):
        strides =[stride] + [1] * (num_block - 1) # strides[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
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
        return x

def ResNet18():
    return ResNet(Residual)      