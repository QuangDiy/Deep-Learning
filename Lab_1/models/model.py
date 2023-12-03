from torch import nn
import torch

class Model1(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.mlp1 = nn.Linear(in_features=784, out_features=10)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x: torch.Tensor):
        x = self.mlp1(x)
        x = self.softmax(x)
        return x

class Model2(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.MLP_1 = nn.Linear(in_features=784, out_features=512)
        self.MLP_2 = nn.Linear(in_features=512, out_features= 256)
        self.MLP_3 = nn.Linear(in_features=256, out_features=10)
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor):
        # x = self.MLP_1(x)
        # x = self.relu(x)
        # x = self.MLP_2(x)
        # x = self.relu(x)
        x = self.relu(self.MLP_1(x))
        x = self.relu(self.MLP_2(x))
        x = self.MLP_3(x)
        x = self.softmax(x)
        return x

    