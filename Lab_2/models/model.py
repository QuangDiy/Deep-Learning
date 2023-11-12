import torch.nn as nn
import torch

class ModelC (nn.Module):
     def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.flatten = nn.Flatten(start_dim=1)
        self.linear1 = nn.Linear(in_features=784, out_features=512)
        self.linear2 =  nn.Linear(in_features=512, out_features=256)
        self.linear3 = nn.Linear(in_features=256, out_features=10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

     def forward(self, x: torch.Tensor):
        x = self.flatten(x)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        x = self.softmax(x)
        return x