import torch
from torch.utils.data import Dataset, DataLoader
from evaluation.metric import predict, compute_score, classification_labels
import torch.nn as nn
import matplotlib.pyplot  as plt
from torchsummary import summary
from models.ResNet import ResNet18



model = ResNet18() 
model.train()
input_size = (3, 32, 32)  
summary(model, input_size=input_size)