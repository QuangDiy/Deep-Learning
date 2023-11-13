from data_utils.dataset import MNISTDataset
import torch
from torch.utils.data import Dataset, DataLoader
from evaluation.metric import predict, compute_score, classification_labels
import torch.nn as nn
import matplotlib.pyplot  as plt
from torchsummary import summary
from models.GoogLeNet import GoogLeNet
from models.lenet import LeNet


training = MNISTDataset('dataset/train-images-idx3-ubyte.gz', 'dataset/train-labels-idx1-ubyte.gz')
train_loader = DataLoader(training, batch_size=128, shuffle=True)


dataiter = iter(train_loader)
images, labels = next(dataiter)

print('Kích thước của ảnh trong batch:', images.size())
print('Kích thước của ảnh đầu tiên trong batch:', images[0].size())

# Hiển thị ảnh đầu tiên trong batch
plt.imshow(images[0].numpy().transpose((1, 2, 0)))
plt.axis('off')
plt.show()


model = LeNet() 
model.train()
input_size = (1, 32, 32)  
summary(model, input_size=input_size)