from data_utils.dataset import MNISTDataset
import torch
from torch.utils.data import Dataset, DataLoader
from evaluation.metric import predict, compute_score, classification_labels
import torch.nn as nn
import matplotlib.pyplot  as plt
from torchsummary import summary
from models.GoogLeNet import GoogLeNet


training = MNISTDataset('dataset/train-images-idx3-ubyte.gz', 'dataset/train-labels-idx1-ubyte.gz')
train_loader = DataLoader(training, batch_size=32, shuffle=True)

# Lấy ra một batch dữ liệu
dataiter = iter(train_loader)
images, labels = next(dataiter)

# Kiểm tra kích thước của ảnh
print('Kích thước của ảnh trong batch:', images.size())

# Kiểm tra kích thước của ảnh đầu tiên trong batch
print('Kích thước của ảnh đầu tiên trong batch:', images[0].size())


model = GoogLeNet()  # Thay đổi giá trị input tùy theo số lượng kênh của ảnh đầu vào

model.train()

input_size = (1, 28, 28)  # (channels, height, width) - Thay đổi giá trị channels tùy theo số lượng kênh của ảnh đầu vào

summary(model, input_size=input_size)
