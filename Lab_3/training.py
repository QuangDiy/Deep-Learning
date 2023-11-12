from data_utils.dataset import MNISTDataset
import torch
from torch.utils.data import Dataset, DataLoader
from evaluation.metric import predict, compute_score, classification_labels
from models.lenet import LeNet
from models.GoogLeNet import GoogLeNet
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

# ./Deep-Learning/Lab_3/
#Config
#---------------------#
train_image_path = './Deep-Learning/Lab_3/dataset/train-images-idx3-ubyte.gz'
train_label_path = './Deep-Learning/Lab_3/dataset/train-labels-idx1-ubyte.gz'
test_image_path = './Deep-Learning/Lab_3/dataset/t10k-images-idx3-ubyte.gz'
test_label_path = './Deep-Learning/Lab_3/dataset/t10k-labels-idx1-ubyte.gz'
n_epochs = 20
batch_size_train = 128
batch_size_test = 128
learning_rate = 0.01
#---------------------#


training = MNISTDataset(train_image_path, train_label_path)
test = MNISTDataset(test_image_path, test_label_path)

train_loader = DataLoader(training, batch_size = batch_size_train, shuffle=True)
test_loader = DataLoader(test, batch_size = batch_size_test, shuffle=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = GoogLeNet().to(device)

optimizer = torch.optim.SGD(
    model.parameters(),
    lr = learning_rate
)

loss_fn = nn.CrossEntropyLoss()

# Train the model

list_loss = []
list_accuracy = []

for epoch in range(n_epochs):
    epoch_loss = 0.0
    epoch_acc = 0.0

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (images, labels) in pbar:
        # forward backward, update
        images = images.to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        acc = (torch.max(outputs, dim=1)[1] == labels).sum().item() / labels.size(0)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc
        
        pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}], loss: {epoch_loss / (i+1):.4f}, accuracy: {epoch_acc / (i+1):.4f}')
        
    epoch_loss /= len(train_loader)
    epoch_acc /= len(train_loader)
    list_loss.append(epoch_loss)
    list_accuracy.append(epoch_acc)
    
    print('Epoch [{}/{}], loss: {:.4f}, accuracy: {:.4f}'.format(epoch+1, n_epochs, epoch_loss, epoch_acc))