from data_utils.dataset import MNISTDataset
import torch
from torch.utils.data import Dataset, DataLoader
from evaluation.metric import predict, compute_score, classification_labels
from models.lenet import LeNet
from models.GoogLeNet import GoogLeNet
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm


training = MNISTDataset('dataset/train-images-idx3-ubyte.gz', 'dataset/train-labels-idx1-ubyte.gz')
train_loader = DataLoader(training, batch_size=32, shuffle=True)

test = MNISTDataset('dataset/t10k-images-idx3-ubyte.gz', 'dataset/t10k-labels-idx1-ubyte.gz')
test_loader = DataLoader(test, batch_size=1, shuffle=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GoogLeNet().to(device)

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)

loss_fn = nn.CrossEntropyLoss()

# Train the model
num_epochs = 20
# total_step = len(train_loader)

list_loss = []
list_accuracy = []

for epoch in range(num_epochs):
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
        
        pbar.set_description(f'Epoch [{epoch+1}/{num_epochs}], loss: {epoch_loss / (i+1):.4f}, accuracy: {epoch_acc / (i+1):.4f}')
        
    epoch_loss /= len(train_loader)
    epoch_acc /= len(train_loader)
    list_loss.append(epoch_loss)
    list_accuracy.append(epoch_acc)
    
    print('Epoch [{}/{}], loss: {:.4f}, accuracy: {:.4f}'.format(epoch+1, num_epochs, epoch_loss, epoch_acc))