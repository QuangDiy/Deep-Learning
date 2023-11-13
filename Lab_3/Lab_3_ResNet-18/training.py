import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from evaluation.metric import predict, compute_score, classification_labels, show_confusion_matrix
from models.ResNet import ResNet18
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from utils.early_stopping import EarlyStopping

#---------------------#
n_epochs = 20
batch_size_train = 128
batch_size_test = 128
learning_rate = 0.01
patience = 3
momentum = 0.9
weight_decay = 0.01
#---------------------#
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_dataset, dev_dataset = train_test_split(train_dataset, test_size=0.2, random_state=42)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=batch_size_train, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = ResNet18().to(device)

optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, weight_decay = weight_decay, momentum = momentum)

loss_fn = nn.CrossEntropyLoss()

early_stopping = EarlyStopping(patience = patience, verbose=True)

list_loss = []
list_accuracy = []
list_dev_loss = []
list_dev_accuracy = []

for epoch in range(n_epochs):
    epoch_loss = 0.0
    epoch_acc = 0.0

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    model.train() 
    for i, (images, labels) in pbar:
        images = images.to(device)
        labels = labels.to(device)

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

    dev_loss = 0.0
    dev_acc = 0.0
    model.eval() 
    with torch.no_grad(): 
        for images, labels in dev_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            acc = (torch.max(outputs, dim=1)[1] == labels).sum().item() / labels.size(0)

            dev_loss += loss.item()
            dev_acc += acc

    dev_loss /= len(dev_loader)
    dev_acc /= len(dev_loader)

    list_dev_loss.append(dev_loss)
    list_dev_accuracy.append(dev_acc)

    print(f"Epoch {epoch+1}/{n_epochs} | Training Loss: {epoch_loss:.6f} | Val Loss: {dev_loss:.6f} | Val Acc: {dev_acc:.6f}")

    early_stopping(dev_loss, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break
#---------------------#
y_pred, y_true = predict(model, test_loader, device)
acc, f1, precision, recall = compute_score(y_pred, y_true)
metrics = classification_labels(y_pred, y_true, num_classes = 10)
# Test set
print("Accuracy: {:.2f} | F1 Score: {:.2f} | Precision: {:.2f} | Recall: {:.2f}".format(acc, f1, precision, recall))
# For each labels
print(f'{"Class":<5} {"F1-score":<10} {"Accuracy":<10} {"Precision":<10} {"Recall":<10}')
for i in range(len(metrics['f1'])):
  print(f'{i:<5} {metrics["f1"][i].item():<10.4f} {metrics["accuracy"][i].item():<10.4f} {metrics["precision"][i].item():<10.4f} {metrics["recall"][i].item():<10.4f}')

# Show confusion_matrix
show_confusion_matrix(y_pred, y_true)

