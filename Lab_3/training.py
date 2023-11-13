from data_utils.dataset import MNISTDataset
import torch
from torch.utils.data import Dataset, DataLoader
from evaluation.metric import predict, compute_score, classification_labels
from models.lenet import LeNet
from models.GoogLeNet import GoogLeNet
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from utils.early_stopping import EarlyStopping
# ./Deep-Learning/Lab_3/
#Config
#---------------------#
# train_image_path = 'dataset/train-images-idx3-ubyte.gz'
# train_label_path = 'dataset/train-labels-idx1-ubyte.gz'
# test_image_path = 'dataset/t10k-images-idx3-ubyte.gz'
# test_label_path = 'dataset/t10k-labels-idx1-ubyte.gz'
train_image_path = './Deep-Learning/Lab_3/dataset/train-images-idx3-ubyte.gz'
train_label_path = './Deep-Learning/Lab_3/dataset/train-labels-idx1-ubyte.gz'
test_image_path = './Deep-Learning/Lab_3/dataset/t10k-images-idx3-ubyte.gz'
test_label_path = './Deep-Learning/Lab_3/dataset/t10k-labels-idx1-ubyte.gz'
n_epochs = 20
batch_size_train = 128
batch_size_test = 128
learning_rate = 0.01
patience = 3
momentum = 0.9
#---------------------#
training = MNISTDataset(train_image_path, train_label_path)
test = MNISTDataset(test_image_path, test_label_path)

dev_size = int(len(training) * 0.2)
train_indices, dev_indices = train_test_split(range(len(training)), test_size=dev_size, random_state=42)

train_loader = DataLoader(Subset(training, train_indices), batch_size=batch_size_train, shuffle=True)
test_loader = DataLoader(test, batch_size = batch_size_test, shuffle=True)
dev_loader = DataLoader(Subset(training, dev_indices), batch_size=batch_size_train, shuffle=True)
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = GoogLeNet().to(device)

optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate,  momentum = momentum)

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
print("Accuracy: {:.2f}".format(acc), end=" | ")
print("F1 Score: {:.2f}".format(f1), end=" | ")
print("Precision: {:.2f}".format(precision), end=" | ")
print("Recall: {:.2f}".format(recall))
# For each labels
print(f'{"Class":<5} {"F1-score":<10} {"Accuracy":<10} {"Precision":<10} {"Recall":<10}')
for i in range(len(metrics['f1'])):
  print(f'{i:<5} {metrics["f1"][i].item():<10.4f} {metrics["accuracy"][i].item():<10.4f} {metrics["precision"][i].item():<10.4f} {metrics["recall"][i].item():<10.4f}')

