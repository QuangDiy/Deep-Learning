from data_utils.dataset import MNISTDataset
import torch
from torch.utils.data import Dataset, DataLoader
from evaluation.metric import evaluate_mnist
from models.model import Model1, Model2
import torch.nn as nn
import matplotlib.pyplot  as plt


training = MNISTDataset('dataset/train-images-idx3-ubyte.gz', 'dataset/train-labels-idx1-ubyte.gz')
train_loader = DataLoader(training, batch_size=32, shuffle=True)

test = MNISTDataset('dataset/t10k-images-idx3-ubyte.gz', 'dataset/t10k-labels-idx1-ubyte.gz')
test_loader = DataLoader(test, batch_size=1, shuffle=True)

model = Model1()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)

loss_fn = nn.CrossEntropyLoss()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Train the model
num_epochs = 20
# total_step = len(train_loader)

list_loss = []
list_accuracy = []

for epoch in range(num_epochs):
    epoch_loss = 0.0
    epoch_acc = 0.0

    for i, (images, labels) in enumerate(train_loader):
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
        
        # if (i+1) % 100 == 0:
        #     print ('Epoch [{}/{}], Step [{}/{}], loss: {:.4f}, accuracy: {:4f}' 
        #            .format(epoch+1, num_epochs, i+1, total_step, loss.item(), acc))
    
    epoch_loss /= len(train_loader)
    epoch_acc /= len(train_loader)
    list_loss.append(epoch_loss)
    list_accuracy.append(epoch_acc)
    
    print('Epoch [{}/{}], loss: {:.4f}, accuracy: {:4f}' .format(epoch+1, num_epochs, epoch_loss, epoch_acc))

# Evaluation model

# accuracy, precision, recall, f1_macro = evaluation(model, test_loader)

# print('Accuracy: {:.2f} %, Precision: {:.2f} %, Recall: {:.2f} %, F1_macro: {:.2f} % '.format(accuracy, precision, recall, f1_macro))

# evaluate_mnist = evaluate_mnist(model, test_loader)
# print(evaluate_mnist)

report_each_class, macro_avg = evaluate_mnist(model, test_loader)

class_accuracy = [report_each_class[i]["accuracy"] for i in range(10)]
class_precision = [report_each_class[i]["precision"] for i in range(10)]
class_recall = [report_each_class[i]["recall"] for i in range(10)]
class_f1 = [report_each_class[i]["f1-score"] for i in range(10)]



print("-" * 60)
print("Evaluation Per Class Results")
print("-" * 60)
print("| Class | Accuracy | Precision | Recall |   F1.  |")
print("|-------|----------|-----------|--------|--------|")
for i in range(10):
    print("|   {}   |".format(i), end="")
    print(f"  {class_accuracy[i]:.4f}  " + "|", end="", sep="")
    print(f"  {class_precision[i]:.4f}   " + "|", end="", sep="")
    print(f" {class_recall[i]:.4f} " + "|", end="", sep="")
    print(f" {class_f1[i]:.4f} " + "|", end="", sep="")
    print()
print("-" * 60)
print("Evaluation Overall Results")
print('Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1_macro: {:.4f}'.format(macro_avg["accuracy"], macro_avg["precision"], macro_avg["recall"], macro_avg["f1-score"]))
print("-" * 60)

epochs = range(1,21)
plt.plot(epochs, list_loss, 'g', label='Training loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, list_accuracy, 'g', label='Training accuracy')
plt.title('Training accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

