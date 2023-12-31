import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

def predict(model, test_loader, device):
    model.eval()
    y_pred = []
    y_true = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    return y_pred, y_true

def compute_score(y_pred: list, y_true: list):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=1)

    return acc, f1, precision, recall

def classification_labels(y_pred: list, y_true: list, num_classes=10):
    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)

    conf_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int)
    for t, p in zip(y_true.view(-1), y_pred.view(-1)):
        conf_matrix[t.long(), p.long()] += 1

    accuracy = torch.diag(conf_matrix) / conf_matrix.sum(1)
    precision = torch.diag(conf_matrix) / conf_matrix.sum(0)
    recall = torch.diag(conf_matrix) / conf_matrix.sum(1)
    f1 = 2 * (precision * recall) / (precision + recall)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
    return metrics

def save_fig(y_pred: list, y_true: list):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')
    plt.ylabel('Labels')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    
    file_name = f"confusion_matrix.png"
    save_path = os.path.join(os.getcwd(), file_name)

    plt.savefig(save_path)


def save_loss_acc_plots(num_epochs, train_loss, train_acc, dev_loss, dev_acc):
    data = {
        "Epochs": list(range(1, num_epochs + 1)),
        "Train Loss": train_loss,
        "Train Accuracy": train_acc,
        "Dev Loss": dev_loss,
        "Dev Accuracy": dev_acc
    }
    df = pd.DataFrame(data)

    sns.set(style="darkgrid")

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x="Epochs", y="Train Loss", label="Train Loss")
    sns.lineplot(data=df, x="Epochs", y="Dev Loss", label="Dev Loss")
    plt.title('Train Loss vs Dev Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    file_name = f"loss_plot.png"
    save_path = os.path.join(os.getcwd(), file_name)
    plt.savefig(save_path)
    plt.close()

    # Biểu đồ Train Accuracy và Dev Accuracy
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x="Epochs", y="Train Accuracy", label="Train Accuracy")
    sns.lineplot(data=df, x="Epochs", y="Dev Accuracy", label="Dev Accuracy")
    plt.title('Train Accuracy vs Dev Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    file_name = f"accuracy_plot.png"
    save_path = os.path.join(os.getcwd(), file_name)
    plt.savefig(save_path)
    plt.close()
