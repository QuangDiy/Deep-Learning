import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

def compute_score(model, test_loader, device):
    y_true = []
    y_pred = []
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())
        
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=1)
    
    return acc, f1, precision, recall


def classification_metrics(model, test_loader, device, num_classes=10):
    y_true = []
    y_pred = []
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

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

    print(f'{"Class":<5} {"F1-score":<10} {"Accuracy":<10} {"Precision":<10} {"Recall":<10}')

    for i in range(len(metrics['f1'])):
      print(f'{i:<5} {metrics["f1"][i].item():<10.4f} {metrics["accuracy"][i].item():<10.4f} {metrics["precision"][i].item():<10.4f} {metrics["recall"][i].item():<10.4f}')

    return metrics