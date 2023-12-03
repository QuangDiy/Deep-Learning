from torch.utils.data import Dataset
import os, torch
import numpy as np
import gzip
import struct
import idx2numpy

# load compressed MNIST gz files and return numpy arrays
def load_data(filename, label=False):
    with gzip.open(filename) as gz:
        struct.unpack('I', gz.read(4))
        n_items = struct.unpack('>I', gz.read(4))
        if not label:
            n_rows = struct.unpack('>I', gz.read(4))[0]
            n_cols = struct.unpack('>I', gz.read(4))[0]
            res = np.frombuffer(gz.read(n_items[0] * n_rows * n_cols), dtype=np.uint8)
            res = res.reshape(n_items[0], n_rows * n_cols)
        else:
            res = np.frombuffer(gz.read(n_items[0]), dtype=np.uint8)
            res = res.reshape(n_items[0], 1)
    return res

class MNISTDataset(Dataset):
    def __init__(self, path_images, path_labels) -> None:
        super().__init__()
        self.load_dataset(path_images, path_labels)
    
    def load_dataset(self, path_images, path_labels):
        images = load_data(path_images, False) / 255.0
        labels = load_data(path_labels, True).reshape(-1)

        self.data = []
        for images, labels in zip(images, labels):
            self.data.append({
                "images": images.astype('float32'),
                "labels": labels
            })
            
    def __getitem__(self, idx):
        images = self.data[idx]["images"]
        labels = self.data[idx]["labels"]

        return images, torch.tensor(int(labels))    
        
    def __len__(self) -> int:
        return len(self.data)