import torch
from typing import List

def collate_fn(items: List[list]) -> torch.Tensor:
    print(len(items['images']))
    images = [item['images'] for item in items]
    labels = [item['labels'] for item in items]

    images = torch.stack(images)
    labels = torch.tensor(labels)

    return images, labels