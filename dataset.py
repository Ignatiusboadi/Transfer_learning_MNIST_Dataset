from torchvision import datasets, transforms

import numpy as np
import pandas as pd
import torch

transform = transforms.Compose([transforms.ToTensor(), ])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)

test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)


def even_odd_split(loader):
    for images, labels in loader:
        images_pixels = images
        labels = labels

    images_flattened = images_pixels.reshape(images.shape[0], -1)

    images_data = pd.DataFrame(images_flattened)
    images_data['label'] = pd.Series(labels)

    even_data = images_data.query('label % 2 == 0')
    odd_data = images_data.query('label % 2 != 0')

    return even_data, odd_data


train_even_data, train_odd_data = even_odd_split(train_loader)
test_even_data, test_odd_data = even_odd_split(test_loader)
