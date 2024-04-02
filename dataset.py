from torchvision import datasets

import numpy as np
import pandas as pd
import torch


train_dataset = datasets.MNIST(root='./data', train=True, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)

test_dataset = datasets.MNIST(root='./data', train=False, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

