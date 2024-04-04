from torchvision import datasets, transforms

import pandas as pd
import torch

transform = transforms.Compose([transforms.ToTensor(), ])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=False)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)

test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)


def even_odd_split(loader):
    for images, labels in loader:
        images_pixels = images
        labels = labels

    images_flattened = images_pixels.reshape(images_pixels.shape[0], -1)

    images_data = pd.DataFrame(images_flattened)
    images_data['label'] = pd.Series(labels)

    even_data = images_data.query('label % 2 == 0')
    odd_data = images_data.query('label % 2 != 0')

    return even_data, odd_data


def to_tensor():
    train_even_data, train_odd_data = even_odd_split(train_loader)
    test_even_data, test_odd_data = even_odd_split(test_loader)

    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print('mps available')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    X_train_even = torch.tensor(train_even_data.drop(columns=['label']).values, dtype=torch.float32, device=device)/255
    y_train_even = torch.tensor(train_even_data['label'].values, dtype=torch.long, device=device)

    X_test_even = torch.tensor(test_even_data.drop(columns=['label']).values, dtype=torch.float32, device=device)/255
    y_test_even = torch.tensor(test_even_data['label'].values, dtype=torch.long, device=device)

    n_samples_even, n_features_even = X_train_even.shape

    X_train_odd = torch.tensor(train_odd_data.drop(columns=['label']).values, dtype=torch.float32, device=device)/255
    y_train_odd = torch.tensor(train_odd_data['label'].values, dtype=torch.long, device=device)

    X_test_odd = torch.tensor(test_odd_data.drop(columns=['label']).values, dtype=torch.float32)/255
    y_test_odd = torch.tensor(test_odd_data['label'].values, dtype=torch.long)

    n_samples_odd, n_features_odd = X_train_odd.shape

    return (X_train_odd, X_train_even, X_test_odd, X_test_even, y_train_odd, y_train_even, y_test_odd, y_test_even,
            n_features_odd, n_features_even)



