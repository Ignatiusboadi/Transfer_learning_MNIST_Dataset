from dataset import *

from matplotlib import pyplot as plt
from NeuralNetwork import NeuralNetwork
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from SoftmaxRegression import SoftmaxRegression, criterion
from torch import nn

import numpy as np
import seaborn as sns

percentages = [.1, .4, .8]

X_train_odd, X_train_even, X_test_odd, X_test_even, y_train_odd, y_train_even, y_test_odd, y_test_even, n_features_odd, n_features_even = to_tensor()

le_odd = LabelEncoder()
le_odd.fit([1, 3, 5, 7, 9])
le_even = LabelEncoder()
le_even.fit([0, 2, 4, 6, 8])

# torch.tensor(train_odd_data['label'].values, dtype=torch.long)
y_train_odd = torch.tensor(le_odd.transform(y_train_odd))
y_train_even = torch.tensor(le_even.transform(y_train_even))


def plot_confusion_even(model, title, file, test_data=X_test_even):
    y_pred_cls = le_even.inverse_transform(torch.argmax(model.forward(test_data), axis=1))
    conf_matrix = confusion_matrix(y_pred_cls, y_test_even, normalize='pred') * 100
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap="Blues", xticklabels=np.arange(0, 10, 2),
                yticklabels=np.arange(0, 10, 2))
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'{title}(%)')
    plt.savefig(f'{file}.png')
    plt.show()


def plot_confusion_odd(model, title, file, test_data=X_test_odd):
    y_pred_cls = le_odd.inverse_transform(torch.argmax(model.forward(test_data), axis=1))
    conf_matrix = confusion_matrix(y_pred_cls, y_test_odd, normalize='pred') * 100
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap="Blues", xticklabels=np.arange(1, 10, 2), yticklabels=np.arange(1, 10, 2))
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'{title}(%)')
    plt.savefig(f'{file}.png')
    plt.show()


n_classes = 5
n_epochs = 1000
lr = 0.1
hidden_layer = 500

# Task 1
# # Softmax regression on Even numbers

for percentage in percentages:
    perc_samples = int(percentage * n_samples)
    even_model = SoftmaxRegression(n_features_even, n_classes)
    optimizer = torch.optim.Adam(even_model.parameters(), lr=lr)
    even_model.fit(X_train_even, y_train_even, X_test_even, y_test_even, n_epochs, criterion, optimizer, le_even)
    plot_confusion_even(even_model, 'Softmax Regression for MNIST Even dataset', 'even')

# Softmax regression on Odd numbers
for percentage in percentages:
    perc_samples = int(percentage * n_samples)
    odd_model = SoftmaxRegression(n_features_odd, n_classes)
    optimizer = torch.optim.Adam(odd_model.parameters(), lr=lr)
    odd_model.fit(X_train_odd, y_train_odd, X_test_odd, y_test_odd, n_epochs, criterion, optimizer, le_odd)
    plot_confusion_odd(odd_model, 'Softmax Regression for MNIST Odd dataset', 'odd')

# Task 2
# Neural Network on Even numbers
nn_even_model = NeuralNetwork(n_features_even, hidden_layer, n_classes)
optimizer = torch.optim.Adam(nn_even_model.parameters(), lr=lr)
nn_even_model.fit(X_train_even, y_train_even, X_test_even, y_test_even, n_epochs, criterion, optimizer, le_even)
plot_confusion_even(nn_even_model, 'Neural Networks with one hidden layer(ReLU) for MNIST Even dataset', 'nn_even')

# Neural Network on Odd numbers
nn_odd_model = NeuralNetwork(n_features_odd, hidden_layer, n_classes)
optimizer = torch.optim.Adam(nn_odd_model.parameters(), lr=lr)
nn_odd_model.fit(X_train_odd, y_train_odd, X_test_odd, y_test_odd, n_epochs, criterion, optimizer, le_odd)
plot_confusion_odd(nn_odd_model, 'Neural Networks with one hidden layer(ReLU) for MNIST Odd dataset', 'nn_odd')


# Task 3
# Extracting the weights from the first layer of the nn model for odd numbers
layer1_weight = nn_odd_model.layer1.weight.data

# Applying the weights and activation of the nn model on the even numbers data
X_train_trans_even = (nn.ReLU()((layer1_weight @ X_train_even.T))).T
X_test_trans_even = (nn.ReLU()((layer1_weight @ X_test_even.T))).T

n_samples, n_features = X_train_trans_even.shape
for percentage in percentages:
    print(f'Transfer learning with {100*percentage}% of dataset.')
    perc_samples = int(percentage * n_samples)
    trans_even_model = SoftmaxRegression(n_features, n_classes)
    optimizer = torch.optim.Adam(trans_even_model.parameters(), lr=lr)
    trans_even_model.fit(X_train_trans_even[:perc_samples], y_train_even[:perc_samples], X_test_trans_even, y_test_even,
                         n_epochs, criterion, optimizer, le_even)
    plot_confusion_even(trans_even_model, f'Transfer Learning for {100*percentage}% MNIST Even dataset',
                        f'{100*percentage}% trans_even', X_test_trans_even)

# Extracting the weights from the first layer of the nn model for odd numbers
layer1_weight = nn_even_model.layer1.weight.data

# Applying the weights and activation of the nn model on the even numbers data
X_train_trans_odd = nn.ReLU()((X_train_odd @ layer1_weight.T))
X_test_trans_odd = nn.ReLU()((X_test_odd @ layer1_weight.T))


n_samples, n_features = X_train_trans_odd.shape
for percentage in percentages:
    print(f'Transfer learning with {100*percentage}% of Odd dataset.')
    perc_samples = int(percentage * n_samples)
    trans_odd_model = SoftmaxRegression(n_features, n_classes)
    optimizer = torch.optim.Adam(trans_odd_model.parameters(), lr=lr)
    trans_odd_model.fit(X_train_trans_odd[:perc_samples], y_train_odd[:perc_samples], X_test_trans_odd,
                        y_test_odd, n_epochs, criterion, optimizer, le_odd)
    plot_confusion_odd(trans_odd_model, f'Transfer learning for {100*percentage}% MNIST Odd dataset',
                       f'{100*percentage}% trans_odd', X_test_trans_odd)
