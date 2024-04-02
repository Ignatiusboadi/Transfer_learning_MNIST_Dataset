from torch import nn, optim
from torch.nn import functional as F

import matplotlib.pyplot as plt
import torch


class NeuralNetwork(nn.Module):
    def __init__(self, in_features, hidden_layer_size, out_features):
        super().__init__()
        self.in_features = in_features
        self.hidden_layer_size = hidden_layer_size
        self.out_features = out_features
        self.relu = nn.ReLU()
        self.layer1 = nn.Linear(self.in_features, self.hidden_layer_size)
        self.layer2 = nn.Linear(self.hidden_layer_size, self.out_features)

    def forward(self, x):
        layer1_linear_output = self.layer1(x)
        layer1_activations = self.relu(layer1_linear_output)
        layer2_linear_output = self.layer2(layer1_activations)
        return F.softmax(layer2_linear_output, dim=1)

    def fit(self, X_train, y_train, X_test, y_test, n_epochs, criterion, optimizer):
        losses = []
        for epoch in range(n_epochs):
            y_pred = self.forward(X_train)
            loss = criterion(y_pred, y_train)
            losses.append(loss)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if  (epoch + 1) % 200 == 0:
                print(f'Epoch {epoch + 1} out of {n_epochs}, loss= {loss.item():.4f}')

        with torch.no_grad():
            y_pred = self.forward(X_test)
            y_pred_cls = torch.argmax(y_pred, axis=1)
            accuracy = torch.mean((y_pred_cls == y_test).float()) * 100
            print(f"Accuracy: {accuracy.item()}")
            plt.plot(losses)
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Losses for each epoch')
            plt.show()
