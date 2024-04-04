from torch import nn, optim

import matplotlib.pyplot as plt
import torch


criterion = nn.CrossEntropyLoss()


class SoftmaxRegression(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

    def fit(self, X_train, y_train, X_test, y_test, n_epochs, criterion, optimizer, label_decoder):
        losses = []
        for epoch in range(n_epochs):
            y_pred = self.forward(X_train)
            loss = criterion(y_pred, y_train)
            losses.append(loss)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            if (epoch + 1) % 100 == 0:
                print(f'Epoch {epoch + 1} out of {n_epochs}, loss= {loss.item():.4f}')

        with torch.no_grad():
            y_pred = self.forward(X_test)
            y_pred_cls = torch.tensor(label_decoder.inverse_transform(torch.argmax(y_pred, axis=1)), dtype=torch.long)
            accuracy = torch.mean((y_pred_cls == y_test).float()) * 100
            print(f"Accuracy: {accuracy.item()}")
            plt.plot(losses)
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Losses for each epoch')
            plt.show()
