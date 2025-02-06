import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt

# define training function
def train(dataloader, model, loss_fn, optimizer, device):
    model.train()
    total_loss = 0

    # iterate through each batch in the dataloader and train model
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch % 5 == 0:
            print(f"Batch {batch + 1}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    print('Average loss:', avg_loss)

    return avg_loss

# define testing function
def test(dataloader, model, device):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            print(pred)

            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # calculate and pring r2 score
    r2 = r2_score(all_targets, all_preds)
    print(f"R-squared: {r2:.4f}")
    return r2

# define regression model
class RegressionModel(nn.Module):
    def __init__(self, input_size):
        super(RegressionModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 2*input_size)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(2*input_size, 2*input_size)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(2*input_size, 1)

    # forward function for model
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        return x
    
# run model
def run_model(epochs, train_loader, test_loader, model, loss_fn, optimizer, device):
    losses = []
    r2 = []

    # train and test for each epoch
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        losses.append(train(train_loader, model, loss_fn, optimizer, device))
        r2.append(test(test_loader, model, device))
    
    # plot loss graph
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.legend()
    plt.show()

    # plot r2 graph
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(r2) + 1), r2, label='R^2')
    plt.xlabel('Epoch')
    plt.ylabel('R^2')
    plt.title('Testing R^2 per Epoch')
    plt.ylim(0,1)
    plt.legend()
    plt.show()