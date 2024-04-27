import os.path

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import pandas as pd
import torch.optim
from sklearn.model_selection import train_test_split
import pyarrow as py
from tqdm import tqdm

from Neural_Network import LinearRegressor
from torch.utils.data import DataLoader, TensorDataset


device = th.device("cuda" if th.cuda.is_available() else "cpu")
save_dir = './nn_result'


class Trainer:
    def __init__(self, model, lr):
        self.model = model.to(device)
        self.optim = th.optim.Adam(model.parameters(), lr=lr)

    def iteration(self, x, y, train=True):
        x, y = x.to(device), y.to(device)
        y_hat = self.model.forward(x)
        sub = y_hat.flatten() - y
        square = th.square(sub)
        loss = th.mean(square)
        if train:
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
        return loss.item()

    def save(self, epoch, save_dir):
        output_path = os.path.join(save_dir, 'ep{:02}.pkl'.format(epoch))
        th.save(self.model.state_dict(), output_path)

    def calculate_mape(self, preds, true_values):
        preds = preds.to(device)
        true_values = true_values.to(device)
        nonzero_mask = true_values != 0
        if th.sum(nonzero_mask) == 0:
            return 0
        return (th.abs(preds[nonzero_mask] - true_values[nonzero_mask]) / th.abs(true_values[nonzero_mask])).mean().item() * 100
    
    def save_model(self, path):
        th.save(self.model.state_dict(), path)


class MLP(nn.Module):
    def __init__(self, dropout_rate, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.norm = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        z = z.to(next(self.parameters()).device)
        z = self.norm(z)
        z = self.fc1(z)
        z = self.relu(z)
        z = self.dropout(z)
        z = self.fc2(z)
        return z

def main():
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    df1 = pd.read_csv('./data/all_data.csv').drop(
        ['Unnamed: 0', 'owner', 'repo', 'Total nloc', 'Avg.NLOC', 'AvgCCN', 'Avg.token', 'Fun Cnt',
         'file threshold cnt', 'Fun Rt', 'nloc Rt', 'Halstead Volume', 'id', 
         'cloneURL', 'Repository', 'Repository URL'], axis=1)
    df2 = df1.dropna()

    y = df2['Maintainability Index'].to_numpy()
    X = df2.drop(['Maintainability Index'], axis=1).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
    train_dataset = TensorDataset(Tensor(X_train), Tensor(y_train))
    test_dataset = TensorDataset(Tensor(X_test), Tensor(y_test))

    dropout_rates = [0.3, 0.4, 0.5]
    hidden_dims = [20, 30, 40]
    learning_rates = [0.1, 0.01, 0.001]
    batch_sizes = [5, 10, 20]

    best_mape = float('inf')
    best_hyperparameters = {}

    for dropout_rate in dropout_rates:
        for hidden_dim in hidden_dims:
            for lr in learning_rates:
                for bs in batch_sizes:
                    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
                    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=True)
                    model = MLP(dropout_rate=dropout_rate, input_dim=19, hidden_dim=hidden_dim, output_dim=1)
                    trainer = Trainer(model, lr)

                    train_losses, test_losses = [], []
                    train_mapes, test_mapes = [], []

                    for epoch in tqdm(range(100)):
                        train_loss = 0
                        model.train()
                        for step, (x, y) in tqdm(enumerate(train_loader)):
                            train_loss += trainer.iteration(x, y)
                        train_loss /= len(train_loader)
                        train_losses.append(train_loss)

                        val_loss = 0
                        val_mape = 0
                        model.eval()
                        for step, (x, y) in tqdm(enumerate(test_loader)):
                            val_loss += trainer.iteration(x, y, train=False)
                            val_mape += trainer.calculate_mape(trainer.model(x), y)
                        val_loss /= len(test_loader)
                        val_mape /= len(test_loader)
                        test_losses.append(val_loss)
                        test_mapes.append(val_mape)

                        print('\nEpoch: {:02} | Train Loss: {:.4f} | Validation Loss: {:.4f} | Validation MAPE: {:.4f}'.format(epoch, train_loss, val_loss, val_mape))

                    avg_mape = sum(test_mapes) / len(test_mapes)

                    if avg_mape < best_mape:
                        best_mape = avg_mape
                        best_hyperparameters = {'dropout_rate': dropout_rate, 'hidden_dim': hidden_dim, 'lr': lr, "batch_size": bs}
                        trainer.save_model("best_mlp_model.pth")

    print("Best MAPE:", best_mape)
    print("Best hyperparameters:", best_hyperparameters)

    # Best MAPE: 10.13765274636015
    # Best hyperparameters: {'dropout_rate': 0.4, 'hidden_dim': 30, 'lr': 0.1, 'batch_size': 20}  

main()