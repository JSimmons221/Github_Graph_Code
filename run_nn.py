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

device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
save_dir = './nn_result'


class Trainer:
    def __init__(self, model, lr):
        self.model = model
        self.optim = torch.optim.Adam(model.parameters(), lr=lr)

    def iteration(self, x, y, train=True):
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
        torch.save(self.model.state_dict(), output_path)


def main():
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = LinearRegressor(3, 18, 30)
    model.to(device)

    df1 = pd.read_csv('D:/1_Data/GraphData/encodings.csv')
    df2 = pd.read_csv('D:/1_Data/GraphData/graph_data.csv')
    df3 = pd.merge(df1, df2).drop(
        ['Unnamed: 0', 'owner', 'repo', 'Total nloc', 'Avg.NLOC', 'AvgCCN', 'Avg.token', 'Fun Cnt',
         'file threshold cnt', 'Fun Rt', 'nloc Rt', 'Halstead Volume'], axis=1)
    df4 = df3.dropna()

    y = df4['Maintainability Index'].to_numpy()
    X = df4.drop(['Maintainability Index'], axis=1).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
    train_dataset = TensorDataset(Tensor(X_train), Tensor(y_train))
    test_dataset = TensorDataset(Tensor(X_test), Tensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=5, shuffle=True)
    trainer = Trainer(model, .1)

    train_losses, test_losses = [], []

    for epoch in tqdm(range(50)):
        train_loss = 0
        model.train()
        for step, (x, y) in tqdm(enumerate(train_loader)):
            train_loss += trainer.iteration(x, y)
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        trainer.save(epoch, save_dir)

        val_loss = 0
        model.eval()
        for step, (x, y) in tqdm(enumerate(test_loader)):
            val_loss += trainer.iteration(x, y, train=False)
        val_loss /= len(test_loader)
        test_losses.append(val_loss)
        print('\nEpoch: {:02} | Train Loss: {:.4f} | Validation Loss: {:.4f}'.format(epoch, train_loss, val_loss))


main()
