import os
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import pandas as pd
import torch.optim
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from skorch import NeuralNetClassifier

from Neural_Network import LinearLayer

device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
save_dir = './nn_2_result'


class MLP(nn.Module):
    def __init__(self, num_layers=3, in_feats=10, hidden_feats=30):
        super(MLP, self).__init__()
        layers = [LinearLayer(in_feats, hidden_feats, F.relu)]
        for i in range(1, num_layers - 1):
            layers.append(LinearLayer(hidden_feats, hidden_feats, F.relu))
        layers.append(LinearLayer(hidden_feats, 1, lambda x: x))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for module in self.layers:
            x = module(x)
        return x

class Trainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def iteration(self, x, y, train=True):
        y_hat = self.model.forward(x)
        sub = y_hat.flatten() - y
        square = th.square(sub)
        loss = th.mean(square)
        if train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss.item()

    def save(self, epoch, save_dir):
        output_path = os.path.join(save_dir, 'ep{:02}.pkl'.format(epoch))
        torch.save(self.model.state_dict(), output_path)


def gridSearch(model, X, y):
    model = NeuralNetClassifier(
        model,
        criterion=nn.MSELoss,
        max_epochs=100,
        batch_size=10,
        verbose=False,
        train_split=None
    )

    param_grid = {
        'optimizer__lr': [0.0001, 0.001, 0.01, 0.1],
        'module__num_layers': [1, 2, 3, 5],
        'module__hidden_feats': [5, 10, 15, 20, 30]
    }
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_result = grid.fit(X, y)

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

def train_model(X, y):

    model = MLP(3, X.shape[1], 30)
    model.to(device)

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


def main():
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    df1 = pd.read_csv('D:/1_Data/GraphData/encodings.csv')
    df2 = pd.read_csv('D:/1_Data/repo.csv').drop(
        ['owner', 'repo','cloneURL','Repository','Repository URL', 'Total nloc', 'Avg.NLOC', 'AvgCCN', 'Avg.token', 'Fun Cnt',
         'file threshold cnt', 'Fun Rt', 'nloc Rt', 'Halstead Volume'], axis=1)
    df3 = pd.merge(df1, df2).astype('float32').dropna()

    y = df3['Maintainability Index'].to_numpy()
    X = df3.drop(['Maintainability Index'], axis=1).to_numpy()

    # gridSearch(MLP,X,y)
    train_model(X,y)

main()