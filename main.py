import os.path

import dgl
import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits as BCELoss
from gae import GAE
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
save_dir = './result'


def collate(samples):
    for g in samples:
        g.to(torch.device(device))
    bg = dgl.batch(samples)
    return bg


class Trainer:
    def __init__(self, model):
        self.model = model
        self.optim = torch.optim.Adam(model.parameters(), lr=0.01)

    def iteration(self, g, train=True):
        g = dgl.add_reverse_edges(g)
        adj = g.adjacency_matrix().to_dense().to(device)
        pos_weight = ((adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())
        adj_rec = self.model.forward(g)
        loss = BCELoss(adj_rec, adj, pos_weight=pos_weight)
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

    model = GAE(19, [16, 14, 10, 8])
    model.to(device)

    graphs = dgl.data.CSVDataset('D:/1_Data/GraphData')
    train_graphs, val_graphs = train_test_split(graphs, test_size=.2)

    train_loader = DataLoader(train_graphs, batch_size=5, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_graphs, batch_size=5, shuffle=True, collate_fn=collate)
    trainer = Trainer(model)

    train_losses, val_losses = [], []

    print("Training Start")
    for epoch in tqdm(range(20)):
        train_loss = 0
        model.train()
        for bg in tqdm(train_loader):
            train_loss += trainer.iteration(bg)
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        trainer.save(epoch, save_dir)

        val_loss = 0
        model.eval()
        for bg in val_loader:
            val_loss += trainer.iteration(bg, train=False)
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        print('\nEpoch: {:02} | Train Loss: {:.4f} | Validation Loss: {:.4f}'.format(epoch, train_loss, val_loss))


main()
