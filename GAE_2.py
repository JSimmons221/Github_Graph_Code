import os
import dgl
import torch.nn.functional
from sklearn.model_selection import train_test_split
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from dgl.dataloading import GraphDataLoader
from dgl.data import CSVDataset
from torch.nn.functional import binary_cross_entropy_with_logits as BCELoss
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ["DGLBACKEND"] = "pytorch"
device = th.device("cuda" if th.cuda.is_available() else "cpu")
# device = th.device("cpu")

save_dir = './result'


def collate(samples):
    for g in samples:
        g.to(torch.device(device))
    bg = dgl.batch(samples)
    return bg


class GAE(nn.Module):
    def __init__(self, in_feats, hidden_GAE):
        super(GAE, self).__init__()
        self.encoder = nn.ModuleList()
        self.batch_norms = nn.ModuleList()  # batch norm layers
        self.reconstructor = nn.ModuleList()
        self.encoder.append(GraphConv(in_feats, hidden_GAE[0], activation=F.relu, allow_zero_in_degree=True))
        self.batch_norms.append(nn.BatchNorm1d(hidden_GAE[0]))  # batch norm layer for first conv layer

        for i in range(1, len(hidden_GAE)):
            self.encoder.append(
                GraphConv(hidden_GAE[i - 1], hidden_GAE[i], activation=F.relu, allow_zero_in_degree=True))
            self.batch_norms.append(nn.BatchNorm1d(hidden_GAE[i]))  # batch norm layer for each hidden layer
            self.reconstructor.append(GraphConv(hidden_GAE[-i], hidden_GAE[-i-1], activation=F.relu, allow_zero_in_degree=True))

        self.reconstructor.append(GraphConv(hidden_GAE[0], in_feats, activation=lambda x: x, allow_zero_in_degree=True))
        self.decoder = InnerProductDecoder()

    def forward(self, g):
        h = g.ndata['feats']
        for conv, bn in zip(self.encoder, self.batch_norms):  # conv and batch norm
            h = conv(g, h)
            h = bn(h)  # batch normalization
            g.ndata['h'] = h  # latent rep

        adj = self.decoder(h)

        for conv in self.reconstructor:
            h = conv(g, h)
            g.ndata['h'] = h

        return adj, h


class InnerProductDecoder(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super(InnerProductDecoder, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, z):
        z = self.dropout(z)
        adj = th.sigmoid(th.matmul(z, z.t()))
        return adj


class Trainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.model.to(device)
        self.optimizer = th.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.8)  # decay .8 every 10 epochs

    def iteration(self, g, train=True):
        g = dgl.add_reverse_edges(g)
        feats = g.ndata['feats']
        adj = g.adjacency_matrix().to_dense().to(device)
        pos_weight = ((adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())
        adj_rec, feat_rec = self.model.forward(g)

        bce = BCELoss(adj_rec, adj, pos_weight=pos_weight)
        mse = torch.nn.functional.mse_loss(feat_rec.float(), feats.float())

        loss = bce + mse

        if train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
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
