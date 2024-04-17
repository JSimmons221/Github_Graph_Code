import os
import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from torch.utils.data import DataLoader
from dgl.dataloading import GraphDataLoader
from dgl.data import CSVDataset
import os

os.environ["DGLBACKEND"] = "pytorch"
device = th.device("cuda" if th.cuda.is_available() else "cpu")
# device = th.device("cpu")

print(device)

ds = CSVDataset('data/1_Data/GraphData')

print("Number of graphs:", len(ds))

class GAE(nn.Module):
    def __init__(self, in_feats, hidden_dims, out_dim):
        super(GAE, self).__init__()
        self.encoder = nn.ModuleList()
        self.encoder.append(GraphConv(in_feats, hidden_dims[0], activation=F.relu, allow_zero_in_degree=True))
        for i in range(1, len(hidden_dims)):
            self.encoder.append(GraphConv(hidden_dims[i-1], hidden_dims[i], activation=F.relu, allow_zero_in_degree=True))
        self.decoder = InnerProductDecoder()
        self.regressor = nn.Linear(hidden_dims[-1], 1)  # regression to a single value

    def forward(self, g, features):
        h = features
        for conv in self.encoder:
            h = conv(g, h)
            g.ndata['h'] = h # latent rep
        h_global = dgl.mean_nodes(g, 'h')
        reconstructed = self.decoder(h)
        pred = self.regressor(h_global)
        return reconstructed, pred

class InnerProductDecoder(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super(InnerProductDecoder, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, z):
        z = self.dropout(z)
        adj = th.sigmoid(th.matmul(z, z.t()))
        return adj

# features = graphs[0].ndata['feats']
# reconstructed, pred = model(graphs[0], features)

def loss_function(reconstructed, adj_label, pred, label):
    loss_recon = F.binary_cross_entropy(reconstructed.view(-1), adj_label.view(-1))
    loss_pred = F.mse_loss(pred, label)
    return loss_recon + loss_pred

class Trainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.model.to(device)
        self.optimizer = th.optim.Adam(model.parameters(), lr=learning_rate)
    
    def train(self, data_loader, epochs=50):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batched_graph, labels in data_loader:
                features = batched_graph.ndata['feats'].to(device)
                labels = labels.to(device).view(-1, 1)  # label shape shape [batch_size, 1]
                
                self.optimizer.zero_grad()
                
                reconstructed, preds = self.model(batched_graph, features)
                
                adj_label = batched_graph.adjacency_matrix().to_dense().to(device)
                
                loss_recon = F.binary_cross_entropy(reconstructed.view(-1), adj_label.view(-1))
                loss_pred = F.mse_loss(preds, labels)  # both [batch_size, 1]
                
                loss = loss_recon + loss_pred
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()

            print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(data_loader)}')

    def save_model(self, path):
        th.save(self.model.state_dict(), path)

graphs = [graph[0].to(device) for graph in ds]  # each tuple's first element is a graph
labels = [graph[1]['target'].float().to(device) for graph in ds]  # second item is target (maintainability index)

model = GAE(graphs[0].ndata['feats'].shape[1], [64, 32], 16).to(device)

data_loader = GraphDataLoader(list(zip(graphs, labels)), batch_size=5, shuffle=True)

model = GAE(in_feats=graphs[0].ndata['feats'].shape[1], hidden_dims=[64, 32], out_dim=1)
trainer = Trainer(model, learning_rate=0.001)

trainer.train(data_loader, epochs=20)

trainer.save_model("model1.pth")
