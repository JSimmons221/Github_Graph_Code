import os
import dgl
from sklearn.model_selection import train_test_split
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from dgl.dataloading import GraphDataLoader
from dgl.data import CSVDataset
from torch.optim.lr_scheduler import StepLR

os.environ["DGLBACKEND"] = "pytorch"
device = th.device("cuda" if th.cuda.is_available() else "cpu")
# device = th.device("cpu")

print(device)

ds = CSVDataset('data/1_Data/GraphData')

print("Number of graphs:", len(ds))

class GAE(nn.Module):
    def __init__(self, in_feats, hidden_GAE, hidden_RGR):
        super(GAE, self).__init__()
        self.encoder = nn.ModuleList()
        self.batch_norms = nn.ModuleList()  # batch norm layers
        self.encoder.append(GraphConv(in_feats, hidden_GAE[0], activation=F.relu, allow_zero_in_degree=True))
        self.batch_norms.append(nn.BatchNorm1d(hidden_GAE[0]))  # batch norm layer for first conv layer
        
        for i in range(1, len(hidden_GAE)): # rest of the hidden layers in a loop to add so I can add normalization for each layer
            self.encoder.append(GraphConv(hidden_GAE[i-1], hidden_GAE[i], activation=F.relu, allow_zero_in_degree=True))
            self.batch_norms.append(nn.BatchNorm1d(hidden_GAE[i]))  # batch norm layer for each hidden layer
        
        self.decoder = InnerProductDecoder()
        
        # regression layers
        reg_dims = hidden_RGR[:]
        if reg_dims[-1] != 1:  # ensure final output of regression is a single value
            reg_dims.append(1)
        self.regressor_layers = nn.ModuleList()
        in_dim = hidden_GAE[-1]
        for dim in reg_dims:
            self.regressor_layers.append(nn.Linear(in_dim, dim))
            in_dim = dim

    def forward(self, g, features):
        h = features
        for conv, bn in zip(self.encoder, self.batch_norms):  # conv and batch norm
            h = conv(g, h)
            h = bn(h)  # batch normalization
            g.ndata['h'] = h  # latent rep
        
        # global graph rep
        h_global = dgl.mean_nodes(g, 'h')
        
        reconstructed = self.decoder(h)
        
        # regression
        for layer in self.regressor_layers[:-1]:
            h_global = F.relu(layer(h_global))
        pred = self.regressor_layers[-1](h_global)
        
        return reconstructed, pred

class InnerProductDecoder(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super(InnerProductDecoder, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, z):
        z = self.dropout(z)
        adj = th.sigmoid(th.matmul(z, z.t()))
        return adj

def loss_function(reconstructed, adj_label, pred, label):
    loss_recon = F.binary_cross_entropy(reconstructed.view(-1), adj_label.view(-1))
    loss_pred = F.mse_loss(pred, label)
    return loss_recon + loss_pred

class Trainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.model.to(device)
        self.optimizer = th.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.8) # decay .8 every 10 epochs

    def train(self, train_loader, val_loader=None, epochs=50):
        for epoch in range(epochs):
            t_loss, t_acc, t_mape = self.run_epoch(train_loader, train=True)
            train_stats = f'Train Loss: {t_loss:.4f}, Train % Correct: {t_acc:.4f}, Train MAPE: {t_mape:.4f}%'
            
            if validate_while_training and val_loader is not None:
                v_loss, v_acc, v_mape = self.run_epoch(val_loader, train=False)
                val_stats = f'Validation Loss: {v_loss:.4f}, Val % Correct: {v_acc:.4f}, Val MAPE: {v_mape:.4f}%'
            else:
                val_stats = "Validation skipped"

            self.scheduler.step()
            print(f'Epoch {epoch+1}/{epochs}, {train_stats},\n {val_stats}')

    def run_epoch(self, data_loader, train=True):
        total_loss = 0
        total_correct = 0
        total_mape = 0
        total_elements = 0
        if train:
            self.model.train()
        else:
            self.model.eval()
        
        for batched_graph, labels in data_loader:
            batched_graph = batched_graph.to(device)
            labels = labels.to(device).view(-1, 1) # label shape shape [batch_size, 1]
            features = batched_graph.ndata['feats']
            
            if train:
                self.optimizer.zero_grad()
            
            reconstructed, preds = self.model(batched_graph, features)
            adj_label = batched_graph.adjacency_matrix().to_dense().to(device)
            loss = loss_function(reconstructed, adj_label, preds, labels)
            if train:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            correct, total = self.percent_correct(reconstructed, adj_label)
            total_correct += correct
            total_elements += total
            total_mape += self.calculate_mape(preds, labels)

        avg_loss = total_loss / len(data_loader)
        accuracy = (total_correct / total_elements) * 100
        mape = (total_mape / len(data_loader))
        return avg_loss, accuracy, mape

    def percent_correct(self, reconstructed, adj_label):
        predicted = (reconstructed > 0.5).float()
        correct = (predicted == adj_label).float().sum()
        return correct.item(), adj_label.numel()

    def calculate_mape(self, preds, true_values):
        nonzero_mask = true_values != 0 # mask for nonzero values
        if th.sum(nonzero_mask) == 0:
            return 0
        return (th.abs(preds[nonzero_mask] - true_values[nonzero_mask]) / th.abs(true_values[nonzero_mask])).mean().item() * 100

    def save_model(self, path):
        th.save(self.model.state_dict(), path)

graphs_labels = [(g[0], g[1]['target'].float()) for g in ds]

validate_while_training = True # False to save VRAM

train_data = graphs_labels
val_data = None
if validate_while_training:
    train_data, val_data = train_test_split(graphs_labels, test_size=0.2)
    val_loader = GraphDataLoader(val_data, batch_size=5, shuffle=False)
train_loader = GraphDataLoader(train_data, batch_size=5, shuffle=True)


# split data
train_data, val_data = train_test_split(graphs_labels, test_size=0.2)
train_loader = GraphDataLoader(train_data, batch_size=5, shuffle=True)
val_loader = GraphDataLoader(val_data, batch_size=5, shuffle=False)


model = GAE(in_feats=graphs_labels[0][0].ndata['feats'].shape[1], hidden_GAE=[64, 32, 16, 16], hidden_RGR=[16, 8])
trainer = Trainer(model, learning_rate=0.001)
trainer.train(train_loader, val_loader if validate_while_training else None, epochs=50)

trainer.save_model("model1.pth")
