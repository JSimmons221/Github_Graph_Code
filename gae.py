import os

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import dgl.data
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dgl.nn import GraphConv

ds = dgl.data.CSVDataset('D:/1_Data/GraphData')


class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h', h}


gcn_msg = fn.copy_u(u="h", out="m")
gcn_reduce = fn.sum(msg="m", out="h")


class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.layer = GraphConv(in_feats, out_feats, norm='both', weight=True, bias=True, allow_zero_in_degree=True)
        self.activation = activation

    def forward(self, g, feats):
        x = self.layer(g, feats)
        x = self.activation(x)
        return x


class GAE(nn.Module):
    def __init__(self, in_dim, hidden_dims):
        super(GAE, self).__init__()
        layers = [GCN(in_dim, hidden_dims[0], F.relu)]
        if len(hidden_dims) >= 2:
            for i in range(1, len(hidden_dims)):
                if i != len(hidden_dims) - 1:
                    layers.append(GCN(hidden_dims[i - 1], hidden_dims[i], F.relu))
                else:
                    layers.append(GCN(hidden_dims[i - 1], hidden_dims[i], lambda x: x))
        else:
            layers = [GCN(in_dim, hidden_dims[0], lambda x: x)]
        self.layers = nn.ModuleList(layers)
        self.decoder = InnerProductDecoder(activation=lambda x: x)

    def forward(self, g):
        h = g.ndata['feats']
        for conv in self.layers:
            h = conv(g, h)
        g.ndata['feats'] = h
        adj_rec = self.decoder(h)
        return adj_rec

    def encode(self, g):
        h = g.ndata['feats']
        for conv in self.layers:
            h = conv(g, h)
        return h


class InnerProductDecoder(nn.Module):
    def __init__(self, activation=th.sigmoid, dropout=0.1):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.activation = activation

    def forward(self, z):
        z = F.dropout(z, self.dropout)
        adj = self.activation(th.mm(z, z.t()))
        return adj


