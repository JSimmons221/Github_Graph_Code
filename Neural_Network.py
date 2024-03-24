import os.path
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class LinearLayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(LinearLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, x):
        lin = self.linear(x)
        out = self.activation(lin)
        return out


class LinearRegressor(nn.Module):
    def __init__(self, num_layers, in_feats, hidden_feats):
        super(LinearRegressor, self).__init__()
        layers = [LinearLayer(in_feats, hidden_feats, F.relu)]
        if num_layers == 1:
            layers = [LinearLayer(in_feats, 1, lambda x:x)]
        else:
            for i in range(1, num_layers - 1):
                layers.append(LinearLayer(hidden_feats, hidden_feats, F.relu))
            layers.append(LinearLayer(hidden_feats, 1, lambda x:x))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        t = x
        for module in self.layers:
            t = module(t)
        return t
