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
g = ds[0]
print(g)
g_1 = dgl.add_reverse_edges(g)
print(g_1)

gcn_msg = fn.copy_u(u="h", out="m")
gcn_reduce = fn.sum(msg="m", out="h")


class Net(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(Net, self).__init__()
        self.layer1 = GraphConv(in_feats, h_feats, norm='both', weight=True, bias=True, allow_zero_in_degree=True)
        self.layer2 = GraphConv(h_feats, in_feats, norm='both', weight=True, bias=True, allow_zero_in_degree=True)

    def forward(self, graph, feats):
        x = F.relu(self.layer1(graph, feats))
        x = self.layer2(graph, x)
        return x


net = Net(19, 100)
print(net)

model = Net(g.ndata["feats"].shape[1], 30)
features = g.ndata["feats"]
values = model.forward(g, features)

values_np = values.detach().numpy()
np.set_printoptions(threshold=np.inf)
mean = np.mean(values_np, axis=0)
print(mean)
