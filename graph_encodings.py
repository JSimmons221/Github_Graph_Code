import numpy as np
import dgl
import pandas as pd
import torch
from gae import GAE

encoder = GAE(19, [16, 14, 10, 8])
encoder.load_state_dict(torch.load('./result/ep19.pkl'))
encodes = []

graphs = dgl.data.CSVDataset('D:/1_Data/GraphData')
i = 0
for g in graphs:
    if i % 50 == 0:
        print(i)

    values = []
    try:
        encoding = encoder.encode(dgl.add_reverse_edges(g))
        encoding_mean = encoding.mean(dim=0)
        values = encoding_mean.tolist()
        values.insert(0, i)
    except:
        print(i)
        print("An error happened somewhere, I will fix this later to be more exact about what happened (maybe)")
        values = [i, None, None, None, None, None, None, None, None]

    encodes.append(values)
    i += 1

df = pd.DataFrame(np.array(encodes), columns=["id", 1, 2, 3, 4, 5, 6, 7, 8])
df.to_csv('D:/1_Data/GraphData/encodings.csv')
