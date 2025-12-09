import os
import glob
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch_geometric
import numpy_indexed as npi
from collections import defaultdict
from torch_geometric.nn import GATv2Conv, global_mean_pool, BatchNorm, Linear
import torch.nn.functional as F





def move_to_cuda(g):
    g.x = g.x.cuda()
    g.edge_index = g.edge_index.cuda()
    g.edge_attr = g.edge_attr.cuda().type(torch.float32)
    return g

def move_to_cpu(g):
    g.x = g.x
    g.edge_index = g.edge_index
    g.edge_attr = g.edge_attr.float()
    g.batch = torch.zeros(g.x.size(0), dtype=torch.long)
    return g

class DeorphaNN(torch.nn.Module):
    def __init__(self, hidden_channels, input_channels=128, gatheads=10, gatdropout=0.5, finaldropout=0.5):
        super(DeorphaNN, self).__init__()
        self.finaldropout = finaldropout
        torch.manual_seed(111)
        #self.norm = GraphNorm(input_channels)
        self.norm = BatchNorm(input_channels)
        #self.norm2 = LayerNorm(hidden_channels)
        #self.conv1 = GCNConv(input_channels, hidden_channels) # edge_dim=1
        self.conv1 = GATv2Conv(input_channels, hidden_channels, dropout=gatdropout, heads=gatheads, concat=False, edge_dim=128)#, edge_dim=128) # edge_dim=1
        #self.conv2 = GATv2Conv(hidden_channels, hidden_channels)
        #self.conv2 = GATv2Conv(hidden_channels, hidden_channels)

        self.pooling = global_mean_pool
        #self.pooling = aggr.MedianAggregation()
        #self.pooling = aggr.MLPAggregation(in_channels=hidden_channels, out_channels=hidden_channels, max_num_elements= hidden_channels, num_layers=2, hidden_channels=128 )

        #self.pooling = aggr.SoftmaxAggregation(learn=True) # pretty good
        #self.pooling = aggr.SetTransformerAggregation(hidden_channels, dropout=0.5, heads=2, concat=False) # pretty good
        #self.pooling = aggr.SoftmaxAggregation(learn=True, channels=hidden_channels)
        #self.pooling = aggr.PowerMeanAggregation(p=0.9, learn=True, channels=hidden_channels)
        #self.pooling = aggr.EquilibriumAggregation(hidden_channels, hidden_channels, num_layers=[64])

        self.lin = Linear(hidden_channels, 2)

    def forward(self, x, edge_index, edge_attr, batch, hidden=False): # edge_attr
        x = self.norm(x)
        x = self.conv1(x, edge_index, edge_attr)
        x = x.relu()
        #x = self.norm2(x)
        #x = self.conv2(x, edge_index)
        #x = x.relu()

        if hidden:
            return x
        x = self.pooling(x, batch)
        x = F.dropout(x, p=self.finaldropout, training=self.training)
        x = self.lin(x)
        return x




import subprocess as sp

def load_models(model_dir="pretrained_models", device="cpu"):
    os.makedirs(model_dir, exist_ok=True)

    base = "https://huggingface.co/datasets/lariferg/DeorphaNN/resolve/main/pretrained_models/"
    files = ["pretrained_0.pth", "pretrained_1.pth", "pretrained_2.pth", "pretrained_3.pth", "pretrained_4.pth", "pretrained_5.pth", "pretrained_6.pth", "pretrained_7.pth", "pretrained_8.pth", "pretrained_9.pth"]
    import requests

    for f in files:
        local_path = os.path.join(model_dir, f)
        if not os.path.exists(local_path):
            url = base + f
            r = requests.get(url)
            with open(local_path, "wb") as out:
                out.write(r.content)


    models = []
    paths = sorted(glob.glob(os.path.join(model_dir, "*.pth")))
    print("Order of model files being loaded:")
    for i, path in enumerate(paths):
        print(i, path)  # <-- print the order here
        w = torch.load(path, map_location="cpu")
        units = w["lin.weight"].shape[1]
        model = DeorphaNN(units)
        model.load_state_dict(w)
        model.eval()
        model.to(device)
        models.append(model)

    return models


models = load_models(device="cpu")
def load_contacts(contact_file, gpcr_len):
    cols = [
        "Atom 1","Atom 2","Clash","Covalent","VdW Clash","VdW","Proximal",
        "Hydrogen Bond","Weak Hydrogen Bond","Halogen Bond","Ionic",
        "Metal Complex","Aromatic","Hydrophobic","Carbonyl",
        "Polar","Weak Polar","Interacting entities"
    ]
    df = pd.read_csv(contact_file, sep="\t", header=None, names=cols)
    df = df[df["Interacting entities"] == "INTER"]
    df = df.drop(columns=['Proximal',"Clash","Covalent","VdW Clash","VdW"])
    df = df[(df[[c for c in df.columns if c not in ["Atom 1","Atom 2","Interacting entities"]]] == 1).any(axis=1)]

    df.insert(0,'Start: chain ID',df['Atom 1'].str.split('/').str[0])
    df.insert(1,'Start: Residue Number',df['Atom 1'].str.split('/').str[1])
    df.insert(2,'End: chain ID',df['Atom 2'].str.split('/').str[0])
    df.insert(3,'End: Residue Number',df['Atom 2'].str.split('/').str[1])
    df = df.drop(columns=['Atom 1','Atom 2'])

    df.loc[df['Start: chain ID']=='B','Start: Residue Number'] = df.loc[df['Start: chain ID']=='B','Start: Residue Number'].astype(int) + gpcr_len
    df.loc[df['End: chain ID']=='B','End: Residue Number'] = df.loc[df['End: chain ID']=='B','End: Residue Number'].astype(int) + gpcr_len
    df = df.drop_duplicates(subset=["Start: Residue Number","End: Residue Number"])

    contacts = list(zip(df["Start: Residue Number"].astype(int), df["End: Residue Number"].astype(int)))
    if contacts:
        arr = np.array(contacts).T
        arr -= 1
    else:
        arr = np.empty((2,0),dtype=int)
    return arr


def build_graph(gpcr_emb, pep_emb, edge_features, h_edge_index):
    gpcr_len = gpcr_emb.shape[0]
    x = np.concatenate([gpcr_emb, pep_emb])
    x = torch.from_numpy(x).float()

    edgeindices = np.arange(gpcr_len)
    sources = npi.remap(h_edge_index[0], edgeindices, np.arange(len(edgeindices)))
    targets = npi.remap(h_edge_index[1], edgeindices, np.arange(len(edgeindices)))

    sourcewherever = np.where(sources >= gpcr_len)[0]
    targetwherever = np.where(targets < gpcr_len)[0]

    newsources = np.array(sources)
    newtargets = np.array(targets)
    newsources[sourcewherever] = targets[sourcewherever]
    newtargets[targetwherever] = sources[targetwherever]
    newtargets -= gpcr_len

    edge_attrs = edge_features[newtargets, newsources, :]

    pep_edge_index = np.vstack([
        np.arange(gpcr_len, gpcr_len + pep_emb.shape[0] - 1),
        np.arange(gpcr_len + 1, gpcr_len + pep_emb.shape[0])
    ])
    pep_edge_attrs = np.ones((pep_edge_index.shape[1], 128)) * edge_attrs.mean(axis=0)

    edge_index = np.concatenate([h_edge_index, pep_edge_index], axis=1)
    edge_index = torch.from_numpy(edge_index).long()
    edge_attrs = torch.from_numpy(np.concatenate([edge_attrs, pep_edge_attrs], axis=0)).float()

    edge_index, edge_attrs = torch_geometric.utils.to_undirected(edge_index, edge_attrs, reduce='mean')
    to_keep = torch.tensor([i for i in range(gpcr_len, len(x))]) #hopping from peptide nodes
    # to_keep = torch.unique(torch.from_numpy(h_edge_index[0])) #hopping from gpcr nodes
       
    nodes, edges, _, _ = torch_geometric.utils.k_hop_subgraph(to_keep, 1, edge_index, relabel_nodes=True, num_nodes=len(x)) #change hop number if needed
    # mask = (nodes >= gpcr_len) | (torch.isin(nodes, to_keep))
    # nodes = nodes[mask]
    edges, new_edge_attrs = torch_geometric.utils.subgraph(nodes, edge_index, edge_attrs, relabel_nodes=True)
    graph = Data(x=x[nodes], edge_index=edges, edge_attr=new_edge_attrs)
    graph = move_to_cpu(graph)
    graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long)
    return graph



def predict(graph):
    preds = []
    with torch.no_grad():
        for model in models:
            logits = model(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
            preds.append(logits.cpu()[:,1].numpy())
    return float(np.mean(np.vstack(preds), axis=0)[0])

import sys

if len(sys.argv) != 2:
    print("usage: python run_batch.py <root_directory>")
    sys.exit(1)

parent_dir = sys.argv[1]
rows = []

for root, dirs, _ in os.walk(parent_dir):
    for folder in sorted(dirs):
        path = os.path.join(root, folder)
        if not os.path.isdir(path):
            continue

        try:
            gpcr_file = glob.glob(os.path.join(path, "*gpcr_T.npy"))[0]
            pep_file = glob.glob(os.path.join(path, "*pep_T.npy"))[0]
            edge_file = glob.glob(os.path.join(path, "*interaction.npy"))[0]
            contact_file = glob.glob(os.path.join(path, "*.contacts"))[0]

            gpcr = np.load(gpcr_file)
            pep = np.load(pep_file)
            edge = np.load(edge_file)
            h_edge_index = load_contacts(contact_file, gpcr.shape[0])
            graph = build_graph(gpcr, pep, edge, h_edge_index)
            score = predict(graph)

            rows.append({"complex": folder, "score": score})
            print(folder, score)

        except Exception as e:
            print(f"Skipping {folder} due to error: {e}")

# write one combined CSV for all folders
output_path = os.path.join(parent_dir, "DeorphaNN_batch_results.csv")
pd.DataFrame(rows).to_csv(output_path, index=False)
print(f"All results written to {output_path}")


