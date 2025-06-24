# deorphann_predict.py

import os, glob, subprocess as sp, argparse
import numpy as np, pandas as pd, torch
import numpy_indexed as npi
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, subgraph
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.nn.norm import BatchNorm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def find_file(d, suffix):
    m = glob.glob(os.path.join(d, f"*{suffix}"))
    if len(m)==0: raise FileNotFoundError(f"{suffix} not found in {d}")
    if len(m)>1: raise ValueError(f"Multiple {suffix} in {d}")
    return m[0]

class DeorphaNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(111)
        self.norm = BatchNorm(128)
        self.conv1 = GATv2Conv(128, hidden_channels, dropout=0.5, heads=10, concat=False, edge_dim=128)
        self.pool = global_mean_pool
        self.lin = Linear(hidden_channels, 2)
        self.finaldropout = 0.5

    def forward(self, x, edge_index, edge_attr, batch, hidden=False):
        x = self.norm(x)
        x = self.conv1(x, edge_index, edge_attr).relu()
        if hidden: return x
        x = self.pool(x, batch)
        x = F.dropout(x, p=self.finaldropout, training=self.training)
        return self.lin(x)

def load_models(model_dir="pretrained_models"):
    os.makedirs(model_dir, exist_ok=True)
    base = "https://huggingface.co/datasets/lariferg/DeorphaNN/resolve/main/pretrained_models/"
    files = [f"pretrained_{i}.pth" for i in range(1,11)]
    for f in files:
        p = os.path.join(model_dir, f)
        if not os.path.exists(p):
            sp.call(["wget","-q","-c", base+f, "-P", model_dir])
    ms = []
    for path in sorted(glob.glob(os.path.join(model_dir,"*.pth"))):
        w = torch.load(path, map_location='cpu', weights_only=True)
        m = DeorphaNN(w['lin.weight'].shape[1])
        m.load_state_dict(w)
        ms.append(m.to(device).eval())
    return ms


def build_graph(gpcr, pep, edges, bonds):
    xg, xp = gpcr, pep
    x = torch.from_numpy(np.concatenate([xg, xp])).float()
    gl = xg.shape[0]
    bonds['source'] = bonds['bgn'].apply(lambda z: z['auth_seq_id'] if z['auth_asym_id']=="A" else z['auth_seq_id']+gl) -1
    bonds['target'] = bonds['end'].apply(lambda z: z['auth_seq_id'] if z['auth_asym_id']=="A" else z['auth_seq_id']+gl) -1
    bonds = bonds.groupby(['source','target'])['contact'].agg(lambda arr: {b for sub in arr for b in sub}).reset_index()
    s,t = bonds['source'].values, bonds['target'].values
    h = np.vstack([s,t])
    ei = np.arange(gl)
    s2 = npi.remap(h[0], ei, np.arange(len(ei)))
    t2 = npi.remap(h[1], ei, np.arange(len(ei)))
    sw = s2 >= gl; tw = t2 < gl
    ns, nt = s2.copy(), t2.copy()
    ns[sw], nt[tw] = t2[sw], s2[tw]
    nt -= gl
    ea = edges[nt, ns, :]
    pe = np.vstack([np.arange(gl, gl+len(xp)-1), np.arange(gl+1, gl+len(xp))])
    pea = np.ones((pe.shape[1],128))*ea.mean(axis=0)
    ei_full = torch.from_numpy(np.concatenate([h, pe],axis=1)).long()
    ea_full = torch.from_numpy(np.concatenate([ea, pea],axis=0)).float()
    to_keep = torch.tensor([i for i in range(gl, gl + xp.shape[0])])
    nodes, _, _, _ = k_hop_subgraph(to_keep, 1, ei_full, relabel_nodes=True, num_nodes=x.shape[0])
    new_edges, new_edge_attrs = subgraph(nodes, ei_full, ea_full, relabel_nodes=True)
    graph = Data(x=x[nodes], edge_index=new_edges, edge_attr=new_edge_attrs)
    graph.x = graph.x.to(device)
    graph.edge_index = graph.edge_index.to(device)
    graph.edge_attr = graph.edge_attr.to(device)
    graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=device)
    return graph

def predict(g, models):
    all_model_scores = []
    with torch.no_grad():
        for model in models:
            out = model(g.x, g.edge_index, g.edge_attr, g.batch)
            binding_score = out[0, 1].item()
            all_model_scores.append(binding_score)
    return sum(all_model_scores) / len(all_model_scores)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root_dir', required=True)
    args = p.parse_args()

    models = load_models()
    os.makedirs('outputs', exist_ok=True)
    results = []
    for d in sorted(glob.glob(os.path.join(args.root_dir,'*/'))):
        try:
            gp = np.load(find_file(d, '*gpcr_T.npy'))
            pp = np.load(find_file(d, '*pep_T.npy'))
            ee = np.load(find_file(d, '*interaction.npy'))
            bb = pd.read_parquet(find_file(d, '*.parquet'))
            g = build_graph(gp, pp, ee, bb)
            score = predict(g, models)
            name = os.path.basename(os.path.abspath(d))
            print(f"{name}: {score:.4f}")
            results.append((name, score))
        except Exception as e:
            print(f"{d} ERROR: {e}")

    df = pd.DataFrame(results, columns=['pair','score'])
    df.to_csv('DeorphaNN_scores.csv', index=False)

if __name__=='__main__':
    main()
