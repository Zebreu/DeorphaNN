
import numpy as np
import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import Data
import numpy_indexed as npi

def move_to_cpu(g):
    g.edge_attr = g.edge_attr.float()
    g.batch = torch.zeros(g.x.size(0), dtype=torch.long)
    return g

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
    df = df[(df[df.columns.difference(["Atom 1","Atom 2","Interacting entities"])] == 1).any(axis=1)]

    df["Start chain"] = df["Atom 1"].str.split("/").str[0]
    df["Start res"] = df["Atom 1"].str.split("/").str[1].astype(int)
    df["End chain"] = df["Atom 2"].str.split("/").str[0]
    df["End res"] = df["Atom 2"].str.split("/").str[1].astype(int)

    df.loc[df["Start chain"] == "B", "Start res"] += gpcr_len
    df.loc[df["End chain"] == "B", "End res"] += gpcr_len
    df = df.drop_duplicates(subset=["Start res","End res"])

    contacts = df[["Start res","End res"]].to_numpy().T - 1
    if contacts.size == 0:
        return np.empty((2,0), dtype=int)
    return contacts

def build_graph(gpcr_emb, pep_emb, edge_features, h_edge_index):
    gpcr_len = gpcr_emb.shape[0]

    x = torch.tensor(np.concatenate([gpcr_emb, pep_emb]), dtype=torch.float32)

    # map contacts
    edgeindices = np.arange(gpcr_len)
    sources = npi.remap(h_edge_index[0], edgeindices, np.arange(len(edgeindices)))
    targets = npi.remap(h_edge_index[1], edgeindices, np.arange(len(edgeindices)))

    s2 = np.where(sources >= gpcr_len)[0]
    t2 = np.where(targets < gpcr_len)[0]

    ns = np.array(sources)
    nt = np.array(targets)
    ns[s2] = targets[s2]
    nt[t2] = sources[t2]
    nt -= gpcr_len

    edge_attrs = edge_features[nt, ns, :]

    # peptide backbone edges
    pep_edge_index = np.vstack([
        np.arange(gpcr_len, gpcr_len + pep_emb.shape[0] - 1),
        np.arange(gpcr_len + 1, gpcr_len + pep_emb.shape[0])
    ])
    pep_edge_attrs = np.ones((pep_edge_index.shape[1], 128)) * edge_attrs.mean(axis=0)

    edge_index = torch.tensor(
        np.concatenate([h_edge_index, pep_edge_index], axis=1),
        dtype=torch.long
    )
    edge_attrs = torch.tensor(
        np.concatenate([edge_attrs, pep_edge_attrs], axis=0),
        dtype=torch.float32
    )

    edge_index, edge_attrs = torch_geometric.utils.to_undirected(
        edge_index, edge_attrs, reduce="mean"
    )

    to_keep = torch.arange(gpcr_len, len(x))
    nodes, edges, _, _ = torch_geometric.utils.k_hop_subgraph(
        to_keep, 1, edge_index, relabel_nodes=True, num_nodes=len(x)
    )
    edges, new_edge_attrs = torch_geometric.utils.subgraph(
        nodes, edge_index, edge_attrs, relabel_nodes=True
    )

    graph = Data(x=x[nodes], edge_index=edges, edge_attr=new_edge_attrs)
    graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long)

    return move_to_cpu(graph)
