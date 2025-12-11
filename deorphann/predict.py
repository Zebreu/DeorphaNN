import numpy as np
import torch

def predict_score(graph, models):
    preds = []
    with torch.no_grad():
        for model in models:
            logits = model(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
            preds.append(logits.cpu()[:,1].numpy())
    return float(np.mean(np.vstack(preds), axis=0)[0])


