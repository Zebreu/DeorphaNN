import os
import glob
import numpy as np
import pandas as pd

from deorphann.loader import load_models
from deorphann.graph import build_graph, load_contacts
from deorphann.predict import predict_score

import sys

def main():
    if len(sys.argv) != 2:
        print("Usage: python deorphann_batch.py <root_directory>")
        sys.exit(1)

    parent_dir = sys.argv[1]
    models = load_models(device="cpu")

    rows = []

    for folder in sorted(os.listdir(parent_dir)):
        path = os.path.join(parent_dir, folder)
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
            h_edges = load_contacts(contact_file, gpcr.shape[0])
            graph = build_graph(gpcr, pep, edge, h_edges)
            score = predict_score(graph, models)

            rows.append({"complex": folder, "score": score})
            print(folder, score)

        except Exception as e:
            print(f"Skipping {folder} due to error: {e}")

    out = os.path.join(parent_dir, "DeorphaNN_batch_results.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"All results written to {out}")

if __name__ == "__main__":
    main()
