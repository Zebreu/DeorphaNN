# This code will average all of the pair representation files in an input directory
# and extract the regions required for DeorphaNN.

import os
import argparse
import numpy as np


def load_pair_repr_files(input_dir):
    pair_repr_files = [
        f for f in os.listdir(input_dir)
        if f.endswith('.npy') and 'pair_repr' in f
    ]
    arrays = []
    for file_name in pair_repr_files:
        file_path = os.path.join(input_dir, file_name)
        try:
            arrays.append(np.load(file_path))
        except Exception as e:
            print(f"Failed to load {file_name}: {e}")
    return arrays


def collapse_T(data):
    X, Y, Z = data.shape
    collapsed = np.zeros((X, Z), dtype=np.float32)
    for i in range(X):
        for z in range(Z):
            row_values = data[i, :, z]
            col_values = data[:, i, z]
            col_values = np.delete(col_values, i)
            collapsed[i, z] = np.mean(np.concatenate([row_values, col_values]))
    return collapsed


def extract_and_save_sections(data, gpcr_length, gpcr, peptide, output_dir):
    gpcr_section = data[:gpcr_length, :gpcr_length, :]
    peptide_section = data[gpcr_length:, gpcr_length:, :]
    pep_gpcr = data[gpcr_length:, :gpcr_length, :]
    gpcr_pep = data[:gpcr_length, gpcr_length:, :]

    gpcr_T = collapse_T(gpcr_section)
    pep_T = collapse_T(peptide_section)
    interaction = (pep_gpcr + np.transpose(gpcr_pep, (1, 0, 2))) / 2.0

    np.save(os.path.join(output_dir, f'{gpcr}_{peptide}_gpcr_T.npy'), gpcr_T)
    np.save(os.path.join(output_dir, f'{gpcr}_{peptide}_pep_T.npy'), pep_T)
    np.save(os.path.join(output_dir, f'{gpcr}_{peptide}_interaction.npy'), interaction)

    print("Saved gpcr_T.npy, pep_T.npy, and interaction.npy")

def main():
    parser = argparse.ArgumentParser(description="Process pair representations for use with DeorphaNN.")
    parser.add_argument('--input_dir', required=True, help='Directory containing *pair_repr* .npy files')
    parser.add_argument('--gpcr_length', type=int, required=True, help='Number of residues in the GPCR')
    parser.add_argument('--gpcr', default='GPCR', help='GPCR name (used in output filenames)')
    parser.add_argument('--peptide', default='peptide', help='Peptide name (used in output filenames)')
    args = parser.parse_args()

    arrays = load_pair_repr_files(args.input_dir)
    if not arrays:
        print("No valid pair representation files found.")
        return

    averaged_data = np.mean(arrays, axis=0).astype(np.float32)
    extract_and_save_sections(averaged_data, args.gpcr_length, args.gpcr, args.peptide, args.input_dir)

    print("Processing complete.")


if __name__ == "__main__":
    main()
