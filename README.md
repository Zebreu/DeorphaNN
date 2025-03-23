# DeorphaNN
DeorphaNN is a graph neural network that predicts peptide agonists for G protein-coupled receptors (GPCRs) by integrating active-state GPCR-peptide predicted structures, interatomic interactions, and deep learning protein representations. 

## Associated paper
DeorphaNN: Virtual screening of GPCR peptide agonists using AlphaFold-predicted active state complexes and deep learning embeddings
https://www.biorxiv.org/content/10.1101/2025.03.19.644234v1

## Data Preprocessing
For each GPCR-peptide query:
1) Use [AlphaFold-multistate](https://github.com/huhlim/alphafold-multistate) to acquire the predicted active state of the GPCR. Trim the top ranked active state structure according to pLDDT and DeepTMHMM identity (template_trim.py).
2) Run AlphaFold-Multimer on your GPCR-peptide complex, using the trimmed active state GPCR as a template. The pdbs must be relaxed--to relax post hoc, use pdb_relax.py. Process the pair representations using process_pair_reps.py to obtain three sets of embeddings, averaged across all models. 
3) Run Arpeggio on the relaxed GPCR-peptide pdb (top ranked by peptide pLDDT) using arpeggio.py

## References
The *C. elegans* dataset was obtained from [Beets, I *et al.* Cell Reports, 2023](https://www.cell.com/cell-reports/fulltext/S2211-1247(23)01069-0?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS2211124723010690%3Fshowall%3Dtrue).

Arpeggio developed by [Jubb HC *et al.* JMB, 2017](https://www.sciencedirect.com/science/article/pii/S0022283616305332?via%3Dihub)

AlphaFold-multistate available at https://github.com/huhlim/alphafold-multistate

Mirdita M, Schütze K, Moriwaki Y, Heo L, Ovchinnikov S and Steinegger M. ColabFold: Making protein folding accessible to all.
Nature Methods (2022) doi: 10.1038/s41592-022-01488-1

Jumper et al. "Highly accurate protein structure prediction with AlphaFold."
Nature (2021) doi: 10.1038/s41586-021-03819-2

Evans et al. "Protein complex prediction with AlphaFold-Multimer."
biorxiv (2021) doi: 10.1101/2021.10.04.463034v1
