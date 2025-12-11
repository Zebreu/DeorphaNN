# DeorphaNN
DeorphaNN is a graph neural network that prioritizes peptide agonists for G protein-coupled receptors (GPCRs) by integrating active-state GPCR-peptide predicted structures, interatomic interactions, and deep learning protein representations. 

## Associated paper
DeorphaNN: Virtual screening of GPCR peptide agonists using AlphaFold-predicted active state complexes and deep learning embeddings
https://www.biorxiv.org/content/10.1101/2025.03.19.644234

Dataset available at https://huggingface.co/datasets/lariferg/DeorphaNN/tree/main 

## 1. Data Preparation (Preprocessing)

Before running DeorphaNN, you need to prepare your GPCR-peptide input files. Optional helper notebooks and scripts are provided to guide this process.

### Steps

1. **Prepare GPCR templates**  
   - Use [AlphaFold-multistate](https://github.com/huhlim/alphafold-multistate) to predict the active state of the GPCR.  
   - Trim the top-ranked active state structure according to pLDDT and DeepTMHMM identity.  
     *Notebook:* [template_trim.ipynb](notebooks/template_trim.ipynb)

2. **Generate GPCR-peptide complexes**  
   - Run AlphaFold-Multimer using the trimmed GPCR as a template.  
   - Save the pair representations and relax the predicted structure with the highest peptide pLDDT.  
   - Determine whether the peptide is within 12.5A of the GPCR binding pocket. Peptides outside the GPCR binding pocket can be assigned −∞ and included in the ranking without running DeorphaNN.
     *Notebook:* [minimum_distance.ipynb](notebooks/minimum_distance.ipynb)
     
3. **Process pair representations**  
   - Average pair representations across all models and pool GPCR, peptide, and interaction regions.
   ```
   python process_pair_reps.py --input_dir "/path/to/pair_repr/files" --gpcr_length <GPCR_LENGTH> --gpcr <GPCR_NAME> --peptide <PEPTIDE_NAME>
   ```
   
4. **Compute peptide-GPCR contacts**  
   - Identify contacts using [Arpeggio](https://github.com/harryjubb/arpeggio) with Docker.  
   - Output: One `.contacts` file for each GPCR-peptide pair.  



## Running DeorphaNN

1. **Organize files**  
   ```
   /your/top/level/directory/
   ├── pair_001/
   │   ├── *_gpcr_T.npy
   │   ├── *_pep_T.npy
   │   ├── *_interaction.npy
   │   └── *.contacts
   ├── pair_002/
   │   ├── ...
   ...
   ```
2. **Install dependencies**
```
pip install -r requirements.txt
```
3. **Run batch inference**
```
python deorphann_batch.py --root_dir /your/top/level/directory/
```

## Model Training
If you want to train DeorphaNN from scratch:
   - See: [DeorphaNN_training.ipynb](https://githubtocolab.com/Zebreu/DeorphaNN/blob/main/DeorphaNN_training.ipynb)

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
