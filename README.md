# PLGR

**PLGR** is a deep learning framework for drug–target binding affinity/interaction prediction. The model integrates molecular graphs, SMILES sequences, and protein sequences through adaptive gating fusion and contrastive ranking objectives.

All benchmark datasets are placed under `./data/`. For feature regeneration, refer to `./preprocessing/` (optional).

## Requirements
- Python 3.11.13
- CUDA 12.8 runtime (installed with `torch==2.8.0+cu128`)
- PyTorch 2.8.0+cu128
- torchvision 0.23.0+cu128
- torchaudio 2.8.0+cu128
- numpy 2.2.6
- pandas 2.3.2
- scikit-learn 1.7.1
- scipy 1.16.1
- numba 0.61.2
- RDKit 2025.3.5
- fair-esm 2.0.0
- transformers 4.56.1

## Dataset Directory Example (BindingDB)
```
data/
  bindingdb_train.csv
  bindingdb_test.csv
  bindingdb/
    compound/
      mol_dict.pkl
      ...  # Molecular feature files
    protein/
      train/<protein_id>.npy
      test/<protein_id>.npy
```
Other datasets (pdb / tdc_dg / davis) follow the same directory structure with filenames adapted per dataset.

## Training and Evaluation (Unified Entry: main.py)
Default data path is `./data`, can be modified with `-d <path>`.

- BindingDB (DTI Classification)  
  `python main.py --objective classification --dataset bindingdb -lr 5e-4 -e 300 -b 128`

- PDB (DTA Regression)  
  `python main.py --objective regression --dataset pdb -lr 1e-4 -e 200 -b 64`

- TDC_DG (DTA Regression)  
  `python main.py --objective regression --dataset tdc_dg -lr 1.5e-4 -e 300 -b 64`

- Davis (DTA Regression)  
  `python main.py --objective regression --dataset davis -lr 2e-4 -e 800 -b 96`

## Optional: Feature Preprocessing Regeneration
To regenerate features from raw data, use scripts in `preprocessing/`:
```bash
# Protein features (example: PDB)
cd preprocessing/protein_pretrain
python protein_dta.py --dataset pdb

# Compound features
cd ..
python compound_pretrain.py --dataset pdb
```
