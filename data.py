import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import pickle
import os
from preprocessing.compound import *
from rdkit import Chem
from smiles_fixer import SMILESFixer
class CPIDataset(Dataset):
    def __init__(self, file_path, compound_data_path, protein_data_path):
        print(f"Loading dataset from {file_path}...")
        self.raw_data = pd.read_csv(file_path)
        smiles_fixer = SMILESFixer()
        valid_indices = []
        fixed_smiles = []
        fixed_count = 0
        for idx, smiles in enumerate(self.raw_data['compound_iso_smiles'].values):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    valid_indices.append(idx)
                    fixed_smiles.append(smiles)
                else:
                    fixed = smiles_fixer.fix_smiles(smiles)
                    if fixed and smiles_fixer.validate_smiles(fixed):
                        valid_indices.append(idx)
                        fixed_smiles.append(fixed)
                        fixed_count += 1
                        print(f"Fixed SMILES at index {idx}: {smiles[:50]}... -> {fixed[:50]}...")
                    else:
                        print(f"Warning: Cannot fix SMILES at index {idx}: {smiles[:100]}...")
            except Exception as e:
                fixed = smiles_fixer.fix_smiles(smiles)
                if fixed and smiles_fixer.validate_smiles(fixed):
                    valid_indices.append(idx)
                    fixed_smiles.append(fixed)
                    fixed_count += 1
                    print(f"Fixed problematic SMILES at index {idx}: {smiles[:50]}... -> {fixed[:50]}...")
                else:
                    print(f"Warning: Error parsing SMILES at index {idx}: {smiles[:100]}..., Error: {e}")
        original_count = len(self.raw_data)
        self.raw_data = self.raw_data.iloc[valid_indices].reset_index(drop=True)
        self.raw_data['compound_iso_smiles'] = fixed_smiles
        print(f"Dataset processing complete:")
        print(f"  Original samples: {original_count}")
        print(f"  Valid samples: {len(valid_indices)}")
        print(f"  Fixed SMILES: {fixed_count}")
        print(f"  Success rate: {len(valid_indices)/original_count*100:.1f}%")
        self.smiles_values = self.raw_data['compound_iso_smiles'].values
        self.sequence_values = self.raw_data['target_sequence'].values
        self.compound_data_path =  compound_data_path
        self.protein_data_path = protein_data_path
        if 'label' in self.raw_data.columns:
            self.label_values = self.raw_data['label'].values
        elif 'affinity' in self.raw_data.columns:
            self.label_values = self.raw_data['affinity'].values
        else:
            raise ValueError("Dataset must contain either 'label' or 'affinity' column")
        self.prot_id = self.raw_data['target_id'].values
        print("Loading compound embeddings...")
        with open(f'{self.compound_data_path}/mol_dict.pkl', 'rb') as file:
            self.loaded_dict = pickle.load(file)
        print("Preloading protein embeddings...")
        self.protein_embeddings = {}
        unique_prot_ids = set(self.prot_id)
        for prot_id in unique_prot_ids:
            try:
                embedding_path = f'{self.protein_data_path}/{str(prot_id)}.npy'
                if os.path.exists(embedding_path):
                    self.protein_embeddings[prot_id] = np.load(embedding_path).squeeze()
                else:
                    print(f"Warning: Protein embedding file not found: {embedding_path}")
            except Exception as e:
                print(f"Error loading protein embedding for {prot_id}: {e}")
        print(f"Loaded {len(self.protein_embeddings)} protein embeddings")
        print("Precomputing compound features...")
        self.compound_features = {}
        self.compound_adj_matrices = {}
        unique_smiles = set(self.smiles_values)
        for i, smiles in enumerate(unique_smiles):
            if i % 100 == 0:
                print(f"Processing compound {i+1}/{len(unique_smiles)}")
            try:
                features, adj = get_mol_features(smiles)
                self.compound_features[smiles] = features
                self.compound_adj_matrices[smiles] = adj
            except Exception as e:
                print(f"Error processing SMILES {smiles}: {e}")
        print(f"Precomputed features for {len(self.compound_features)} compounds")
    def __len__(self):
        return len(self.raw_data)
    def __getitem__(self, idx):
        smiles = self.smiles_values[idx]
        label = self.label_values[idx]
        prot_id = self.prot_id[idx]
        try:
            if smiles not in self.compound_features:
                raise ValueError(f"Compound features not found for SMILES: {smiles}")
            compound_node_features = self.compound_features[smiles]
            compound_adj_matrix = self.compound_adj_matrices[smiles]
            if smiles not in self.loaded_dict:
                raise ValueError(f"Compound embedding not found for SMILES: {smiles}")
            smiles_embedding = self.loaded_dict[smiles]
            if prot_id not in self.protein_embeddings:
                raise ValueError(f"Protein embedding not found for protein ID: {prot_id}")
            protein_seq_embedding = self.protein_embeddings[prot_id]
            return {
                'COMPOUND_NODE_FEAT': compound_node_features,
                'COMPOUND_ADJ': compound_adj_matrix,
                'COMPOUND_EMBEDDING': smiles_embedding,
                'PROTEIN_EMBEDDING': protein_seq_embedding,
                'LABEL': label,
            }
        except Exception as e:
            print(f"Error processing sample at index {idx} with SMILES '{smiles}', protein ID '{prot_id}': {e}")
            raise RuntimeError(f"Failed to process sample at index {idx}: {e}")
    def collate_fn(self, batch):
        batch_size = len(batch)
        compound_node_nums = [item['COMPOUND_NODE_FEAT'].shape[0] for item in batch]
        compound_smiles_nums = [item['COMPOUND_EMBEDDING'].shape[0] for item in batch]
        protein_node_nums = [item['PROTEIN_EMBEDDING'].shape[0] for item in batch]
        max_compound_len = max(compound_node_nums)
        max_protein_len = max(protein_node_nums)
        max_smiles_len = max(compound_smiles_nums)
        compound_node_features = torch.zeros(
            (batch_size, max_compound_len, batch[0]['COMPOUND_NODE_FEAT'].shape[1]),
            pin_memory=True
        )
        compound_adj_matrix = torch.zeros(
            (batch_size, max_compound_len, max_compound_len),
            pin_memory=True
        )
        compound_embedding = torch.zeros(
            (batch_size, max_smiles_len, batch[0]['COMPOUND_EMBEDDING'].shape[1]),
            pin_memory=True
        )
        protein_seq_embedding = torch.zeros(
            (batch_size, max_protein_len, batch[0]['PROTEIN_EMBEDDING'].shape[1]),
            pin_memory=True
        )
        labels = []
        for i, item in enumerate(batch):
            v = item['COMPOUND_NODE_FEAT']
            compound_node_features[i, :v.shape[0], :] = torch.FloatTensor(v)
            v = item['COMPOUND_ADJ']
            compound_adj_matrix[i, :v.shape[0], :v.shape[0]] = torch.FloatTensor(v)
            v = item['COMPOUND_EMBEDDING']
            compound_embedding[i, :v.shape[0], :] = torch.FloatTensor(v)
            v = item['PROTEIN_EMBEDDING']
            protein_seq_embedding[i, :v.shape[0], :] = torch.FloatTensor(v)
            labels.append(item['LABEL'])
        compound_node_nums = torch.LongTensor(compound_node_nums)
        compound_smiles_length = torch.LongTensor(compound_smiles_nums)
        protein_node_nums = torch.LongTensor(protein_node_nums)
        labels = torch.tensor(labels, dtype=torch.float32)
        return {
            'COMPOUND_NODE_FEAT': compound_node_features,
            'COMPOUND_ADJ': compound_adj_matrix,
            'COMPOUND_NODE_NUM': compound_node_nums,
            'COMPOUND_SMILES_LENGTH': compound_smiles_length,
            'COMPOUND_EMBEDDING': compound_embedding,
            'PROTEIN_EMBEDDING': protein_seq_embedding,
            'PROTEIN_NODE_NUM': protein_node_nums,
            'LABEL': labels,
        }
if __name__ == "__main__":
    print("Testing CPIDataset...")
    data_path = 'data/'
    davis_csv = data_path + 'davis.csv'
    davis_compound = data_path + 'davis/compound'
    davis_protein = data_path + 'davis/protein'
    if not os.path.exists(davis_csv):
        print(f"Error: {davis_csv} not found")
        exit(1)
    if not os.path.exists(davis_compound):
        print(f"Error: {davis_compound} not found")
        exit(1)
    if not os.path.exists(davis_protein):
        print(f"Error: {davis_protein} not found")
        exit(1)
    try:
        train_set = CPIDataset(davis_csv, davis_compound, davis_protein)
        print(f"Dataset loaded successfully with {len(train_set)} samples")
        item = train_set[3]
        print('Test Item:')
        print('Compound Node Feature Shape:', item['COMPOUND_NODE_FEAT'].shape)
        print('Compound Adjacency Matrix Shape:', item['COMPOUND_ADJ'].shape)
        print('Compound EMBEDDING Shape:', item['COMPOUND_EMBEDDING'].shape)
        print('Protein Embedding Shape:', item['PROTEIN_EMBEDDING'].shape)
        print('Label:', item['LABEL'])
        train_loader = DataLoader(
            train_set,
            batch_size=8,
            collate_fn=train_set.collate_fn,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )
        print('Test Batch:')
        for batch in train_loader:
            print('Batch loaded successfully!')
            print('Compound Node Feature Shape:', batch['COMPOUND_NODE_FEAT'].shape)
            print('Compound Adjacency Matrix Shape:', batch['COMPOUND_ADJ'].shape)
            print('Compound Node Numbers Shape:', batch['COMPOUND_NODE_NUM'].shape)
            print('Protein Embedding Shape:', batch['PROTEIN_EMBEDDING'].shape)
            print('Protein Node Numbers Shape:', batch['PROTEIN_NODE_NUM'].shape)
            print('Label Shape:', batch['LABEL'].shape)
            break
        print("All tests passed!")
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
