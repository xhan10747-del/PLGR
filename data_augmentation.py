import numpy as np
import random
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
class SMILESAugmentation:
    @staticmethod
    def randomize_smiles(smiles, n_variants=1):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [smiles] * n_variants
        variants = []
        for _ in range(n_variants):
            try:
                random_smiles = Chem.MolToSmiles(mol, doRandom=True)
                variants.append(random_smiles)
            except:
                variants.append(smiles)
        return variants if variants else [smiles]
class MolecularGraphAugmentation:
    @staticmethod
    def add_noise_to_features(features, noise_level=0.01):
        if isinstance(features, torch.Tensor):
            noise = torch.randn_like(features) * noise_level
            return features + noise
        else:
            noise = np.random.randn(*features.shape) * noise_level
            return features + noise
    @staticmethod
    def edge_dropout(adj_matrix, dropout_rate=0.1):
        if isinstance(adj_matrix, torch.Tensor):
            mask = torch.rand_like(adj_matrix) > dropout_rate
            mask = mask * mask.t()
            mask.fill_diagonal_(1)
            return adj_matrix * mask.float()
        else:
            mask = np.random.rand(*adj_matrix.shape) > dropout_rate
            mask = mask * mask.T
            np.fill_diagonal(mask, 1)
            return adj_matrix * mask
    @staticmethod
    def node_dropout(features, adj_matrix, dropout_rate=0.05):
        num_nodes = features.shape[0]
        keep_nodes = int(num_nodes * (1 - dropout_rate))
        if keep_nodes < num_nodes:
            keep_indices = np.random.choice(num_nodes, keep_nodes, replace=False)
            keep_indices = np.sort(keep_indices)
            if isinstance(features, torch.Tensor):
                aug_features = features[keep_indices]
                aug_adj = adj_matrix[keep_indices][:, keep_indices]
            else:
                aug_features = features[keep_indices]
                aug_adj = adj_matrix[keep_indices][:, keep_indices]
            return aug_features, aug_adj
        return features, adj_matrix
class DataAugmentor:
    def __init__(self,
                 use_smiles_randomization=True,
                 use_feature_noise=True,
                 use_edge_dropout=True,
                 use_node_dropout=False,
                 feature_noise_level=0.01,
                 edge_dropout_rate=0.1,
                 node_dropout_rate=0.05):
        self.use_smiles_randomization = use_smiles_randomization
        self.use_feature_noise = use_feature_noise
        self.use_edge_dropout = use_edge_dropout
        self.use_node_dropout = use_node_dropout
        self.feature_noise_level = feature_noise_level
        self.edge_dropout_rate = edge_dropout_rate
        self.node_dropout_rate = node_dropout_rate
        self.smiles_aug = SMILESAugmentation()
        self.graph_aug = MolecularGraphAugmentation()
    def augment_graph(self, features, adj_matrix, training=True):
        if not training:
            return features, adj_matrix
        aug_features = features
        aug_adj = adj_matrix
        if self.use_node_dropout and random.random() < 0.3:
            aug_features, aug_adj = self.graph_aug.node_dropout(
                aug_features, aug_adj, self.node_dropout_rate
            )
        if self.use_feature_noise:
            aug_features = self.graph_aug.add_noise_to_features(
                aug_features, self.feature_noise_level
            )
        if self.use_edge_dropout and random.random() < 0.5:
            aug_adj = self.graph_aug.edge_dropout(
                aug_adj, self.edge_dropout_rate
            )
        return aug_features, aug_adj
    def augment_smiles(self, smiles, n_variants=1):
        if not self.use_smiles_randomization:
            return [smiles]
        return self.smiles_aug.randomize_smiles(smiles, n_variants)
def get_default_augmentor(training_mode='standard'):
    if training_mode == 'aggressive':
        return DataAugmentor(
            use_smiles_randomization=True,
            use_feature_noise=True,
            use_edge_dropout=True,
            use_node_dropout=False,
            feature_noise_level=0.02,
            edge_dropout_rate=0.15,
            node_dropout_rate=0.0
        )
    elif training_mode == 'conservative':
        return DataAugmentor(
            use_smiles_randomization=True,
            use_feature_noise=True,
            use_edge_dropout=False,
            use_node_dropout=False,
            feature_noise_level=0.005,
            edge_dropout_rate=0.0,
            node_dropout_rate=0.0
        )
    elif training_mode == 'cold_target':
        return DataAugmentor(
            use_smiles_randomization=True,
            use_feature_noise=True,
            use_edge_dropout=True,
            use_node_dropout=False,
            feature_noise_level=0.025,
            edge_dropout_rate=0.20,
            node_dropout_rate=0.0
        )
    elif training_mode == 'cold_target_drug':
        return DataAugmentor(
            use_smiles_randomization=True,
            use_feature_noise=True,
            use_edge_dropout=True,
            use_node_dropout=False,
            feature_noise_level=0.035,
            edge_dropout_rate=0.30,
            node_dropout_rate=0.0
        )
    else:
        return DataAugmentor(
            use_smiles_randomization=True,
            use_feature_noise=True,
            use_edge_dropout=True,
            use_node_dropout=False,
            feature_noise_level=0.01,
            edge_dropout_rate=0.1,
            node_dropout_rate=0.0
        )
if __name__ == "__main__":
    print("Testing SMILES Augmentation...")
    smiles = "CCO"
    aug = SMILESAugmentation()
    variants = aug.randomize_smiles(smiles, n_variants=5)
    print(f"Original: {smiles}")
    print(f"Variants: {variants}")
    print("\nTesting Graph Augmentation...")
    features = np.random.rand(10, 78)
    adj = np.eye(10)
    graph_aug = MolecularGraphAugmentation()
    aug_features = graph_aug.add_noise_to_features(features, 0.01)
    print(f"Feature noise added: {np.mean(np.abs(aug_features - features)):.6f}")
    aug_adj = graph_aug.edge_dropout(adj, 0.1)
    print(f"Edges dropped: {np.sum(adj) - np.sum(aug_adj)}")
