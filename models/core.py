import math
import torch
from models.mlp_modules import MultiLayerMLP
from models.gnn import MultiGIN
from torch import nn
import torch.nn.functional as F
from models.decoder import DecoderLayer
from torch.nn.utils.weight_norm import weight_norm
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
class AttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super(AttentionPooling, self).__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x, mask=None):
        attn_scores = self.attention_net(x)
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            attn_scores = attn_scores.masked_fill(mask_expanded == 0, -1e4)
        attn_weights = F.softmax(attn_scores, dim=1)
        pooled = (x * attn_weights).sum(dim=1)
        return pooled, attn_weights
class HierarchicalPooling(nn.Module):
    def __init__(self, input_dim):
        super(HierarchicalPooling, self).__init__()
        self.attn_pool = AttentionPooling(input_dim)
        self.fusion = nn.Sequential(
            nn.Linear(input_dim * 3, input_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim * 2, input_dim * 2)
        )
    def forward(self, x, node_num=None):
        B, N, D = x.shape
        node_mask = None
        if node_num is not None:
            if not isinstance(node_num, torch.Tensor):
                node_num = torch.tensor(node_num, device=x.device)
            elif node_num.device != x.device:
                node_num = node_num.to(x.device)
            node_mask = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
            node_mask = (node_mask < node_num.unsqueeze(1)).float()
        if node_mask is not None:
            mask_expanded = node_mask.unsqueeze(-1)
            sum_x = (x * mask_expanded).sum(dim=1)
            count = mask_expanded.sum(dim=1).clamp(min=1)
            mean_pool = sum_x / count
        else:
            mean_pool = x.mean(dim=1)
        if node_mask is not None:
            x_masked = x.clone()
            x_masked[node_mask == 0] = -1e4
            max_pool = x_masked.max(dim=1)[0]
        else:
            max_pool = x.max(dim=1)[0]
        attn_pool, _ = self.attn_pool(x, node_mask)
        combined = torch.cat([mean_pool, max_pool, attn_pool], dim=-1)
        fused = self.fusion(combined)
        return fused
class LinearAttention(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=32, heads=10):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.linear_first = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.linear_second = torch.nn.Linear(self.hidden_dim, self.heads)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x, masks):
        sentence_att = F.tanh(self.linear_first(x))
        sentence_att = self.linear_second(sentence_att)
        sentence_att = sentence_att.transpose(1, 2)
        minus_inf = -1e4 * torch.ones_like(sentence_att)
        e = torch.where(masks > 0.5, sentence_att, minus_inf)
        att = self.softmax(e)
        sentence_embed = att @ x
        avg_sentence_embed = torch.sum(sentence_embed, 1) / self.heads
        return avg_sentence_embed
class ImprovedLinearAttention(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, heads=10, dropout=0.1):
        super(ImprovedLinearAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        assert hidden_dim % heads == 0, f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})"
        self.q_proj = nn.Linear(input_dim, hidden_dim)
        self.k_proj = nn.Linear(input_dim, hidden_dim)
        self.v_proj = nn.Linear(input_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, input_dim)
        self.ln = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.scale = math.sqrt(self.head_dim)
    def forward(self, x, masks):
        B, N, D = x.shape
        residual = x
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        importance_scores = (Q * K).sum(dim=-1, keepdim=True) / self.scale
        if masks is not None:
            mask_avg = masks.mean(dim=1, keepdim=True).transpose(1, 2)
            importance_scores = importance_scores.masked_fill(mask_avg < 0.5, -1e4)
        importance_weights = torch.softmax(importance_scores, dim=1)
        importance_weights = self.dropout(importance_weights)
        weighted_v = V * importance_weights
        pooled_features = weighted_v.sum(dim=1)
        output = self.out_proj(pooled_features)
        residual_pooled = residual.mean(dim=1)
        output = self.ln(residual_pooled + self.dropout(output))
        return output, importance_weights
class GatedFusion(nn.Module):
    def __init__(self, graph_dim, seq_dim, output_dim, dropout=0.1):
        super(GatedFusion, self).__init__()
        self.graph_dim = graph_dim
        self.seq_dim = seq_dim
        self.output_dim = output_dim
        self.graph_proj = nn.Linear(graph_dim, output_dim)
        self.seq_proj = nn.Linear(seq_dim, output_dim)
        self.gate = nn.Sequential(
            nn.Linear(graph_dim + seq_dim, output_dim),
            nn.Sigmoid()
        )
        self.fusion_transform = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )
        self.ln = nn.LayerNorm(output_dim)
    def forward(self, graph_feat, seq_feat):
        graph_proj = self.graph_proj(graph_feat)
        seq_proj = self.seq_proj(seq_feat)
        concat = torch.cat([graph_feat, seq_feat], dim=-1)
        gate_weight = self.gate(concat)
        fused = gate_weight * graph_proj + (1 - gate_weight) * seq_proj
        fused = self.fusion_transform(fused)
        fused = self.ln(fused + graph_proj + seq_proj)
        return fused
class PLGR(torch.nn.Module):
    def __init__(self, args):
        super(PLGR, self).__init__()
        self.gnn_dim = args.compound_gnn_dim
        self.dropout = args.dropout
        self.decoder_dim = args.decoder_dim
        self.decoder_heads = args.decoder_heads
        self.compound_text_dim = args.compound_text_dim
        self.compound_structure_dim = args.compound_structure_dim
        self.protein_dim = args.protein_dim
        self.linear_heads = args.linear_heads
        self.linears_hidden_dim = args.linear_hidden_dim
        self.feedforward_dim = args.pf_dim
        self.encoder_heads = args.encoder_heads
        self.encoder_layers = args.encoder_layers
        self.protein_pretrained_dim = args.protein_pretrained_dim
        self.compound_pretrained_dim = args.compound_pretrained_dim
        self.objective = args.objective
        from models.gnn import MultiGINWithJK
        self.drug_gin = MultiGINWithJK(
            self.compound_structure_dim,
            self.gnn_dim,
            jk_mode='cat'
        )
        self.hierarchical_pool = HierarchicalPooling(self.gnn_dim)
        self.gated_fusion = GatedFusion(
            graph_dim=self.gnn_dim * 2 + self.compound_text_dim,
            seq_dim=self.compound_text_dim,
            output_dim=self.decoder_dim,
            dropout=self.dropout
        )
        self.fusion_residual_proj = nn.Linear(self.compound_text_dim, self.decoder_dim)
        self.fusion_ln = nn.LayerNorm(self.decoder_dim)
        self.cross_atten = DecoderLayer(self.decoder_dim, self.decoder_heads, self.dropout)
        improved_hidden_dim = self.linear_heads * 5
        self.drug_attn = ImprovedLinearAttention(
            self.compound_text_dim,
            improved_hidden_dim,
            self.linear_heads,
            dropout=self.dropout
        )
        self.target_attn = ImprovedLinearAttention(
            self.protein_dim,
            improved_hidden_dim,
            self.linear_heads,
            dropout=self.dropout
        )
        self.inter_attn_one = ImprovedLinearAttention(
            self.protein_dim,
            improved_hidden_dim,
            self.linear_heads,
            dropout=self.dropout
        )
        optimized_feedforward_dim = max(256, self.feedforward_dim // 2)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.compound_text_dim, dim_feedforward=optimized_feedforward_dim, nhead=self.encoder_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.encoder_layers)
        self.encoder_layer2 = nn.TransformerEncoderLayer(d_model=self.protein_dim, dim_feedforward=optimized_feedforward_dim, nhead=self.encoder_heads)
        self.transformer_encoder2 = nn.TransformerEncoder(self.encoder_layer2, num_layers=self.encoder_layers)
        self.fc1 = nn.Linear(self.gnn_dim, self.compound_text_dim)
        self.fc2 = nn.Linear(self.protein_pretrained_dim,self.protein_dim)
        self.fc3 = nn.Linear(self.compound_pretrained_dim,self.compound_text_dim)
        self.drug_ln = nn.LayerNorm(self.compound_text_dim)
        self.target_ln = nn.LayerNorm(self.protein_dim)
        if self.objective in ('regression', 'classification'):
            output_dim = 1 if self.objective == 'regression' else 2
            self.lin = self._build_output_mlp(output_dim)
        else:
            raise ValueError(f"Unsupported objective: {self.objective}")

    def _build_output_mlp(self, out_dim: int):
        return nn.Sequential(
            nn.Linear(self.protein_dim * 3, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(self.dropout * 0.5),
            nn.Linear(256, out_dim)
        )
    def generate_masks(self, adj, adj_sizes, n_heads):
        out = torch.ones(adj.shape[0], adj.shape[1])
        max_size = adj.shape[1]
        if isinstance(adj_sizes, int):
            out[0, adj_sizes:max_size] = 0
        else:
            for e_id, drug_len in enumerate(adj_sizes):
                out[e_id, drug_len: max_size] = 0
        out = out.unsqueeze(1).expand(-1, n_heads, -1)
        return out.to(device=adj.device)
    def make_masks(self, atom_num, protein_num, compound_max_len, protein_max_len):
        batch_size = len(atom_num)
        device = next(self.parameters()).device
        compound_mask = torch.zeros((batch_size, compound_max_len), device=device)
        protein_mask = torch.zeros((batch_size, protein_max_len), device=device)
        for i in range(batch_size):
            compound_mask[i, :atom_num[i]] = 1
            protein_mask[i, :protein_num[i]] = 1
        compound_mask = compound_mask.unsqueeze(1).unsqueeze(2)
        protein_mask = protein_mask.unsqueeze(1).unsqueeze(2)
        return compound_mask, protein_mask
    def encode_joint_features(self, data):
        device = next(self.parameters()).device
        compound_x = data['COMPOUND_NODE_FEAT'].to(device, non_blocking=True)
        compound_adj = data['COMPOUND_ADJ'].to(device, non_blocking=True)
        compound_emb = data['COMPOUND_EMBEDDING'].to(device, non_blocking=True)
        target_emb = data['PROTEIN_EMBEDDING'].to(device, non_blocking=True)
        compound_smiles_max_len = data['COMPOUND_EMBEDDING'].shape[1]
        compound_node_max_len = data['COMPOUND_NODE_FEAT'].shape[1]
        node_mask, smiles_mask = self.make_masks(
            data["COMPOUND_NODE_NUM"],
            data["COMPOUND_SMILES_LENGTH"],
            compound_node_max_len,
            compound_smiles_max_len,
        )
        compound_graph_feat = self.drug_gin(compound_x, compound_adj)
        del compound_x, compound_adj
        xd_f1 = self.drug_ln(self.fc1(compound_graph_feat))
        smiles_input = self.fc3(compound_emb)
        compound_smiles = self.transformer_encoder(smiles_input)
        del compound_emb
        xd_f2 = self.drug_ln(compound_smiles)
        del compound_smiles
        graph_pooled = self.hierarchical_pool(
            compound_graph_feat,
            data["COMPOUND_NODE_NUM"]
        )
        graph_pooled = graph_pooled.unsqueeze(1).expand(-1, xd_f2.size(1), -1)
        attention_weights = torch.sigmoid(self.fc1(compound_graph_feat)).mean(dim=1, keepdim=True)
        del compound_graph_feat
        attention_weights = attention_weights.expand(-1, xd_f2.size(1), -1)
        weighted_seq_feat = xd_f2 * attention_weights
        del attention_weights
        graph_features = torch.cat([graph_pooled, weighted_seq_feat], dim=-1)
        del graph_pooled, weighted_seq_feat
        fused_features = self.gated_fusion(graph_features, xd_f2)
        del graph_features
        residual = self.fusion_residual_proj(xd_f2)
        del xd_f2
        fused_features = self.fusion_ln(fused_features + residual)
        del residual
        compound_mask = self.generate_masks(fused_features, data["COMPOUND_SMILES_LENGTH"], self.linear_heads)
        xd = self.cross_atten(fused_features, xd_f1, smiles_mask, node_mask)
        del fused_features, xd_f1
        xd_attn, drug_weights = self.drug_attn(xd, compound_mask)
        self.drug_weights = drug_weights
        target_emb_proj = self.fc2(target_emb)
        del target_emb
        seq_emb = self.transformer_encoder2(target_emb_proj)
        del target_emb_proj
        xt = self.target_ln(seq_emb)
        del seq_emb
        protein_mask = self.generate_masks(xt, data["PROTEIN_NODE_NUM"], self.linear_heads)
        xt_attn, target_weights = self.target_attn(xt, protein_mask)
        self.target_weights = target_weights
        cat_f = torch.cat([xt, xd], dim=1)
        del xt, xd
        cat_mask = torch.cat([protein_mask, compound_mask], dim=-1)
        del protein_mask, compound_mask
        cat_attn, inter_weights = self.inter_attn_one(cat_f, cat_mask)
        self.inter_weights = inter_weights
        del cat_f, cat_mask
        joint_features = torch.cat([xd_attn, cat_attn, xt_attn], dim=-1)
        self.joint_features = joint_features
        out = self.lin(joint_features)
        del xd_attn, cat_attn, xt_attn, joint_features
        return out
    def predict_from_joint_features(self, joint_features):
        return self.lin(joint_features)
    def forward(self, data):
        return self.encode_joint_features(data)
