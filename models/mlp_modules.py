import torch
import torch.nn as nn
import torch.nn.functional as F
class ThreeLayerMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, activation='relu'):
        super(ThreeLayerMLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)
        self.dropout_layer = nn.Dropout(dropout)
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'tanh':
            self.activation = torch.tanh
        else:
            self.activation = F.relu
        self._init_weights()
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    def forward(self, x, adj=None):
        h1 = self.layer1(x)
        h1 = self.activation(h1)
        h1 = self.dropout_layer(h1)
        h2 = self.layer2(h1)
        h2 = self.activation(h2)
        h2 = self.dropout_layer(h2)
        h3 = self.layer3(h2)
        return h3
class MLPPositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super(MLPPositionwiseFeedforward, self).__init__()
        self.hid_dim = hid_dim
        self.pf_dim = pf_dim
        self.mlp = ThreeLayerMLP(hid_dim, pf_dim, hid_dim, dropout)
    def forward(self, x):
        return self.mlp(x)
class MultiLayerMLP(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super(MultiLayerMLP, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mlp = ThreeLayerMLP(in_dim, out_dim, out_dim, dropout)
    def forward(self, inputs, adj=None):
        return self.mlp(inputs)
class ResidualMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(ResidualMLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mlp = ThreeLayerMLP(input_dim, hidden_dim, output_dim, dropout)
        self.residual_proj = None
        if input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim)
    def forward(self, x, adj=None):
        out = self.mlp(x)
        if self.residual_proj is not None:
            residual = self.residual_proj(x)
        else:
            residual = x
        return out + residual
class AdaptiveMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(AdaptiveMLP, self).__init__()
        self.base_mlp = ThreeLayerMLP(input_dim, hidden_dim, output_dim, dropout)
        self.attention_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    def forward(self, x, adj=None):
        attention_weights = self.attention_layer(x)
        mlp_out = self.base_mlp(x)
        weighted_out = mlp_out * attention_weights
        return weighted_out
