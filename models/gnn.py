import math
import torch
import torch.nn as nn
import torch.nn.functional as F
class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, add_self=True, normalize_embedding=False, dropout=0.0, bias=True):
        super(GraphConv, self).__init__()
        self.add_self = add_self
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.bias = None
        self.init_weights()
    def init_weights(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.constant(self.bias.data, 0.0)
    def forward(self, x, adj):
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        y = torch.matmul(adj, x)
        if self.add_self:
            y += x
        y = torch.matmul(y, self.weight)
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
        return y
class ResGCN(nn.Module):
    def __init__(self, args, in_dim, out_dim):
        super(ResGCN, self).__init__()
        self.res_conv1 = GraphConv(in_dim, out_dim)
        self.res_conv2 = GraphConv(out_dim, out_dim)
        self.res_conv3 = GraphConv(out_dim, out_dim)
    def forward(self, inputs, adj):
        outputs = self.res_conv1(inputs, adj)
        outputs = self.res_conv2(outputs, adj)
        outputs = self.res_conv3(outputs, adj)
        return outputs
class SimpleGCN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SimpleGCN, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
    def forward(self, inputs, adj):
        return torch.bmm(adj, self.fc(inputs))
class MultiGCN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MultiGCN, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim, bias=True)
        self.fc2 = nn.Linear(out_dim, out_dim, bias=True)
        self.fc3 = nn.Linear(out_dim, out_dim, bias=True)
    def forward(self, inputs, adj):
        outputs = F.relu(self.fc1(torch.bmm(adj, inputs)))
        outputs = F.relu(self.fc2(torch.bmm(adj, outputs)))
        outputs = F.relu(self.fc3(torch.bmm(adj, outputs)))
        return outputs
class GCNLayer(nn.Module):
    def __init__(self,input_features,output_features,bias=False):
        super(GCNLayer,self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weights = nn.Parameter(torch.FloatTensor(input_features,output_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_features))
        else:
            self.register_parameter('bias',None)
        self.reset_parameters()
    def reset_parameters(self):
        std = 1./math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-std,std)
        if self.bias is not None:
            self.bias.data.uniform_(-std,std)
    def forward(self,adj,x):
        support = torch.mm(x,self.weights)
        output = torch.spmm(adj,support)
        if self.bias is not None:
            return output+self.bias
        return output
class GCN(nn.Module):
    def __init__(self,input_size,hidden_size,dropout,bias=False):
        super(GCN,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.gcn1 = GCNLayer(input_size,hidden_size,bias=bias)
        self.gcn2 = GCNLayer(hidden_size,hidden_size,bias=bias)
        self.gcn3 = GCNLayer(hidden_size,hidden_size,bias=bias)
        self.dropout = dropout
    def forward(self,adj,x):
        x = F.relu(self.gcn1(adj,x))
        x = F.dropout(x,self.dropout,training=self.training)
        x = F.relu(self.gcn2(adj, x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gcn3(adj, x))
        x = F.dropout(x, self.dropout, training=self.training)
        return x
class GINLayer(nn.Module):
    def __init__(self, input_features, output_features, eps=0.0, train_eps=False):
        super(GINLayer, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.mlp = nn.Sequential(
            nn.Linear(input_features, output_features),
            nn.ReLU(),
            nn.Linear(output_features, output_features)
        )
        if train_eps:
            self.eps = nn.Parameter(torch.tensor([eps], dtype=torch.float32))
        else:
            self.register_buffer('eps', torch.tensor([eps], dtype=torch.float32))
    def forward(self, adj, x):
        if len(x.shape) == 3:
            neighbor_sum = torch.bmm(adj, x)
            out = (1 + self.eps) * x + neighbor_sum
        else:
            neighbor_sum = torch.mm(adj, x)
            out = (1 + self.eps) * x + neighbor_sum
        out = self.mlp(out)
        return out
class GIN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, eps=0.0, train_eps=False, bias=False):
        super(GIN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.gin1 = GINLayer(input_size, hidden_size, eps=eps, train_eps=train_eps)
        self.gin2 = GINLayer(hidden_size, hidden_size, eps=eps, train_eps=train_eps)
        self.gin3 = GINLayer(hidden_size, hidden_size, eps=eps, train_eps=train_eps)
        self.gin4 = GINLayer(hidden_size, hidden_size, eps=eps, train_eps=train_eps)
        self.gin5 = GINLayer(hidden_size, hidden_size, eps=eps, train_eps=train_eps)
    def forward(self, adj, x):
        x = F.relu(self.gin1(adj, x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gin2(adj, x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gin3(adj, x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gin4(adj, x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gin5(adj, x))
        x = F.dropout(x, self.dropout, training=self.training)
        return x
class ResidualGINLayer(nn.Module):
    def __init__(self, input_features, output_features, eps=0.0, train_eps=False):
        super(ResidualGINLayer, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.gin = GINLayer(input_features, output_features, eps, train_eps)
        self.residual_proj = None
        if input_features != output_features:
            self.residual_proj = nn.Linear(input_features, output_features, bias=False)
        self.bn = nn.BatchNorm1d(output_features)
    def forward(self, adj, x):
        out = self.gin(adj, x)
        if self.residual_proj is not None:
            residual = self.residual_proj(x)
        else:
            residual = x
        out = out + residual
        if len(out.shape) == 3:
            B, N, D = out.shape
            out = out.transpose(1, 2)
            out = self.bn(out)
            out = out.transpose(1, 2)
        else:
            out = out.transpose(0, 1)
            out = self.bn(out)
            out = out.transpose(0, 1)
        return out
class MultiGINWithJK(nn.Module):
    def __init__(self, in_dim, out_dim, eps=0.0, train_eps=False, jk_mode='cat'):
        super(MultiGINWithJK, self).__init__()
        self.jk_mode = jk_mode
        self.out_dim = out_dim
        self.gin1 = ResidualGINLayer(in_dim, out_dim, eps, train_eps)
        self.gin2 = ResidualGINLayer(out_dim, out_dim, eps, train_eps)
        self.gin3 = ResidualGINLayer(out_dim, out_dim, eps, train_eps)
        self.gin4 = ResidualGINLayer(out_dim, out_dim, eps, train_eps)
        self.gin5 = ResidualGINLayer(out_dim, out_dim, eps, train_eps)
        if jk_mode == 'cat':
            self.jk_fusion = nn.Sequential(
                nn.Linear(out_dim * 5, out_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(out_dim * 2, out_dim)
            )
        elif jk_mode == 'lstm':
            self.jk_lstm = nn.LSTM(out_dim, out_dim, batch_first=True, bidirectional=False)
    def forward(self, inputs, adj):
        layer_outputs = []
        x = F.relu(self.gin1(adj, inputs))
        layer_outputs.append(x)
        x = F.relu(self.gin2(adj, x))
        layer_outputs.append(x)
        x = F.relu(self.gin3(adj, x))
        layer_outputs.append(x)
        x = F.relu(self.gin4(adj, x))
        layer_outputs.append(x)
        x = F.relu(self.gin5(adj, x))
        layer_outputs.append(x)
        if self.jk_mode == 'cat':
            jk_out = torch.cat(layer_outputs, dim=-1)
            jk_out = self.jk_fusion(jk_out)
        elif self.jk_mode == 'max':
            jk_out = torch.stack(layer_outputs, dim=0).max(dim=0)[0]
        elif self.jk_mode == 'lstm':
            B, N, D = layer_outputs[0].shape
            stacked = torch.stack(layer_outputs, dim=2)
            stacked = stacked.view(B * N, 5, D)
            _, (h_n, _) = self.jk_lstm(stacked)
            jk_out = h_n[-1].view(B, N, D)
        else:
            jk_out = layer_outputs[-1]
        return jk_out
class MultiGIN(nn.Module):
    def __init__(self, in_dim, out_dim, eps=0.0, train_eps=False):
        super(MultiGIN, self).__init__()
        self.gin1 = GINLayer(in_dim, out_dim, eps=eps, train_eps=train_eps)
        self.gin2 = GINLayer(out_dim, out_dim, eps=eps, train_eps=train_eps)
        self.gin3 = GINLayer(out_dim, out_dim, eps=eps, train_eps=train_eps)
        self.gin4 = GINLayer(out_dim, out_dim, eps=eps, train_eps=train_eps)
        self.gin5 = GINLayer(out_dim, out_dim, eps=eps, train_eps=train_eps)
    def forward(self, inputs, adj):
        outputs = F.relu(self.gin1(adj, inputs))
        outputs = F.relu(self.gin2(adj, outputs))
        outputs = F.relu(self.gin3(adj, outputs))
        outputs = F.relu(self.gin4(adj, outputs))
        outputs = F.relu(self.gin5(adj, outputs))
        return outputs
