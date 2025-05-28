import torch
from torch import nn
from axial_attention import AxialAttention, AxialPositionalEmbedding
import copy

class IrisAxisVisionEncoderLayer(nn.Module):
    def __init__(self, num_heads=8, num_axial_dim=128, dropout=0.1, num_norm_eps=1e-5):
        super(IrisAxisVisionEncoderLayer, self).__init__()
        self.axis_attention = AxialAttention(dim = num_axial_dim, dim_index = 1, heads = num_heads, num_dimensions = 3)
        self.norm = nn.LayerNorm(num_axial_dim, eps=num_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(num_axial_dim, num_axial_dim)
        self.linear2 = nn.Linear(num_axial_dim, num_axial_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        pos_embedding = AxialPositionalEmbedding(dim=x.shape[1], shape=(x.shape[2], x.shape[3], x.shape[4])).to(x.device)
        x += pos_embedding(x)
        residual = x
        x = self.axis_attention(x)
        x = self.dropout1(x)
        x = x + residual
        x = torch.permute(x, (0, 2, 3, 4, 1))
        x = self.norm(x)
        residual = x
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout2(x)
        x = self.linear2(x)
        x = self.dropout3(x)
        x = x + residual
        x = self.norm(x)
        x = torch.permute(x, (0, 4, 1, 2, 3))
        return x


class IrisAxisVisionEncoder(nn.Module):
    def __init__(self, num_heads=8, num_layers=4, num_axial_dim=128):
        super(IrisAxisVisionEncoder, self).__init__()
        self.attentions = nn.ModuleList([copy.deepcopy(IrisAxisVisionEncoderLayer(num_heads, num_axial_dim)) for i in range(num_layers)])

    def forward(self, x):
        for attention in self.attentions:
            x = attention(x)
        return x