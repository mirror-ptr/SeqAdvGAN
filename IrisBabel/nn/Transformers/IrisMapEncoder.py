import torch
from torch import nn
from .PositionalEncoding import PositionalEncoding

class IrisMapEncoder(nn.Module):
    def __init__(self, input_dim=1024, output_dim=4096, num_head=8, num_encoder_layer=4):
        super(IrisMapEncoder, self).__init__()
        # Positional Encoding
        self.pos_enc = PositionalEncoding(d_model=input_dim, max_len=400)
        #Transformers
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_head, batch_first=True)
        encoder_layer.self_attn.batch_first = True
        self.transformers = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layer)
        # Projection Layer
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Input Shape: (B, W, H, N, D)
        B, W, H, N, D = x.shape
        # Object Pooling
        x = torch.mean(x, dim=3)
        x = x.view(B, -1, D)
        # Positional Encoding
        x = self.pos_enc(x)
        # Transformers
        x = self.transformers(x)
        # Final Pooling
        x = torch.mean(x, dim=1)
        # Projection
        x = self.proj(x)
        return x

