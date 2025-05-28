import torch
from axial_attention import AxialAttention
from .IrisAxisVisionEncoder import IrisAxisVisionEncoder
from torch import nn
from IrisBabel.nn.CNN import Bottleneck3d, Bottleneck, IrisXeonNet, IrisXeonWeight

class IrisAttentionTransformers(nn.Module):
    def __init__(self, inject_model=None):
        super(IrisAttentionTransformers, self).__init__()
        self.norm1 = nn.BatchNorm3d(128)
        self.attn = IrisAxisVisionEncoder(num_heads=8, num_layers=8, num_axial_dim=128)
        self.xeon_net = IrisXeonNet()
        self.xeon_weight = IrisXeonWeight()
        self.inject_model = inject_model

    def forward(self, x):
        # CNN Downsample
        x1 = self.xeon_net(x)
        if self.inject_model is not None:
            x1 = self.inject_model(x1)
        # Axial Transformer
        x2 = self.norm1(x1)
        x2 = self.attn(x2)
        x2 = self.xeon_weight(x2)
        x2 = torch.squeeze(x2, 1)

        return x2
