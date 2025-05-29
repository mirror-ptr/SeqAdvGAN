import torch
from torch import nn
from .IrisXeonCls import IrisXeonCls

class IrisTriggerNet(nn.Module):
    def __init__(self):
        super(IrisTriggerNet, self).__init__()
        self.xeon_cls = IrisXeonCls()
        self.norm1 = nn.BatchNorm3d(128)

    def forward(self, x, w):
        x = self.norm1(x)
        x = self.xeon_cls(x, w)
        x = x.squeeze(1)
        return x