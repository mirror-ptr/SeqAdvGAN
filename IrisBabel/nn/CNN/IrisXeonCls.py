import torch
from torch import nn
from .Bottleneck import Bottleneck

class IrisXeonCls(nn.Module):
    def __init__(self):
        super(IrisXeonCls, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.cls_bottleneck1 = Bottleneck(128, 32, 1)
        self.cls_bottleneck2 = Bottleneck(128, 32, 1)
        self.cls_bottleneck3 = Bottleneck(128, 32, 1)
        self.cls_bottleneck4 = Bottleneck(128, 32, 1)
        self.linear1 = nn.Linear(128, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, 8)
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(0.1)
        self.norm4 = nn.BatchNorm2d(128)
        self.softmax = nn.Softmax(dim=3)

    def forward(self, x, w):
        x = torch.mean(x, dim = 2)
        w = w.view(w.size(0), 1, w.size(1), w.size(2))
        w = torch.repeat_interleave(w, 128, dim=1)
        x = x * w
        x = self.norm4(x)
        x1 = self.cls_bottleneck1(x)
        x1 = self.cls_bottleneck2(x1)
        x1 = self.cls_bottleneck3(x1)
        x1 = self.cls_bottleneck4(x1)
        x1 = torch.permute(x1, (0, 2, 3, 1))
        x1 = self.relu(x1)
        x1 = self.dropout(x1)
        x1 = self.linear1(x1)
        x1 = self.dropout(x1)
        x1 = self.relu(x1)
        x1 = self.linear2(x1)
        x1 = self.dropout(x1)
        x1 = self.relu(x1)
        x1 = self.linear3(x1)
        x1 = self.dropout(x1)
        x1 = self.relu(x1)
        x1 = self.linear4(x1)
        x1 = self.dropout(x1)
        x1 = self.relu(x1)
        x1 = self.softmax(x1)
        return x1