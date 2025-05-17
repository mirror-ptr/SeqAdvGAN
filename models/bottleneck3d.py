import torch
from torch import nn

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck3d(nn.Module):
    expansion = 4
    def __init__(self, channel, factor=4, stride=1, downsample=None):
        super(Bottleneck3d, self).__init__()
        self.mid_channel = channel // factor
        self.conv1 = nn.Conv3d(channel, self.mid_channel, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(self.mid_channel)
        self.conv2 = nn.Conv3d(self.mid_channel, self.mid_channel, kernel_size=(1, 3, 3), stride=stride, padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(self.mid_channel)
        self.conv3 = nn.Conv3d(self.mid_channel, channel, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(channel)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out