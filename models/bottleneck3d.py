# 定义了 3D 卷积网络的 Bottleneck 模块，用于构建深度残差结构。
# 包含 2D 和 3D 版本的 Bottleneck。本项目主要使用 Bottleneck3d。

import torch
from torch import nn
from typing import Optional # Import Optional for type hinting

# Standard 2D Bottleneck module (likely not used in this 3D project)
class Bottleneck(nn.Module):
    """2D Bottleneck module (standard ResNet style). Likely not used in this 3D project."""
    expansion: int = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        out += residual # 残差连接
        out = self.relu(out)

        return out

class Bottleneck3d(nn.Module):
    """
    用于 3D 卷积网络的 Bottleneck 模块。
    包含三个 3D 卷积层和一个残差连接。第一个和第三个卷积是 1x1x1，用于通道数的缩减和恢复（类似瓶颈）。
    第二个卷积是 1x3x3（时间维度核大小为 1），主要在空间维度进行特征提取。
    注意：与标准 ResNet Bottleneck 不同，此实现中输入和输出通道数相同。
    """
    expansion: int = 4 # 扩展因子，虽然在此实现中输出通道不直接等于 planes * expansion

    def __init__(self,
                 channel: int, # 输入和输出通道数
                 factor: int = 4, # 通道缩减因子：中间层通道数为 channel // factor
                 stride: int = 1, # 空间卷积的步长 (H, W)；序列维度步长固定为 1
                 downsample: Optional[nn.Module] = None # 用于残差连接维度匹配的下采样模块
                ):
        """
        初始化 Bottleneck3d 模块。

        Args:
            channel (int): Bottleneck 模块的输入和输出通道数。
                           注意：此实现中输入和输出通道数始终相同。
            factor (int): 通道缩减因子。中间的卷积层通道数为 channel // factor。
            stride (int): 空间维度 (H, W) 的卷积步长。会影响第二个卷积层以及 downsample 模块。
                          时间维度 (N) 的步长固定为 1。
            downsample (Optional[nn.Module]): 一个可选的 nn.Module，用于当输入和残差连接的输出形状不匹配时，
                                             对残差连接进行下采样（例如，通过 1x1x1 卷积和步长）。
        """
        super(Bottleneck3d, self).__init__()
        # 计算中间层的通道数
        self.mid_channel: int = channel // factor
        # 如果 channel 不能被 factor 整除，这里可能会有微小差异，但通常 channel 设计为 factor 的倍数。

        # 第一个 1x1x1 卷积：用于缩减通道数 (channel -> mid_channel)
        self.conv1 = nn.Conv3d(channel, self.mid_channel, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(self.mid_channel)

        # 第二个 1x3x3 卷积：在空间维度进行特征提取，保持序列维度不变。步长由 stride 参数控制。
        # padding=(0, 1, 1) 确保空间维度尺寸不变 (当 stride=1 时)
        self.conv2 = nn.Conv3d(self.mid_channel, self.mid_channel, kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(self.mid_channel)

        # 第三个 1x1x1 卷积：用于恢复通道数 (mid_channel -> channel)
        self.conv3 = nn.Conv3d(self.mid_channel, channel, kernel_size=1, bias=False) # 输出通道数与输入相同
        self.bn3 = nn.BatchNorm3d(channel)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor: # Type hint input and output tensors
        """
        前向传播函数。

        Args:
            x (torch.Tensor): 输入张量，形状通常为 (B, C, N, H, W)。

        Returns:
            torch.Tensor: 经过 Bottleneck 处理后的输出张量，形状与输入 x 相同 (B, C, N, H, W)。
        """
        # 获取残差连接的输入
        residual: torch.Tensor = x

        # 路径 1：卷积序列 (Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> Conv -> BN)
        out: torch.Tensor = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # 如果 downsample 模块存在 (通常在 stride=1 但输入和残差维度不匹配时)
        if self.downsample is not None:
            # 对残差连接应用 downsample 模块以匹配维度
            residual = self.downsample(x)

        # 添加残差连接
        out += residual
        # 应用最终的 ReLU 激活函数
        out = self.relu(out)

        return out