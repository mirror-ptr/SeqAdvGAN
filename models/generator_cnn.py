import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple # Import Tuple for type hints

from models.bottleneck3d import Bottleneck3d
from utils.layers import Lambda

class Generator(nn.Module):
    """
    基于 3D 卷积和 3D Bottleneck 的生成器模型。
    用于生成针对 ATN 模型特征层的对抗性扰动。
    模型采用编码-解码架构，中间包含多个 Bottleneck 块。
    输出扰动通过 Tanh 激活函数并缩放到 [-epsilon, epsilon] 范围。
    """
    def __init__(self,
                 in_channels: int = 128, # 输入特征的通道数
                 out_channels: int = 128, # 输出扰动的通道数
                 num_bottlenecks: int = 4, # Bottleneck 块的数量
                 base_channels: int = 32, # 编码器和解码器中卷积层的基准通道数
                 epsilon: float = 0.03 # L-inf 范数约束上限
                ):
        """
        初始化生成器模型。

        Args:
            in_channels (int): 输入张量的通道数。在 Stage 1 中应与 ATN 特征通道数匹配。
            out_channels (int): 输出扰动张量的通道数。在 Stage 1 中应与 ATN 特征通道数匹配。
            num_bottlenecks (int): 在编码器和解码器之间的 Bottleneck 模块的数量。
            base_channels (int): 决定编码器和解码器各层通道数的基准值（例如，第一层为 base_channels * 4）。
            epsilon (float): 生成扰动的 L-infinity 范数上限。输出扰动将被裁剪到 [-epsilon, epsilon] 范围内。
        """
        super(Generator, self).__init__()
        # 扰动上限，用于 L-inf 约束
        self.epsilon: float = epsilon

        # 通道数：在 Stage 1 中用于匹配 ATN 特征层的输入输出通道数
        # assert in_channels == 128 and out_channels == 128 # 确保通道数符合预期

        # Encoder (编码器): 逐步减小空间维度 (H, W)，增加通道数，提取高级特征
        # (B, in_channels, N, H, W) -> (B, base_channels*16, N, H/4, W/4)
        self.encoder = nn.Sequential(
            # 第一个卷积层：stride=(1, 2, 2) 在空间维度减半
            nn.Conv3d(in_channels, base_channels * 4, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(base_channels * 4), # 批量归一化
            nn.ReLU(inplace=True), # ReLU 激活函数

            # 第二个卷积层：stride=(1, 2, 2) 在空间维度再次减半
            nn.Conv3d(base_channels * 4, base_channels * 8, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(base_channels * 8),
            nn.ReLU(inplace=True),

            # 第三个卷积层：stride=(1, 1, 1) 保持空间尺寸
            nn.Conv3d(base_channels * 8, base_channels * 16, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(base_channels * 16),
            nn.ReLU(inplace=True),
        )

        # Bottleneck Blocks (瓶颈块): 包含残差连接，用于深度特征转换，保持空间和序列维度不变
        # 输入通道数与编码器输出通道数一致
        encoder_out_channels = base_channels * 16
        current_bottleneck_input_channels = encoder_out_channels

        self.bottlenecks = nn.Sequential() # 使用 nn.Sequential 包含多个 Bottleneck 实例
        for i in range(num_bottlenecks):
            # 添加 Bottleneck3d 模块
            # Bottleneck3d 接受单个 channel 参数，内部会乘以 expansion factor (通常为 4)
            self.bottlenecks.add_module(
                f'bottleneck_{i}', # 模块名称，例如 bottleneck_0, bottleneck_1, ...
                Bottleneck3d(channel=current_bottleneck_input_channels, factor=Bottleneck3d.expansion, stride=1) # stride=1 保持空间尺寸
                # 注意：这里的 Bottleneck3d(channel=...) 似乎期望的是 Bottleneck3d 内部 bottleneck 的通道数，而不是整个块的输入通道数。
                # 根据 models/bottleneck3d.py 的定义，输入 channel 会被分成 channel, channel // factor * expansion
                # 这里的 channel 应该对应 Bottleneck3d 类定义中的 channel 参数。
                # 如果 Bottleneck3d 的实现是标准的 ResNet 风格，channel 应该是 Bottleneck 块中间层的通道数（原始论文中的 planes 参数）。
                # 根据 Bottleneck3d 的 __init__ 定义 `inplanes` 和 `planes`，这里的 `channel` 参数可能对应 `inplanes`。
                # 假设这里的 `channel` 参数是 Bottleneck 块的输入通道数，即 `current_bottleneck_input_channels`。
                # Bottleneck3d(inplanes, planes, ...), where planes = channel // factor
                # 根据 models/bottleneck3d.py: __init__(self, inplanes, planes, ...), conv1 uses inplanes, conv2 uses planes, conv3 uses planes * self.expansion
                # 并且 Bottleneck3d(channel=...) 调用了 super().__init__(channel, channel // factor, ...)
                # 这表明这里的 channel 参数对应的是 Bottleneck3d 中的 inplanes 和 planes 参数。这个接口设计有点特殊。
                # 假设 channel=current_bottleneck_input_channels 是正确的用法。
            )
            # 如果需要改变 Bottleneck 之间的通道数，这里需要更新 current_bottleneck_input_channels
            # 根据 models/bottleneck3d.py 的实现，Bottleneck3d 的输出通道是 planes * expansion。
            # 所以如果需要通道匹配，下一个 Bottleneck 的输入应该是上一个的输出。
            # current_bottleneck_input_channels = current_bottleneck_input_channels // Bottleneck3d.expansion * Bottleneck3d.expansion # 保持通道数不变
            # 但看起来原始代码中的循环并没有改变通道数，假设所有 Bottleneck 块的输入输出通道数相同。

        # Decoder (解码器): 逐步增大空间维度，减小通道数，重建扰动图
        # 输入通道数与最后一个 Bottleneck 块的输出通道数一致 (假设 Bottleneck 块保持通道数不变)
        decoder_input_channels = current_bottleneck_input_channels # 理论上应是最后一个 Bottleneck 的输出通道数
        # 假设 Bottleneck3d(channel=C) 的输出通道也是 C (如果 factor * expansion = 1, 这通常不是 Bottleneck 的设计)
        # 或者说，Bottleneck3d(channel=C, factor=4) 的输出通道是 (C//4)*4 = C (如果 C 是 4 的倍数)。是的，根据 Bottleneck3d 定义，inplanes=channel, planes=channel//factor, conv3_out = planes*expansion = (channel//factor)*expansion
        # 在 models/bottleneck3d.py 中，factor=4, expansion=4，所以 (channel//4)*4 = channel (假设 channel 是 4 的倍数)。
        # 所以 Bottleneck3d(channel) 输入输出通道数是 channel，这与标准 ResNet Bottleneck 块不同（标准 Bottleneck 的 plains * expansion 是其输出通道数，通常大于输入）。
        # 这里的实现可能期望 channel 参数就是输入通道数，且输出通道数与输入相同。

        # (B, decoder_input_channels, N, H/4, W/4) -> (B, out_channels, N, H, W)
        self.decoder = nn.Sequential(
            # 第一个反卷积层：stride=(1, 2, 2) 在空间维度加倍
            nn.ConvTranspose3d(decoder_input_channels, base_channels * 8, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), output_padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(base_channels * 8),
            nn.ReLU(inplace=True),

            # 第二个反卷积层：stride=(1, 2, 2) 在空间维度加倍
            nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True),

            # 第三个反卷积层：stride=(1, 1, 1) 保持空间尺寸
            # 输出通道数匹配期望的扰动通道数
            nn.ConvTranspose3d(base_channels * 4, out_channels, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Output Layer (输出层): 最终生成扰动，并进行 Tanh 激活和缩放
        decoder_out_channels = out_channels # 理论上与解码器最后一个反卷积层的输出通道数一致
        self.output_layer = nn.Sequential(
            # 1x1x1 卷积用于最终通道调整 (如果需要)
            nn.Conv3d(decoder_out_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.Tanh(), # Tanh 激活函数，将输出值范围映射到 [-1, 1]
            # Lambda 层用于将 Tanh 的输出乘以 epsilon，将扰动范围缩放到 [-epsilon, epsilon]
            Lambda(lambda x: x * self.epsilon) # 输出范围 [-epsilon, epsilon]
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor: # Type hint input and output tensors
        """
        前向传播函数。

        Args:
            x (torch.Tensor): 输入特征张量，形状为 (B, C, N, H, W)。
                              在 Stage 1 中，这是从 ATN 特征提取器获得的原始特征。

        Returns:
            torch.Tensor: 生成的对抗性扰动，形状与输入 x 相同 (B, C, N, H, W)，
                          值范围在 [-epsilon, epsilon] 内。
        """
        # 输入张量的尺寸，用于后续上采样以匹配尺寸
        input_size_n_h_w: Tuple[int, int, int] = x.size()[2:] # 获取 N, H, W 尺寸

        # 编码器前向传播
        encoded: torch.Tensor = self.encoder(x)
        # Debug 日志：Encoded shape: {}
        # print(f"Debug: Encoded shape: {encoded.shape}") # Debug print

        # Bottleneck 块前向传播
        bottleneck_out: torch.Tensor = self.bottlenecks(encoded)
        # Debug 日志：Bottleneck output shape: {}
        # print(f"Debug: Bottleneck output shape: {bottleneck_out.shape}") # Debug print

        # 解码器前向传播
        decoded: torch.Tensor = self.decoder(bottleneck_out)
        # Debug 日志：Decoded shape before adjustment: {}
        # print(f"Debug: Decoded shape before adjustment: {decoded.shape}") # Debug print

        # 尺寸匹配: 解码器的输出空间尺寸可能与输入不完全匹配，需要通过插值进行调整
        # 比较解码器输出的空间和序列尺寸与输入尺寸是否一致
        if decoded.shape[2:] != input_size_n_h_w:
            # 使用三线性插值 (trilinear) 对 5D 张量进行上采样
            # size 参数应该是一个 Tuple (N', H', W')
            # align_corners=False 通常是更推荐的选择
            adjusted_decoded: torch.Tensor = F.interpolate(decoded, size=input_size_n_h_w, mode='trilinear', align_corners=False) # 三线性插值
            # Debug 日志：Adjusted decoded shape after interpolation: {}
            # print(f"Debug: Adjusted decoded shape after interpolation: {adjusted_decoded.shape}") # Debug print
        else:
            # 如果尺寸已经匹配，则无需调整
            adjusted_decoded: torch.Tensor = decoded
            # Debug 日志：Decoded shape already matches input.
            # print("Debug: Decoded shape already matches input.") # Debug print

        # 最终扰动计算：通过输出层 (1x1x1 Conv -> Tanh -> Scale)
        # Debug 日志：Adjusted decoded shape before output layer: {}
        # print(f"Debug: Adjusted decoded shape before output layer: {adjusted_decoded.shape}") # Debug print
        delta: torch.Tensor = self.output_layer(adjusted_decoded)
        
        # 验证输出扰动的形状与输入特征形状是否一致
        assert delta.shape == x.shape, f"Output delta shape {delta.shape} does not match input shape {x.shape}"
        return delta