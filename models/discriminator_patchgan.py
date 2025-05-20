import torch
import torch.nn as nn

class PatchDiscriminator3d(nn.Module):
    def __init__(self, in_channels=128, base_channels=64):
        super(PatchDiscriminator3d, self).__init__()
        # 3D PatchGAN 结构
        # 目标是将 (B, C, N, H, W) 输入映射到 (B, 1, patch_N, patch_H, patch_W)
        # 根据ATN输出的特征尺寸(B, 128, 5, 3, 3)调整网络结构

        # layer1: 空间卷积，不改变空间尺寸，不改变序列长度
        # kernel_size=(seq_k, h_k, w_k), stride=(seq_s, h_s, w_s)
        self.layer1 = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),
            nn.InstanceNorm3d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
        ) # Output: (B, base_channels, 5, 3, 3)

        # layer2: 序列下采样，空间卷积不改变尺寸
        self.layer2 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1), bias=False),
            nn.InstanceNorm3d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
        ) # Output: (B, base_channels*2, 3, 3, 3)

        # layer3: 序列下采样，空间卷积不改变尺寸
        self.layer3 = nn.Sequential(
            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1), bias=False),
            nn.InstanceNorm3d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
        ) # Output: (B, base_channels*4, 2, 3, 3)

        # layer4: 序列下采样，空间卷积不改变尺寸
        self.layer4 = nn.Sequential(
            nn.Conv3d(base_channels * 4, base_channels * 8, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
        ) # Output: (B, base_channels*8, 1, 3, 3)

        # output_layer: 将通道数变为 1，空间和序列尺寸保持
        self.output_layer = nn.Conv3d(base_channels * 8, 1, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)) # Output: (B, 1, 1, 3, 3)

        # 注意：PatchGAN 通常在损失函数中直接使用原始输出 (logits)，而不是在这里加 Sigmoid。
        # 但为了与 gan_losses.py 中的 BCELoss 兼容，暂时在这里添加 Sigmoid。
        # 如果后面切换到 BCEWithLogitsLoss，可以移除 Sigmoid。
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (B, C, N, H, W)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # output_layer 会将通道数变为 1
        # 空间和序列维度会根据卷积参数变化
        patch_output = self.output_layer(out)

        # 应用 Sigmoid 得到概率值 (B, 1, patch_N, patch_H, patch_W)
        prediction = self.sigmoid(patch_output)

        return prediction

# 示例用法:
# discriminator_patch = PatchDiscriminator3d(in_channels=128)
# dummy_input = torch.randn(4, 128, 5, 3, 3) # (B, C, N, H, W) based on ATN output
# output = discriminator_patch(dummy_input)
# print(output.shape) # 期望形状为 (B, 1, 1, 3, 3) 