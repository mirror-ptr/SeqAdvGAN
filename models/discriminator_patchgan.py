import torch
import torch.nn as nn

class PatchDiscriminator3d(nn.Module):
    def __init__(self, in_channels=128, base_channels=64):
        super(PatchDiscriminator3d, self).__init__()
        # 3D PatchGAN 结构
        # 目标是将 (B, C, N, H, W) 输入映射到 (B, 1, patch_N, patch_H, patch_W)
        # 通过使用步长卷积在空间维度 (H, W) 和序列维度 (N) 上进行下采样

        # 初始层：可能只在空间下采样，保留序列长度
        # kernel_size=(seq_k, h_k, w_k), stride=(seq_s, h_s, w_s)
        self.layer1 = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 第二层：开始在序列和空间维度下采样
        self.layer2 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 第三层：进一步下采样
        self.layer3 = nn.Sequential(
            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=(3, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 第四层：可能只在空间下采样或不做下采样，准备输出
        # 这一层决定了 Patch 的最终空间大小
        self.layer4 = nn.Sequential(
            nn.Conv3d(base_channels * 4, base_channels * 8, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=False), # Stride (1,2,2) for spatial downsampling
            nn.BatchNorm3d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 输出层：将通道数变为 1，表示每个 patch 的真实性得分
        # kernel_size=(s_k, h_k, w_k), stride=1, padding=... depends on desired output patch size and previous layer output
        # 为了简单起见，我们使用 3x4x4 的核，步长为 1，padding 调整以匹配 Patch 大小
        self.output_layer = nn.Conv3d(base_channels * 8, 1, kernel_size=(3, 4, 4), stride=1, padding=(1, 1, 1))

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
# dummy_input = torch.randn(4, 128, 16, 64, 64) # (B, C, N, H, W)
# output = discriminator_patch(dummy_input)
# print(output.shape) # 期望形状为 (4, 1, patch_N, patch_H, patch_W) - 实际取决于卷积参数 