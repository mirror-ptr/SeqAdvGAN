import torch
import torch.nn as nn

class SequenceDiscriminatorCNN(nn.Module):
    """
    基于 3D 卷积网络的序列判别器模型。
    设计用于区分真实序列特征和生成器生成的伪造序列特征。
    采用一系列 3D 卷积层提取特征，并通过一个分类器预测输入是真实的还是伪造的。
    """
    def __init__(self,
                 in_channels: int = 128, # 输入特征的通道数
                 base_channels: int = 64 # 判别器中卷积层的基准通道数
                ):
        """
        初始化序列判别器模型。

        Args:
            in_channels (int): 输入张量的通道数。在 Stage 1 中应与 ATN 特征通道数匹配。
            base_channels (int): 决定卷积各层通道数的基准值（例如，第一层为 base_channels）。
        """
        super(SequenceDiscriminatorCNN, self).__init__()

        # Features (特征提取层): 一系列 3D 卷积层，用于从输入序列中提取判别特征
        # 输入形状通常为 (B, C, N, H, W)，其中 C 是 in_channels
        self.features = nn.Sequential(
            # 第一个卷积层：stride=(1, 2, 2) 在空间维度减半
            nn.Conv3d(in_channels, base_channels, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(base_channels), # 批量归一化
            nn.LeakyReLU(0.2, inplace=True), # LeakyReLU 激活函数，常用于 GAN 判别器

            # 第二个卷积层：stride=(1, 2, 2) 在空间维度再次减半
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 第三个卷积层：stride=(1, 2, 2) 在空间维度减半
            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # 第四个卷积层：stride=(1, 2, 2) 在空间维度减半
            nn.Conv3d(base_channels * 4, base_channels * 8, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # AdaptiveAvgPool3d: 将特征图的空间和序列维度缩减到 1x1x1
            nn.AdaptiveAvgPool3d(output_size=(1, 1, 1)),
            
        )

        # 计算经过特征提取后展平的通道数
        final_channels_after_conv: int = base_channels * 8 

        # Classifier (分类器): 展平特征并通过全连接层输出判别结果
        self.classifier = nn.Sequential(
            nn.Flatten(), # 展平 (B, C', 1, 1, 1) 到 (B, C')
            nn.Linear(final_channels_after_conv, 1), # 全连接层，输出一个标量（判断真假的得分）
            nn.Sigmoid() # Sigmoid 激活函数，将得分映射到 [0, 1] 范围内的概率
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor: # Type hint input and output tensors
        """
        前向传播函数。

        Args:
            x (torch.Tensor): 输入特征张量，形状为 (B, C, N, H, W)。
                              在 Stage 1 中，这是从 ATN 特征提取器获得的原始特征或生成器产生的对抗特征。

        Returns:
            torch.Tensor: 判别器对输入的判断结果，形状为 (B, 1)。
                          Sigmoid 激活后，值在 [0, 1] 范围，接近 1 表示判断为真，接近 0 表示判断为假。
        """
        # 提取特征
        features: torch.Tensor = self.features(x)
        # Debug 日志：Features shape after conv layers: {}
        # print(f"Debug: Features shape after conv layers: {features.shape}") # Debug print

        # 分类预测
        prediction: torch.Tensor = self.classifier(features)
        # Debug 日志：Prediction shape after classifier: {}
        # print(f"Debug: Prediction shape after classifier: {prediction.shape}") # Debug print

        return prediction
