import torch
import torch.nn as nn

class SequenceDiscriminatorCNN(nn.Module):
    def __init__(self, in_channels=128, base_channels=64):
        super(SequenceDiscriminatorCNN, self).__init__()

        # 卷积层
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(base_channels),
            nn.LeakyReLU(0.2, inplace=True), # GAN 的判别器常用 LeakyReLU

            nn.Conv3d(base_channels, base_channels * 2, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(base_channels * 4, base_channels * 8, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool3d(output_size=(1, 1, 1)),
            
        )

        final_channels_after_conv = base_channels * 8 

        # 展平层和全连接
        self.classifier = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(final_channels_after_conv, 1), 
            nn.Sigmoid() # 输出概率
        )


    def forward(self, x):
        # 提取特征
        features = self.features(x)

        # 分类预测
        prediction = self.classifier(features)

        return prediction
