import torch
import torch.nn as nn
import torch.nn.functional as F

from models.bottleneck3d import Bottleneck3d
from utils.layers import Lambda

class Generator(nn.Module):
    def __init__(self, in_channels=128, out_channels=128, num_bottlenecks=4, base_channels=32, epsilon=0.03):
        super(Generator, self).__init__()
        # 扰动上限
        self.epsilon = epsilon

        # 通道数：for ATN
        # assert in_channels == 128 and out_channels == 128

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, base_channels * 4, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True),

            nn.Conv3d(base_channels * 4, base_channels * 8, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(base_channels * 8),
            nn.ReLU(inplace=True),

            nn.Conv3d(base_channels * 8, base_channels * 16, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(base_channels * 16),
            nn.ReLU(inplace=True),
        )

        # Bottleneck 
        encoder_out_channels = base_channels * 16
        current_bottleneck_input_channels = encoder_out_channels

        self.bottlenecks = nn.Sequential()
        for i in range(num_bottlenecks):
            self.bottlenecks.add_module(
                f'bottleneck_{i}',
                Bottleneck3d(channel=current_bottleneck_input_channels, factor=Bottleneck3d.expansion, stride=1)
            )

        # Decoder
        decoder_input_channels = current_bottleneck_input_channels #512
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(decoder_input_channels, base_channels * 8, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), output_padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(base_channels * 8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(base_channels * 4, out_channels, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

        # 输出层
        decoder_out_channels = out_channels # 128
        self.output_layer = nn.Sequential(
            nn.Conv3d(decoder_out_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.Tanh(), #[-1, 1]
            Lambda(lambda x: x * self.epsilon)#[-epsilon, epsilon]
        )


    def forward(self, x):
        # (B, 128, N, W, H)
        input_size_n_h_w = x.size()[2:]

        encoded = self.encoder(x)
        bottleneck_out = self.bottlenecks(encoded)
        decoded = self.decoder(bottleneck_out)

        # 尺寸匹配 
        if decoded.shape[2:] != input_size_n_h_w:
            adjusted_decoded = F.interpolate(decoded, size=input_size_n_h_w, mode='trilinear', align_corners=False) # 三线性插值
        else:
            adjusted_decoded = decoded

        # 最终扰动
        print(f"Debug: Adjusted decoded shape before output layer: {adjusted_decoded.shape}")
        delta = self.output_layer(adjusted_decoded)
        
        assert delta.shape == x.shape, f"Output delta shape {delta.shape} does not match input shape {x.shape}"
        return delta