from torchvision.models import mobilenet_v3_small
from torch import nn

class IrisMobileNet(nn.Module):
    def __init__(self, input_channels=1024, output_stride=8):
        super().__init__()
        self.model = mobilenet_v3_small(pretrained=True)
        # Modify the input channel of the first conv layer
        self.model.features[0][0] = nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1)
        # Control the minimum of the sample rate with change the stride
        if output_stride == 16:
            self.model.features[3].block[0][0].stride = (1, 1)

    def forward(self, x):
        x = self.model.features(x)
        return x