from torch import nn

class IrisRelativePositionDecoder(nn.Module):
    def __init__(self, input_dim=4096):
        super(IrisRelativePositionDecoder, self).__init__()
        self.projection = nn.Linear(input_dim, 2)
        self.sigmoid = nn.Sigmoid()
        self.shortcut = nn.Linear(input_dim, 2) if input_dim != 2 else nn.Identity()
        if isinstance(self.shortcut, nn.Linear):
            nn.init.eye_(self.shortcut.weight)

    def forward(self, x):
        mid = self.projection (x) + self.shortcut(x)
        output = self.sigmoid(mid)
        return output.cpu().view(2)