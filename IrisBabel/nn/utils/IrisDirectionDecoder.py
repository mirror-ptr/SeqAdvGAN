from torch import nn
import torch

class IrisDirectionDecoder(nn.Module):
    def __init__(self, input_dim=4096):
        super(IrisDirectionDecoder, self).__init__()
        linear_list = []
        linear_list.append(nn.Linear(input_dim, 64))
        linear_list.append(nn.Linear(64, 4))
        self.projection = nn.Sequential(*linear_list)
        self.shortcut = nn.Linear(input_dim, 4) if input_dim != 4 else nn.Identity()
        if isinstance(self.shortcut, nn.Linear):
            nn.init.eye_(self.shortcut.weight)

    def forward(self, x):
        x = self.projection(x) + self.shortcut(x)
        direction_probs = torch.nn.functional.softmax(x, dim=-1)
        direction_id = torch.argmax(direction_probs, dim=-1)
        return direction_id, direction_probs