from torch import nn
import torch
import numpy as np

class IrisTargetDecoder(nn.Module):
    def __init__(self, input_dim=4096, num_max_targets = 20, temperature=1.0):
        super(IrisTargetDecoder, self).__init__()
        linear_list = []
        linear_list.append(nn.Linear(input_dim, 64))
        linear_list.append(nn.Linear(64, num_max_targets))
        self.projection = nn.Sequential(*linear_list)
        self.temperature = temperature
        self.shortcut = nn.Linear(input_dim, num_max_targets) if input_dim != num_max_targets else nn.Identity()
        if isinstance(self.shortcut, nn.Linear):
            nn.init.eye_(self.shortcut.weight)

    def forward(self, x, target_mask=None):
        if target_mask is not None:
            x = torch.masked_fill(x, target_mask, np.float32('-inf'))
        output = self.projection(x) + self.shortcut(x)
        target_probs = torch.nn.functional.softmax(output, dim=-1)
        target_id = torch.argmax(output, dim=-1)
        return target_id, target_probs

