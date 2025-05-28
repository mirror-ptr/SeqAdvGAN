from torch import nn
import torch

class IrisActionDecoder(nn.Module):
    def __init__(self, input_dim=4096, action_dim=5):
        super(IrisActionDecoder, self).__init__()
        linear_list = []
        linear_list.append(nn.Linear(input_dim, 64))
        linear_list.append(nn.Linear(64, action_dim))
        self.projection = nn.Sequential(*linear_list)
        self.shortcut = nn.Linear(input_dim, action_dim) if input_dim != action_dim else nn.Identity()
        if isinstance(self.shortcut, nn.Linear):
            nn.init.eye_(self.shortcut.weight)

    def forward(self, x):
        # Logit ResNet
        x = self.projection(x) + self.shortcut(x)
        action_probs = torch.nn.functional.softmax(x, dim=-1)
        action_id = torch.argmax(action_probs, dim=-1)
        return action_id, action_probs