import torch

class ActionVocabulary:
    def __init__(self, num_targets=20):
        super(ActionVocabulary, self).__init__()
        self.action_embedding = torch.stack([torch.linspace(0, 1, 5)]*170, 0).permute(1, 0)
        self.targets_embedding = torch.stack([torch.linspace(0, 1, num_targets+1)]*170, 0).permute(1, 0)
        self.direction_embedding = torch.stack([torch.linspace(0, 1, 5)]*170, 0).permute(1, 0)

    def action_to_vector(self, action_idx, action_target, action_position, action_direction):
        device = action_idx.device
        embedding_action = torch.cat([self.action_embedding[action_idx.to(torch.device("cpu"))].squeeze(), self.targets_embedding[action_target.to(torch.device("cpu"))].squeeze(), action_position.to(torch.device("cpu")), self.direction_embedding[action_direction.to(torch.device("cpu"))].squeeze()])
        return embedding_action.to(device)