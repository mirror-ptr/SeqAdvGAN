import random

from .nn.Transformers import IrisRTDetrVisionEncoder, IrisBabelTransformer
from .nn.utils import IrisActionDecoder, IrisDirectionDecoder, IrisTargetDecoder, IrisRelativePositionDecoder
from .utils import ActionVocabulary
import torch
import matplotlib.pyplot as plt
from torchsummary import summary

class IrisBabelModel:
    def __init__(self, perception_layer_type="rtdetr", memory_length=100, num_mid_dim=512, num_actions=5, num_targets=20):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory_length = memory_length
        self.perception_layer_type = perception_layer_type
        self.num_mid_dim = num_mid_dim
        self.num_actions = num_actions
        self.num_targets = num_targets
        if self.perception_layer_type == "rtdetr":
            self.perception_layer = IrisRTDetrVisionEncoder(output_dim=num_mid_dim, device=self.device)
            self.perception_layer.to(self.device)
        else:
            raise NotImplementedError("Unsupported perception layer type: {}".format(self.perception_layer_type))
        self.babel_transformer = IrisBabelTransformer(input_dim=num_mid_dim)
        self.babel_transformer.to(self.device)
        self.action_decoder = IrisActionDecoder(input_dim=num_mid_dim, action_dim=num_actions)
        self.action_decoder.to(self.device)
        self.target_decoder = IrisTargetDecoder(input_dim=num_mid_dim, num_max_targets=num_targets)
        self.target_decoder.to(self.device)
        self.rel_position_decoder = IrisRelativePositionDecoder(input_dim=num_mid_dim)
        self.rel_position_decoder.to(self.device)
        self.direction_decoder = IrisDirectionDecoder(input_dim=num_mid_dim)
        self.direction_decoder.to(self.device)
        self.feature_memory = torch.zeros(1, 3, self.num_mid_dim).to(self.device)
        self.action_memory = torch.zeros(1, 3, self.num_mid_dim).to(self.device)
        self.backbone_optimizer = None
        self.backbone_learning_rate = 1e-3
        self.backbone_mse_function = torch.nn.MSELoss()
        self.backbone_cross_entropy_function = torch.nn.CrossEntropyLoss()
        self.perception_optimizer = None
        self.perception_learning_rate = 1e-3
        self.perception_loss_function = None
        self.action_vocabulary = ActionVocabulary()
        self.lock_feature_memory = False
        self.lock_action_memory = False

    def reset_memory(self):
        self.feature_memory = torch.zeros(1, 3, self.num_mid_dim).to(self.device)
        self.action_memory = torch.zeros(1, 3, self.num_mid_dim).to(self.device)

    def push_feature_memory(self, feature):
        if self.lock_feature_memory:
            return
        feature = feature.detach().view(1, 1, self.num_mid_dim)
        self.feature_memory = torch.cat((self.feature_memory, feature), dim=1)
        if self.feature_memory.size(1) > self.memory_length:
            temp_prev_memory = self.feature_memory[:, :(self.feature_memory.size(1) - self.memory_length)]
            temp_prev_memory_pooled = torch.mean(temp_prev_memory, dim=1).view(1, 1, self.num_mid_dim)
            temp_next_memory = self.feature_memory[:, (self.feature_memory.size(1) - self.memory_length):]
            self.feature_memory = torch.cat((temp_prev_memory_pooled, temp_next_memory), dim=1)

    def push_action_memory(self, action):
        if self.lock_action_memory:
            return
        action = action.detach().view(1, 1, self.num_mid_dim)
        self.action_memory = torch.cat((self.action_memory, action), dim=1)
        if self.action_memory.size(1) > self.memory_length:
            temp_prev_memory = self.action_memory[:, :(self.action_memory.size(1) - self.memory_length)]
            temp_prev_memory_pooled = torch.mean(temp_prev_memory, dim=1).view(1, 1, self.num_mid_dim)
            temp_next_memory = self.action_memory[:, (self.action_memory.size(1) - self.memory_length):]
            self.action_memory = torch.cat((temp_prev_memory_pooled, temp_next_memory), dim=1)

    def pred(self, x):
        feature = self.perception_layer(x)
        self.push_feature_memory(feature)
        pred = self.babel_transformer(self.feature_memory, self.action_memory)
        action_idx, _ = self.action_decoder(pred[:, -1])
        action_target, _ = self.target_decoder(pred[:, -1])
        action_position = self.rel_position_decoder(pred[:, -1])
        action_direction, _ = self.direction_decoder(pred[:, -1])
        action_embedding = self.action_vocabulary.action_to_vector(action_idx, action_target, action_position, action_direction)
        self.push_action_memory(action_embedding)
        return action_idx.to("cpu").item(), action_target.to("cpu").item(), action_position.to("cpu"), action_direction.to("cpu").item()

    def train_backbone_start(self, optimizer="AdamW", lr=0.001):
        self.reset_memory()
        if optimizer == "AdamW":
            self.backbone_optimizer = torch.optim.AdamW(self.babel_transformer.parameters(), lr=lr)
        else:
            raise NotImplementedError("Unsupported optimizer: {}".format(optimizer))
        self.backbone_learning_rate = lr
        torch.set_grad_enabled(True)

    def train_perception_start(self, optimizer="AdamW", lr=0.001, loss_function="CrossEntropyLoss"):
        pass

    def train_stop(self):
        self.reset_memory()
        torch.set_grad_enabled(False)

    # Shape of x: [B, ..., ..., ...]
    # Shape of label: [B, D]
    def train_backbone_step(self, x):
        feature = self.perception_layer(x)
        self.push_feature_memory(feature)
        pred = self.babel_transformer(self.feature_memory, self.action_memory)
        action_idx, action_idx_probs = self.action_decoder(pred[:, -1])
        action_target, action_target_probs = self.target_decoder(pred[:, -1])
        action_position = self.rel_position_decoder(pred[:, -1])
        action_direction, action_direction_probs = self.direction_decoder(pred[:, -1])
        action_embedding = self.action_vocabulary.action_to_vector(action_idx, action_target, action_position, action_direction)
        self.push_action_memory(action_embedding)
        return action_idx, action_target, action_position, action_direction, action_idx_probs, action_target_probs, action_direction_probs

    def backbone(self, x):
        return self.train_backbone_step(x)

    @staticmethod
    def plot_gradient_flow(model, title="Gradient Flow Analysis"):
        gradients = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                gradients.append((name, grad_norm))

        plt.figure(figsize=(10, 6))
        plt.barh([n for n, _ in gradients], [g for _, g in gradients])
        plt.title(title)
        plt.xlabel("Gradient L2 Norm")
        plt.show()

    def zero_grad(self):
        self.backbone_optimizer.zero_grad()

    def backbone_optimizer_step(self):
        self.backbone_optimizer.step()

    def calculate_loss(self, action_idx, action_idx_probs, action_target_probs, action_position_probs, action_direction_probs, label):
        loss = None

        label_action_probs = torch.zeros_like(action_idx_probs)
        label_action_probs[0][int(label[0, 0].item())] = 1.0
        action_idx_loss = self.backbone_cross_entropy_function(action_idx_probs, label_action_probs)
        return action_idx_loss

        if action_idx != label[:, 0]:
            label_action_probs = torch.zeros_like(action_idx_probs)
            label_action_probs[0][int(label[0, 0].item())] = 1.0
            action_idx_loss = self.backbone_cross_entropy_function(action_idx_probs, label_action_probs)
            rnd = random.randint(0, 15)
            if label[:, 0] != 0 or (label[:, 0] == 0 and rnd == 0):
                self.backbone_optimizer.zero_grad()
                action_idx_loss.backward()
                self.backbone_optimizer.step()
            loss = action_idx_loss.item()
        elif action_idx == 0:
            label_action_probs = torch.zeros_like(action_idx_probs)
            label_action_probs[0][int(label[0, 0].item())] = 1.0
            action_idx_loss = self.backbone_cross_entropy_function(action_idx_probs, label_action_probs)
            # self.backbone_optimizer.zero_grad()
            # action_idx_loss.backward()
            # self.backbone_optimizer.step()
            loss = action_idx_loss.item()
        elif action_idx == 1:   # Deploy
            label_target_probs = torch.zeros_like(action_target_probs)
            label_target_probs[0][int(label[0, 1].item())] = 1.0
            label_direction_probs = torch.zeros_like(action_direction_probs)
            label_direction_probs[0][int(label[0, 4].item())] = 1.0
            action_target_loss = self.backbone_cross_entropy_function(action_target_probs, label_target_probs)
            action_position_loss = self.backbone_mse_function(action_position_probs.to(self.device), torch.tensor([label[:, 2], label[:, 3]]).to(self.device))
            action_direction_loss = self.backbone_cross_entropy_function(action_direction_probs, label_direction_probs)
            loss_factor = [0.4, 0.3, 0.3]
            action_loss_loss = action_target_loss * loss_factor[0] + action_position_loss * loss_factor[1] + action_direction_loss * loss_factor[2]
            self.backbone_optimizer.zero_grad()
            action_loss_loss.backward()
            self.backbone_optimizer.step()
            loss = action_loss_loss.item()
        elif action_idx == 2:   # Withdraw
            label_target_probs = torch.zeros_like(action_target_probs)
            label_target_probs[0][int(label[0, 1].item())] = 1.0
            action_target_loss = self.backbone_cross_entropy_function(action_target_probs, label_target_probs)
            self.backbone_optimizer.zero_grad()
            action_target_loss.backward()
            self.backbone_optimizer.step()
            loss = action_target_loss.item()
        elif action_idx == 3:   # Activate Skill
            label_target_probs = torch.zeros_like(action_target_probs)
            label_target_probs[0][int(label[0, 1].item())] = 1.0
            action_target_loss = self.backbone_cross_entropy_function(action_target_probs, label_target_probs)
            self.backbone_optimizer.zero_grad()
            action_target_loss.backward()
            self.backbone_optimizer.step()
            loss = action_target_loss.item()
        elif action_idx == 4:   # Deactivate Skill
            label_target_probs = torch.zeros_like(action_target_probs)
            label_target_probs[0][int(label[0, 1].item())] = 1.0
            action_target_loss = self.backbone_cross_entropy_function(action_target_probs, label_target_probs)
            self.backbone_optimizer.zero_grad()
            action_target_loss.backward()
            self.backbone_optimizer.step()
            loss = action_target_loss.item()
        # self.plot_gradient_flow(self.babel_transformer)
        return loss

    def calculate_loss_num(self, action_idx, action_idx_probs, action_target_probs, action_position_probs, action_direction_probs, label):
        loss = None
        if action_idx != label[:, 0]:
            label_action_probs = torch.zeros_like(action_idx_probs)
            label_action_probs[0][int(label[0, 0].item())] = 1.0
            action_idx_loss = self.backbone_cross_entropy_function(action_idx_probs, label_action_probs)
            loss = action_idx_loss.item()
        elif action_idx == 0:
            label_action_probs = torch.zeros_like(action_idx_probs)
            label_action_probs[0][int(label[0, 0].item())] = 1.0
            action_idx_loss = self.backbone_cross_entropy_function(action_idx_probs, label_action_probs)
            loss = action_idx_loss.item()
        elif action_idx == 1:  # Deploy
            label_target_probs = torch.zeros_like(action_target_probs)
            label_target_probs[0][int(label[0, 1].item())] = 1.0
            label_direction_probs = torch.zeros_like(action_direction_probs)
            label_direction_probs[0][int(label[0, 4].item())] = 1.0
            action_target_loss = self.backbone_cross_entropy_function(action_target_probs, label_target_probs)
            action_position_loss = self.backbone_mse_function(action_position_probs.to(self.device),
                                                              torch.tensor([label[:, 2], label[:, 3]]).to(self.device))
            action_direction_loss = self.backbone_cross_entropy_function(action_direction_probs, label_direction_probs)
            loss_factor = [0.4, 0.3, 0.3]
            action_loss_loss = action_target_loss * loss_factor[0] + action_position_loss * loss_factor[
                1] + action_direction_loss * loss_factor[2]
            loss = action_loss_loss.item()
        elif action_idx == 2:  # Withdraw
            label_target_probs = torch.zeros_like(action_target_probs)
            label_target_probs[0][int(label[0, 1].item())] = 1.0
            action_target_loss = self.backbone_cross_entropy_function(action_target_probs, label_target_probs)
            loss = action_target_loss.item()
        elif action_idx == 3:  # Activate Skill
            label_target_probs = torch.zeros_like(action_target_probs)
            label_target_probs[0][int(label[0, 1].item())] = 1.0
            action_target_loss = self.backbone_cross_entropy_function(action_target_probs, label_target_probs)
            loss = action_target_loss.item()
        elif action_idx == 4:  # Deactivate Skill
            label_target_probs = torch.zeros_like(action_target_probs)
            label_target_probs[0][int(label[0, 1].item())] = 1.0
            action_target_loss = self.backbone_cross_entropy_function(action_target_probs, label_target_probs)
            loss = action_target_loss.item()
        return loss

    def lock_memory(self):
        self.lock_action_memory = True
        self.lock_feature_memory = True

    def unlock_memory(self):
        self.lock_action_memory = False
        self.lock_feature_memory = False

    def summary(self):
        print("IrisRTDetrVisionEncoder:")
        print(self.perception_layer)
        print("IrisBabelTransformer:")
        print(self.babel_transformer)
        print("ActionDecoder:")
        print(self.action_decoder)
        print("TargetDecoder:")
        print(self.target_decoder)
        print("RelativePositionDecoder:")
        print(self.rel_position_decoder)
        print("DirectionDecoder:")
        print(self.direction_decoder)

    def snapshot(self, snapshot_name="", snapshot_backbone=True, snapshot_perception=True):
        pass

    def export_model(self, export_path="model", snapshot_backbone=True, snapshot_perception=True, separate_export=True):
        pass

    def load_model(self, model_path):
        pass

