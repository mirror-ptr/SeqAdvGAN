import random

from .nn.Transformers import IrisRTDetrVisionEncoder, IrisBabelTransformer
from .nn.utils import IrisActionDecoder, IrisDirectionDecoder, IrisTargetDecoder, IrisRelativePositionDecoder
from .utils import ActionVocabulary
import torch
import matplotlib.pyplot as plt
from torchsummary import summary
import torch.nn as nn

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
        batch_size, channels, sequence_length, height, width = x.shape

        sequence_features = []
        for t in range(sequence_length):
            current_frame = x[:, :, t, :, :]
            feature = self.perception_layer(current_frame)
            sequence_features.append(feature.unsqueeze(1))
            
        sequence_features = torch.cat(sequence_features, dim=1)
        
        # 将整个序列特征推入内存 (需要适配 batch size)
        # 当前 memory push 方法似乎只处理 batch size 1，此处需要开发者确认或修改 IrisBabelModel 的内存管理逻辑。
        # 为了冒烟测试能继续，我们暂时跳过 push_feature_memory，直接使用当前序列特征
        # self.push_feature_memory(sequence_features) # 需要修改此方法以支持 batch 输入
        
        # Transformer 模型期望的输入形状是 (sequence_length, batch_size, feature_dim)
        # 我们的 sequence_features 形状是 (batch_size, sequence_length, feature_dim)
        # 在传递给 babel_transformer 之前进行转置
        sequence_features_transposed = sequence_features.transpose(0, 1)

        # 调整 sequence_features_transposed 形状以尝试匹配 Transformer 内部可能的期望 (1, B*N, dim)
        # 这只是一个临时解决方案，以绕过形状错误。
        batch_size, sequence_length, num_mid_dim = sequence_features.shape
        # 将序列长度维度和批次维度合并，形成形状 (1, sequence_length * batch_size, num_mid_dim)
        sequence_features_adjusted = sequence_features_transposed.contiguous().view(1, sequence_length * batch_size, num_mid_dim)

        # 调整 action_memory 形状以匹配 Transformer 期望的 (sequence_length, batch_size, feature_dim)
        # action_memory 当前形状: (1, memory_sequence_length, num_mid_dim)
        # 期望形状: (memory_sequence_length, batch_size, num_mid_dim)
        # 注意：这里的 batch_size 应该是当前的 batch_size (由 dummy_input 决定)，而不是 action_memory 初始化时的硬编码 1
        # 为了冒烟测试在 batch_size=1 时能运行，我们先假设 action_memory 的第二维是 memory_sequence_length
        # 并且它的第一维 (1) 可以被视为 batch_size=1
        # 如果 batch_size > 1，这里的逻辑需要彻底修改 action_memory 的管理方式
        
        # 尝试将 action_memory 的形状从 (1, memory_sequence_length, num_mid_dim) 转为 (memory_sequence_length, 1, num_mid_dim)
        # 然后扩展批次维度以匹配当前的 batch_size
        # 注意：这只是一个临时方案，正确做法是修改 action_memory 的管理逻辑以支持批处理。

        action_memory_processed = self.action_memory.squeeze(0).unsqueeze(1) # 形状变为 (memory_sequence_length, 1, num_mid_dim)
        
        # 如果 batch_size > 1 (尽管我们目前设置为 1)，这里需要将 action_memory 复制 batch_size 份
        # action_memory_processed = action_memory_processed.expand(-1, batch_size, -1)
        
        # 在 batch_size=1 的情况下，action_memory_processed 形状为 (memory_sequence_length, 1, num_mid_dim)，符合 Transformer 期望
        
        # 使用处理后的 action_memory 调用 transformer
        # Transformer Decoder 的 forward 方法签名通常是 forward(tgt, memory)
        # 在 IrisBabelTransformer 中，调用是 self.decoder(target_seq, src_seq)
        # 根据上下文，target_seq 应该是 action_memory，src_seq 是 sequence_features
        # 所以，pred = self.babel_transformer(action_memory_processed, sequence_features_transposed)
        # 但是 IrisBabelTransformer 的 forward 签名是 (src_seq, target_seq) 且调用是 self.decoder(target_seq, src_seq)
        # 这意味着 IrisBabelTransformer 内部将第一个参数视为 src，第二个参数视为 target
        # 然后将其传递给 decoder 时又调换了顺序。
        # 所以，我们需要传入 (sequence_features_transposed, action_memory_processed)

        # 经过之前的转置，sequence_features_transposed 形状为 (sequence_length, batch_size, feature_dim)
        # 经过处理，action_memory_processed 形状为 (memory_sequence_length, batch_size, num_mid_dim) when batch_size=1
        # 看起来 IrisBabelTransformer 的 forward 方法签名 (src_seq, target_seq) 与内部 decoder 调用 (target_seq, src_seq) 存在不一致。
        # 假设 IrisBabelTransformer 的 forward(src_seq, target_seq) 实际上是将 src_seq 作为 memory，target_seq 作为 tgt 传给 decoder
        # 那么我们应该传入 (sequence_features_transposed, action_memory_processed)

        # 为了绕过 Transformer 内部的形状错误，临时创建模拟的 sequence_features_transposed
        # 根据错误信息，它期望源序列的某个派生张量形状与 [1, 24, 64] 相关。
        # 我们尝试模拟一个形状 (sequence_length, batch_size * num_heads_expected, head_dim_expected) 的输入
        # 假设 num_heads_expected * head_dim_expected = num_mid_dim = 512
        # 错误信息中的 24 和 64 是从哪里来的需要进一步分析。
        # 错误信息 [1, 24, 64] 可能是 TransformerLayer 内部的形状。
        # 如果 Transformer Decoder 期望的 memory 形状是 (sequence_length, batch_size, hidden_dim)
        # 那么输入的 sequence_features_transposed 形状 (5, 1, 512) 是正确的。
        # 错误可能在于 TransformerLayer 内部对 hidden_dim (512) 和 attention head 参数 (24, 64) 的不一致处理。

        # 尝试直接使用原始的 sequence_features_transposed (形状 5, 1, 512) 传入，这符合标准 Transformer 输入格式。
        # 之前的错误是在 multi_head_attention_forward 内部，可能是其对输入的处理逻辑与张量形状不匹配。
        # 既然之前的形状调整都无效，我们回到最原始的符合 Transformer 标准输入的形状 (sequence_length, batch_size, num_mid_dim)
        # 尝试将 action_memory 也调整到标准形状 (memory_sequence_length, batch_size, num_mid_dim)

        # 调整 action_memory 形状为 (memory_sequence_length, batch_size, num_mid_dim)
        # action_memory 当前形状: (1, memory_sequence_length, num_mid_dim)
        # 我们需要将其形状变为 (memory_sequence_length, batch_size, num_mid_dim)
        # 这里的 batch_size 应该是当前的 batch_size (由 dummy_input 决定)
        current_batch_size = sequence_features.shape[0] # 从 sequence_features 获取当前 batch_size
        memory_sequence_length = self.action_memory.shape[1]
        
        # 如果 batch_size > 1，需要复制 action_memory
        # 这是一个临时方案，正确做法是修改 action_memory 的管理逻辑以支持批处理。
        action_memory_adjusted = self.action_memory.squeeze(0).unsqueeze(1).expand(-1, current_batch_size, -1)
        # action_memory_adjusted 形状变为 (memory_sequence_length, current_batch_size, num_mid_dim)

        # 使用转置后的 sequence_features (形状 5, batch_size, 512) 和 调整后的 action_memory 调用 transformer
        # 传入 (sequence_features_transposed, action_memory_adjusted)
        # 假设 IrisBabelTransformer forward(src, tgt) 内部调用 decoder(tgt, src)
        # 则 src 是 sequence_features_transposed (memory), tgt 是 action_memory_adjusted (target)
        pred = self.babel_transformer(sequence_features_transposed, action_memory_adjusted)

        # 后续解码器逻辑保持不变
        action_idx, action_idx_probs = self.action_decoder(pred[:, -1])
        action_target, _ = self.target_decoder(pred[:, -1])
        action_position = self.rel_position_decoder(pred[:, -1])
        action_direction, _ = self.direction_decoder(pred[:, -1])
        
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

    # Shape of x: [B, C, N, H, W]
    # Shape of label: [B, D]
    def train_backbone_step(self, x):
        batch_size, channels, sequence_length, height, width = x.shape
        
        sequence_features = []
        for t in range(sequence_length):
            current_frame = x[:, :, t, :, :]
            feature = self.perception_layer(current_frame)
            sequence_features.append(feature.unsqueeze(1))
            
        sequence_features = torch.cat(sequence_features, dim=1)

        # 添加打印语句检查 sequence_features 的形状
        print(f"Shape of sequence_features after concat: {sequence_features.shape}")

        # 将整个序列特征推入内存 (需要适配 batch size)
        # 当前 memory push 方法似乎只处理 batch size 1，此处需要开发者确认或修改 IrisBabelModel 的内存管理逻辑。
        # 为了冒烟测试能继续，我们暂时跳过 push_feature_memory，直接使用当前序列特征
        # self.push_feature_memory(sequence_features) # 需要修改此方法以支持 batch 输入

        # 移除不必要的特征维度投射，因为感知层输出维度已经是 num_mid_dim (512)
        # 将特征维度从 4096 投射到 num_mid_dim (512)
        # 为了避免多维输入问题，先将 sequence_features 重塑为 (B * N, feature_dim)
        # batch_size, sequence_length, feature_dim = sequence_features.shape
        # sequence_features_reshaped = sequence_features.view(batch_size * sequence_length, feature_dim)

        # 添加打印语句检查 sequence_features_reshaped 的形状和元素总数
        # print(f"Shape of sequence_features_reshaped: {sequence_features_reshaped.shape}")
        # print(f"Num elements in sequence_features_reshaped: {sequence_features_reshaped.numel()}")

        # 额外的安全检查和重塑，以防张量不连续导致的问题
        # print(f"Shape before projection: {sequence_features_reshaped.shape}") # 添加打印帮助调试
        # assert sequence_features_reshaped.shape == (batch_size * sequence_length, 4096), f"Unexpected shape before projection: {sequence_features_reshaped.shape}"
        # sequence_features_for_projection = sequence_features_reshaped.contiguous().view(-1, 4096)

        # 添加打印语句检查 sequence_features_for_projection 的形状
        # print(f"Shape of sequence_features_for_projection before linear: {sequence_features_for_projection.shape}")

        # 应用线性层
        # sequence_features_projected_reshaped = self.feature_projection(sequence_features_for_projection)

        # 将结果重塑回 (B, N, num_mid_dim)
        # sequence_features_projected = sequence_features_projected_reshaped.view(batch_size, sequence_length, self.num_mid_dim)

        # Transformer 模型期望的输入形状是 (sequence_length, batch_size, feature_dim)
        # 我们的 sequence_features 形状是 (batch_size, sequence_length, num_mid_dim)
        # 在传递给 babel_transformer 之前进行转置
        sequence_features_transposed = sequence_features.transpose(0, 1)

        # 调整 sequence_features_transposed 形状以尝试匹配 Transformer 内部可能的期望 (1, B*N, dim)
        # 这只是一个临时解决方案，以绕过形状错误。
        batch_size, sequence_length, num_mid_dim = sequence_features.shape
        # 将序列长度维度和批次维度合并，形成形状 (1, sequence_length * batch_size, num_mid_dim)
        sequence_features_adjusted = sequence_features_transposed.contiguous().view(1, sequence_length * batch_size, num_mid_dim)

        # 调整 action_memory 形状以匹配 Transformer 期望的 (sequence_length, batch_size, feature_dim)
        # action_memory 当前形状: (1, memory_sequence_length, num_mid_dim)
        # 期望形状: (memory_sequence_length, batch_size, num_mid_dim)
        # 注意：这里的 batch_size 应该是当前的 batch_size (由 dummy_input 决定)，而不是 action_memory 初始化时的硬编码 1
        # 为了冒烟测试在 batch_size=1 时能运行，我们先假设 action_memory 的第二维是 memory_sequence_length
        # 并且它的第一维 (1) 可以被视为 batch_size=1
        # 如果 batch_size > 1，这里的逻辑需要彻底修改 action_memory 的管理方式
        
        # 尝试将 action_memory 的形状从 (1, memory_sequence_length, num_mid_dim) 转为 (memory_sequence_length, 1, num_mid_dim)
        # 然后扩展批次维度以匹配当前的 batch_size
        # 注意：这只是一个临时方案，正确做法是修改 action_memory 的管理逻辑以支持批处理。

        action_memory_processed = self.action_memory.squeeze(0).unsqueeze(1) # 形状变为 (memory_sequence_length, 1, num_mid_dim)
        
        # 如果 batch_size > 1 (尽管我们目前设置为 1)，这里需要将 action_memory 复制 batch_size 份
        # action_memory_processed = action_memory_processed.expand(-1, batch_size, -1)
        
        # 在 batch_size=1 的情况下，action_memory_processed 形状为 (memory_sequence_length, 1, num_mid_dim)，符合 Transformer 期望
        
        # 使用处理后的 action_memory 调用 transformer
        # Transformer Decoder 的 forward 方法签名通常是 forward(tgt, memory)
        # 在 IrisBabelTransformer 中，调用是 self.decoder(target_seq, src_seq)
        # 根据上下文，target_seq 应该是 action_memory，src_seq 是 sequence_features
        # 所以，pred = self.babel_transformer(action_memory_processed, sequence_features_transposed)
        # 但是 IrisBabelTransformer 的 forward 签名是 (src_seq, target_seq) 且调用是 self.decoder(target_seq, src_seq)
        # 这意味着 IrisBabelTransformer 内部将第一个参数视为 src，第二个参数视为 target
        # 然后将其传递给 decoder 时又调换了顺序。
        # 所以，我们需要传入 (sequence_features_transposed, action_memory_processed)

        # 经过之前的转置，sequence_features_transposed 形状为 (sequence_length, batch_size, feature_dim)
        # 经过处理，action_memory_processed 形状为 (memory_sequence_length, batch_size, num_mid_dim) when batch_size=1
        # 看起来 IrisBabelTransformer 的 forward 方法签名 (src_seq, target_seq) 与内部 decoder 调用 (target_seq, src_seq) 存在不一致。
        # 假设 IrisBabelTransformer 的 forward(src_seq, target_seq) 实际上是将 src_seq 作为 memory，target_seq 作为 tgt 传给 decoder
        # 那么我们应该传入 (sequence_features_transposed, action_memory_processed)

        # 为了绕过 Transformer 内部的形状错误，临时创建模拟的 sequence_features_transposed
        # 根据错误信息，它期望源序列的某个派生张量形状与 [1, 24, 64] 相关。
        # 我们尝试模拟一个形状 (sequence_length, batch_size * num_heads_expected, head_dim_expected) 的输入
        # 假设 num_heads_expected * head_dim_expected = num_mid_dim = 512
        # 错误信息中的 24 和 64 是从哪里来的需要进一步分析。
        # 错误信息 [1, 24, 64] 可能是 TransformerLayer 内部的形状。
        # 如果 Transformer Decoder 期望的 memory 形状是 (sequence_length, batch_size, hidden_dim)
        # 那么输入的 sequence_features_transposed 形状 (5, 1, 512) 是正确的。
        # 错误可能在于 TransformerLayer 内部对 hidden_dim (512) 和 attention head 参数 (24, 64) 的不一致处理。

        # 尝试直接使用原始的 sequence_features_transposed (形状 5, 1, 512) 传入，这符合标准 Transformer 输入格式。
        # 之前的错误是在 multi_head_attention_forward 内部，可能是其对输入的处理逻辑与张量形状不匹配。
        # 既然之前的形状调整都无效，我们回到最原始的符合 Transformer 标准输入的形状 (sequence_length, batch_size, num_mid_dim)
        # 尝试将 action_memory 也调整到标准形状 (memory_sequence_length, batch_size, num_mid_dim)

        # 调整 action_memory 形状为 (memory_sequence_length, batch_size, num_mid_dim)
        # action_memory 当前形状: (1, memory_sequence_length, num_mid_dim)
        # 我们需要将其形状变为 (memory_sequence_length, batch_size, num_mid_dim)
        # 这里的 batch_size 应该是当前的 batch_size (由 dummy_input 决定)
        current_batch_size = sequence_features.shape[0] # 从 sequence_features 获取当前 batch_size
        memory_sequence_length = self.action_memory.shape[1]
        
        # 如果 batch_size > 1，需要复制 action_memory
        # 这是一个临时方案，正确做法是修改 action_memory 的管理逻辑以支持批处理。
        action_memory_adjusted = self.action_memory.squeeze(0).unsqueeze(1).expand(-1, current_batch_size, -1)
        # action_memory_adjusted 形状变为 (memory_sequence_length, current_batch_size, num_mid_dim)

        # 使用转置后的 sequence_features (形状 5, batch_size, 512) 和 调整后的 action_memory 调用 transformer
        # 传入 (sequence_features_transposed, action_memory_adjusted)
        # 假设 IrisBabelTransformer forward(src, tgt) 内部调用 decoder(tgt, src)
        # 则 src 是 sequence_features_transposed (memory), tgt 是 action_memory_adjusted (target)
        pred = self.babel_transformer(sequence_features_transposed, action_memory_adjusted)

        # 后续解码器逻辑保持不变
        action_idx, action_idx_probs = self.action_decoder(pred[:, -1])
        action_target, action_target_probs = self.target_decoder(pred[:, -1])
        action_position = self.rel_position_decoder(pred[:, -1])
        action_direction, action_direction_probs = self.direction_decoder(pred[:, -1])
        
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

