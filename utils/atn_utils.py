import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# TODO: 这是最关键的一步！
# 你需要从你的"朋友"那里获取完整的 ATN 模型类定义。
# 假设这个模型类定义在 models/atn_model.py 中，并且名为 `AttentiveTriggerNetwork`。
# from models.atn_model import AttentiveTriggerNetwork # 替换为你的实际 ATN 模型类

# ⚠️ 临时占位符 ATN 模型类，请务必替换为你的实际模型！
# 这个类需要能够接收模型输入相关的尺寸参数，并包含一个 feature_extractor 子模块
# 以及生成 'decision' 和 'attention' 输出的逻辑。
class AttentiveTriggerNetwork(nn.Module):
    def __init__(self, in_channels, sequence_length, height, width, num_heads):
        super().__init__()
        # 这是一个模拟的 ATN 模型结构，仅用于演示如何加载特征头和处理输入输出
        # 你的实际 ATN 模型会复杂得多，包含 Transformer 块等，并且输入输出维度可能不同。
        
        self.in_channels = in_channels
        self.sequence_length = sequence_length
        self.height = height
        self.width = width
        
        # --- 模拟特征提取器 (假定是 iris_feature_13_ce_4bn_20f_best.pth 对应的部分) ---
        # 根据你朋友的特征头实际结构来定义这个 IrisFeatureHead 类或使用他们的代码
        # 假定这是一个 3D CNN 特征提取器，处理 (B, C, T, H, W) 输入
        class IrisFeatureHead(nn.Module):
            def __init__(self, in_channels_feature, out_channels_feature=128, seq_len=16, height=64, width=64):
                super().__init__()
                # 这里的层结构是模拟的，需要替换为你朋友的实际特征头模型结构
                self.conv1 = nn.Conv3d(in_channels_feature, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1))
                self.relu = nn.ReLU(inplace=True)
                self.conv2 = nn.Conv3d(64, out_channels_feature, kernel_size=(1, 3, 3), padding=(0, 1, 1))
                # 模拟下采样，假设 spatial 2x2 下采样，序列无下采样
                self.conv_down = nn.Conv3d(out_channels_feature, out_channels_feature, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))

            def forward(self, x):
                # x shape: (B, C, T, H, W)
                out = self.relu(self.conv1(x))
                out = self.relu(self.conv2(out))
                features = self.relu(self.conv_down(out)) # 模拟特征输出 (B, 128, T, H/2, W/2)
                return features
        
        # 使用 ATN 期望的输入通道来初始化特征提取器
        self.feature_extractor = IrisFeatureHead(in_channels_feature=self.in_channels, 
                                                 seq_len=self.sequence_length,
                                                 height=self.height,
                                                 width=self.width) 

        # --- 模拟决策层 ---
        # 从 feature_extractor 的输出 (B, 128, T, H_feat, W_feat) 生成决策图 (B, H_dec, W_dec)
        # 假定决策图空间维度是 H/4, W/4
        # 需要根据你的实际 ATN 模型结构来确定
        self.decision_layer = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)), # spatial downsample 2x2
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 32, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)), # spatial downsample 2x2 -> H/4, W/4
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=1) # Output 1 channel score
        )
        
        # --- 模拟注意力层 ---
        # 从 feature_extractor 的输出 (B, 128, T, H_feat, W_feat) 生成注意力矩阵 (B, heads, N, N)
        # N 是序列长度，即 self.sequence_length
        # 假定注意力关注的是序列维度 (N x N)，并且有 4 个头
        self.num_heads = 4 # 与 mock 保持一致，也从 cfg 中获取
        
        # 模拟一个投影层，将特征映射到注意力相关的维度
        # 这里的输入维度需要根据 feature_extractor 和 decision_layer 之后的特征图尺寸来确定
        # 假定 feature_extractor 输出 (B, 128, T, H/2, W/2)
        # 将空间维度展平：128 * (H/2) * (W/2)
        # 然后对序列维度 T 进行某种处理，再生成注意力
        
        # 这是一个高度简化的模拟：直接从展平的特征预测注意力矩阵
        # 实际的 Transformer 结构会非常不同
        simulated_attention_input_dim = 128 * (height // 2) * (width // 2) * sequence_length # 展平所有特征维度
        simulated_attention_output_dim = self.num_heads * sequence_length * sequence_length
        
        self.attention_projection = nn.Linear(simulated_attention_input_dim, simulated_attention_output_dim)
        

    def forward(self, x):
        # x shape: (B, C, T, H, W)
        # 确保输入形状与模型期望一致
        assert x.shape[1:] == (self.in_channels, self.sequence_length, self.height, self.width), \
            f"Input shape mismatch. Expected ({x.shape[0]}, {self.in_channels}, {self.sequence_length}, {self.height}, {self.width}), got {x.shape}"

        # 1. 特征提取
        features = self.feature_extractor(x) # 假设输出 (B, 128, T_feat, H_feat, W_feat)
        
        # 2. 决策图生成
        # 决策层处理 features
        decision_map_raw = self.decision_layer(features) # 假设输出 (B, 1, T_dec, H_dec, W_dec)
        # 对序列维度求平均或取某个时间步，并去除单通道维度
        decision_map = decision_map_raw.mean(dim=2).squeeze(1) # -> (B, H_dec, W_dec)
        
        # 3. 注意力图生成
        # 将 feature_extractor 的输出展平用于模拟注意力投影
        B, C_feat, T_feat, H_feat, W_feat = features.shape
        flat_features = features.view(B, -1) # (B, C_feat * T_feat * H_feat * W_feat)
        
        # 确保投影层输入维度匹配
        expected_attn_proj_in_dim = 128 * T_feat * H_feat * W_feat
        if flat_features.shape[1] != expected_attn_proj_in_dim:
            print(f"Warning: Attention projection input shape mismatch. Expected {expected_attn_proj_in_dim}, got {flat_features.shape[1]}. Re-initializing attention_projection.")
            # 如果维度不匹配，重新初始化投影层 (这通常意味着你的模拟结构与实际不符)
            self.attention_projection = nn.Linear(flat_features.shape[1], simulated_attention_output_dim).to(x.device)
            flat_features = self.attention_projection(flat_features) # Recalculate with new layer

        attention_matrix_flat = flat_features

        # 如果 T_feat != self.sequence_length，需要处理（例如插值或调整模拟结构）
        # 这里简单起见，如果维度不匹配，直接返回一个与 input N 匹配的随机矩阵作为警告
        if T_feat != self.sequence_length:
             print(f"Warning: Feature sequence length T_feat ({T_feat}) != Input sequence length N ({self.sequence_length}). Attention matrix shape may be incorrect. Returning simulated matrix.")
             attention_matrix_flat = torch.randn(B, 4, self.sequence_length, self.sequence_length, device=x.device) * 0.01 # Return matrix matching input N

        attention_matrix = attention_matrix_flat.view(B, self.num_heads, self.sequence_length, self.sequence_length)

        return {'decision': decision_map, 'attention': attention_matrix}

def load_atn_model(model_path, device, in_channels, sequence_length, height, width, num_heads, load_feature_head_only=False):
    """
    加载 ATN 模型，并将特征头权重加载到模型中。
    model_path: 指向模型权重的路径。
    device: 计算设备。
    in_channels, sequence_length, height, width, num_heads: 用于初始化完整的 ATN 模型。
    load_feature_head_only (bool): 如果为 True，只加载特征头权重（假设模型结构支持）。
    """
    print(f"Loading ATN model and feature head from {model_path}")

    # 实例化完整的 ATN 模型，传入初始化参数
    # TODO: 替换为你的真实 AttentiveTriggerNetwork 类
    # 如果你的 ATN 模型初始化需要不同的参数集，请调整这里
    atn_model = AttentiveTriggerNetwork(
        in_channels=in_channels,
        sequence_length=sequence_length,
        height=height,
        width=width,
        num_heads=num_heads
    ).to(device)

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device, weights_only=False)

        try:
            if load_feature_head_only:
                # 假设特征提取器是 atn_model 的一个子模块，名为 'feature_extractor'
                # 并且 model_path 直接指向特征头的权重文件
                # 如果你的特征头权重文件和完整模型的权重文件不同，请调整 config 中的 model_path
                atn_model.feature_extractor.load_state_dict(state_dict, strict=True)
                print("Successfully loaded feature head state dict to ATN model's feature_extractor.")
            else:
                # 加载完整的模型权重
                # 如果权重文件的键名与模型的不完全匹配，可能需要调整 strict=False
                atn_model.load_state_dict(state_dict, strict=True)
                print("Successfully loaded full ATN model state dict.")
        except RuntimeError as e:
            print(f"Warning: Could not load state dict with strict=True. Error: {e}")
            print("Attempting to load with strict=False...")
            # 如果键名不匹配，尝试非严格加载，并打印缺失和意外的键
            if load_feature_head_only:
                missing_keys, unexpected_keys = atn_model.feature_extractor.load_state_dict(state_dict, strict=False)
                print("Loaded feature head state dict with strict=False.")
            else:
                missing_keys, unexpected_keys = atn_model.load_state_dict(state_dict, strict=False)
                print("Loaded full ATN model state dict with strict=False.")

            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")

    else:
        print(f"Warning: ATN model weights not found at {model_path}. Using randomly initialized model.")

    atn_model.eval()
    return atn_model

def get_atn_outputs(atn_model, x, return_features=False, return_decision=True, return_attention=True):
    """
    获取 ATN 模型的指定输出。

    Args:
        atn_model: ATN 模型实例。
        x (torch.Tensor): 输入张量 (B, C, T, H, W) 或 (B, C, N, W, H) - 根据模型期望调整。
        return_features (bool): 是否返回特征提取器输出。
        return_decision (bool): 是否返回决策图。
        return_attention (bool): 是否返回注意力矩阵。

    Returns:
        dict: 包含指定输出的字典。
              可能包含 'features', 'decision', 'attention'。
    """
    atn_model.eval()
    with torch.no_grad():
        # 假设 ATN model 的 forward 方法可以接受参数来控制返回的输出
        # TODO: 根据你的实际 ATN model.forward 方法签名进行调整
        outputs = atn_model(x, return_features=return_features, return_decision=return_decision, return_attention=return_attention)

    # 确保返回的是一个字典，并且包含所有请求的输出
    result = {}
    if return_features and 'features' in outputs:
        result['features'] = outputs['features']
    if return_decision and 'decision' in outputs:
        result['decision'] = outputs['decision']
    if return_attention and 'attention' in outputs:
        result['attention'] = outputs['attention']

    # 如果模型forward不支持这些参数或返回结构不同，需要在这里适配
    # 例如：如果 atn_model(x) 总是返回一个包含所有输出的字典
    # result['features'] = outputs.get('features')
    # result['decision'] = outputs.get('decision')
    # result['attention'] = outputs.get('attention')

    return result

# TODO: 获取原始特征层数据的占位符
def get_original_features(original_input):
    """
    占位符：通过 ATN 模型获取原始输入的特征层。
    这可能需要修改 ATN 模型以暴露中间层的输出。
    目前模拟为原始输入本身。
    """
    print("Placeholder: Getting original features from ATN.")
    # 模拟返回原始输入作为特征层
    return original_input # 假设特征层就是输入 (B, 128, N, W, H)

# TODO: 获取对抗样本下 ATN 输出的占位符
def get_atn_outputs(atn_model, adversarial_input):
    """
    占位符：获取对抗样本下的 ATN 输出 (决策图和注意力矩阵)。
    """
    print("Placeholder: Getting ATN outputs for adversarial input.")
    # 调用模拟的 ATN 模型
    return atn_model(adversarial_input) 