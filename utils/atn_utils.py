import torch
import torch.nn as nn
import torch.nn.functional as F
import os


# 导入你的 IrisXeonNet 类
# 确保你的 Python 环境可以找到 IrisBabel 模块
try:
    from IrisBabel.nn.CNN.IrisXeonNet import IrisXeonNet
    print("Successfully imported IrisXeonNet from IrisBabel.")
except ModuleNotFoundError as e:
    print(f"Error importing IrisXeonNet: {e}")
    print("Please ensure the IrisBabel module is correctly placed in the project root or in Python path.")
    IrisXeonNet = None # Set to None if import fails


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
    加载 ATN 模型 (假设权重文件对应 IrisXeonNet)。
    model_path: 指向模型权重的路径。
    device: 计算设备。
    in_channels, sequence_length, height, width, num_heads: 用于初始化模型 (如果 IrisXeonNet 需要这些参数，请调整)。
    load_feature_head_only (bool): 在 IrisXeonNet 的情况下，此参数可能不适用，因为 IrisXeonNet 似乎本身就是特征提取器。
                                  保留此参数用于兼容性，但在加载 IrisXeonNet 时其行为可能不同。
    """
    print(f"Loading ATN model (assuming IrisXeonNet structure) from {model_path}")

    if IrisXeonNet is None:
        print("IrisXeonNet class not found due to import error. Cannot load ATN model.")
        return None

    # 实例化 IrisXeonNet 模型
    # TODO: 检查 IrisXeonNet 的 __init__ 方法需要哪些参数，并根据你的配置 cfg.model.atn 提供。
    # 根据 IrisBabel/nn/CNN/IrisXeonNet.py 文件内容，__init__ 只需要 num_classes (默认为 1000)。
    # 如果你的权重文件需要特定的 num_classes 来匹配结构，请在此处指定。
    # 这里暂且使用默认值，如果加载失败，可能需要调整。
    try:
        # 实例化模型。注意：这里假设 IrisXeonNet 的初始化不需要 cfg 中的所有 atn_* 参数
        # 如果你的 IrisXeonNet 实际需要这些参数，请在这里传递。
        atn_model = IrisXeonNet().to(device)
        print("Instantiated IrisXeonNet model.")
    except Exception as e:
        print(f"Error instantiating IrisXeonNet: {e}")
        return None


    if os.path.exists(model_path):
        try:
            # 加载权重，使用 weights_only=False 和 strict=False
            # weights_only=False 解决 _pickle.UnpicklingError (ModuleNotFoundError)
            # strict=False 解决键名不完全匹配的问题
            state_dict = torch.load(model_path, map_location=device, weights_only=False)
            print(f"Loaded state_dict from {model_path} with weights_only=False.")

            # 由于我们假设权重文件直接对应 IrisXeonNet，直接加载到模型
            # 使用 strict=False 以处理键名不完全匹配的情况
            missing_keys, unexpected_keys = atn_model.load_state_dict(state_dict, strict=False)
            print("Attempted to load state dict to IrisXeonNet with strict=False.")

            if missing_keys:
                print(f"Missing keys in loaded state_dict: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys in loaded state_dict: {unexpected_keys}")

            # 如果 missing_keys 包含关键层（如 conv 层或 BN 层），加载可能没有成功
            if missing_keys and any(k.startswith(('conv', 'bn', 'bottleneck')) for k in missing_keys):
                print("Warning: Significant model keys are missing. State dict might not match IrisXeonNet structure.")


        except Exception as e:
            print(f"Error loading state dict to IrisXeonNet: {e}")
            print("ATN model might not be loaded correctly.")
            # 如果加载失败，可以返回 None 或随机初始化的模型
            return None # Return None if loading fails critically

    else:
        print(f"Warning: ATN model weights not found at {model_path}. Using randomly initialized IrisXeonNet.")

    atn_model.eval()
    return atn_model

def get_atn_outputs(atn_model, x, return_features=False, return_decision=True, return_attention=True):
    """
    获取 IrisXeonNet 模型的指定输出。
    由于 IrisXeonNet 当前看起来只返回特征，此函数将只返回 features。
    未来如果使用完整的 ATN 模型，需要修改此函数。

    Args:
        atn_model: ATN 模型实例 (现在是 IrisXeonNet 实例)。
        x (torch.Tensor): 输入张量 (B, C, T, H, W) 或 (B, C, N, W, H) - 根据模型期望调整。

    Returns:
        dict: 包含指定输出的字典。
              目前只返回 'features'。
    """
    if atn_model is None:
        print("Warning: ATN model is None. Cannot get ATN outputs.")
        return {}

    atn_model.eval()
    with torch.no_grad():
        # 假设 IrisXeonNet 的 forward 方法接受 (B, C, T, H, W) 或 (B, C, N, W, H)
        # 根据 IrisXeonNet.py 文件，forward 方法内部会 permute 输入
        # 它似乎期望输入是 (B, C, H, W) 然后 permute 成 (B, W, C, H, W)??
        # 不，forward 的第一行是 x = torch.permute(x, (0, 4, 1, 2, 3))
        # 如果输入是 (B, C, T, H, W)，permute 后变成 (B, W, C, T, H)
        # 这看起来与 conv3d 的输入形状不符 (conv3d 期望 (B, C_in, D, H_in, W_in))
        # IrisXeonNet Conv3d kernels are (1, 3, 3), stride (1, 3, 3)
        # 这意味着 Conv3d 操作主要在 spatial (H, W) 和 channel (C) 维度上进行
        # 序列维度 T (或 N) 似乎被当作 depth 维度 D
        # 如果输入 x 是 (B, C, T, H, W)，那么 permute (0, 4, 1, 2, 3) 得到 (B, W, C, T, H)
        # 再 permute (0, 2, 3, 4, 1) 将其变回 (B, C, T, H, W)，然后 Conv3d 的输入 channel 是 C。
        # 这与 IrisXeonNet 的 conv1x1_1(3, 16, ...) 不符，它的输入通道是 3。
        # 看来 IrisXeonNet 期望的输入形状可能不是标准的 (B, C, T, H, W) 格式。

        # ⚠️ 紧急修正：根据 IrisXeonNet.py 中的 forward 方法，它接收一个张量 `x`
        # 然后进行 `x = torch.permute(x, (0, 4, 1, 2, 3))`
        # Conv3d(3, 16, (1, 3, 3), stride=(1, 3, 3)) 意味着 Conv3d 期望输入形状是 (B, C_in, D, H_in, W_in)，其中 C_in=3。
        # 如果原始输入是 (B, C, T, H, W)，permute 后是 (B, W, C, T, H)。
        # 要让 C_in 变成 3，原始输入的第 3 个维度 (索引 2) 应该是通道 C。
        # 这意味着 IrisXeonNet 的 `forward(x)` 期望 `x` 的形状是 (B, T, H, W, C) 吗？
        # 如果是 (B, T, H, W, C)，permute (0, 4, 1, 2, 3) -> (B, C, T, H, W)
        # 这与我们数据加载器返回的 (C, T, H, W) 或训练循环中的 (B, C, T, H, W) 相匹配。
        # 所以，假设 IrisXeonNet 期望的输入是 (B, T, H, W, C)，并且 forward 方法内部的 permute 是为了将其转换为 Conv3d 习惯的 (B, C, T, H, W)。
        # 但是，在 train_generator.py 中，我们已经将数据 permute 成了 (B, C, T, W, H) 或 (B, C, T, H, W)。
        # 如果 IrisXeonNet 期望 (B, T, H, W, C)，那么在调用 atn_model(x) 之前，我们需要将 (B, C, T, H, W) 转换为 (B, T, H, W, C)。
        # 这意味着 permute (0, 2, 3, 4, 1)
        # 假设 x 输入到 get_atn_outputs 是 (B, C, T, H, W)
        # 为了匹配 IrisXeonNet 的 forward 内部 permute 前的形状，我们需要先 permute
        # x_for_atn = x.permute(0, 2, 3, 4, 1) # (B, C, T, H, W) -> (B, T, H, W, C)

        # 再次查看 train_generator.py 里面的 permute
        # real_x = batch_data.to(device) # real_x shape: (B, C, T, H, W)
        # real_x = real_x.permute(0, 1, 2, 4, 3) # (B, C, T, H, W) -> (B, C, T, W, H) to match Generator/Discriminator expectation
        # adversarial_x = real_x + delta # delta is (B, C, T, W, H)
        # adversarial_x = torch.clamp(adversarial_x, 0, 1) # Still (B, C, T, W, H)
        # original_atn_outputs = get_atn_outputs(atn_model, real_x, ...) # passing (B, C, T, W, H) ? No, passing real_x -> (B, C, T, W, H)
        # adversarial_atn_outputs = get_atn_outputs(atn_model, real_x, ...) # Still passing real_x. THIS IS WRONG! Should pass adversarial_x!
        # Also, the input to get_atn_outputs is expected to be in original image space (B, C, T, H, W), not feature space.
        # The Generator takes real_x in feature space (B, 128, N, W, H) and outputs delta (B, 128, N, W, H).
        # THEN adversarial_features = original_features + delta.

        # Let's re-evaluate the get_atn_outputs usage in train_generator.py
        # It is called with get_atn_outputs(atn_model, real_x, return_features=True, ...)
        # and get_atn_outputs(atn_model, real_x, return_features=True, ...) - both called with real_x
        # this is wrong. It should be original_atn_outputs = get_atn_outputs(atn_model, real_x)
        # AND adversarial_atn_outputs = get_atn_outputs(atn_model, adversarial_x)
        # And ATN takes original image data (B, C, T, H, W), not permuted or feature data.
        # The Generator/Discriminator operate on features (B, 128, N, W, H).

        # Okay, let's correct the train_generator.py logic *and* assume get_atn_outputs expects (B, C, T, H, W) and IrisXeonNet expects (B, T, H, W, C).
        # In train_generator.py, the data loader returns (C, T, H, W), stacked by DataLoader to (B, C, T, H, W).
        # It is then permuted to (B, C, T, W, H) for Generator/Discriminator. This permute seems wrong based on Generator kernel sizes.
        # Generator kernel (1, 4, 4) with stride (1, 2, 2) padding (0, 1, 1) on input (B, C, N, W, H) downsamples W and H by 2.
        # If input was (B, C, N, H, W), it would downsample H and W by 2. The typical convention is (B, C, D, H, W).
        # So let's assume Generator expects (B, C, N, H, W) and data loader returns (B, C, T, H, W) with N=T.

        # Correction in train_generator.py:
        # Remove `real_x = real_x.permute(0, 1, 2, 4, 3)`
        # Calculate delta: `delta = generator(original_features)` - Generator input is features.
        # Calculate adversarial_features: `adversarial_features = original_features + delta`.
        # Clamp adversarial_features.
        # Discriminator takes features.

        # Now, how to get original_features from the ATN model?
        # The ATN model (IrisXeonNet) takes original image data (B, C, T, H, W)
        # and returns features (B, 128, T', H', W').

        # Okay, the `get_atn_outputs` function should take the original image `real_x` (B, C, T, H, W)
        # It should pass this to `atn_model`.
        # The `atn_model` (IrisXeonNet) expects input shape (B, T, H, W, C), permutes it to (B, C, T, H, W) internally.
        # So, inside `get_atn_outputs`, we need to permute `x` from (B, C, T, H, W) to (B, T, H, W, C) before passing it to `atn_model.forward`.

        x_for_atn_forward = x.permute(0, 2, 3, 4, 1) # (B, C, T, H, W) -> (B, T, H, W, C)

        # Call the actual ATN model (IrisXeonNet) forward
        # IrisXeonNet().forward(x) returns the feature tensor (B, 128, T', H', W')
        # It does not return decision or attention maps directly.
        # So, this function can *only* return features if the model is IrisXeonNet.
        features = atn_model(x_for_atn_forward) # This calls IrisXeonNet.forward

    # Return only features for now, as IrisXeonNet doesn't output others
    result = {'features': features}

    # If we ever get the full ATN model that wraps IrisXeonNet and adds decision/attention heads,
    # this function would need to be updated to call the full ATN model's forward,
    # which hopefully returns a dict like {'features':..., 'decision':..., 'attention':...}
    # For now, print warnings if decision or attention were requested but not available.
    if return_decision:
        print("Warning: Decision map requested but IrisXeonNet does not return it directly.")
    if return_attention:
         print("Warning: Attention matrix requested but IrisXeonNet does not return it directly.")


    return result


# Keep the placeholder functions below if they are still used elsewhere,
# but they should ideally be replaced with logic that uses the actual ATN model outputs.

# TODO: 获取原始特征层数据的占位符 - NOW USES REAL ATN
# def get_original_features(original_input):
#    ...

# TODO: 获取对抗样本下 ATN 输出的占位符 - NOW USES REAL ATN
# def get_atn_outputs(atn_model, adversarial_input):
#    ... 