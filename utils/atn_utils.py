import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import Dict, Any, Optional, Tuple # 导入类型提示相关的模块


# 导入你的 IrisXeonNet 类
# 确保你的 Python 环境可以找到 IrisBabel 模块
try:
    from IrisBabel.nn.CNN.IrisXeonNet import IrisXeonNet
   #print("Successfully imported IrisXeonNet from IrisBabel.")
except ModuleNotFoundError as e:
    print(f"Error importing IrisXeonNet: {e}")
    print("Please ensure the IrisBabel module is correctly placed in the project root or in Python path.")
    IrisXeonNet = None # 如果导入失败，将 IrisXeonNet 设为 None


# ⚠️ 临时占位符 ATN 模型类，请务必替换为你的实际模型！
# 这个类需要能够接收模型输入相关的尺寸参数，并包含一个 feature_extractor 子模块
# 以及生成 'decision' 和 'attention' 输出的逻辑。
class AttentiveTriggerNetwork(nn.Module):
    """
    Attentive Trigger Network (ATN) 的占位符模型类。
    该类模拟了 ATN 的基本结构，包括特征提取器、决策层和注意力层，
    用于在开发和测试阶段替代实际的 ATN 模型。
    
    ⚠️ 重要提示：这是一个临时实现，其内部结构和逻辑需要根据实际使用的 ATN 模型
    (如 IrisXeonNet 或其完整版本) 进行替换和调整，以确保输入输出维度和行为正确匹配。
    """
    def __init__(
        self, 
        in_channels: int, # 输入张量的通道数
        sequence_length: int, # 输入张量的序列（时间）长度
        height: int, # 输入张量的高度
        width: int, # 输入张量的宽度
        num_heads: int # 注意力机制的头部数量
    ):
        """
        初始化占位符 ATN 模型。

        Args:
            in_channels (int): 输入张量的通道数。
            sequence_length (int): 输入张量的序列（时间）长度。
            height (int): 输入张量的高度。
            width (int): 输入张量的宽度。
            num_heads (int): 注意力机制的头部数量。
        """
        super().__init__()
        # 将初始化参数保存为成员变量
        self.in_channels = in_channels
        self.sequence_length = sequence_length
        self.height = height
        self.width = width
        
        # --- 模拟特征提取器 (假定是 iris_feature_13_ce_4bn_20f_best.pth 对应的部分) ---
        # 根据你朋友的特征头实际结构来定义这个 IrisFeatureHead 类或使用他们的代码
        # 假定这是一个 3D CNN 特征提取器，处理 (B, C, T, H, W) 输入
        class IrisFeatureHead(nn.Module):
            """
            IrisXeonNet 特征提取部分的模拟类。
            这个类模拟了实际 IrisXeonNet 模型中的特征提取部分，
            其结构和输出维度需要与实际加载的权重文件匹配。
            """
            def __init__(
                self, 
                in_channels_feature: int, # 输入特征提取器的通道数
                out_channels_feature: int = 128, # 输出特征图的通道数
                seq_len: int = 16, # 输入序列长度 (虽然这里模拟结构中 Conv3d kernel size seq dim 是 1)
                height: int = 64, # 输入高度
                width: int = 64 # 输入宽度
            ):
                """
                初始化模拟的 IrisXeonNet 特征提取头。

                Args:
                    in_channels_feature (int): 输入特征提取器的通道数。
                    out_channels_feature (int): 输出特征图的通道数，默认为 128。
                    seq_len (int): 输入序列长度，默认为 16。注意模拟结构中 Conv3d 对序列维度操作。
                    height (int): 输入高度，默认为 64。
                    width (int): 输入宽度，默认为 64。
                """
                super().__init__()
                self.seq_len = seq_len # Store seq_len

                # 这里的层结构是模拟的，需要替换为你朋友的实际特征头模型结构
                # Conv3d 参数: in_channels, out_channels, kernel_size, stride, padding
                # kernel_size=(1, 3, 3) 意味着在序列维度上 kernel size 是 1，不对序列长度做卷积
                # padding=(0, 1, 1) 意味着在序列维度上没有 padding，在空间维度有 padding
                self.conv1 = nn.Conv3d(in_channels_feature, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1))
                self.relu = nn.ReLU(inplace=True)
                # Note: conv2 output channels match out_channels_feature
                self.conv2 = nn.Conv3d(64, out_channels_feature, kernel_size=(1, 3, 3), padding=(0, 1, 1))
                # 模拟下采样，假设 spatial 2x2 下采样，序列无下采样
                # Conv3d stride=(1, 2, 2) 意味着在序列维度 stride 是 1，在空间维度 stride 是 2
                # padding=(0, 1, 1) 在序列维度无 padding，空间维度有 padding
                # kernel_size=(1, 4, 4) 空间 kernel size 较大，结合 stride 实现下采样
                self.conv_down = nn.Conv3d(out_channels_feature, out_channels_feature, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """
                特征提取器前向传播。

                Args:
                    x (torch.Tensor): 输入张量，形状期望为 (B, C, T, H, W)。

                Returns:
                    torch.Tensor: 提取的特征张量，形状为 (B, out_channels_feature, T_feat, H_feat, W_feat)。
                                  其中 T_feat=T, H_feat=H/2, W_feat=W/2 (取决于具体的卷积和下采样配置)。
                """
                out = self.relu(self.conv1(x))
                out = self.relu(self.conv2(out))
                features = self.relu(self.conv_down(out)) # 模拟特征输出 (B, 128, T, H/2, W/2)
                
                # Add a check to ensure sequence length is preserved in the simulated feature head
                if features.shape[2] != x.shape[2]:
                     print(f"Warning: Simulated IrisFeatureHead changed sequence length from {x.shape[2]} to {features.shape[2]}")
                     # If sequence length changes, this simulation is inaccurate. For now, return as is.

                return features
        
        # 使用 ATN 期望的输入通道和序列长度来初始化特征提取器
        # 这里的 in_channels 应该是输入到整个 ATN 模型的通道数，而不是特征提取器期望的通道数。
        # TODO: 确认实际 IrisXeonNet 的特征提取器期望的输入通道数 (似乎是 3)
        # 根据 load_atn_model 函数中的注释，IrisXeonNet Conv3d 输入通道是 3，这与 train_generator.py 中的输入数据形状 (B, C, T, H, W) 不符，除非 C=3。
        # 假设这里 in_channels 是指输入到 ATN 的总通道数，而特征提取器实际期望的是 3 通道输入（例如 RGB 图像）。
        # 这意味着在将原始输入传递给 feature_extractor 之前，可能需要进行通道适配或 IrisXeonNet 本身处理多通道输入。
        # 鉴于 IrisXeonNet 的 conv1x1_1(3, 16, ...)，很可能它期望 3 通道输入。
        # 这里的模拟应该匹配 IrisXeonNet 实际接收的输入。
        # 如果 IrisXeonNet.py 中的 forward 期望输入 (B, T, H, W, C)，那么在调用 self.feature_extractor 之前需要 permute。
        # 但是模拟的 IrisFeatureHead 期望输入 (B, C, T, H, W)。
        # 结论：模拟类 IrisFeatureHead 的输入通道应与实际 IrisXeonNet 的特征提取第一层期望的通道数一致，这似乎是 3。
        # 但整个 ATN 占位符接收的 in_channels 是配置中的输入通道数。
        # 这是一个模拟和实际模型之间需要仔细核对的地方。
        # 为了匹配模拟结构的 Conv3d(in_channels_feature, ...)，这里的 in_channels_feature 应该设置为输入数据的通道数 (self.in_channels)
        # 如果实际 IrisXeonNet 需要 3 通道，那么这里模拟就不完全准确，或者 IrisXeonNet 在内部处理多通道。
        # 暂时按照模拟结构的定义来初始化 feature_extractor，输入通道数使用传入的 self.in_channels。
        # 并使用传入的 self.sequence_length 初始化模拟特征提取器的 seq_len 参数
        self.feature_extractor = IrisFeatureHead(in_channels_feature=self.in_channels, 
                                                 seq_len=self.sequence_length,
                                                 height=self.height,
                                                 width=self.width) 

        # --- 模拟决策层 ---
        # 从 feature_extractor 的输出 (B, 128, T_feat, H_feat, W_feat) 生成决策图 (B, H_dec, W_dec)
        # 假定决策图空间维度是 H/4, W/4，这通过两个 stride 为 (1, 2, 2) 的 Conv3d 实现空间 4x4 下采样
        # 决策层处理 feature_extractor 的输出，其通道数是 128。
        # 需要根据你的实际 ATN 模型结构来确定决策层的输入输出维度和结构。
        # Note: Decision layer simulation assumes input features channels are 128.
        self.decision_layer = nn.Sequential(
            # 第一个 Conv3d: 输入通道 128，输出通道 64，kernel (1,4,4), stride (1,2,2), padding (0,1,1)
            # 空间维度: (H_feat - 4 + 2*1)/2 + 1 ≈ H_feat/2
            # 序列维度: T_feat 不变
            nn.Conv3d(128, 64, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)), # spatial downsample 2x2
            nn.ReLU(inplace=True),
            # 第二个 Conv3d: 输入通道 64，输出通道 32，kernel (1,4,4), stride (1,2,2), padding (0,1,1)
            # 空间维度: (H_feat/2 - 4 + 2*1)/2 + 1 ≈ H_feat/4
            # 序列维度: T_feat 不变
            nn.Conv3d(64, 32, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)), # spatial downsample 2x2 -> H_feat/4, W_feat/4
            nn.ReLU(inplace=True),
            # 最后一个 Conv3d: 输入通道 32，输出通道 1，kernel (1,1,1), stride (1,1,1), padding (0,0,0)
            # 只改变通道数，不改变空间和序列维度
            nn.Conv3d(32, 1, kernel_size=1) # Output 1 channel score per spatial-temporal location
        )
        
        # --- 模拟注意力层 ---
        # 从 feature_extractor 的输出 (B, 128, T_feat, H_feat, W_feat) 生成注意力矩阵 (B, heads, N, N)
        # N 是输入序列长度，即 self.sequence_length
        # 假定注意力关注的是序列维度 (N x N)，并且有 num_heads 个头
        self.num_heads = num_heads # 与 config 中保持一致
        
        # 模拟一个投影层，将展平的特征映射到用于计算注意力的维度。
        # 实际的 Transformer 注意力机制不是简单通过一个全连接层实现的，而是 Q, K, V 的线性投影和 scaled dot-product attention。
        # 这里的模拟是一个高度简化的代替。
        
        # Feature extractor output shape: (B, C_feat, T_feat, H_feat, W_feat)
        # The attention projection should map from the flattened features to the flattened attention matrix.
        # The input dimension to the projection depends on the *actual* output shape of the feature extractor.
        # We will calculate this dynamically in the forward pass for robustness in the simulation.
        
        # Note: The expected output dimension of the attention projection should be self.num_heads * self.sequence_length * self.sequence_length
        # (B, heads, N, N), where N is the input sequence length self.sequence_length
        
        # We cannot define the Linear layer here with a fixed input dimension if the feature extractor output shape is uncertain.
        # Instead, we will define the linear layer in the forward pass or adapt the simulation.
        # Given the previous error, the issue was likely mismatch between assumed feature shape and actual.
        # Let's redefine the projection layer to be created dynamically or sized based on the *assumed* output shape.
        # Based on the simulation description, T_feat = T, H_feat = H/2, W_feat = W/2.
        # The number of feature channels C_feat is 128.
        
        # Recalculate the simulated_attention_input_dim based on the assumed output shape of the feature extractor
        # C_feat = 128 (from IrisFeatureHead out_channels)
        # T_feat = self.sequence_length (from IrisFeatureHead simulation description)
        # H_feat = self.height // 2
        # W_feat = self.width // 2
        simulated_feature_spatial_h = self.height // 2
        simulated_feature_spatial_w = self.width // 2
        
        # Check if spatial dimensions are positive before calculating flattened size
        if simulated_feature_spatial_h <= 0 or simulated_feature_spatial_w <= 0:
             # Print warning and set flattened size to 0 if spatial dimensions are non-positive
             print(f"Warning: Calculated simulated feature spatial dimensions are non-positive: ({simulated_feature_spatial_h}, {simulated_feature_spatial_w}). Cannot calculate flattened attention input size.")
             simulated_attention_input_dim = 0
        else:
             simulated_attention_input_dim = 128 * self.sequence_length * simulated_feature_spatial_h * simulated_feature_spatial_w


        # Attention matrix shape is (B, heads, N, N), where N is the input sequence length.
        # Flattened attention matrix shape is (B, heads * N * N).
        simulated_attention_output_dim = self.num_heads * self.sequence_length * self.sequence_length
        
        # Define the simulated attention projection layer if input dimension is positive
        self.attention_projection = None # Initialize to None
        if simulated_attention_input_dim > 0:
            self.attention_projection = nn.Linear(simulated_attention_input_dim, simulated_attention_output_dim)
        else:
             print("Warning: Simulated attention input dimension is 0. Attention projection layer not created.")

        

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        占位符 ATN 模型前向传播。

        Args:
            x (torch.Tensor): 输入张量，形状期望为 (B, C, T, H, W)。
                              其中 C=self.in_channels, T=self.sequence_length, H=self.height, W=self.width。

        Returns:
            Dict[str, torch.Tensor]: 包含模型输出的字典。
                                     键 'decision' 对应决策图张量，形状为 (B, H_dec, W_dec)。
                                     键 'attention' 对应注意力矩阵张量，形状为 (B, heads, N, N)。
                                     如果计算失败或输入无效，可能返回包含空张量或引发错误。
        """
        # x shape: (B, C, T, H, W)
        # 确保输入形状与模型期望一致
        # 注意：这里检查的是输入到此占位符 ATN 类的形状，而不是实际 IrisXeonNet 期望的形状。
        expected_shape = (x.shape[0], self.in_channels, self.sequence_length, self.height, self.width)
        if x.shape != expected_shape:
             print(f"Warning: Input shape mismatch for placeholder ATN. Expected {expected_shape}, got {x.shape}")
             # 在实际应用中，这里应该根据实际 ATN 模型进行形状调整或抛出错误。
             # 暂时继续处理，但可能会导致后续错误。
             # 如果输入形状与期望不符，且无法继续，返回空字典
             # return {}

        # 1. 特征提取
        # 模拟的 feature_extractor 期望输入形状 (B, C, T, H, W)
        # 输出形状: (B, 128, T_feat, H_feat, W_feat)
        features = self.feature_extractor(x) 
        
        # 检查特征提取器输出的形状是否符合预期，尤其是序列长度 T_feat 和空间维度 H_feat, W_feat
        B, C_feat, T_feat, H_feat, W_feat = features.shape
        # Check if feature extractor maintained sequence length
        if T_feat != self.sequence_length:
             print(f"Warning: Feature extractor output sequence length mismatch. Expected {self.sequence_length}, got {T_feat}.")
             # This mismatch was the cause of the previous error.
             # The simulated attention projection layer's input dimension was calculated based on the *expected* T_feat (self.sequence_length).
             # If the actual T_feat is different, the Linear layer will have the wrong size.
             # To handle this in the simulation, we can either:
             # a) Return None or an empty dict if T_feat != self.sequence_length
             # b) Recreate the attention_projection layer dynamically (less efficient but more flexible for simulation)
             # c) Assume the feature extractor *should* maintain sequence length and this warning indicates an issue with the simulation itself or the actual ATN model.
             # Let's choose option a) for now to highlight the issue with the simulation or actual feature extractor behavior.
             print("Error: Feature extractor did not maintain sequence length. Cannot proceed with attention calculation. Returning empty dict.")
             # Return empty dict if the feature extractor output shape is inconsistent with simulation assumptions
             return {}

        # Check simulated spatial dimensions
        simulated_feature_spatial_h = self.height // 2
        simulated_feature_spatial_w = self.width // 2
        if (H_feat, W_feat) != (simulated_feature_spatial_h, simulated_feature_spatial_w):
             print(f"Warning: Feature spatial shape mismatch. Expected {simulated_feature_spatial_h, simulated_feature_spatial_w}, got {(H_feat, W_feat)}")
             # This mismatch might also cause issues with subsequent layers (decision/attention)
             # Decide how to handle: proceed with potentially incorrect shapes or return None/empty dict
             # Let's proceed for now, but be aware this indicates a potential issue.


        # 2. 决策图生成
        # 决策层处理 features (B, 128, T_feat, H_feat, W_feat)
        # 模拟的 decision_layer 期望输入通道 128，输出通道 1，空间维度下采样 4x4
        # 输出形状: (B, 1, T_feat, H_dec, W_dec)
        # H_dec ≈ H_feat / 4, W_dec ≈ W_feat / 4
        # Note: Decision layer simulation assumes input features channels are 128.
        if C_feat != 128:
             print(f"Warning: Feature channels mismatch for decision layer. Expected 128, got {C_feat}")
             # Decision layer simulation might fail or produce incorrect results.
             # Decide how to handle: proceed or return None/empty dict
             # Let's proceed for now.

        decision_map_raw = self.decision_layer(features)
        
        # 对序列维度求平均并去除单通道维度，得到最终决策图 (B, H_dec, W_dec)
        # 注意：这里假设决策图只与空间位置有关，与序列步骤无关。
        # 如果实际决策图与序列步骤有关，需要修改此处的逻辑。
        # 这里的 .mean(dim=2) 假设 T_feat > 0
        if T_feat > 0:
             # Check if the decision layer output has the expected dimensions before mean/squeeze
             if decision_map_raw.ndim == 5 and decision_map_raw.shape[1] == 1:
                decision_map = decision_map_raw.mean(dim=2).squeeze(1) # -> (B, H_dec, W_dec)
             else:
                  print(f"Warning: Decision layer raw output has unexpected shape {decision_map_raw.shape}. Cannot calculate final decision map. Returning simulated empty tensor.")
                  # Simulate an empty tensor with expected spatial dimensions if calculation fails
                  # H_dec = H_feat / 4, W_dec = W_feat / 4
                  simulated_h_dec = H_feat // 4
                  simulated_w_dec = W_feat // 4
                  decision_map = torch.empty(B, simulated_h_dec, simulated_w_dec, device=features.device) * float('nan') # Use NaN to mark simulated data
        else:
             # 如果序列长度为 0，无法计算平均，返回空张量或处理错误
             print("Warning: Feature sequence length T_feat is 0. Cannot compute decision map mean. Returning simulated empty tensor.")
             # Simulate an empty tensor with expected spatial dimensions
             # H_dec = H_feat / 4, W_dec = W_feat / 4. If H_feat or W_feat is also 0, this might fail.
             simulated_h_dec = H_feat // 4 if H_feat > 0 else 0
             simulated_w_dec = W_feat // 4 if W_feat > 0 else 0
             decision_map = torch.empty(B, simulated_h_dec, simulated_w_dec, device=features.device) * float('nan') # Return empty tensor with expected spatial shape

        # 3. 注意力图生成
        # 模拟的注意力层期望从展平的特征生成注意力矩阵
        # 特征形状: (B, C_feat, T_feat, H_feat, W_feat)
        # 展平除了批量维度外的所有维度
        # Ensure feature dimensions are positive before flattening
        if C_feat > 0 and T_feat > 0 and H_feat > 0 and W_feat > 0:
            flat_features_for_attn = features.view(B, -1) # (B, C_feat * T_feat * H_feat * W_feat)
            expected_attn_proj_in_dim = C_feat * T_feat * H_feat * W_feat
             
             # Check if the attention projection layer was created (i.e., input dimension was positive)
            if self.attention_projection is None:
                  print("Warning: Attention projection layer was not created during init. Cannot calculate attention matrix. Returning simulated matrix.")
                  # Return a simulated attention matrix if projection layer is missing
                  attention_matrix = torch.randn(B, self.num_heads, self.sequence_length, self.sequence_length, device=x.device) * float('nan') # Use NaN to mark simulated data
            elif self.attention_projection.in_features is not None and flat_features_for_attn.shape[1] != self.attention_projection.in_features:
                 print(f"Warning: Attention projection input shape mismatch. Expected {self.attention_projection.in_features}, got {flat_features_for_attn.shape[1]}. This indicates an issue with simulated feature extractor output shape consistency.")
                 # If dimension mismatch occurs, the Linear layer will fail. Return simulated data.
                 print("Error: Flattened feature dimension mismatch with attention projection input dimension. Returning simulated matrix.")
                 attention_matrix = torch.randn(B, self.num_heads, self.sequence_length, self.sequence_length, device=x.device) * float('nan') # Use NaN to mark simulated data
            else:
                 # Use the simulated attention projection layer to map flattened features to flattened attention matrix dimension
                attention_matrix_flat = self.attention_projection(flat_features_for_attn)

                # 将展平的注意力矩阵重塑回 (B, heads, N, N) 的形状
                # 这里的 N 应该对应于注意力机制关注的序列长度。在 Transformer 中通常是输入的序列长度 (self.sequence_length)。
                expected_attn_matrix_shape = (B, self.num_heads, self.sequence_length, self.sequence_length)
                # Check if the flattened attention matrix size matches the expected reshaped size
                if attention_matrix_flat.shape[1] == self.num_heads * self.sequence_length * self.sequence_length:
                    attention_matrix = attention_matrix_flat.view(expected_attn_matrix_shape)
                    # 注意：模拟的注意力矩阵没有经过 Softmax 或其他归一化，实际的注意力会归一化。
                else:
                    # If the flattened dimension doesn't match the expected attention matrix dimension, something is wrong.
                    print(f"Warning: Simulated attention matrix flat dimension mismatch. Expected {self.num_heads * self.sequence_length * self.sequence_length}, got {attention_matrix_flat.shape[1]}. Returning simulated matrix.")
                    attention_matrix = torch.randn(B, self.num_heads, self.sequence_length, self.sequence_length, device=x.device) * float('nan') # Use NaN to mark simulated data

        else:
             # If feature dimensions are non-positive, cannot flatten or compute attention
             print(f"Warning: Feature dimensions are non-positive ({C_feat}, {T_feat}, {H_feat}, {W_feat}). Cannot compute attention matrix. Returning simulated matrix.")
             # Return a simulated attention matrix with NaN values
             attention_matrix = torch.randn(B, self.num_heads, self.sequence_length, self.sequence_length, device=x.device) * float('nan') # Use NaN to mark simulated data


        # 返回包含决策图和注意力矩阵的字典
        atn_outputs = {}
        # Always include decision map, even if simulated with NaN
        atn_outputs['decision'] = decision_map
        # Always include attention matrix, even if simulated with NaN
        atn_outputs['attention'] = attention_matrix

        # Note: Features are not returned by default, only decision and attention as per the return dictionary description.
        # If features are needed, modify this return part based on the return_features flag in get_atn_outputs.

        return atn_outputs

def load_atn_model(
    model_path: str, # 模型权重文件的路径
    device: torch.device, # 计算设备 (例如 'cuda' 或 'cpu')
    in_channels: int, # 输入张量的通道数 (用于初始化模型)
    sequence_length: int, # 输入张量的序列长度 (用于初始化模型)
    height: int, # 输入张量的高度 (用于初始化模型)
    width: int, # 输入张量的宽度 (用于初始化模型)
    num_heads: int, # 注意力头部数量 (用于初始化模型)
    load_feature_head_only: bool = False # 是否只加载特征提取头 (对于 IrisXeonNet 可能不适用)
) -> Optional[nn.Module]:
    """
    加载 ATN 模型权重。目前假设权重文件对应于 IrisXeonNet 结构。
    此函数负责实例化模型并加载预训练权重。

    Args:
        model_path (str): 指向模型权重文件的路径 (.pth 文件)。
        device (torch.device): 模型加载到的计算设备 (如 'cuda' 或 'cpu')。
        in_channels (int): 输入数据的通道数。用于初始化 IrisXeonNet (如果需要)。
        sequence_length (int): 输入数据的序列长度。用于初始化 IrisXeonNet (如果需要)。
        height (int): 输入数据的高度。用于初始化 IrisXeonNet (如果需要)。
        width (int): 输入数据的宽度。用于初始化 IrisXeonNet (如果需要)。
        num_heads (int): 注意力头部数量。用于初始化 IrisXeonNet (如果需要)。
        load_feature_head_only (bool): 是否只加载模型的特征提取头部分的权重。对于 IrisXeonNet 可能不适用，
                                      其权重通常对应整个网络。默认为 False。

    Returns:
        Optional[nn.Module]: 成功加载并加载权重的 ATN 模型实例 (IrisXeonNet)。
                             如果 IrisXeonNet 类未找到或加载权重失败，返回 None。
    """
    print(f"Loading ATN model (assuming IrisXeonNet structure) from {model_path}")

    # 检查 IrisXeonNet 类是否成功导入
    if IrisXeonNet is None:
        print("IrisXeonNet class not found due to import error. Cannot load ATN model.")
        return None

    # 实例化 IrisXeonNet 模型
    # TODO: 检查 IrisXeonNet 的 __init__ 方法需要哪些参数，并根据你的配置 cfg.model.atn 提供。
    # 根据 IrisBabel/nn/CNN/IrisXeonNet.py 文件内容，__init__ 只需要 num_classes (默认为 1000)。
    # 如果你的权重文件需要特定的 num_classes 或其他参数来匹配结构，请在此处指定。
    # 目前假设只需要默认参数或少数参数。
    try:
        # 实例化模型。注意：这里假设 IrisXeonNet 的初始化不需要 cfg 中的所有 atn_* 参数
        # 如果你的 IrisXeonNet 实际需要这些参数，请在这里传递。
        # 根据 IrisXeonNet 的forward方法对输入形状的处理，它可能不需要精确的 sequence_length, height, width 参数初始化。
        atn_model = IrisXeonNet().to(device)
        print("Instantiated IrisXeonNet model.")
    except Exception as e:
        print(f"Error instantiating IrisXeonNet: {e}")
        return None # 如果实例化失败，返回 None


    # 加载模型权重文件
    if os.path.exists(model_path):
        try:
            # 加载权重，使用 weights_only=False 和 strict=False
            # weights_only=False 解决 _pickle.UnpicklingError (ModuleNotFoundError) 问题。
            # strict=False 允许加载 state_dict 中的键名与模型当前结构不完全匹配，
            # 对于迁移学习或部分加载有用，但也可能隐藏问题。
            state_dict = torch.load(model_path, map_location=device, weights_only=False)
            print(f"Loaded state_dict from {model_path} with weights_only=False.")

            # 由于我们假设权重文件直接对应 IrisXeonNet，直接加载到模型
            # 使用 strict=False 以处理键名不完全匹配的情况
            missing_keys, unexpected_keys = atn_model.load_state_dict(state_dict, strict=False)
            print("Attempted to load state dict to IrisXeonNet with strict=False.")

            # 打印加载过程中缺失或意外的键，以供调试
            if missing_keys:
                print(f"Missing keys in loaded state_dict: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys in loaded state_dict: {unexpected_keys}")

            # 如果 missing_keys 包含关键层（如 conv 层或 BN 层），可能表示加载没有成功匹配模型结构
            # 这里的判断可能需要根据实际 IrisXeonNet 的模块命名来调整
            if missing_keys and any(k.startswith(('conv', 'bn', 'bottleneck', 'layer')) for k in missing_keys):
                print("Warning: Significant model keys (like conv/bn/layer) are missing. State dict might not match IrisXeonNet structure well.")
                # 考虑是否在此处返回 None，取决于对加载失败的容忍度
                # return None # 如果认为关键层缺失意味着加载失败


        except Exception as e:
            print(f"Error loading state dict to IrisXeonNet: {e}")
            print("ATN model might not be loaded correctly or state dict is incompatible.")
            # 如果加载失败，返回 None 或根据需要处理
            return None # Return None if loading fails critically

    else:
        # 如果权重文件不存在，打印警告并返回随机初始化的模型实例
        print(f"Warning: ATN model weights not found at {model_path}. Using randomly initialized IrisXeonNet.")
        # 此时返回的模型是随机初始化的，可能无法进行有效的特征提取或攻击
        # 依赖于调用的训练或评估脚本是否能处理这种情况

    # 将模型设置为评估模式，这通常是加载预训练权重后进行推理或评估的标准做法
    atn_model.eval()
    # 返回加载或随机初始化的模型实例
    return atn_model

def get_atn_outputs(
    atn_model: Optional[nn.Module], # ATN 模型实例，可以是 IrisXeonNet 或占位符
    x: torch.Tensor, # 输入张量
    return_features: bool = False, # 是否返回特征提取层的输出
    return_decision: bool = True, # 是否返回决策图输出
    return_attention: bool = True # 是否返回注意力矩阵输出
) -> Dict[str, torch.Tensor]:
    """
    通过 ATN 模型获取指定的输出 (特征、决策图、注意力矩阵)。
    此函数适配了当前占位符 ATN 模型和可能的 IrisXeonNet 实际模型。
    
    Args:
        atn_model (Optional[nn.Module]): ATN 模型实例。如果为 None，将打印警告并返回空字典。
        x (torch.Tensor): 输入张量。对于占位符 ATN，形状期望为 (B, C, T, H, W)。
                          对于实际 IrisXeonNet，输入形状需要根据其 forward 方法的要求进行调整（例如可能的 permute）。
                          **重要**: 训练脚本中 `real_x` 和 `adversarial_x` 的形状是 (B, C, T, W, H)。
                          这意味着输入到此函数的 `x` 形状可能是 (B, C, T, W, H)。
                          然而，模拟的 `AttentiveTriggerNetwork` 和其中的 `IrisFeatureHead` 期望输入是 (B, C, T, H, W)。
                          这存在形状不匹配的问题。需要根据实际 `IrisXeonNet` 期望的输入形状进行调整。
                          **暂时假设 IrisXeonNet 期望输入是 (B, C, T, H, W)，并且其内部 Conv3d 卷积核已适配 W/H 顺序，
                          或者在内部进行了 permute 调整回 (B, C, T, H, W) 以匹配其 Conv3d kernel (1,3,3) 等。**
                          为了与训练脚本中的数据形状一致，假设输入 `x` 为 (B, C, T, W, H)。
                          如果实际 `IrisXeonNet` 期望 (B, C, T, H, W)，则在调用 atn_model(x) 前需要进行 permute。
                          `x = x.permute(0, 1, 2, 4, 3)` 将 (B, C, T, W, H) 转换为 (B, C, T, H, W)。
                          **为了兼容性，在调用模型前进行形状检查和可能的 permute。**

        return_features (bool): 如果为 True，则返回特征提取层的输出。默认为 False。
        return_decision (bool): 如果为 True，则返回决策图输出。默认为 True。
        return_attention (bool): 如果为 True，则返回注意力矩阵输出。默认为 True。

    Returns:
        Dict[str, torch.Tensor]: 包含指定输出的字典。可能的键包括 'features', 'decision', 'attention'。
                                 如果某个输出未指定返回或模型无法提供，则该键不会出现在字典中。
                                 如果模型为 None 或输入无效，返回空字典。
    """
    if atn_model is None:
        print("Warning: ATN model is None. Cannot get ATN outputs.")
        return {}

    # 将模型设置为评估模式并进入 no_grad 上下文，以进行推理
    atn_model.eval()
    with torch.no_grad():
        # 检查输入张量的形状
        # 根据训练脚本 train_generator.py，输入到这里的 x 形状是 (B, C, T, W, H)
        # ATN 模型 (IrisXeonNet 或占位符) 可能期望 (B, C, T, H, W)
        # 检查输入形状，如果与期望的 (B, C, T, H, W) 不符，尝试 permute W 和 H 维度。
        # 期望形状示例: (B, C, T, H, W)
        # 实际可能形状: (B, C, T, W, H)
        B, C, T, dim1, dim2 = x.shape

        # 假设期望的是 H, W 顺序，检查 dim1 和 dim2 是否是期望的 H 和 W
        # 注意：这里需要知道模型期望的 H 和 W。
        # 从 config 中获取模型期望的输入高度和宽度
        # atn_height = atn_model.height if hasattr(atn_model, 'height') else dim1 # 尝试从模型实例获取，否则用输入张量的第一个空间维度
        # atn_width = atn_model.width if hasattr(atn_model, 'width') else dim2 # 尝试从模型实例获取，否则用输入张量的第二个空间维度
        # 更好的方法是依赖于模型自身的形状检查或明确知道其输入要求。

        # 假设模型期望输入形状是 (B, C, T, H, W)
        # 如果输入是 (B, C, T, W, H)，则需要交换最后两个维度
        # 根据 train_generator.py 中的注释，数据加载后是 (B, C, T, H, W)，然后被 permute 成 (B, C, T, W, H) 用于 Generator/Discriminator。
        # 但是，传给 get_atn_outputs 的是原始 real_x，其形状是 (B, C, T, H, W)。
        # 结论：输入到 get_atn_outputs 的 x 形状是 (B, C, T, H, W)。
        # ATN 模型 (IrisXeonNet) 的 forward 方法内部会 permute 成 (B, W, C, T, H)，这看起来不太对劲。
        # 鉴于 IrisXeonNet 的 Conv3d 内核是 (1, 3, 3)，它最可能期望输入是 (B, C, T, H, W) 或 (B, C, D, H, W) 格式。
        # 假设实际 IrisXeonNet 期望输入是 (B, C, T, H, W)，那么 get_atn_outputs 的输入形状是正确的。
        # 如果 IrisXeonNet 的 forward 方法有问题（permute 导致形状不匹配 Conv3d），则需要修改 IrisXeonNet.py。
        # 在这里，我们假设输入 x 是 (B, C, T, H, W)，并直接传递给 atn_model。

        try:
            # 调用 ATN 模型的前向传播
            # atn_model 应该返回一个字典，包含 'decision' 和 'attention'
            # 如果模型是 IrisXeonNet，它目前只返回特征，需要修改其 forward 方法来返回所有输出
            # 暂时假设 atn_model 返回一个字典，或者根据 load_feature_head_only 参数处理

            if hasattr(atn_model, 'forward') and callable(atn_model.forward):
                 # 检查模型是否是我们的占位符 ATN
                 if isinstance(atn_model, AttentiveTriggerNetwork):
                      # 如果是占位符，直接调用其 forward 方法，它返回包含所有输出的字典
                      atn_outputs = atn_model(x) # x 形状 (B, C, T, H, W)
                 elif isinstance(atn_model, IrisXeonNet):
                      # print(f"Debug: Input shape to IrisXeonNet: {x.shape}")
                      features = atn_model(x)
                      # 构建返回字典，只包含特征
                      atn_outputs = {'features': features}
                      # 由于加载的是特征头，不期望有决策和注意力输出
                      if return_decision:
                          print("Warning: get_atn_outputs - return_decision is True, but loading a feature head. Returning None for decision.")
                          atn_outputs['decision'] = None # 返回 None 或空张量表示没有决策输出
                      if return_attention:
                          print("Warning: get_atn_outputs - return_attention is True, but loading a feature head. Returning None for attention.")
                          atn_outputs['attention'] = None # 返回 None 或空张量表示没有注意力输出

                 else:
                      # 如果模型既不是占位符也不是 IrisXeonNet，且有 forward 方法
                      print(f"Warning: Unrecognized ATN model type {type(atn_model)}. Attempting to call forward but output structure is unknown.")
                      # 直接调用 forward，并尝试根据返回类型适配
                      model_output = atn_model(x)
                      atn_outputs = {}
                      # 如果模型返回单个张量，无法确定是特征、决策还是注意力
                      if isinstance(model_output, torch.Tensor):
                           print("Warning: ATN model returned a single tensor. Cannot distinguish outputs.")
                           # 尝试根据形状猜测是什么，但风险很高
                           # 例如，如果形状是 (B, num_classes)，可能是分类 logits
                           # 如果形状是 (B, C, T, H, W)，可能是特征
                           # 为了安全，不自动放入字典
                      elif isinstance(model_output, dict):
                           # 如果返回字典，直接使用
                           atn_outputs = model_output
                      else:
                           print(f"Warning: ATN model returned unexpected type {type(model_output)}. Expected dict or Tensor.")

            else:
                 # 如果模型没有 forward 方法，无法调用
                 print("Warning: ATN model instance does not have a callable forward method.")
                 atn_outputs = {}


        except Exception as e:
            # 捕获模型前向传播中的错误
            print(f"Error during ATN model forward pass: {e}")
            # 在出错时返回空字典或其他指示错误的标记
            return {}

    # 根据 return_* flags 过滤并返回指定输出
    # 注意：如果实际模型无法提供某个输出 (如 IrisXeonNet 当前情况)，即使 flag 为 True 也不会返回。
    final_outputs = {}
    if return_features and 'features' in atn_outputs and atn_outputs['features'] is not None:
        final_outputs['features'] = atn_outputs['features']
    if return_decision and 'decision' in atn_outputs and atn_outputs['decision'] is not None:
        final_outputs['decision'] = atn_outputs['decision']
    if return_attention and 'attention' in atn_outputs and atn_outputs['attention'] is not None:
        final_outputs['attention'] = atn_outputs['attention']
        
    # 打印返回的字典键，以便调试
    # print(f"Debug: get_atn_outputs returning keys: {list(final_outputs.keys())}")

    return final_outputs

# 示例用法 (在训练或评估脚本中): # 添加注释以说明示例用途
# from utils.atn_utils import load_atn_model, get_atn_outputs
# import torch
#
# # 从配置中加载 ATN 模型参数和权重路径
# # model_config = cfg.model.atn
# model_path = 'path/to/your/atn_weights.pth' # 实际权重文件路径
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# in_channels = 3 # 示例输入通道数
# sequence_length = 16 # 示例序列长度
# height = 64 # 示例高度
# width = 64 # 示例宽度
# num_heads = 4 # 示例注意力头部数量
#
# # 加载 ATN 模型
# atn_model = load_atn_model(
#     model_path=model_path,
#     device=device,
#     in_channels=in_channels,
#     sequence_length=sequence_length,
#     height=height,
#     width=width,
#     num_heads=num_heads,
#     load_feature_head_only=False # 根据需要调整
# )
#
# if atn_model is not None:
#     # 准备示例输入数据 (B, C, T, H, W)
#     # 确保输入数据形状与模型期望一致
#     dummy_input = torch.randn(2, in_channels, sequence_length, height, width).to(device)
#
#     # 获取 ATN 输出
#     # 根据训练阶段和需要，选择要返回的输出
#     # 例如，在阶段 1 训练 Generator 时，只需要特征
#     # 在阶段 2 训练 Generator 时，需要决策图和注意力图
#     atn_outputs = get_atn_outputs(
#         atn_model,
#         dummy_input,
#         return_features=True, # 例如需要特征
#         return_decision=False, # 例如不需要决策图
#         return_attention=False # 例如不需要注意力图
#     )
#
#     # 处理获取的输出
#     if 'features' in atn_outputs:
#         features = atn_outputs['features']
#         print(f"Got features with shape: {features.shape}")
#     if 'decision' in atn_outputs:
#          decision = atn_outputs['decision']
#          print(f"Got decision map with shape: {decision.shape}")
#     if 'attention' in atn_outputs:
#          attention = atn_outputs['attention']
#          print(f"Got attention matrix with shape: {attention.shape}")
# else:
#     print("Failed to load ATN model.") 