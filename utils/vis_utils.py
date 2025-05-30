import sys
import os
# 将项目根目录添加到 Python 路径，以便导入其他模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from typing import Optional # 导入 Optional
from typing import List # 导入 List
import traceback
from typing import Any # 导入 Any

# 导入正则化损失中的范数计算函数
# 假设 regularization_losses.py 在 losses 文件夹下
from losses.regularization_losses import linf_norm, l2_norm

# 辅助函数：将特征层转换为可可视化格式
# 这需要根据 ATN 的特征层含义来决定如何可视化 128 个通道和 N 个序列步骤。
# 简单示例：选取一个序列步骤，对通道进行平均或选取某个通道子集
def feature_to_visualizable(features: Optional[torch.Tensor], sequence_step: int = 0, mode: str = 'first_3_channels', channel_subset: Optional[List[int]] = None) -> torch.Tensor:
    """
    将 5D 特征张量 (B, C, N, H, W) 转换为可直接用于可视化的 4D 图像张量 (B, C', H, W)。
    根据指定的模式 (mode) 和序列步骤 (sequence_step) 从原始特征中提取信息。
    
    支持多种可视化模式，以便从不同角度观察特征层的内容和结构。

    Args:
        features (Optional[torch.Tensor]): 输入的 5D 特征张量，形状为 (B, C, N, H, W)。如果为 None 或空，返回空张量。
        sequence_step (int): 指定要可视化的序列（时间）步骤索引 (0-indexed)。如果超出范围，默认为第一个步骤 (0)。
        mode (str): 可视化模式。可选值包括：
                    'first_3_channels': 选取前 3 个通道作为 RGB (如果通道数 C >= 3)，不足 3 个通道时会重复现有通道。
                    'l2_norm': 计算空间维度 (H, W) 上每个位置在所有通道上的 L2 范数，结果形状为 (B, 1, H, W)。用于表示特征空间中各位置的激活强度。
                    'mean': 计算空间维度 (H, W) 上每个位置在所有通道上的平均值，结果形状为 (B, 1, H, W)。用于表示特征空间中各位置的平均激活水平。
                    'channel_subset': 可视化指定的通道子集。需要提供 `channel_subset` 参数。
        channel_subset (Optional[List[int]], optional): 当 mode 为 'channel_subset' 时，指定要可视化的通道索引列表。如果列表为空或包含无效索引，会给出警告。

    Returns:
        torch.Tensor: 可可视化格式的 4D 张量，形状为 (B, C', H, W)。
                      返回的张量已归一化到 [0, 1] 范围，并且设备与输入张量相同（如果输入非空）。
                      如果输入无效或无法可视化，返回形状为 (0,) 的空张量。
    """
    # 检查输入特征是否为None或空张量
    if features is None or features.numel() == 0:
        # print("Warning: Input features are None or empty. Cannot visualize features.") # 避免过多打印
        return torch.empty(0) # 返回空张量

    # 确保输入张量是 5D (B, C, N, H, W)
    if features.ndim != 5:
         print(f"Warning: Features tensor must be 5D (B, C, N, H, W), got {features.ndim}D. Cannot visualize features.")
         return torch.empty(0, device=features.device) # 返回设备正确的空张量

    B, C, N, H, W = features.shape
    
    # 检查序列维度 N 是否为零
    if N == 0:
        # print("Warning: Sequence dimension N is 0. Cannot visualize features.") # 避免过多打印
        return torch.empty(0, device=features.device) # 返回设备正确的空张量

    # 检查并调整 sequence_step，确保在有效范围内
    if sequence_step >= N or sequence_step < 0:
        # print(f"Warning: sequence_step {sequence_step} is out of bounds for N={N}. Defaulting to 0.") # 避免过多打印
        sequence_step = 0 # 默认可视化第一个步骤

    # 选取指定序列步骤的特征切片，形状变为 (B, C, H, W)
    features_step = features[:, :, sequence_step, :, :]

    visualizable_features: torch.Tensor # 预定义变量类型以便后续赋值

    # 根据可视化模式处理特征切片
    if mode == 'first_3_channels':
        # 如果通道数 C >= 3，选取前 3 个通道作为 RGB
        if C >= 3:
            visualizable_features = features_step[:, :3, :, :] # (B, 3, H, W)
        # 如果通道数 C == 1，重复第一通道 3 次以形成 3 通道图像
        elif C == 1:
            visualizable_features = features_step.repeat(1, 3, 1, 1) # (B, 1, H, W) -> (B, 3, H, W)
        else:
            # 如果通道数 C 是 2 或其他小于 3 的情况（且 C > 0），重复第一通道 3 次
            if C > 0:
                visualizable_features = features_step[:, 0, :, :].unsqueeze(1).repeat(1, 3, 1, 1) # (B, 1, H, W) -> (B, 3, H, W)
            else: # C == 0
                # 警告：特征通道 C=0。无法创建 first_3_channels 可视化特征。
                print("Warning: Feature channels C=0. Cannot create visualizable features for 'first_3_channels'.")
                return torch.empty(0, device=features.device) # 返回设备正确的空张量

    elif mode == 'l2_norm':
        # 计算空间维度上每个位置在所有通道上的 L2 范数
        # 结果形状为 (B, 1, H, W)，keepdim=True 保留通道维度
        l2_norm_map = torch.linalg.norm(features_step, dim=1, keepdim=True) # (B, 1, H, W)
        visualizable_features = l2_norm_map

    elif mode == 'mean':
        # 计算空间维度上每个位置在所有通道上的平均值
        # 结果形状为 (B, 1, H, W)，keepdim=True 保留通道维度
        mean_map = torch.mean(features_step, dim=1, keepdim=True) # (B, 1, H, W)
        visualizable_features = mean_map

    elif mode == 'channel_subset':
        # 检查是否提供了通道子集列表且列表不为空
        if channel_subset is None or not channel_subset:
            # 警告：'channel_subset' 模式需要提供 'channel_subset' 列表。将使用 'l2_norm' 模式替代。
            print("Warning: 'channel_subset' mode requires 'channel_subset' list. Using 'l2_norm' mode instead.")
            # 如果未提供有效子集，退回使用 L2 Norm 模式
            return feature_to_visualizable(features, sequence_step, mode='l2_norm')

        # 过滤出有效的通道索引
        valid_channels = [c for c in channel_subset if 0 <= c < C]
        # 如果没有有效的通道索引，无法可视化
        if not valid_channels:
            # 警告：指定的通道子集 {} 对于通道数 C={} 无效。无法可视化通道子集。
            print(f"Warning: None of the specified channel subset {channel_subset} are valid for C={C}. Cannot visualize channel subset.")
            return torch.empty(0, device=features.device) # 返回设备正确的空张量

        # 选取指定的通道子集，形状为 (B, num_subset_channels, H, W)
        visualizable_features = features_step[:, valid_channels, :, :]

        # 为了使用 make_grid 可视化，输出张量通常需要是 1 或 3 通道
        # 如果子集只有一个通道，形状已经是 (B, 1, H, W)，无需处理
        if visualizable_features.shape[1] == 1:
            pass
        # 如果子集通道数是 2，重复第一个通道使其成为 3 通道 (用于 RGB 显示)
        elif visualizable_features.shape[1] == 2:
            # 警告：通道子集有 2 个通道。重复第一个通道以使其成为 3 通道用于可视化。
            print("Warning: Channel subset has 2 channels. Repeating the first channel to make it 3 for visualization.")
            visualizable_features = torch.cat([visualizable_features, visualizable_features[:, :1, :, :]], dim=1) # (B, 3, H, W)
        # 如果子集通道数 > 3，只取前三个通道进行 RGB 可视化
        elif visualizable_features.shape[1] > 3:
            # 警告：通道子集有 {} 个通道。将每个选定通道作为独立的灰度图像可视化。
            print(f"Warning: Channel subset has {visualizable_features.shape[1]} channels. Visualizing each selected channel as a separate grayscale image.")
            # Collect each channel as a separate image (B, 1, H, W)
            channel_images = [visualizable_features[:, i, :, :].unsqueeze(1) for i in range(visualizable_features.shape[1])]
            # Stack these single-channel images. Resulting shape: (B * num_subset_channels, 1, H, W)
            # This structure is suitable for make_grid
            visualizable_features = torch.cat(channel_images, dim=0) # Concatenate along batch dimension
        # 如果子集通道数是 0 (尽管前面检查了 non_empty)，理论上不可能到达
        elif visualizable_features.shape[1] == 0:
            # 警告：通道子集为空。无法可视化。
            print("Warning: Channel subset is empty. Cannot visualize.")
            return torch.empty(0, device=features.device) # 返回设备正确的空张量

    else:
        # 警告：不支持的可视化模式：{}。请选择 'first_3_channels', 'l2_norm', 'mean', 或 'channel_subset'。将使用 'l2_norm' 模式替代。
        print(f"Warning: Unsupported visualization mode: {mode}. Choose 'first_3_channels', 'l2_norm', 'mean', or 'channel_subset'. Using 'l2_norm' instead.")
        # 如果模式无效，退回使用 L2 Norm 模式
        return feature_to_visualizable(features, sequence_step, mode='l2_norm')


    # 将可视化特征张量的值缩放到 [0, 1] 范围以便显示
    # 如果可视化张量为空，则无需归一化
    if visualizable_features.numel() == 0:
        return visualizable_features # 返回空张量

    # 对于 'l2_norm' 和 'mean' 模式，通常在整个批次上进行归一化，以显示批次内的相对强度。
    # 对于 'first_3_channels' 或 'channel_subset'，按样本独立归一化可能更合适，以保持颜色感知一致性。
    # 决定：对所有模式按样本独立归一化，这样更通用，避免批次间颜色/亮度差异过大。

    normalized_batch = []
    # 遍历批次中的每个样本 (每个图像)
    for img in visualizable_features: # img shape (C', H, W)
        min_val = img.min()
        max_val = img.max()
        # 避免除以零或数值不稳定
        if max_val > min_val + 1e-8:
            normalized_img = (img - min_val) / (max_val - min_val + 1e-8)
        else:
            # 如果所有值都相同，图像将是单色，设置为全零（显示为黑色）
            normalized_img = torch.zeros_like(img, device=img.device)
        normalized_batch.append(normalized_img)
    
    # 将处理后的样本重新堆叠成一个批次张量
    visualizable_features = torch.stack(normalized_batch, dim=0)

    # 确保返回 float 类型张量
    return visualizable_features.float()


# 可视化训练损失
def visualize_training_losses(writer: SummaryWriter, loss_dict: dict, step: int):
    """
    将训练过程中的各种损失值记录到 TensorBoard 的 Scalars 选项卡中。

    Args:
        writer (SummaryWriter): TensorBoard 的 SummaryWriter 对象，用于写入日志。
        loss_dict (dict): 一个字典，键是损失名称 (str)，值是对应的损失值 (可以是标量张量或 Python 数字)。
        step (int): 当前的训练全局步数 (global step)，用作 TensorBoard 图表的横轴。
    """
    # 遍历损失字典中的每一项
    for name, loss_val in loss_dict.items():
        # 检查损失值是否是 PyTorch 张量
        if isinstance(loss_val, torch.Tensor):
            # 检查张量是否是标量 (ndim == 0)
            if loss_val.ndim == 0:
                # 如果是标量张量，直接将其转换为 Python 数字并记录
                writer.add_scalar(f'Loss/{name}', loss_val.item(), step)
            else:
                 # 如果张量不是标量，计算其平均值并记录（例如 Batch Loss 张量）
                 # print(f"Warning: Loss tensor '{name}' is not scalar ({loss_val.shape}). Logging mean.") # 避免过多打印
                 writer.add_scalar(f'Loss/{name}', loss_val.mean().item(), step) # 记录平均值

        # 如果损失值不是张量，假设它是 Python 数字
        else:
            writer.add_scalar(f'Loss/{name}', loss_val, step)

def visualize_perturbation_norms(writer: SummaryWriter, delta: Optional[torch.Tensor], step: int):
    """
    计算并记录生成的扰动张量 (delta) 的 L-infinity 范数和 L2 范数的平均值到 TensorBoard。
    这些指标用于监控扰动的大小，评估其隐蔽性和攻击强度。

    Args:
        writer (SummaryWriter): TensorBoard 的 SummaryWriter 对象。
        delta (Optional[torch.Tensor]): 生成的扰动张量，形状与原始输入图像相同 (B, C, H, W)。如果为 None 或空，则跳过记录。
        step (int): 当前的训练全局步数。
    """
    # 检查输入扰动张量是否为None或空
    if delta is None or delta.numel() == 0:
        # print("Warning: Delta tensor is None or empty. Skipping perturbation norm visualization.") # 避免过多打印
        return

    # 确保扰动张量是 5D (B, C, N, H, W)
    if delta.ndim != 5:
         print(f"Warning: Delta tensor is not 5D ({delta.ndim}D). Cannot calculate perturbation norms.")
         return

    with torch.no_grad(): # 在 no_grad 模式下计算范数，不影响梯度
        # 计算每个样本的 L-infinity 范数和 L2 范数 (返回形状为 (B,) 的张量)
        # linf_norm 和 l2_norm 函数已经在 losses.regularization_losses 中定义
        linf_norms = linf_norm(delta)
        l2_norms = l2_norm(delta)

        # 计算这些范数在批次维度上的平均值 (得到标量)
        avg_linf = linf_norms.mean().item()
        avg_l2 = l2_norms.mean().item()

        # 将平均范数记录到 TensorBoard
        writer.add_scalar('Perturbation/Linf_Norm_Avg', avg_linf, step) # 范数名称改为带有 _Avg 后缀，与其他评估指标一致
        writer.add_scalar('Perturbation/L2_Norm_Avg', avg_l2, step)   # 范数名称改为带有 _Avg 后缀

def visualize_samples_and_outputs(writer: SummaryWriter,
                                  original_image: Optional[torch.Tensor],   # 原始图像
                                  original_features: Optional[torch.Tensor], # 原始特征 (Stage 1)
                                  feature_delta: Optional[torch.Tensor],    # 特征扰动 (Stage 1)
                                  adversarial_features: Optional[torch.Tensor], # 对抗特征 (Stage 1)
                                  original_decision_map: Optional[torch.Tensor], # 原始决策图 (Stage 1)
                                  adversarial_decision_map: Optional[torch.Tensor], # 对抗决策图 (Stage 1)
                                  step: int,
                                  num_samples: int = 4,
                                  sequence_step_to_vis: int = 0,
                                  visualize_decision_diff: bool = True): # 是否可视化决策图差异 (主要用于 Stage 2)
    """
    可视化原始图像、对抗样本、扰动、特征层、决策图等到 TensorBoard。
    Modified for Stage 2 to focus on image inputs.

    Args:
        writer (SummaryWriter): TensorBoard 的 SummaryWriter 对象。
        original_image (Optional[torch.Tensor]): 原始输入图像批次 (B, C, T, H, W).
        # The following arguments are primarily for Stage 1 and may not be used in Stage 2 visualization.
        original_features (Optional[torch.Tensor]): Original ATN features (B, C, N, H, W).
        feature_delta (Optional[torch.Tensor]): Stage 1 feature perturbation (B, C, N, H, W).
        adversarial_features (Optional[torch.Tensor]): Adversarial ATN features (B, C, N, H, W).
        original_decision_map (Optional[torch.Tensor]): Original decision map (B, H, W) or (B, N, H, W).
        adversarial_decision_map (Optional[torch.Tensor]): Adversarial decision map (B, H, W) or (B, N, H, W).
        step (int): 当前的训练全局步数。
        num_samples (int): 要可视化的样本数量。
        sequence_step_to_vis (int): 要可视化的序列步骤索引。
        visualize_decision_diff (bool): Whether to visualize the difference between original and adversarial decision maps (primarily for Stage 1).
    """
    # --- Add Debug Prints for Tensor Shapes and numel() --- (Keep for debugging if needed)
    # print(f"DEBUG VIS: Step {step}")
    # print(f"DEBUG VIS: original_image shape: {original_image.shape if original_image is not None else None}, numel: {original_image.numel() if original_image is not None else 0}")
    # print(f"DEBUG VIS: original_features shape: {original_features.shape if original_features is not None else None}, numel: {original_features.numel() if original_features is not None else 0}")
    # print(f"DEBUG VIS: feature_delta shape: {feature_delta.shape if feature_delta is not None else None}, numel: {feature_delta.numel() if feature_delta is not None else 0}")
    # print(f"DEBUG VIS: adversarial_features shape: {adversarial_features.shape if adversarial_features is not None else None}, numel: {adversarial_features.numel() if adversarial_features is not None else 0}")
    # print(f"DEBUG VIS: original_decision_map shape: {original_decision_map.shape if original_decision_map is not None else None}, numel: {original_decision_map.numel() if original_decision_map is not None else 0}")
    # print(f"DEBUG VIS: adversarial_decision_map shape: {adversarial_decision_map.shape if adversarial_decision_map is not None else None}, numel: {adversarial_decision_map.numel() if adversarial_decision_map is not None else 0}")
    # --- End Debug Prints ---

    # This function is primarily for Stage 1 visualization of features and decision maps.
    # For Stage 2 (Pixel Attack), we will primarily use visualize_stage2_pixel_attack.
    # However, this function can still be used to visualize the original image input if needed.
    # We will retain the image visualization part and remove the feature/decision map parts.

    # --- 可视化原始图像 (如果可用) ---
    # 假设原始图像是 5D (B, C_img, T, H_img, W_img)
    if original_image is not None and original_image.numel() > 0 and original_image.ndim == 5:
        # 确定实际要显示的样本数量
        batch_size = original_image.shape[0]
        num_samples_to_show = min(num_samples, batch_size)

        # 检查序列步骤是否有效，并提取指定步骤的图像 (B, C_img, H_img, W_img)
        img_seq_len = original_image.shape[2]
        vis_step_img = min(sequence_step_to_vis, img_seq_len - 1) # 确保步骤在范围内
        vis_step_img = max(0, vis_step_img)

        original_image_step = original_image[:num_samples_to_show, :, vis_step_img, :, :].squeeze(2).cpu().detach() # Remove sequence dim, move to CPU, detach
        # make_grid needs input shape (B, C, H, W)
        # Assuming original image is 3 channels (RGB)
        if original_image_step.shape[1] == 3:
            grid_image = make_grid(original_image_step, nrow=num_samples_to_show, normalize=True)
            writer.add_image('Samples/Original_Image', grid_image, step)
        else:
            # 警告：原始图像不是 3 通道 ({})。跳过原始图像可视化。
            print(f"Warning: Original image is not 3 channels ({original_image_step.shape[1]}). Skipping original image visualization in visualize_samples_and_outputs.")

    # --- Remove Stage 1 specific visualizations ---
    # Commenting out or removing the logic for visualizing features and decision maps.
    # The _process_decision_map helper can also be removed or kept if needed elsewhere.
    pass # Placeholder for removed code

# Keep visualize_attention_maps as it might be useful for analysis
# Keep visualize_training_losses and visualize_perturbation_norms

# Ensure visualize_stage2_pixel_attack is correctly defined and used in the training script
# This function should handle visualization of original images, adversarial images, pixel deltas, and TriggerNet outputs.

# TODO: 可视化注意力图
def visualize_attention_maps(writer: SummaryWriter,
                             attention_matrix_orig: Optional[torch.Tensor], # 原始注意力矩阵 (B, head, N, N)
                             attention_matrix_adv: Optional[torch.Tensor], # 对抗注意力矩阵 (B, head, N, N)
                             step: int,
                             num_samples: int = 1,
                             num_heads_to_vis: int = 1):
    """
    可视化原始注意力和对抗注意力矩阵到 TensorBoard。

    Args:
        writer (SummaryWriter): TensorBoard 的 SummaryWriter 对象。
        attention_matrix_orig (Optional[torch.Tensor]): 原始样本的注意力矩阵批次 (B, head, N, N)。
        attention_matrix_adv (Optional[torch.Tensor]): 对抗样本的注意力矩阵批次 (B, head, N, N)。
        step (int): 当前的训练全局步数。
        num_samples (int): 要可视化的样本数量。
        num_heads_to_vis (int): 要可视化的注意力头数量。
    """
    # print(f"Debug: visualize_attention_maps called at step {step}") # Debug print

    # 检查输入注意力矩阵是否为None或空，并获取批量大小和头部数量
    if attention_matrix_orig is None or attention_matrix_orig.numel() == 0:
        # print("Warning: Original attention matrix is None or empty. Skipping attention map visualization.") # 避免过多打印
        return

    # 确保注意力矩阵是 4D (B, head, N, N)
    if attention_matrix_orig.ndim != 4:
         print(f"Warning: Original attention matrix is not 4D ({attention_matrix_orig.ndim}D). Skipping attention map visualization.")
         return

    B, head, N1, N2 = attention_matrix_orig.shape
    if N1 != N2:
         print(f"Warning: Original attention matrix spatial dimensions are not square ({N1}x{N2}). Skipping attention map visualization.")
         return
    if head == 0:
         print("Warning: Attention matrix has 0 heads. Skipping attention map visualization.")
         return

    # 确保对抗注意力矩阵存在、非空且形状匹配原始矩阵
    if attention_matrix_adv is None or attention_matrix_adv.numel() == 0 or attention_matrix_adv.shape != attention_matrix_orig.shape:
         # 警告：对抗注意力矩阵为None、空或形状与原始矩阵不匹配 ({} vs {})。只可视化原始注意力图。
         print(f"Warning: Adversarial attention matrix is None/empty or shape mismatch ({attention_matrix_adv.shape if attention_matrix_adv is not None else 'None'} vs {attention_matrix_orig.shape}). Visualizing original attention maps only.")
         attention_matrix_adv = None # 将对抗矩阵设为 None，只可视化原始


    # 确定实际要显示的样本数量和头部数量
    num_samples_to_show = min(num_samples, B)
    num_heads_to_show = min(num_heads_to_vis, head)
    # print(f"Debug: Visualizing {num_samples_to_show} samples and {num_heads_to_show} heads.") # Debug print

    # 将要显示的张量移动到 CPU (如果不在 CPU 上) 并选取切片
    attention_matrix_orig_cpu = attention_matrix_orig[:num_samples_to_show, :num_heads_to_show].cpu().detach()
    attention_matrix_adv_cpu = attention_matrix_adv[:num_samples_to_show, :num_heads_to_show].cpu().detach() if attention_matrix_adv is not None else None

    # 遍历选定的样本和注意力头进行可视化
    for i in range(num_samples_to_show):
        for j in range(num_heads_to_show):
            # 获取当前样本和头部的原始注意力矩阵 (N, N)
            orig_attn = attention_matrix_orig_cpu[i, j, :, :]

            # 将注意力矩阵转换为图像格式
            # 注意力矩阵值通常在 [0, 1] 或经过 Softmax 后，可以直接作为灰度图或热力图可视化
            # 为了在 TensorBoard 中显示为图像，需要增加通道维度 (1, N, N)
            # 可以使用 make_grid 将单个注意力图转为网格 (虽然只有一个图)
            
            # 可视化原始注意力图
            # make_grid 需要输入形状 (B, C, H, W)，这里 Batch=1, C=1
            # 注意力图通常归一化到 [0, 1]
            grid_orig_attn = make_grid(orig_attn.unsqueeze(0).unsqueeze(0), nrow=1, normalize=True)
            writer.add_image(f'Attention_Maps/Sample{i}_Head{j}/Original', grid_orig_attn, step)

            # 如果对抗注意力矩阵可用，可视化对抗注意力图和差异图
            if attention_matrix_adv_cpu is not None:
                # 获取当前样本和头部的对抗注意力矩阵 (N, N)
                adv_attn = attention_matrix_adv_cpu[i, j, :, :]

                # 可视化对抗注意力图
                grid_adv_attn = make_grid(adv_attn.unsqueeze(0).unsqueeze(0), nrow=1, normalize=True)
                writer.add_image(f'Attention_Maps/Sample{i}_Head{j}/Adversarial', grid_adv_attn, step)

                # 可视化原始注意力图与对抗注意力图的差异 (绝对差异)
                attn_diff = torch.abs(adv_attn - orig_attn)
                # 归一化差异图以便可视化 (0到最大差异)
                if attn_diff.numel() > 0:
                     min_val = attn_diff.min()
                     max_val = attn_diff.max()
                     if max_val > min_val + 1e-8:
                          normalized_diff_map = (attn_diff - min_val) / (max_val - min_val + 1e-8)
                     else:
                          normalized_diff_map = torch.zeros_like(attn_diff, device=attn_diff.device)

                     grid_attn_diff = make_grid(normalized_diff_map.unsqueeze(0).unsqueeze(0), nrow=1, normalize=False) # 差异图通常不需要 normalize 到 [0,1]
                     writer.add_image(f'Attention_Maps/Sample{i}_Head{j}/Difference', grid_attn_diff, step)

    # TODO: 可视化 Top-K 位置的变化或值下降 (如果相关)
    # 这可能需要计算并绘制 Top-K 索引的热力图或标记图
    # 或者在 visualize_samples_and_outputs 函数中叠加标记

def visualize_stage2_pixel_attack(writer: SummaryWriter,
                                    original_images: torch.Tensor,
                                    adversarial_images: torch.Tensor,
                                    pixel_deltas: torch.Tensor,
                                    original_trigger_output: Optional[torch.Tensor], # TriggerNet output
                                    adversarial_trigger_output: Optional[torch.Tensor], # TriggerNet output
                                    step: int,
                                    cfg: Any, # Add cfg parameter to access config
                                    num_samples: int = 4,
                                    sequence_steps_to_vis: Optional[List[int]] = None): # Still accept list for flexibility or future use
    """
    可视化 Stage 2 像素级攻击训练过程的关键元素：
    原始图像、对抗图像、像素扰动，以及原始和对抗样本的 TriggerNet 输出。
    根据配置中的 visualize_mode (logging.visualize_mode)，选择使用 add_video 或 make_grid 进行可视化。

    Args:
        writer (SummaryWriter): TensorBoard 的 SummaryWriter 对象。
        original_images (torch.Tensor): 原始输入图像批次 (B, C, H, W) 或 (B, C, T, H, W)。
        adversarial_images (torch.Tensor): 添加扰动后的对抗图像批次 (B, C, H, W) 或 (B, C, T, H, W)。
        pixel_deltas (torch.Tensor): 生成的像素级扰动批次 (B, C, H, W) 或 (B, C, T, H, W)。
        original_trigger_output (Optional[torch.Tensor]): 原始样本通过 ATN 后的 TriggerNet 输出。
        adversarial_trigger_output (Optional[torch.Tensor]): 对抗样本通过 ATN 后的 TriggerNet 输出。
        step (int): 当前的训练全局步数。
        cfg (Any): 配置对象，用于读取 visualize_mode 等配置。
        num_samples (int): 要可视化的样本数量。对于视频可视化，这是要可视化的视频数量。
        sequence_steps_to_vis (Optional[List[int]], optional): (仅用于 image_grid 模式) 用于指定要可视化的序列步骤列表。当输入为 4D 时，此参数被忽略。如果为 None，对于 5D 输入，在 image_grid 模式下默认可视化第一个步骤 (0)。
    """
    # Ensure tensors are 4D (B, C, H, W) or 5D (B, C, T, H, W)

    # Get visualization mode from config
    visualize_mode = getattr(cfg.logging, 'visualize_mode', 'video') # Default to video if not specified

    # Determine batch size and number of samples to show
    batch_size = original_images.shape[0]
    num_samples_to_show = min(num_samples, batch_size)

    # Check if input is 5D (has sequence dimension)
    is_5d_input = original_images.ndim == 5

    if visualize_mode == "video" and is_5d_input:
        # --- Video Visualization Mode (for 5D input) ---
        # Input original/adversarial images are (B, T, H, W, C)
        # Input pixel deltas are (B, C, T, H, W)
        # add_video expects (N, T, C, H, W)

        # Slice batch for all inputs
        original_videos = original_images[:num_samples_to_show].cpu().detach()
        adversarial_videos = adversarial_images[:num_samples_to_show].cpu().detach()
        pixel_delta_videos = pixel_deltas[:num_samples_to_show].cpu().detach()

        # Ensure images are 3 channels (RGB) for video visualization and permute to (N, T, C, H, W)
        # Original/Adversarial images are (B, T, H, W, C), need to permute to (N, T, C, H, W)
        if original_videos.shape[4] != 3:
             print(f"Warning: Original input videos are not 3 channels ({original_videos.shape[4]}). Skipping video visualization.")
             return
        # Permute (N, T, H, W, C) to (N, T, C, H, W)
        original_videos_for_vis = original_videos.permute(0, 1, 4, 2, 3)
        adversarial_videos_for_vis = adversarial_videos.permute(0, 1, 4, 2, 3)

        # Pixel deltas are (B, C, T, H, W), need to permute to (N, T, C, H, W)
        if pixel_delta_videos.shape[1] != 3:
             print(f"Warning: Pixel delta videos are not 3 channels ({pixel_delta_videos.shape[1]}). Skipping video visualization.")
             return
        # Permute (N, C, T, H, W) to (N, T, C, H, W)
        pixel_delta_videos_for_vis = pixel_delta_videos.permute(0, 2, 1, 3, 4)


        # Ensure there are videos to show after slicing and permutation
        if original_videos_for_vis.shape[0] == 0 or original_videos_for_vis.shape[1] == 0:
            print(f"Warning: No complete video sequences to show after slicing batch ({original_videos_for_vis.shape[0]} samples, {original_videos_for_vis.shape[1]} frames). Skipping video visualization.")
            return

        # --- Visualize Videos ---
        # Original Images Video
        writer.add_video(f'Stage2_Visualization/Videos/Original_Images', original_videos_for_vis, step, fps=getattr(cfg.logging, 'video_fps', 4))

        # Adversarial Images Video
        writer.add_video(f'Stage2_Visualization/Videos/Adversarial_Images', adversarial_videos_for_vis, step, fps=getattr(cfg.logging, 'video_fps', 4))

        # Pixel Deltas Video (Normalize for visualization)
        # Normalize based on the batch's min/max over the whole video
        min_delta = pixel_delta_videos_for_vis.min() if pixel_delta_videos_for_vis.numel() > 0 else 0.0
        max_delta = pixel_delta_videos_for_vis.max() if pixel_delta_videos_for_vis.numel() > 0 else 1.0
        if max_delta > min_delta + 1e-8:
             normalized_delta_videos = (pixel_delta_videos_for_vis - min_delta) / (max_delta - min_delta + 1e-8)
        else:
             normalized_delta_videos = torch.zeros_like(pixel_delta_videos_for_vis, device=pixel_delta_videos_for_vis.device)

        # add_video expects values in [0, 255] uint8 or [0, 1] float. Normalizing to [0, 1].
        writer.add_video(f'Stage2_Visualization/Videos/Pixel_Deltas', normalized_delta_videos, step, fps=getattr(cfg.logging, 'video_fps', 4))

        # --- Visualize TriggerNet Outputs Video (If available and 5D) ---
        # Assume TriggerNet output is 5D (B, C_out, N, H', W') for video visualization
        # Need to check TriggerNet output shape and potentially permute to (N, T, C_out, H', W')
        if original_trigger_output is not None and adversarial_trigger_output is not None and original_trigger_output.ndim == 5 and adversarial_trigger_output.ndim == 5:
             if original_trigger_output.shape[0] >= num_samples_to_show and original_trigger_output.shape[2] > 0:
                  # Slice batch
                  original_trigger_output_vis_sliced = original_trigger_output[:num_samples_to_show].cpu().detach() # (N, C_out, T, H', W')
                  adversarial_trigger_output_vis_sliced = adversarial_trigger_output[:num_samples_to_show].cpu().detach() # (N, C_out, T, H', W')

                  # Permute to (N, T, C_out, H', W') for add_video
                  original_trigger_videos = original_trigger_output_vis_sliced.permute(0, 2, 1, 3, 4)
                  adversarial_trigger_videos = adversarial_trigger_output_vis_sliced.permute(0, 2, 1, 3, 4)

                  # Process channels for visualization (ensure 1 or 3 channels)
                  def _process_trigger_output_channels_for_video(trigger_videos: torch.Tensor) -> Optional[torch.Tensor]:
                       # Input is (N, T, C_out, H', W')
                       N_vid, T_vid, C_out_vid, H_vid, W_vid = trigger_videos.shape
                       if C_out_vid == 1:
                            return trigger_videos # Already 1 channel
                       elif C_out_vid >= 3:
                             return trigger_videos[:, :, :3, :, :] # Take first 3 channels
                       elif C_out_vid > 0: # Take mean across channels
                            # Ensure mean is over channel dim (index 2)
                            return torch.mean(trigger_videos, dim=2, keepdim=True).repeat(1, 1, 3, 1, 1) # Mean over C_out dim, then repeat to 3 channels
                       else:
                            print("Warning: TriggerNet output has 0 channels. Cannot visualize.")
                            return None

                  processed_orig_trigger_videos = _process_trigger_output_channels_for_video(original_trigger_videos)
                  processed_adv_trigger_videos = _process_trigger_output_channels_for_video(adversarial_trigger_videos)

                  if processed_orig_trigger_videos is not None and processed_orig_trigger_videos.shape[2] in [1, 3]: # Check processed channels
                       # Normalize for visualization ([0, 1] range)
                       min_val = processed_orig_trigger_videos.min() if processed_orig_trigger_videos.numel() > 0 else 0.0
                       max_val = processed_orig_trigger_videos.max() if processed_orig_trigger_videos.numel() > 0 else 1.0
                       if max_val > min_val + 1e-8:
                            normalized_orig_trigger = (processed_orig_trigger_videos - min_val) / (max_val - min_val + 1e-8)
                       else:
                            normalized_orig_trigger = torch.zeros_like(processed_orig_trigger_videos, device=processed_orig_trigger_videos.device)

                       writer.add_video(f'Stage2_Visualization/Videos/Original_TriggerNet_Output', normalized_orig_trigger, step, fps=getattr(cfg.logging, 'video_fps', 4))

                  if processed_adv_trigger_videos is not None and processed_orig_trigger_videos is not None and processed_adv_trigger_videos.shape == processed_orig_trigger_videos.shape and processed_adv_trigger_videos.shape[2] in [1, 3]: # Check processed channels
                       # Normalize for visualization ([0, 1] range)
                       min_val = processed_adv_trigger_videos.min() if processed_adv_trigger_videos.numel() > 0 else 0.0
                       max_val = processed_adv_trigger_videos.max() if processed_adv_trigger_videos.numel() > 0 else 1.0
                       if max_val > min_val + 1e-8:
                            normalized_adv_trigger = (processed_adv_trigger_videos - min_val) / (max_val - min_val + 1e-8)
                       else:
                            normalized_adv_trigger = torch.zeros_like(processed_adv_trigger_videos, device=processed_adv_trigger_videos.device)

                       writer.add_video(f'Stage2_Visualization/Videos/Adversarial_TriggerNet_Output', normalized_adv_trigger, step, fps=getattr(cfg.logging, 'video_fps', 4))

                       # TriggerNet Output Difference Video
                       if processed_orig_trigger_videos.device != processed_adv_trigger_videos.device:
                            # print("Warning: Original and adversarial TriggerNet output videos are on different devices. Moving adversarial to original device.")
                            processed_adv_trigger_videos = processed_adv_trigger_videos.to(processed_orig_trigger_videos.device)

                       trigger_output_diff_videos = torch.abs(processed_adv_trigger_videos - processed_orig_trigger_videos)
                       if trigger_output_diff_videos.numel() > 0:
                            min_diff = trigger_output_diff_videos.min() if trigger_output_diff_videos.numel() > 0 else 0.0
                            max_diff = trigger_output_diff_videos.max() if trigger_output_diff_videos.numel() > 0 else 1.0
                            if max_diff > min_diff + 1e-8:
                                 normalized_diff_videos = (trigger_output_diff_videos - min_diff) / (max_diff - min_diff + 1e-8)
                            else:
                                 normalized_diff_videos = torch.zeros_like(trigger_output_diff_videos, device=trigger_output_diff_videos.device)

                            # Ensure difference video is 1 or 3 channels before adding
                            if normalized_diff_videos.shape[2] in [1, 3]:
                                 writer.add_video(f'Stage2_Visualization/Videos/TriggerNet_Output_Difference', normalized_diff_videos, step, fps=getattr(cfg.logging, 'video_fps', 4))
                            else:
                                 print(f"Warning: TriggerNet output difference video is not 1 or 3 channels ({normalized_diff_videos.shape[2]}). Skipping visualization.")
                       else:
                            print("Warning: TriggerNet output difference videos are empty. Skipping visualization.")
                  elif processed_adv_trigger_videos is not None: # If original is None but adversarial is not
                       print("Warning: Original TriggerNet output videos are None but adversarial are not. Cannot visualize.")
                  elif processed_orig_trigger_videos is not None and processed_adv_trigger_videos is None: # If adversarial is None but original is not
                       print("Warning: Adversarial TriggerNet output videos are None but original are not. Cannot visualize.")
             else:
                 print(f"Warning: TriggerNet output batch size ({original_trigger_output.shape[0]}) is less than num_samples_to_show ({num_samples_to_show}) or sequence length ({original_trigger_output.shape[2]}) is 0. Skipping TriggerNet output video visualization.")
        elif original_trigger_output is not None or adversarial_trigger_output is not None: # If either exists but is not 5D
             print(f"Warning: TriggerNet output is not 5D ({original_trigger_output.ndim if original_trigger_output is not None else adversarial_trigger_output.ndim}D). Skipping TriggerNet output video visualization.")

    elif visualize_mode == "image_grid":
        # --- Image Grid Visualization Mode ---
        # Input original/adversarial images are (B, T, H, W, C) or (B, C, H, W)
        # Input pixel deltas are (B, C, T, H, W) or (B, C, H, W)
        # make_grid expects (N, C, H, W) - where N is total images in grid

        # Determine which sequence steps to visualize for image grid (use sequence_steps_to_vis parameter)
        # If input is 5D (B, T, H, W, C), sequence length is T (index 1)
        # If input is 4D (B, C, H, W), sequence length is 1
        seq_len = original_images.shape[1] if original_images.ndim == 5 else 1 # Sequence length is 1 for 4D input (B, C, H, W) where T=1
        # Let's refine seq_len check based on shape (B, T, H, W, C)
        seq_len_images = original_images.shape[1] if original_images.ndim == 5 else 1 # T is at index 1 for (B, T, H, W, C)
        seq_len_deltas = pixel_deltas.shape[2] if pixel_deltas.ndim == 5 else 1 # T is at index 2 for (B, C, T, H, W)

        if original_images.ndim == 5: # If input is 5D (B, T, H, W, C)
             valid_seq_len = seq_len_images
        elif original_images.ndim == 4: # If input is 4D (B, C, H, W)
             valid_seq_len = 1
        else:
             valid_seq_len = 0 # Not 4D or 5D, no valid sequence length
             print(f"Warning: Input images for image grid are not 4D or 5D ({original_images.ndim}D). Skipping image grid visualization.")
             return


        if sequence_steps_to_vis is None or not isinstance(sequence_steps_to_vis, list) or not sequence_steps_to_vis:
             # If no list is provided, default to visualizing the first step (0)
             steps_to_visualize = [0]
        else:
             # Filter out invalid steps and sort the valid ones based on images sequence length
             steps_to_visualize = sorted([s for s in sequence_steps_to_vis if 0 <= s < valid_seq_len])

        if not steps_to_visualize:
             print(f"Warning: No valid sequence steps provided for image grid visualization (available steps 0-{valid_seq_len-1}). Skipping visualization.")
             return # Skip visualization if no valid steps

        original_images_vis_list = []
        adversarial_images_vis_list = []
        pixel_deltas_vis_list = []

        # Extract and collect the specified sequence steps for visualization
        for vis_step in steps_to_visualize:
             # Handle both 4D (B, C, H, W) and 5D (B, T, H, W, C) inputs when extracting step
             if original_images.ndim == 5: # Input (B, T, H, W, C), extract step from T (index 1)
                  original_images_step = original_images[:num_samples_to_show, vis_step, :, :, :].cpu().detach() # (B_vis, H, W, C)
                  # Adversarial images also (B, T, H, W, C)
                  adversarial_images_step = adversarial_images[:num_samples_to_show, vis_step, :, :, :].cpu().detach() # (B_vis, H, W, C)
                  # Pixel deltas are (B, C, T, H, W), need to extract step from T (index 2)
                  # Ensure the step is also valid for delta sequence length
                  if vis_step < seq_len_deltas:
                       pixel_deltas_step = pixel_deltas[:num_samples_to_show, :, vis_step, :, :].cpu().detach() # (B_vis, C, H, W)
                  else:
                       print(f"Warning: Pixel delta step {vis_step} out of bounds for delta sequence length {seq_len_deltas}. Skipping this delta step visualization.")
                       pixel_deltas_step = None # Set to None to skip this delta step

             elif original_images.ndim == 4 and vis_step == 0: # For 4D (B, C, H, W), only step 0 is valid
                  # Original/Adversarial images are (B, C, H, W)
                  original_images_step = original_images[:num_samples_to_show, :, :, :].cpu().detach() # (B_vis, C, H, W)
                  adversarial_images_step = adversarial_images[:num_samples_to_show, :, :, :].cpu().detach() # (B_vis, C, H, W)
                  # Pixel deltas are also (B, C, H, W)
                  pixel_deltas_step = pixel_deltas[:num_samples_to_show, :, :, :].cpu().detach() # (B_vis, C, H, W)

             else:
                  # Should not happen if invalid steps are filtered, but as safeguard
                  print(f"Error: Unexpected input dimension ({original_images.ndim}D) or invalid step ({vis_step}) for image grid visualization.")
                  continue # Skip this step

             # Ensure the extracted tensors are valid and 3/C channels
             # For 5D input -> (B_vis, H, W, C), check C at index 3
             # For 4D input -> (B_vis, C, H, W), check C at index 1
             img_channels_idx = 3 if original_images.ndim == 5 else 1 # Determine channel index based on original input dim
             delta_channels_idx = 1 # Delta is always (B, C, T, H, W) or (B, C, H, W), channel is at index 1 after slicing

             if original_images_step is not None and adversarial_images_step is not None and original_images_step.shape[img_channels_idx] in [1, 3] and adversarial_images_step.shape[img_channels_idx] in [1, 3]:
                   # Permute (B_vis, H, W, C) to (B_vis, C, H, W) if needed (for 5D input original format)
                   if original_images.ndim == 5:
                         original_images_step_permuted = original_images_step.permute(0, 3, 1, 2) # (B_vis, H, W, C) -> (B_vis, C, H, W)
                         adversarial_images_step_permuted = adversarial_images_step.permute(0, 3, 1, 2) # (B_vis, H, W, C) -> (B_vis, C, H, W)
                   else: # 4D input is already (B_vis, C, H, W)
                        original_images_step_permuted = original_images_step
                        adversarial_images_step_permuted = adversarial_images_step

                   original_images_vis_list.append(original_images_step_permuted) # Append (B_vis, C, H, W)
                   adversarial_images_vis_list.append(adversarial_images_step_permuted) # Append (B_vis, C, H, W)
             else:
                  # Update warning to be more specific about the channel index being checked
                  orig_channels = original_images_step.shape[img_channels_idx] if original_images_step is not None and original_images_step.ndim > img_channels_idx else 'N/A'
                  adv_channels = adversarial_images_step.shape[img_channels_idx] if adversarial_images_step is not None and adversarial_images_step.ndim > img_channels_idx else 'N/A'
                  print(f"Warning: Original or adversarial image step is None or channel count ({orig_channels}/{adv_channels}) at index {img_channels_idx} is not 1 or 3 for step {vis_step}. Skipping.")

             if pixel_deltas_step is not None and pixel_deltas_step.shape[delta_channels_idx] in [1, 3]:
                  # Delta is already (B, C, H, W) for 5D input after slicing T, or is (B, C, H, W) for 4D input
                  pixel_deltas_vis_list.append(pixel_deltas_step) # Append (B_vis, C, H, W)
             elif pixel_deltas_step is not None:
                  delta_channels = pixel_deltas_step.shape[delta_channels_idx] if pixel_deltas_step.ndim > delta_channels_idx else 'N/A'
                  print(f"Warning: Pixel delta step is not 1 or 3 channels ({delta_channels}) at index {delta_channels_idx} for step {vis_step}. Skipping.")

        # Concatenate the list of tensors along the batch dimension for make_grid
        # Only proceed if there are images to show
        if original_images_vis_list:
            original_images_grid_input = torch.cat(original_images_vis_list, dim=0) # Shape (num_samples_to_show * num_steps_to_vis, C, H, W)
            adversarial_images_grid_input = torch.cat(adversarial_images_vis_list, dim=0)
            # Only concatenate deltas if there are valid delta steps
            if pixel_deltas_vis_list:
                 pixel_deltas_grid_input = torch.cat(pixel_deltas_vis_list, dim=0)
            else:
                 pixel_deltas_grid_input = None # No deltas to show in grid

            num_images_in_grid = original_images_grid_input.shape[0] # Total number of images in the grid
            vis_steps_for_log = steps_to_visualize # Use the list for log naming
        else:
             print("Warning: No images collected for image grid visualization. Skipping.")
             return

        # Ensure there are images in the grid
        if num_images_in_grid == 0:
             print("Warning: No images to show in the grid. Skipping visualization.")
             return

        # --- Visualize Images and Deltas (Image Grid) ---
        # Assuming images are 3-channel (RGB) or 1-channel (grayscale). Check channel dimension (index 1).
        if original_images_grid_input.shape[1] in [1, 3]:
             # Original Images Grid
             grid_orig_img = make_grid(original_images_grid_input, nrow=len(steps_to_visualize), normalize=True)
             writer.add_image(f'Stage2_Visualization/Image_Grids/Original_Images_Steps_{vis_steps_for_log}', grid_orig_img, step)

             # Adversarial Images Grid
             grid_adv_img = make_grid(adversarial_images_grid_input, nrow=len(steps_to_visualize), normalize=True)
             writer.add_image(f'Stage2_Visualization/Image_Grids/Adversarial_Images_Steps_{vis_steps_for_log}', grid_adv_img, step)

             # Pixel Deltas Grid (Normalize for visualization)
             if pixel_deltas_grid_input is not None and pixel_deltas_grid_input.shape[1] in [1, 3]: # Check if deltas are available and 1/3 channels
                 min_delta = pixel_deltas_grid_input.min() if pixel_deltas_grid_input.numel() > 0 else 0.0
                 max_delta = pixel_deltas_grid_input.max() if pixel_deltas_grid_input.numel() > 0 else 1.0
                 if max_delta > min_delta + 1e-8:
                      normalized_deltas_grid = (pixel_deltas_grid_input - min_delta) / (max_delta - min_delta + 1e-8)
                 else:
                      normalized_deltas_grid = torch.zeros_like(pixel_deltas_grid_input, device=pixel_deltas_grid_input.device)

                 grid_deltas = make_grid(normalized_deltas_grid, nrow=len(steps_to_visualize), normalize=False)
                 writer.add_image(f'Stage2_Visualization/Image_Grids/Pixel_Deltas_Steps_{vis_steps_for_log}', grid_deltas, step)
             elif pixel_deltas_grid_input is not None:
                  print(f"Warning: Pixel deltas grid is not 1 or 3 channels ({pixel_deltas_grid_input.shape[1]}). Skipping delta grid visualization.")
             # If pixel_deltas_grid_input is None, the warning was already printed when collecting steps

        else:
             print(f"Warning: Images or deltas are not 1 or 3 channels. Skipping image grid visualization.")

        # --- Visualize TriggerNet Outputs Grid (If available) ---
        # Assume TriggerNet output is a spatial map (B, C_out, N, H', W') or (B, C_out, H', W') or (B, H', W')
        # Process and concatenate TriggerNet outputs for selected steps
        processed_orig_trigger_list = []
        processed_adv_trigger_list = []

        if original_trigger_output is not None and adversarial_trigger_output is not None:
             # Determine TriggerNet output sequence length
             trigger_seq_len = original_trigger_output.shape[2] if original_trigger_output.ndim == 5 else 1 # N is at index 2 for (B, C, N, H, W)
             # Let's refine trigger_seq_len based on shape
             if original_trigger_output.ndim == 5: # (B, C_out, N, H', W')
                  trigger_valid_seq_len = original_trigger_output.shape[2]
             elif original_trigger_output.ndim in [3, 4]: # (B, H', W') or (B, C_out, H', W'), sequence length is 1
                  trigger_valid_seq_len = 1
             else:
                  trigger_valid_seq_len = 0 # Not supported shape
                  print(f"Warning: TriggerNet output for image grid is not 3D, 4D, or 5D ({original_trigger_output.ndim}D). Skipping TriggerNet output grid visualization.")

             # Determine which sequence steps to visualize for TriggerNet (use the same steps as images if valid)
             steps_to_visualize_trigger = sorted([s for s in steps_to_visualize if s < trigger_valid_seq_len])

             if not steps_to_visualize_trigger:
                  if trigger_valid_seq_len > 0: # Only print if TriggerNet output had a valid sequence length but no steps were in common
                       print(f"Warning: No valid sequence steps provided for TriggerNet output image grid visualization (available steps 0-{trigger_valid_seq_len-1}). Skipping.")
                  # If no valid steps for trigger, but valid steps for images were found, log this warning and continue
             else:
                 for vis_step_trigger in steps_to_visualize_trigger:
                      # Handle extracting step from 4D, 5D, or 3D TriggerNet output
                      if original_trigger_output.ndim == 5: # (B, C_out, N, H', W'), extract N at index 2
                           # Extract step (B, C_out, H', W')
                           orig_trigger_step = original_trigger_output[:num_samples_to_show, :, vis_step_trigger, :, :].cpu().detach()
                           adv_trigger_step = adversarial_trigger_output[:num_samples_to_show, :, vis_step_trigger, :, :].cpu().detach()
                      elif original_trigger_output.ndim == 4 and vis_step_trigger == 0: # For 4D (B, C_out, H', W'), only step 0
                           # Use directly (B, C_out, H', W')
                           orig_trigger_step = original_trigger_output[:num_samples_to_show, :, :, :].cpu().detach()
                           adv_trigger_step = adversarial_trigger_output[:num_samples_to_show, :, :, :].cpu().detach()
                      elif original_trigger_output.ndim == 3 and vis_step_trigger == 0: # For 3D (B, H', W'), only step 0
                           # Add channel dim (B, 1, H', W')
                           orig_trigger_step = original_trigger_output[:num_samples_to_show, :, :].unsqueeze(1).cpu().detach()
                           adv_trigger_step = adversarial_trigger_output[:num_samples_to_show, :, :].unsqueeze(1).cpu().detach()
                      else: # Should not happen if steps are filtered
                           print(f"Error: Unexpected TriggerNet output dimension ({original_trigger_output.ndim}D) or invalid step ({vis_step_trigger}) for image grid visualization.")
                           continue # Skip this step

                      # Process channels (ensure 1 or 3 channels for make_grid)
                      def _process_channels_for_grid(trigger_output_step_4d: torch.Tensor) -> Optional[torch.Tensor]:
                           # Input is (B_vis, C_out, H', W')
                           if trigger_output_step_4d.shape[1] == 1:
                                return trigger_output_step_4d
                           elif trigger_output_step_4d.shape[1] >= 3: # Take first 3 channels
                                return trigger_output_step_4d[:, :3, :, :]
                           elif trigger_output_step_4d.shape[1] > 0: # Take mean across channels, then repeat to 3
                                return torch.mean(trigger_output_step_4d, dim=1, keepdim=True).repeat(1, 3, 1, 1)
                           else:
                                print("Warning: Processed TriggerNet step has 0 channels. Cannot visualize.")
                                return None

                      processed_orig_step = _process_channels_for_grid(orig_trigger_step)
                      processed_adv_step = _process_channels_for_grid(adv_trigger_step)

                      if processed_orig_step is not None and processed_adv_step is not None and processed_orig_step.shape == processed_adv_step.shape and processed_orig_step.shape[1] in [1, 3]: # Check processed channels
                           processed_orig_trigger_list.append(processed_orig_step) # Append processed step (B_vis, C', H', W')
                           processed_adv_trigger_list.append(processed_adv_step) # Append processed step (B_vis, C', H', W')
                      elif processed_orig_step is not None or processed_adv_step is not None: # If one is None or shapes/channels mismatch
                           print(f"Warning: Original or adversarial TriggerNet step outputs for step {vis_step_trigger} have issues (None, shape mismatch, or wrong channels). Skipping this step.")

             # Concatenate the list of processed tensors along the batch dimension
             if processed_orig_trigger_list:
                  processed_orig_trigger_grid_input = torch.cat(processed_orig_trigger_list, dim=0) # Shape (num_samples_to_show * num_valid_steps, C', H', W')
                  processed_adv_trigger_grid_input = torch.cat(processed_adv_trigger_list, dim=0)
                  num_trigger_images_in_grid = processed_orig_trigger_grid_input.shape[0]
             else:
                  print("Warning: No TriggerNet output images collected for image grid visualization. Skipping.")
                  num_trigger_images_in_grid = 0

             # Ensure there are images in the trigger grid
             if num_trigger_images_in_grid > 0:
                  # Normalize the combined batch of outputs for visualization ([0, 1] range)
                  normalized_orig_trigger_list = []
                  normalized_adv_trigger_list = []

                  for img in processed_orig_trigger_grid_input:
                       min_val = img.min() if img.numel() > 0 else 0.0
                       max_val = img.max() if img.numel() > 0 else 1.0
                       if max_val > min_val + 1e-8:
                            normalized_img = (img - min_val) / (max_val - min_val + 1e-8)
                       else:
                            normalized_img = torch.zeros_like(img, device=img.device)
                       normalized_orig_trigger_list.append(normalized_img)

                  for img in processed_adv_trigger_grid_input:
                       min_val = img.min() if img.numel() > 0 else 0.0
                       max_val = img.max() if img.numel() > 0 else 1.0
                       if max_val > min_val + 1e-8:
                            normalized_img = (img - min_val) / (max_val - min_val + 1e-8)
                       else:
                            normalized_img = torch.zeros_like(img, device=img.device)
                       normalized_adv_trigger_list.append(normalized_img)

                  # Stack normalized images
                  normalized_orig_trigger_grid = torch.stack(normalized_orig_trigger_list, dim=0).float()
                  normalized_adv_trigger_grid = torch.stack(normalized_adv_trigger_list, dim=0).float()

                  # Ensure tensors are on CPU before make_grid
                  normalized_orig_trigger_grid = normalized_orig_trigger_grid.cpu()
                  normalized_adv_trigger_grid = normalized_adv_trigger_grid.cpu()

                  # Visualize Original TriggerNet Output Grid
                  grid_orig_trigger = make_grid(normalized_orig_trigger_grid, nrow=len(steps_to_visualize_trigger), normalize=False)
                  writer.add_image(f'Stage2_Visualization/Image_Grids/Original_TriggerNet_Output_Steps_{steps_to_visualize_trigger}', grid_orig_trigger, step)

                  # Visualize Adversarial TriggerNet Output Grid
                  grid_adv_trigger = make_grid(normalized_adv_trigger_grid, nrow=len(steps_to_visualize_trigger), normalize=False)
                  writer.add_image(f'Stage2_Visualization/Image_Grids/Adversarial_TriggerNet_Output_Steps_{steps_to_visualize_trigger}', grid_adv_trigger, step)

                  # Visualize TriggerNet Output Difference Grid
                  # Ensure both tensors are on the same device before calculating difference
                  if normalized_orig_trigger_grid.device != normalized_adv_trigger_grid.device:
                       # print("Warning: Original and adversarial TriggerNet output grids are on different devices for difference calculation. Attempting to move adversarial to original device.")
                       normalized_adv_trigger_grid = normalized_adv_trigger_grid.to(normalized_orig_trigger_grid.device)

                  trigger_output_diff_grid = torch.abs(normalized_adv_trigger_grid - normalized_orig_trigger_grid)
                  if trigger_output_diff_grid.numel() > 0:
                       min_diff = trigger_output_diff_grid.min() if trigger_output_diff_grid.numel() > 0 else 0.0
                       max_diff = trigger_output_diff_grid.max() if trigger_output_diff_grid.numel() > 0 else 1.0
                       if max_diff > min_diff + 1e-8:
                            normalized_diff_grid = (trigger_output_diff_grid - min_diff) / (max_diff - min_diff + 1e-8)
                       else:
                            normalized_diff_grid = torch.zeros_like(trigger_output_diff_grid, device=trigger_output_diff_grid.device)

                       # Ensure difference grid is 1 or 3 channels before adding
                       if normalized_diff_grid.shape[1] in [1, 3]:
                            grid_trigger_diff = make_grid(normalized_diff_grid.cpu(), nrow=len(steps_to_visualize_trigger), normalize=False)
                            writer.add_image(f'Stage2_Visualization/Image_Grids/TriggerNet_Output_Difference_Steps_{steps_to_visualize_trigger}', grid_trigger_diff, step)
                       else:
                            print(f"Warning: TriggerNet output difference grid is not 1 or 3 channels ({normalized_diff_grid.shape[1]}). Skipping visualization.")
                  else:
                       print("Warning: TriggerNet output difference grid map is empty. Skipping visualization.")
             elif original_trigger_output is not None or adversarial_trigger_output is not None: # If one is None or processed lists are empty
                  print("Warning: Original or adversarial TriggerNet output is not None, but processed lists are empty after filtering/processing. Skipping image grid visualization.")
        elif original_trigger_output is not None or adversarial_trigger_output is not None: # If either exists but is not supported shape (3D, 4D, 5D)
             print(f"Warning: TriggerNet output is not None, but is not 3D, 4D, or 5D ({original_trigger_output.ndim if original_trigger_output is not None else adversarial_trigger_output.ndim}D) for image grid visualization. Skipping.")

    # If visualize_mode is neither "video" nor "image_grid", print a warning
    # Also print warning if video mode requested for non-5D input
    elif visualize_mode == "video" and not is_5d_input:
         print(f"Warning: Video visualization requested but input is not 5D ({original_images.ndim}D). Input must be 5D for video visualization.")
    elif visualize_mode not in ["video", "image_grid"]:
        print(f"Warning: Unsupported visualize_mode '{visualize_mode}' specified in config. Skipping visualization.")


# 示例用法 (在训练或评估脚本中): # 添加注释以说明示例用途
# from utils.vis_utils import visualize_training_losses, visualize_perturbation_norms, visualize_samples_and_outputs, visualize_attention_maps, visualize_stage2_pixel_attack
# from torch.utils.tensorboard import SummaryWriter
# import os
#
# # 初始化 TensorBoard SummaryWriter
# # log_dir 应该从配置中读取
# log_dir = 'runs/experiment_name' # 示例日志目录
# os.makedirs(log_dir, exist_ok=True)
# writer = SummaryWriter(log_dir)
#
# # 在训练循环中:
# # ... (模型前向传播，计算损失) ...
# # 假设你有原始图像 (B, C, T, H, W) 和对应的对抗样本、扰动、以及 ATN 输出
# # original_image, adversarial_image, pixel_delta, original_trigger_output, adversarial_trigger_output
#
# # 每隔 M 步或每个 epoch 结束时可视化样本和输出
# current_step = ... # 当前全局训练步数
# vis_interval = ... # 可视化间隔
# num_vis_samples = ... # 可视化样本数量
#
# if current_step % vis_interval == 0 or current_step == total_steps:
#      # 从配置中读取可视化模式和要可视化的时间步列表
#      visualize_mode = cfg.logging.get('visualize_mode', 'video') # Default to video
#      sequence_steps_to_vis = cfg.logging.get('sequence_steps_to_vis', None)
#
#      visualize_stage2_pixel_attack(
#          writer=writer,
#          original_images=original_image, # 原始图像 (B, C, T, H, W) 或 (B, C, H, W)
#          adversarial_images=adversarial_image, # 对抗图像 (B, C, T, H, W) 或 (B, C, H, W)
#          pixel_deltas=pixel_delta, # 像素扰动 (B, C, T, H, W) 或 (B, C, H, W)
#          original_trigger_output=original_trigger_output, # 原始 TriggerNet 输出
#          adversarial_trigger_output=adversarial_trigger_output, # 对抗 TriggerNet 输出
#          step=current_step,
#          cfg=cfg, # Pass the config object
#          num_samples=num_vis_samples, # Number of samples/videos to visualize
#          sequence_steps_to_vis=sequence_steps_to_vis # Pass the list of steps for image_grid mode
#      )
#
#      # 其他可视化，如损失、扰动范数等，按原间隔记录
#      # ... (visualize_training_losses, visualize_perturbation_norms) ...
#
# # 训练结束时关闭 writer
# writer.close()





