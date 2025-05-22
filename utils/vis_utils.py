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
                                  original_features: Optional[torch.Tensor], # 原始特征
                                  feature_delta: Optional[torch.Tensor],    # 特征扰动
                                  adversarial_features: Optional[torch.Tensor], # 对抗特征
                                  original_decision_map: Optional[torch.Tensor], # 原始决策图
                                  adversarial_decision_map: Optional[torch.Tensor], # 对抗决策图
                                  step: int,
                                  num_samples: int = 4,
                                  sequence_step_to_vis: int = 0,
                                  visualize_decision_diff: bool = True): # 是否可视化决策图差异 (主要用于 Stage 2)
    """
    可视化原始图像、对抗样本、扰动、特征层、决策图等到 TensorBoard。

    Args:
        writer (SummaryWriter): TensorBoard 的 SummaryWriter 对象。
        original_image (Optional[torch.Tensor]): 原始输入图像批次 (B, C, H, W)。
        original_features (Optional[torch.Tensor]): 原始样本通过 ATN 的特征层输出 (B, C, N, H, W)。
        feature_delta (Optional[torch.Tensor]): 阶段1 生成的特征层扰动 (B, C, N, H, W)。
        adversarial_features (Optional[torch.Tensor]): 对抗样本通过 ATN 的特征层输出 (B, C, N, H, W)。
        original_decision_map (Optional[torch.Tensor]): 原始样本通过 ATN 的决策图输出 (B, H, W)。
        adversarial_decision_map (Optional[torch.Tensor]): 对抗样本通过 ATN 的决策图输出 (B, H, W)。
        step (int): 当前的训练全局步数。
        num_samples (int): 要可视化的样本数量。
        sequence_step_to_vis (int): 要可视化的特征层的序列步骤索引。
        visualize_decision_diff (bool): 是否可视化原始决策图和对抗决策图的差异。
    """
    # # --- Add Debug Prints for Tensor Shapes and numel() ---
    # print(f"DEBUG VIS: Step {step}")
    # print(f"DEBUG VIS: original_image_cpu shape: {original_image.shape if original_image is not None else None}, numel: {original_image.numel() if original_image is not None else 0}")
    # print(f"DEBUG VIS: original_features_cpu shape: {original_features.shape if original_features is not None else None}, numel: {original_features.numel() if original_features is not None else 0}")
    # print(f"DEBUG VIS: feature_delta_cpu shape: {feature_delta.shape if feature_delta is not None else None}, numel: {feature_delta.numel() if feature_delta is not None else 0}")
    # print(f"DEBUG VIS: adversarial_features_cpu shape: {adversarial_features.shape if adversarial_features is not None else None}, numel: {adversarial_features.numel() if adversarial_features is not None else 0}")
    # print(f"DEBUG VIS: original_decision_map_cpu shape: {original_decision_map.shape if original_decision_map is not None else None}, numel: {original_decision_map.numel() if original_decision_map is not None else 0}")
    # print(f"DEBUG VIS: adversarial_decision_map_cpu shape: {adversarial_decision_map.shape if adversarial_decision_map is not None else None}, numel: {adversarial_decision_map.numel() if adversarial_decision_map is not None else 0}")
    # # --- End Debug Prints ---

    # 尝试从任意非空张量中获取批量大小和设备
    batch_size = 0
    device = 'cpu'
    valid_tensors = [original_image, original_features, feature_delta, adversarial_features, original_decision_map, adversarial_decision_map]
    for t in valid_tensors:
        if t is not None and t.numel() > 0:
            batch_size = t.shape[0]
            device = t.device # 获取张量所在设备
            break # 找到第一个有效张量即停止

    # 如果没有有效张量，无法进行可视化
    if batch_size == 0:
        # print("Warning: No valid input tensors with batch dimension. Skipping sample visualization.") # 避免过多打印
        return

    # 确定实际要显示的样本数量
    num_samples_to_show = min(num_samples, batch_size)
    # print(f"Debug: Visualizing {num_samples_to_show} samples.") # Debug print

    # 将要显示的张量移动到 CPU (如果不在 CPU 上) 并选取切片
    # 使用 .detach() 断开计算图，使用 .cpu() 移动到 CPU
    original_image_cpu = original_image[:num_samples_to_show].cpu().detach() if original_image is not None else None
    original_features_cpu = original_features[:num_samples_to_show].cpu().detach() if original_features is not None and original_features.numel() > 0 else None # Empty tensor if numel==0
    feature_delta_cpu = feature_delta[:num_samples_to_show].cpu().detach() if feature_delta is not None and feature_delta.numel() > 0 else None # Empty tensor if numel==0
    adversarial_features_cpu = adversarial_features[:num_samples_to_show].cpu().detach() if adversarial_features is not None and adversarial_features.numel() > 0 else None # Empty tensor if numel==0
    original_decision_map_cpu = original_decision_map[:num_samples_to_show].cpu().detach() if original_decision_map is not None else None
    adversarial_decision_map_cpu = adversarial_decision_map[:num_samples_to_show].cpu().detach() if adversarial_decision_map is not None else None

    # --- 可视化原始图像 (如果可用) ---
    # 假设原始图像是 5D (B, C_img, T, H_img, W_img)
    if original_image_cpu is not None and original_image_cpu.ndim == 5:
        # 检查序列步骤是否有效，并提取指定步骤的图像 (B, C_img, H_img, W_img)
        img_seq_len = original_image_cpu.shape[2]
        vis_step_img = min(sequence_step_to_vis, img_seq_len - 1) # 确保步骤在范围内
        vis_step_img = max(0, vis_step_img)

        original_image_step = original_image_cpu[:, :, vis_step_img, :, :].squeeze(2) # 移除序列维度
        # make_grid 需要输入形状 (B, C, H, W)
        # 假设原始图像是 3 通道 (RGB)
        if original_image_step.shape[1] == 3:
            grid_image = make_grid(original_image_step, nrow=num_samples_to_show, normalize=True)
            writer.add_image('Samples/Original_Image', grid_image, step)
        else:
            # 警告：原始图像不是 3 通道 ({})。跳过原始图像可视化。
            print(f"Warning: Original image is not 3 channels ({original_image_step.shape[1]}). Skipping original image visualization.")

    # --- 可视化特征相关的图 (如果特征可用) ---
    # 假设特征是 5D (B, C_feat, N, H_feat, W_feat)
    if original_features_cpu is not None and original_features_cpu.numel() > 0 and original_features_cpu.ndim == 5:
        # 检查序列步骤是否有效，并用于特征可视化
        feat_seq_len = original_features_cpu.shape[2]
        vis_step_feat = min(sequence_step_to_vis, feat_seq_len - 1)
        vis_step_feat = max(0, vis_step_feat)
        
        # 可视化原始特征
        # 使用 feature_to_visualizable 将 5D 特征转换为 4D 可视化张量
        vis_orig_features = feature_to_visualizable(original_features_cpu, sequence_step=vis_step_feat, mode='l2_norm') # 示例：可视化 L2 范数
        if vis_orig_features.numel() > 0:
            # make_grid 需要输入形状 (B, C, H, W)
            # vis_orig_features 的形状是 (B, 1, H, W) 或 (B, 3, H, W)
            grid_orig_features = make_grid(vis_orig_features, nrow=num_samples_to_show, normalize=True)
            writer.add_image(f'Features/Original_Seq{vis_step_feat}_L2Norm', grid_orig_features, step)

        # 可视化特征扰动 Delta (如果可用)
        if feature_delta_cpu is not None and feature_delta_cpu.numel() > 0 and feature_delta_cpu.ndim == 5:
             # 确保 delta 和 features 形状匹配
             if feature_delta_cpu.shape == original_features_cpu.shape:
                  vis_feature_delta = feature_to_visualizable(feature_delta_cpu, sequence_step=vis_step_feat, mode='l2_norm') # 示例：可视化 L2 范数
                  if vis_feature_delta.numel() > 0:
                      grid_feature_delta = make_grid(vis_feature_delta, nrow=num_samples_to_show, normalize=True)
                      writer.add_image(f'Features/Feature_Perturbation_Delta_Seq{vis_step_feat}_L2Norm', grid_feature_delta, step)
             else:
                  # 警告：特征扰动 delta 形状 {} 与原始特征形状 {} 不匹配。跳过扰动可视化。
                  print(f"Warning: Feature delta shape {feature_delta_cpu.shape} != Original features shape {original_features_cpu.shape}. Skipping delta visualization.")

        # 可视化对抗性特征 (如果可用)
        if adversarial_features_cpu is not None and adversarial_features_cpu.numel() > 0 and adversarial_features_cpu.ndim == 5:
             # 确保对抗特征和原始特征形状匹配
             if adversarial_features_cpu.shape == original_features_cpu.shape:
                  vis_adv_features = feature_to_visualizable(adversarial_features_cpu, sequence_step=vis_step_feat, mode='l2_norm') # 示例：可视化 L2 范数
                  if vis_adv_features.numel() > 0:
                      grid_adv_features = make_grid(vis_adv_features, nrow=num_samples_to_show, normalize=True)
                      writer.add_image(f'Features/Adversarial_Seq{vis_step_feat}_L2Norm', grid_adv_features, step)

                  # 可视化原始特征与对抗特征的差异
                  # 计算差异的 L2 范数图
                  feature_diff = adversarial_features_cpu - original_features_cpu
                  # 计算差异的 L2 范数图 (空间维度上)
                  feature_diff_l2_norm_map = torch.linalg.norm(feature_diff[:, :, vis_step_feat, :, :].squeeze(2), dim=1, keepdim=True) # (B, 1, H, W)
                  
                  # 归一化差异图以便可视化 (0到最大差异)
                  if feature_diff_l2_norm_map.numel() > 0:
                       min_val = feature_diff_l2_norm_map.min()
                       max_val = feature_diff_l2_norm_map.max()
                       if max_val > min_val + 1e-8:
                            normalized_diff_map = (feature_diff_l2_norm_map - min_val) / (max_val - min_val + 1e-8)
                       else:
                            normalized_diff_map = torch.zeros_like(feature_diff_l2_norm_map, device=feature_diff_l2_norm_map.device)

                       grid_feature_diff = make_grid(normalized_diff_map, nrow=num_samples_to_show, normalize=False)
                       writer.add_image(f'Features/Feature_Difference_Seq{vis_step_feat}_L2Norm', grid_feature_diff, step)
             else:
                  # 警告：对抗特征形状 {} 与原始特征形状 {} 不匹配。跳过对抗特征和差异可视化。
                  print(f"Warning: Adversarial features shape {adversarial_features_cpu.shape} != Original features shape {original_features_cpu.shape}. Skipping adversarial features and difference visualization.")

        # 可视化特征直方图 (每个样本)
        if original_features_cpu.numel() > 0:
            # 确保特征是 5D
            if original_features_cpu.ndim == 5:
                 # 选择一个序列步骤进行直方图可视化
                 hist_step_feat = min(sequence_step_to_vis, original_features_cpu.shape[2] - 1)
                 hist_step_feat = max(0, hist_step_feat)
                 
                 # 对原始特征、特征扰动和对抗特征绘制直方图 (展平除批量外的所有维度)
                 if original_features_cpu.numel() > 0:
                      # original_features_cpu shape: (B, C, N, H, W)
                      # 展平除批量维度外的所有维度 for histogram
                      original_features_flat_hist_combined = original_features_cpu[:, :, hist_step_feat, :, :].reshape(-1).float().cpu().numpy()
                      # 绘制原始特征的合并直方图
                      writer.add_histogram(f'Histograms/Original_Features_Seq{hist_step_feat}', original_features_flat_hist_combined, step)
                 
                 if feature_delta_cpu is not None and feature_delta_cpu.numel() > 0 and feature_delta_cpu.shape == original_features_cpu.shape:
                       # 展平选定序列步骤的所有样本的扰动数据，以便绘制合并直方图
                       feature_delta_flat_hist_combined = feature_delta_cpu[:, :, hist_step_feat, :, :].reshape(-1).float().cpu().numpy()
                       # 绘制特征扰动的合并直方图
                       writer.add_histogram(f'Histograms/Feature_Perturbation_Delta_Seq{hist_step_feat}', feature_delta_flat_hist_combined, step)

                 if adversarial_features_cpu is not None and adversarial_features_cpu.numel() > 0 and adversarial_features_cpu.shape == original_features_cpu.shape:
                       # 展平选定序列步骤的所有样本的对抗特征数据，以便绘制合并直方图
                       adversarial_features_flat_hist_combined = adversarial_features_cpu[:, :, hist_step_feat, :, :].reshape(-1).float().cpu().numpy()
                       # 绘制对抗特征的合并直方图
                       writer.add_histogram(f'Histograms/Adversarial_Features_Seq{hist_step_feat}', adversarial_features_flat_hist_combined, step)

                 # 可视化通道子集 (仅对原始特征和对抗特征)
                 # 定义要可视化的通道子集 (示例：前 3 个，中间某个范围，最后一个)
                 # 可以从 config 中读取这个列表
                 # channel_subset_to_vis = [0, 1, 2] # Example
                 # if original_features_cpu.shape[1] > 10: channel_subset_to_vis.extend([10, 20, 30])
                 # if original_features_cpu.shape[1] > 100: channel_subset_to_vis.extend([100, original_features_cpu.shape[1]-1])

                 # 示例：可视化一些固定索引的通道子集，确保索引不越界
                 channel_indices = [0, 1, 2, 50, 100, original_features_cpu.shape[1] - 1] # 示例索引
                 valid_channel_subset = [idx for idx in channel_indices if 0 <= idx < original_features_cpu.shape[1]]

                 if valid_channel_subset:
                      # 可视化原始特征的通道子集
                      vis_orig_features_subset = feature_to_visualizable(original_features_cpu, sequence_step=vis_step_feat, mode='channel_subset', channel_subset=valid_channel_subset)
                      if vis_orig_features_subset.numel() > 0:
                          # make_grid 将所有选定的通道并排放置
                          grid_orig_features_subset = make_grid(vis_orig_features_subset, nrow=num_samples_to_show, normalize=True)
                          writer.add_image(f'Feature_Channel_Subsets/Original_Seq{vis_step_feat}_Channels{valid_channel_subset}', grid_orig_features_subset, step)
                      
                      # 可视化对抗特征的通道子集 (如果可用)
                      if adversarial_features_cpu is not None and adversarial_features_cpu.numel() > 0 and adversarial_features_cpu.shape == original_features_cpu.shape:
                           vis_adv_features_subset = feature_to_visualizable(adversarial_features_cpu, sequence_step=vis_step_feat, mode='channel_subset', channel_subset=valid_channel_subset)
                           if vis_adv_features_subset.numel() > 0:
                               grid_adv_features_subset = make_grid(vis_adv_features_subset, nrow=num_samples_to_show, normalize=True)
                               writer.add_image(f'Feature_Channel_Subsets/Adversarial_Seq{vis_step_feat}_Channels{valid_channel_subset}', grid_adv_features_subset, step)


    # --- 可视化决策图 (如果可用) ---
    # 假设决策图是 3D (B, H, W) 或 (B, W, H)，并且是单通道
    if original_decision_map_cpu is not None and original_decision_map_cpu.numel() > 0 and original_decision_map_cpu.ndim == 3:
         # 确保决策图是单通道 (B, 1, H, W) 或 (B, H, W) 但需要扩展通道维度
         # 如果是 (B, H, W)，添加一个通道维度 (B, 1, H, W)
         if original_decision_map_cpu.shape[1] != 1:
             print("Warning: Original decision map is not single-channel. Assuming shape (B, H, W) and adding channel dim.")
             original_decision_map_cpu = original_decision_map_cpu.unsqueeze(1) # (B, 1, H, W)

         # 确保对抗决策图形状匹配 (如果可用)
         if adversarial_decision_map_cpu is not None and adversarial_decision_map_cpu.numel() > 0 and adversarial_decision_map_cpu.shape == original_decision_map_cpu.shape:
              # 可视化原始决策图
              # 决策图通常不需要标准化到 [0,1]，直接显示其相对值即可
              grid_orig_decision = make_grid(original_decision_map_cpu, nrow=num_samples_to_show, normalize=True)
              writer.add_image('Decision_Maps/Original', grid_orig_decision, step)

              # 可视化对抗决策图
              grid_adv_decision = make_grid(adversarial_decision_map_cpu, nrow=num_samples_to_show, normalize=True)
              writer.add_image('Decision_Maps/Adversarial', grid_adv_decision, step)

              # 可视化决策图的差异 (如果启用)
              if visualize_decision_diff:
                   # 计算决策图的差异图 (绝对差异)
                   decision_diff_map = torch.abs(adversarial_decision_map_cpu - original_decision_map_cpu)
                   # 归一化差异图以便可视化 (0到最大差异)
                   if decision_diff_map.numel() > 0:
                        min_val = decision_diff_map.min()
                        max_val = decision_diff_map.max()
                        if max_val > min_val + 1e-8:
                             normalized_diff_map = (decision_diff_map - min_val) / (max_val - min_val + 1e-8)
                        else:
                             normalized_diff_map = torch.zeros_like(decision_diff_map, device=decision_diff_map.device)

                        grid_decision_diff = make_grid(normalized_diff_map, nrow=num_samples_to_show, normalize=False)
                        writer.add_image('Decision_Maps/Difference', grid_decision_diff, step)
         else:
              # 警告：对抗决策图为None或形状与原始决策图不匹配 ({} vs {})。跳过对抗决策图和差异可视化。
              print(f"Warning: Adversarial decision map is None or shape mismatch ({adversarial_decision_map_cpu.shape if adversarial_decision_map_cpu is not None else 'None'} vs {original_decision_map_cpu.shape}). Skipping adversarial decision map and difference visualization.")
    elif original_decision_map_cpu is not None:
         # 警告：原始决策图为None或空或维度不是 3D ({})。跳过决策图可视化。
         print(f"Warning: Original decision map is None/empty or not 3D ({original_decision_map_cpu.ndim}D). Skipping decision map visualization.")

    # TODO: 可视化注意力图 (这通常在 Stage 2 更相关) -> 由 visualize_attention_maps 函数处理
    # visualize_attention_maps 函数需要原始和对抗的注意力矩阵作为输入

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

# 示例用法 (在训练或评估脚本中): # 添加注释以说明示例用途
# from utils.vis_utils import visualize_training_losses, visualize_perturbation_norms, visualize_samples_and_outputs, visualize_attention_maps
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
# # 假设你有一个包含各种损失的字典 loss_dict
# # loss_dict = {'Generator_Total': g_loss, 'Discriminator_Total': d_loss, 'Generator_GAN': g_gan_loss, ...}
#
# # 每隔 N 步记录损失和扰动范数
# current_step = ... # 当前全局训练步数
# if current_step % log_interval == 0:
#      visualize_training_losses(writer, loss_dict, current_step)
#      # 假设 delta 是 generator 输出的扰动
#      visualize_perturbation_norms(writer, delta, current_step)
#
# # 每隔 M 步或每个 epoch 结束时可视化样本和输出
# if current_step % vis_interval == 0 or current_step == total_steps:
#      # 假设你有原始图像、ATN 输出 (特征、决策、注意力) 和对抗样本/输出
#      # original_image: 原始图像输入 (B, C_img, T, H_img, W_img)
#      # original_atn_outputs = get_atn_outputs(atn_model, original_image, ...)
#      # original_features = original_atn_outputs.get('features')
#      # original_decision_map = original_atn_outputs.get('decision')
#      # original_attention_matrix = original_atn_outputs.get('attention')
#
#      # # 生成对抗样本/扰动并获取对抗输出
#      # delta = generator(original_input or original_features) # 根据阶段不同
#      # adversarial_input or adversarial_features = original_input/features + delta
#      # adversarial_atn_outputs = get_atn_outputs(atn_model, adversarial_input or adversarial_features, ...)
#      # adversarial_features = adversarial_atn_outputs.get('features')
#      # adversarial_decision_map = adversarial_atn_outputs.get('decision')
#      # adversarial_attention_matrix = adversarial_atn_outputs.get('attention')
#
#      visualize_samples_and_outputs(
#          writer,
#          original_image=original_image, # 原始图像 (如果 Stage 2 使用)
#          original_features=original_features, # 原始 ATN 特征
#          feature_delta=delta, # 阶段 1 的特征扰动
#          adversarial_features=adversarial_features, # 对抗 ATN 特征
#          original_decision_map=original_decision_map, # 原始决策图 (如果 Stage 2 使用)
#          adversarial_decision_map=adversarial_decision_map, # 对抗决策图 (如果 Stage 2 使用)
#          step=current_step,
#          num_samples=4, # 从 config 中获取
#          sequence_step_to_vis=0, # 从 config 中获取
#          visualize_decision_diff=True # 从 config 中获取 (Stage 2 usually True)
#      )
#
#      # 可视化注意力图 (通常在 Stage 2 更相关)
#      if original_attention_matrix is not None and adversarial_attention_matrix is not None:
#           visualize_attention_maps(
#               writer,
#               attention_matrix_orig=original_attention_matrix,
#               attention_matrix_adv=adversarial_attention_matrix,
#               step=current_step,
#               num_samples=2, # 从 config 中获取
#               num_heads_to_vis=2 # 从 config 中获取
#           )
#
# # 训练结束时关闭 writer
# writer.close()





