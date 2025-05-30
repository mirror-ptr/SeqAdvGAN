import sys
import os
# 将项目根目录添加到 Python 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
# 可能需要安装 piq 库来计算 LPIPS 或 SSIM: pip install piq
import torch.nn.functional as F # Import F for F.mse_loss
# from piq import LPIPS, ssim # Import piq metrics inside the function

# 扰动范数
from losses.regularization_losses import linf_norm, l2_norm

# 导入 ATN 工具
from utils.atn_utils import get_atn_outputs

# Import mask loading/generation utility if needed
# from utils.mask_utils import load_or_generate_mask # Placeholder
from typing import Optional, Dict, Any # Import Optional, Dict, Any
from typing import List # Import List (if needed)


# TODO: 定义攻击成功率的衡量标准
def calculate_attack_success_rate(
    original_map: torch.Tensor,
    adversarial_map: torch.Tensor,
    success_threshold: float,
    success_criterion: str = 'mse_diff_threshold', # Add success criterion parameter
    higher_is_better_orig: bool = True,
    topk_k: int = 10
) -> float:
    """
    计算攻击成功率。

    本函数根据原始和对抗样本的输出图以及预设的成功标准来判定一次攻击是否成功。
    支持多种成功标准，包括基于MSE差异、平均值变化和Top-K位置/值的变化。

    Args:
        original_map (torch.Tensor): 原始样本在ATN模型下的输出图（如特征图、决策图或注意力矩阵）。
                                    形状为 (B, ...) ，其中 B 是批量大小。
        adversarial_map (torch.Tensor): 对抗样本在ATN模型下的输出图。
                                    形状与 `original_map` 相同。
        success_threshold (float): 用于判断攻击是否成功的阈值。其含义取决于 success_criterion。
        success_criterion (str): 攻击成功判定的标准。可选：
                                 - 'mse_diff_threshold': 对抗样本和原始样本输出图的 MSE 差异大于阈值。
                                 - 'mean_change_threshold': 输出图平均值变化大于阈值 (有方向性，取决于 higher_is_better_orig)。
                                 - 'topk_value_drop': 原始输出图 Top-K 位置的值在对抗样本中平均下降超过阈值 (适用于注意力或展平的特征)。
                                 - 'topk_position_change': 原始输出图 Top-K 位置在对抗样本中的 Top-K 位置是否不同 (适用于注意力或展平的特征)。
                                 - 'classification_change': ATN 最终分类结果发生变化 (TODO: 待实现)。
                                 - 'decision_map_divergence': 输出图分布变化 (KL/JS 散度) 大于阈值 (TODO: 不直接适用于连续输出，考虑移除或修改)。
        higher_is_better_orig (bool): 仅用于 'mean_change_threshold' 标准。指示原始输出图中值越高是否越好。攻击目标是使其变差。
        topk_k (int): 当 `success_criterion` 为 'topk_value_drop' 或 'topk_position_change' 时，
                    用于Top-K计算的参数K。

    Returns:
        float: 计算出的攻击成功率 (范围在 0.0 到 1.0 之间)。
    """
    # 检查输入是否为None
    if original_map is None or adversarial_map is None:
        # 警告：原始或对抗地图为None，对于标准'{}'返回0.0成功率。
        print(f"Warning: Original or adversarial map is None for criterion '{success_criterion}'. Returning 0.0 success rate.")
        return 0.0

    # 确保输入不是标量
    if original_map.ndim == 0 or adversarial_map.ndim == 0:
         # 警告：原始或对抗地图是标量，对于标准'{}'无法计算成功率。
         print(f"Warning: Original or adversarial map is scalar for criterion '{success_criterion}'. Cannot calculate success rate on scalars.")
         return 0.0

    # 确保输入形状匹配
    assert original_map.shape == adversarial_map.shape, f"Input map shapes mismatch: {original_map.shape} vs {adversarial_map.shape}"
    batch_size = original_map.shape[0]
    # 处理空批量情况
    if batch_size == 0: return 0.0

    successful_attacks = 0

    # --- 不同成功标准的实现 --- #

    if success_criterion == 'mse_diff_threshold':
        # MSE 差异：对抗样本与原始样本的 MSE 差异大于阈值即视为攻击成功
        # 需要处理不同输入维度 (B, H, W) 或 (B, head, N, N)
        # 展平除批量维度外的所有维度，用于计算每个样本的MSE
        original_flat = original_map.view(batch_size, -1)
        adversarial_flat = adversarial_map.view(batch_size, -1)

        # 计算每个样本的MSE差异
        mse_diff = torch.mean((adversarial_flat - original_flat)**2, dim=1) # 在展平维度上取平均
        # 统计攻击成功的样本数量
        successful_attacks = (mse_diff > success_threshold).sum().item()

    elif success_criterion == 'mean_change_threshold':
        # 平均值变化：输出图的平均值变化大于指定阈值
        # 此标准主要适用于决策图或特征图，不适用于注意力图
        # 期望输入维度至少为2D (H, W) 或 3D (B, H, W)
        if original_map.ndim not in [2, 3]:
            # 警告：平均变化标准期望2D/3D输入 (H, W) 或 (B, H, W)，得到 {}D。跳过。
            print(f"Warning: Mean change criterion expects 2D/3D input (H, W) or (B, H, W), got {original_map.ndim}D. Skipping.")
            return 0.0

        # 确保输入形状兼容，展平除批量维度外的所有维度
        original_flat = original_map.view(batch_size, -1)
        adversarial_flat = adversarial_map.view(batch_size, -1)

        # 计算每个样本的平均值
        original_mean = original_flat.mean(dim=1)
        adversarial_mean = adversarial_flat.mean(dim=1)

        # 根据 higher_is_better_orig 计算变化量
        if higher_is_better_orig: # 原始值越高越好，希望对抗后降低 (original - adversarial > threshold)
            change = original_mean - adversarial_mean
        else: # 原始值越低越好，希望对抗后升高 (adversarial - original > threshold)
            change = adversarial_mean - original_mean

        # 统计攻击成功的样本数量
        successful_attacks = (change > success_threshold).sum().item()

    elif success_criterion == 'topk_value_drop':
        # 针对注意力图：原始 top-K 位置的注意力值在对抗样本中显著下降
        # 要求输入至少有2个空间/序列维度 (N, N) 或批量形式 (B, head, N, N)
        if original_map.ndim < 2:
            # 警告：Top-K 值下降标准要求至少2个维度，得到 {}D。跳过。
            print(f"Warning: Top-K value drop criterion requires at least 2 dimensions (e.g., flattened attention or spatial), got {original_map.ndim}D. Skipping.")
            return 0.0

        # 展平到 (B, num_features) 形状，以便在最后一个维度上找到 Top-K
        if original_map.ndim == 4: # (B, head, N, N) -> (B, head * N * N)
            original_flat = original_map.view(batch_size, -1)
            adversarial_flat = adversarial_map.view(batch_size, -1)
        elif original_map.ndim == 3: # (B, H, W) -> (B, H*W)
             original_flat = original_map.view(batch_size, -1)
             adversarial_flat = adversarial_map.view(batch_size, -1)
        elif original_map.ndim == 2: # 假设 (B, N*N) 或 (B, H*W)
             original_flat = original_map
             adversarial_flat = adversarial_map
        else:
             # 警告：Top-K 值下降标准不支持 {}D 输入。跳过。
             print(f"Warning: Top-K value drop criterion unsupported for {original_map.ndim}D input. Skipping.")
             return 0.0

        # 确保 topk_k 不超过扁平化后的维度大小
        effective_k = min(topk_k, original_flat.shape[-1])
        if effective_k <= 0: return 0.0 # 避免 k 无效时出错

        # 在原始 map 的扁平化维度上找到 Top-K 值和索引
        # original_topk_values shape (B, effective_k)
        # original_topk_indices shape (B, effective_k)
        # largest=True 表示找最大的 K 个值，sorted=False 可以提高一点速度
        original_topk_values, original_topk_indices = torch.topk(
            original_flat,
            k=effective_k,
            dim=-1,       # 在最后一个维度上找 Top-K (展平后的)
            largest=True, # 找最大的 K 个值
            sorted=False
        )

        # 获取对抗样本在原始 Top-K 索引位置上的值
        # 使用 gather 函数
        # adversarial_values_at_original_topk shape (B, head, k) or (B, k) depending on original_flat.ndim
        # 需要根据原始输入的维度来确定 gather 的 dim 参数
        # 对于 (B, num_features)，gather 的 dim 应为 1
        # 对于 (B, head, num_features)，gather 的 dim 应为 2
        gather_dim = -1 # 默认在最后一个维度收集
        if original_attention_matrix is not None and original_attention_matrix.ndim == 4: # (B, head, N, N)
             gather_dim = -1 # 在 head * N * N 展平后的维度上收集
        # Note: The original implementation was just dim=-1, which works for both (B, num_features) and (B, head, num_features) after flattening head and spatial dims.
        # Let's keep dim=-1 for consistency with the flatten approach.

        adversarial_values_at_original_topk = torch.gather(
            adversarial_flat,
            dim=-1,       # 在展平后的最后一个维度上根据索引收集
            index=original_topk_indices # 使用原始 Top-K 的索引
        )

        # 计算下降比例 (原始值 - 对抗值) / 原始值
        # 避免除以零，如果原始值为 0，则下降比例也为 0
        # 使用一个小的epsilon避免数值不稳定
        value_drop = torch.where(
            original_topk_values.abs() > 1e-6, # 避免除以接近零的值
            (original_topk_values - adversarial_values_at_original_topk) / (original_topk_values.abs() + 1e-8), # 使用 abs() for robust division
            torch.zeros_like(original_topk_values) # 如果原始值接近零，下降比例为零
        )

        # 计算每个样本的平均下降比例
        mean_value_drop_per_sample = value_drop.mean(dim=-1) # 在 Top-K 维度上取平均，形状 (B,)

        # 攻击成功 if 平均下降比例 > 阈值
        successful_attacks = (mean_value_drop_per_sample > success_threshold).sum().item()

    elif success_criterion == 'topk_position_change':
        # 针对注意力图：原始 top-K 注意力位置与对抗样本 top-K 注意力位置的交集小于阈值
        # 要求输入至少有2个空间/序列维度 (N, N) 或批量形式 (B, head, N, N)
        if original_map.ndim < 2:
            # 警告：Top-K 位置变化标准要求至少2个维度，得到 {}D。跳过。
            print(f"Warning: Top-K position change criterion requires at least 2 dimensions, got {original_map.ndim}D. Skipping.")
            return 0.0

        # 展平到 (B, num_features) 或 (B, head, num_features) 形状
        if original_map.ndim == 4: # (B, head, N, N) -> (B, head * N * N)
            original_flat = original_map.view(batch_size, -1)
            adversarial_flat = adversarial_map.view(batch_size, -1)
        elif original_map.ndim == 3: # (B, H, W) -> (B, H*W)
             original_flat = original_map.view(batch_size, -1)
             adversarial_flat = adversarial_map.view(batch_size, -1)
        elif original_map.ndim == 2: # 假设 (B, N*N) 或 (B, H*W)
             original_flat = original_map
             adversarial_flat = adversarial_map
        else:
             # 警告：Top-K 位置变化标准不支持 {}D 输入。跳过。
             print(f"Warning: Top-K position change criterion unsupported for {original_map.ndim}D input. Skipping.")
             return 0.0

        # 确保 topk_k 有效
        effective_k = min(topk_k, original_flat.shape[-1])
        if effective_k <= 0: return 0.0

        # 在原始和对抗 map 的扁平化维度上找到 Top-K 索引
        _, original_topk_indices = torch.topk(
            original_flat,
            k=effective_k,
             dim=-1,
            largest=True,
            sorted=False
        )
        _, adversarial_topk_indices = torch.topk(
            adversarial_flat,
            k=effective_k,
             dim=-1,
            largest=True,
            sorted=False
        )

        # 遍历每个样本，将索引转换为集合，计算交集比例
        for i in range(batch_size):
            orig_set = set(original_topk_indices[i].cpu().numpy())
            adv_set = set(adversarial_topk_indices[i].cpu().numpy())

            # 计算交集大小
            intersection_size = len(orig_set.intersection(adv_set))
            # 计算交集大小与有效 k 的比例
            intersection_ratio = intersection_size / effective_k if effective_k > 0 else 0.0

            # 攻击成功 if 交集比例低于阈值 (即位置变化显著)
            if intersection_ratio < success_threshold:
                successful_attacks += 1

    # elif success_criterion == 'classification_change':
    #      # TODO: 实现分类变化标准
    #      print("Warning: 'classification_change' criterion is not implemented.")
    #      return 0.0

    else:
        # 警告：不支持的成功标准：{}。返回0.0成功率。
        print(f"Warning: Unsupported success criterion: {success_criterion}. Returning 0.0 success rate.")
        return 0.0

    # 返回攻击成功率 (成功样本数 / 总样本数)
    return successful_attacks / batch_size

# Calculate perturbation norms (usually called directly in training script or logged via vis_utils)
# from losses.regularization_losses import linf_norm, l2_norm

# TODO: 将特征层转换为适合感知评估的图像格式
# 这可能是最困难的部分，需要朋友的帮助或相关文献的支持。
def features_to_perceptual_input(features: torch.Tensor, sequence_step: int = 0) -> torch.Tensor:
    """
    将 (B, C, N, W, H) 或 (B, C, N, H, W) 特征张量转换为 (B, 3, H', W')
    或 (B, 1, H', W') 图像格式，以便进行 LPIPS 或 SSIM 感知评估。
    这需要深入理解 ATN 特征如何映射回图像空间，或者找到一种通用的特征可视化方法。

    Args:
        features (torch.Tensor): 输入特征张量，形状为 (B, C, N, W, H) 或 (B, C, N, H, W)。
        sequence_step (int): 选择序列中哪个步骤进行转换 (0- indexed)。

    Returns:
        torch.Tensor: 适合感知评估的图像张量，例如 (B, 3, H', W') 或 (B, 1, H', W')。
                      注意：这是一个占位符实现；在 ATN 特征上直接计算感知指标的有效性未知。
    """
    # 检查输入特征是否为None或空
    if features is None or features.numel() == 0:
        # 警告：特征的序列维度为空。无法转换为感知指标的输入。
        print("Warning: Empty sequence dimension in features. Cannot convert for perceptual metrics.")
        # 返回一个空的张量，形状匹配预期输出，以便后续计算能够处理
        # 假设最后两个维度是空间维度 (H, W)
        # 需要至少3个维度 (B, C, N, H, W) 或 (B, C, N, W, H)
        if features is not None and features.ndim >= 4:
             # 假设最后两个维度是空间维度 H, W (或 W, H)
             spatial_dims = features.shape[-2:]
             # 返回一个空的 tensor，形状为 (0, 3, H, W)，设备和数据类型与输入相同
             return torch.empty(0, 3, spatial_dims[0], spatial_dims[1], device=features.device, dtype=features.dtype)
        else:
             # 如果特征为None或维度不足，返回任意形状的空 tensor
             return torch.empty(0, 3, 64, 64, device='cpu', dtype=torch.float32) # 使用默认形状

    # print("注意：正在使用 features_to_perceptual_input 的占位符实现。")
    # 根据 Generator/Discriminator 的定义，假设特征形状是 (B, C, N, H, W)
    # 需要处理潜在的 (B, C, N, W, H) 形状
    if features.ndim != 5:
         # 警告：特征张量不是5D ({})，无法转换为感知输入的格式。
         print(f"Warning: Features tensor is not 5D ({features.ndim}D) for perceptual input conversion.")
         # 尝试处理 (B, C, H, W) 形状的输入，为其添加序列维度
         if features.ndim == 4: # 假设输入是 (B, C, H, W)
              # 假设4D输入是 (B, C, H, W)，添加序列维度以兼容后续检查。
              print("Assuming 4D input is (B, C, H, W), adding sequence dim for compatibility check.")
              features = features.unsqueeze(2) # 形状变为 (B, C, 1, H, W)
              B, C, N, H, W = features.shape # 重新读取形状
         else:
              # 如果形状不兼容，返回默认形状的空 tensor
              return torch.empty(0, 3, 64, 64, device=features.device, dtype=features.dtype)

    B, C, N, H, W = features.shape
    # 检查序列维度 N 是否为空
    if N == 0:
        # 警告：特征的序列维度为空。无法转换为感知指标的输入。
        print("Warning: Empty sequence dimension in features. Cannot convert for perceptual metrics.")
        # 返回形状为 (0, 3, H, W) 的空 tensor
        return torch.empty(0, 3, H, W, device=features.device, dtype=features.dtype)

    # 检查 sequence_step 是否越界
    if sequence_step >= N or sequence_step < 0:
        # 警告：sequence_step {} 超出范围 {}，将使用第一个步骤（0）。
        print(f"Warning: sequence_step {sequence_step} is out of bounds for N={N}. Defaulting to 0.")
        sequence_step = 0 # 默认使用第一个步骤

    # 提取指定序列步骤的特征切片 (B, C, H, W)
    features_step = features[:, :, sequence_step, :, :] # 形状为 (B, C, H, W)

    # 简单的转换逻辑：如果通道数 C >= 3，则取前3个通道；否则，取第一个通道并复制到3个通道
    # (以满足 LPIPS/SSIM 通常需要3通道输入的特性)
    if C >= 3:
        perceptual_img = features_step[:, :3, :, :] # 形状为 (B, 3, H, W)
    elif C == 1:
        perceptual_img = features_step.repeat(1, 3, 1, 1) # 形状为 (B, 1, H, W) -> (B, 3, H, W)
    else: # 例如 C=2，取第一个通道并复制到3个通道
        # 确保通道维度存在，即使 C=0，在 unsqueeze/repeat 之前进行检查
        if C > 0:
            # 取第一个通道，增加一个通道维度，然后复制3次
            perceptual_img = features_step[:, 0, :, :].unsqueeze(1).repeat(1, 3, 1, 1) # 形状为 (B, 3, H, W)
        else:
             # 警告：特征通道 C=0。无法创建感知图像。
             print("Warning: Feature channels C=0. Cannot create perceptual image.")
             # 返回形状为 (0, 3, H, W) 的空 tensor
             return torch.empty(0, 3, H, W, device=features.device, dtype=features.dtype)


    # 将像素值归一化到感知指标通常期望的范围，例如 [0, 1]
    # 这是一个简单的 min-max 归一化，可能需要根据实际的 ATN 特征分布进行调整。
    min_val = perceptual_img.min() if perceptual_img.numel() > 0 else 0.0
    max_val = perceptual_img.max() if perceptual_img.numel() > 0 else 1.0

    # 避免除以零或数值不稳定
    if max_val > min_val + 1e-8:
        perceptual_img = (perceptual_img - min_val) / (max_val - min_val)
    else:
        # 处理所有值都相同或 tensor 为空的情况
        perceptual_img = torch.zeros_like(perceptual_img) # 或填充 0.5 表示中等灰色

    # LPIPS/SSIM 通常期望 float 类型输入
    return perceptual_img.float() # 确保返回 float 类型


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """
    计算两张图片（张量形式）的 PSNR (Peak Signal-to-Noise Ratio)。

    Args:
        img1 (torch.Tensor): 第一张图片张量，形状通常为 (B, C, H, W)。
        img2 (torch.Tensor): 第二张图片张量，形状与 img1 相同。
        data_range (float): 像素值的范围 (例如 1.0 表示 [0, 1] 或 255.0 表示 [0, 255])。

    Returns:
        torch.Tensor: 每张图片对的 PSNR 值 (形状 B)。如果输入无效则返回包含 NaN 的张量。
    """
    # 检查输入图片是否为None或为空
    if img1 is None or img2 is None or img1.numel() == 0 or img2.numel() == 0:
        # 警告：用于计算PSNR的输入图像为None或为空。
        print("Warning: Input images for PSNR are None or empty.")
        # 返回包含NaN的张量或适当处理
        # 确定批量大小以返回正确大小的包含NaN的张量
        batch_size = max(img1.shape[0] if img1 is not None else 0, img2.shape[0] if img2 is not None else 0)
        if batch_size == 0: return torch.tensor([]) # 如果没有批量维度信息，返回空张量
        # 返回一个填充NaN的张量，设备与输入相同（如果可用）
        device = img1.device if img1 is not None else (img2.device if img2 is not None else 'cpu')
        return torch.full((batch_size,), float('nan'), device=device)


    # Ensure shapes match except batch dimension
    if img1.shape[1:] != img2.shape[1:]:
        print(f"Error: Image shapes for PSNR calculation mismatch: {img1.shape} vs {img2.shape}")
        # Return NaN tensor for the batch size
        return torch.full((img1.shape[0],), float('nan'), device=img1.device)


    mse = torch.mean((img1 - img2) ** 2, dim=(-3, -2, -1)) # Average over C, H, W dimensions
    # Avoid log(0)
    mse = torch.clamp(mse, min=1e-10)
    psnr = 20 * torch.log10(data_range / torch.sqrt(mse))
    return psnr

def calculate_perceptual_metrics(
    original_features: torch.Tensor,
    adversarial_features: torch.Tensor,
    device: torch.device,
    sequence_step_to_vis: int = 0
) -> dict:
    """
    计算 LPIPS, SSIM, PSNR 等感知评估指标。

    注意：这些指标计算在将 ATN 特征转换为图像格式后，其有效性未知。

    Args:
        original_features (torch.Tensor): 原始特征层张量，形状为 (B, C, N, W, H) 或 (B, C, N, H, W)。
        adversarial_features (torch.Tensor): 对抗性特征层张量，形状与 original_features 相同。
        device (torch.device): 计算设备（例如 'cuda' 或 'cpu'）。
        sequence_step_to_vis (int): 选择序列中的哪个步骤（时间步）进行可视化和感知评估 (0-indexed)。

    Returns:
        dict: 包含感知指标的字典。如果计算失败或 piq 库不可用，相关指标的值为 NaN。
    """
    metrics = {}

    # 将特征转换为适合感知评估的图像格式 (B, 3, H', W')
    # 需要指定用于2D图像比较的序列步骤
    original_img_perceptual = features_to_perceptual_input(original_features, sequence_step=sequence_step_to_vis).to(device)
    adversarial_img_perceptual = features_to_perceptual_input(adversarial_features, sequence_step=sequence_step_to_vis).to(device)


    # Check if converted tensors are empty or shapes mismatch
    if original_img_perceptual.shape[0] == 0 or adversarial_img_perceptual.shape[0] == 0 or original_img_perceptual.shape != adversarial_img_perceptual.shape:
        print("Warning: Converted perceptual input tensors are empty or shapes mismatch. Skipping perceptual metric calculations.")
        metrics['psnr'] = float('nan')
        metrics['lpips'] = float('nan')
        metrics['ssim'] = float('nan')
        return metrics


    # PSNR Calculation
    try:
        # PSNR can be calculated on single-channel or multi-channel images
        psnr_vals = calculate_psnr(original_img_perceptual, adversarial_img_perceptual, data_range=1.0) # Assuming data is normalized to [0, 1]
        # calculate_psnr returns a tensor of shape (B,), take the mean, ignore NaNs
        metrics['psnr'] = np.nanmean(psnr_vals.cpu().numpy()) if psnr_vals.numel() > 0 else float('nan')
    except Exception as e:
        print(f"Error calculating PSNR: {e}")
        metrics['psnr'] = float('nan')


    # LPIPS and SSIM Calculation (requires piq)
    # try:
    #     from piq import LPIPS, ssim # type: ignore
    #     piq_available = True
    #     # print("piq library found.")
    # except ImportError:
    #     # print("Warning: 'piq' library not found. Skipping LPIPS/SSIM. Install with 'pip install piq'")
    #     piq_available = False
    # except Exception as e:
    #     print(f"Warning: Error importing piq: {e}. Skipping LPIPS/SSIM.")
    #     piq_available = False

    # if piq_available:
    #     # print("Attempting to calculate perceptual metrics (LPIPS, SSIM) using piq...")
    #     try:
    #         # LPIPS: Input range is typically [0,1] or [-1,1], depending on the pretrained model.
    #         # AlexNet LPIPS usually requires 3-channel RGB images.
    #         # features_to_perceptual_input attempts to ensure 3-channel output.
    #         # The LPIPS class in piq by default normalizes input from [0,1] to [-1,1].
    #         # So, providing [0,1] should be fine.
    #         if original_img_perceptual.shape[1] == 3:
    #              # Ensure LPIPS metric is on the correct device
    #              # LPIPS metric initialization compatible with piq 0.8.0
    #              lpips_metric = LPIPS().to(device) # Re-initialize or move
    #              # lpips_metric expects tensors on the same device
    #              lpips_val = lpips_metric(original_img_perceptual * 2 - 1, adversarial_img_perceptual * 2 - 1).mean().item() # Map [0,1] to [-1,1]
    #              metrics['lpips'] = lpips_val
    #         else:
    #              print(f"Warning: LPIPS expects 3 channels, got {original_img_perceptual.shape[1]}. Skipping LPIPS.")
    #              metrics['lpips'] = float('nan')

    #         # SSIM: Input range is typically [0,1] or [0, 255] (data_range must match).
    #         # SSIM can be calculated on single-channel or multi-channel images.
    #         # perceptual_img is already float [0, 1]
    #         ssim_val = ssim(original_img_perceptual, adversarial_img_perceptual, data_range=1., reduction='mean').item()
    #         metrics['ssim'] = ssim_val
    #     except Exception as e:
    #         print(f"Error calculating LPIPS/SSIM using piq: {e}")
    #         metrics['lpips'] = float('nan')
    #         metrics['ssim'] = float('nan')
    # else: # piq not available
    #     metrics['lpips'] = float('nan')
    #     metrics['ssim'] = float('nan')

    metrics['lpips'] = float('nan') # Explicitly set LPIPS to NaN
    metrics['ssim'] = float('nan') # Explicitly set SSIM to NaN


    return metrics

def calculate_image_perceptual_metrics(
    original_images: torch.Tensor,
    adversarial_images: torch.Tensor,
    device: torch.device
) -> dict:
    """
    计算两批图像之间的 PSNR, SSIM, LPIPS 等感知评估指标。

    Args:
        original_images (torch.Tensor): 原始图像批次 (B, C, H, W)。
        adversarial_images (torch.Tensor): 对抗图像批次 (B, C, H, W)。
        device (torch.device): 计算设备（例如 'cuda' 或 'cpu'）。

    Returns:
        dict: 包含感知指标的字典。如果计算失败或 piq 库不可用，相关指标的值为 NaN。
    """
    metrics = {}

    # Check if input images are None, empty, or shapes mismatch
    if original_images is None or adversarial_images is None or original_images.numel() == 0 or adversarial_images.numel() == 0 or original_images.shape != adversarial_images.shape:
        print("Warning: Input images for perceptual metrics are None, empty, or shapes mismatch. Skipping perceptual metric calculations.")
        metrics['psnr'] = float('nan')
        metrics['lpips'] = float('nan')
        metrics['ssim'] = float('nan')
        return metrics

    # PSNR Calculation
    try:
        # calculate_psnr expects input in [0, 1] range, which our images should be
        psnr_vals = calculate_psnr(original_images, adversarial_images, data_range=1.0)
        # calculate_psnr returns a tensor of shape (B,), take the mean, ignore NaNs
        metrics['psnr'] = float(np.nanmean(psnr_vals.cpu().numpy())) if psnr_vals.numel() > 0 else float('nan')
    except Exception as e:
        print(f"Error calculating PSNR: {e}")
        metrics['psnr'] = float('nan')

    # LPIPS and SSIM Calculation (requires piq)
    try:
        # Attempt to import piq inside the function to avoid hard dependency if not used
        from piq import LPIPS, ssim # type: ignore
        piq_available = True
        # print("piq library found for image perceptual metrics.") # Debug print
    except ImportError:
        # print("Warning: 'piq' library not found. Skipping LPIPS/SSIM for images.") # Debug print
        piq_available = False
    except Exception as e: # Catch other potential import errors
         print(f"Warning: Error importing piq for image perceptual metrics: {e}. Skipping LPIPS/SSIM.")
         piq_available = False

    if piq_available:
        # print("Attempting to calculate image perceptual metrics (LPIPS, SSIM) using piq...") # Debug print
        try:
            # LPIPS: Input range is typically [0,1] or [-1,1]. piq.LPIPS by default normalizes [0,1] to [-1,1].
            # It requires 3-channel RGB images.
            if original_images.shape[1] == 3:
                 lpips_metric = LPIPS().to(device) # Initialize and move to device
                 # LPIPS metric expects tensors on the same device as itself.
                 # Map [0,1] to [-1,1] for LPIPS
                 lpips_val = lpips_metric(original_images * 2 - 1, adversarial_images * 2 - 1).mean().item()
                 metrics['lpips'] = lpips_val
            else:
                 print(f"Warning: LPIPS expects 3 channels for images, got {original_images.shape[1]}. Skipping LPIPS.")
                 metrics['lpips'] = float('nan')

            # SSIM: Input range typically [0,1] or [0, 255] (data_range must match). Images are [0, 1].
            ssim_val = ssim(original_images, adversarial_images, data_range=1., reduction='mean').item()
            metrics['ssim'] = ssim_val
            # print("Image perceptual metrics calculation successful.") # Debug print
        except Exception as e:
            print(f"Error calculating LPIPS/SSIM for images using piq: {e}")
            metrics['lpips'] = float('nan')
            metrics['ssim'] = float('nan')
    else: # piq not available
        metrics['lpips'] = float('nan')
        metrics['ssim'] = float('nan')

    return metrics

# --- Example of how to use in train/eval script ---
def evaluate_model(
    generator: Optional[torch.nn.Module],
    discriminator: Optional[torch.nn.Module],
    atn_model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    cfg: Any, # Use Any or EasyDict if available
    current_train_stage: int
) -> Dict[str, float]:
    """
    在评估数据集上评估生成器和判别器的性能。

     Args:
        generator (Optional[torch.nn.Module]): 生成器模型。在 Stage 1 或 Stage 2 评估时需要生成器产生的对抗样本。
        discriminator (Optional[torch.nn.Module]): 判别器模型。如果需要评估判别器性能（如得分），则需要。
        atn_model (torch.nn.Module): ATN 模型（冻结）。
        dataloader (torch.utils.data.DataLoader): 评估数据加载器。
        device (torch.device): 计算设备。
        cfg (Any): 配置对象。
        current_train_stage (int): 当前训练阶段 (1 或 2)。

    Returns:
        Dict[str, float]: 包含评估指标名称及其对应值的字典。
                          例如：{'Feature_L2_Diff_Avg': 0.5, 'TriggerNet_ASR': 0.8, 'PSNR_Avg': 25.0, ...}
    """
    # Ensure ATN model is in evaluation mode and on the correct device
    # Note: IrisBabelModel handles its own device placement internally.
    # Ensure other models are in eval mode if they exist
    if generator is not None: generator.eval()
    if discriminator is not None: discriminator.eval()

    print(f"Starting evaluation for Stage {current_train_stage}...")

    # Initialize evaluation metrics dictionary and lists to accumulate batch results
    eval_metrics_lists: Dict[str, list] = {}

    # Use torch.no_grad() context manager, ensure gradients are not computed during evaluation
    with torch.no_grad():
        # Iterate over the evaluation dataloader
        from tqdm import tqdm # Import tqdm here for local use in this function
        for i, batch_data in tqdm(enumerate(dataloader), total=len(dataloader), desc="Evaluating"):
            # Check if batch is None or empty
            if batch_data is None or batch_data.size(0) == 0:
                 print(f"Warning: Skipping empty evaluation batch {i}.")
                 continue # Skip empty batch

            # Original image data (B, T, H, W, C) from dataset
            real_x_original_format = batch_data.to(device) # Shape (B, T, H, W, C)

            # Permute original images to (B, C, T, H, W) if Stage 2 Generator expects this
            # In Stage 2, Generator takes original images (B, C, T, H, W) and outputs pixel delta (B, C, T, H, W)
            # ATN models take (B, T, H, W, C) input.
            # Let's work with both formats as needed.
            real_x_permuted = real_x_original_format.permute(0, 4, 1, 2, 3) # (B, T, H, W, C) -> (B, C, T, H, W)

            # --- Generate Adversarial Samples (Stage 1 or Stage 2) ---
            # This depends on the generator and the stage
            # Stage 1: Generator inputs Features, outputs Feature Delta -> Adversarial Features
            # Stage 2: Generator inputs Images, outputs Pixel Delta -> Adversarial Images

            adversarial_x = None # Will hold either adversarial features or adversarial images
            pixel_delta = None # Will hold pixel delta if Stage 2
            adversarial_features = None # Will hold adversarial features if Stage 1
            original_features = None # Will hold original features if needed

            if generator is not None: # If a generator is provided
                 if current_train_stage == 1:
                      # Stage 1 evaluation needs original features first
                      original_atn_outputs_for_gen = get_atn_outputs(
                         atn_model,
                         real_x_original_format, # ATN takes (B, T, H, W, C)
                         return_features=True, return_decision=False, return_attention=False
                      )
                      original_features = original_atn_outputs_for_gen.get('features') # Expected shape (B, C_feat, N_feat, H_feat, W_feat)

                      if original_features is not None and original_features.numel() > 0:
                           # Generator takes original features (B, C_feat, N_feat, H_feat, W_feat)
                           feature_delta = generator(original_features.to(device)) # Feature delta (B, C_feat, N_feat, H_feat, W_feat)
                           adversarial_features = original_features.to(device) + feature_delta # Adversarial features
                           # Ensure adversarial features are valid (e.g., within certain bounds if applicable)
                           # Not typically bounded like pixel values
                      else:
                           print(f"Warning: Evaluation batch {i} - Failed to get original features for Stage 1 generator input. Skipping batch for metrics depending on adversarial features.")
                           # Skip batch if features are missing for Stage 1 gen
                           continue # Skip to next batch

                 elif current_train_stage == 2:
                      # Stage 2 evaluation: Generator takes original images, outputs pixel delta
                      # Generator expects (B, C, T, H, W)
                      pixel_delta = generator(real_x_permuted) # Pixel delta (B, 3, T, H, W)
                      # Apply L-inf constraint for evaluation if generator didn't guarantee it
                      epsilon = getattr(cfg.model.generator, 'epsilon', None)
                      if epsilon is not None:
                           pixel_delta = torch.clamp(pixel_delta, -epsilon, epsilon)

                      # Calculate adversarial images: original images + pixel delta
                      # Ensure original images are in the same format as delta (B, C, T, H, W)
                      adversarial_x = real_x_permuted + pixel_delta
                      # Clamp adversarial images to [0, 1] range
                      adversarial_x = torch.clamp(adversarial_x, 0, 1) # Adversarial images (B, 3, T, H, W)

                      # In Stage 2 eval, we might also need ATN outputs for both original and adversarial images
                      # ATN models expect (B, T, H, W, C) input format
                      # Permute adversarial_x back to (B, T, H, W, C) for ATN
                      adversarial_x_original_format = adversarial_x.permute(0, 2, 3, 4, 1) # (B, 3, T, H, W) -> (B, T, H, W, 3)

                      # Get ATN outputs for original and adversarial images
                      original_atn_outputs = get_atn_outputs(
                           atn_model,
                           real_x_original_format, # Original images (B, T, H, W, C)
                           return_features=True, return_decision=True, return_attention=True
                      )
                      adversarial_atn_outputs = get_atn_outputs(
                           atn_model,
                           adversarial_x_original_format, # Adversarial images (B, T, H, W, C)
                           return_features=True, return_decision=True, return_attention=True
                      )

                      # Extract outputs needed for Stage 2 metrics
                      original_trigger_output = original_atn_outputs.get('trigger_output') # TriggerNet output
                      adversarial_trigger_output = adversarial_atn_outputs.get('trigger_output') # TriggerNet output
                      # original_attention_map = original_atn_outputs.get('attention') # Attention map (Optional metric)
                      # adversarial_attention_map = adversarial_atn_outputs.get('attention') # Attention map (Optional metric)

                 else: # Should not happen based on script entry point, but for completeness
                      print(f"Error: evaluate_model called with unknown training stage {current_train_stage}. Cannot proceed.")
                      return {} # Return empty dict for error
            else: # If no generator is provided, evaluate original data or pre-saved adversarial data
                 print("Warning: No generator provided to evaluate_model. Evaluating original data only, or assuming adversarial data is implicitly handled.")
                 # If no generator, can only evaluate original ATN outputs and potentially pre-loaded adversarial data
                 # For this implementation, we assume evaluation requires generated adversarial samples.
                 # If you need to evaluate pre-saved samples, modify this logic to load them instead.
                 print("Error: Generator is None, cannot perform evaluation requiring adversarial samples.")
                 return {} # Return empty dict if generator is missing

            # --- Calculate Metrics based on Training Stage --- #
            # Metrics to calculate for the current batch
            batch_metrics: Dict[str, float] = {}

            if current_train_stage == 1:
                # Stage 1 Metrics: Feature Difference, GAN evaluation (optional)
                if original_features is not None and adversarial_features is not None:
                    # Calculate perceptual metrics on features (using placeholder function)
                    # stage_1_seq_step = getattr(cfg.evaluation, 'sequence_step_to_vis', 0) # Use config step for feature vis if needed
                    # feature_perceptual_metrics = calculate_perceptual_metrics(
                    #     original_features=original_features, # shape (B, C, N, H, W)
                    #     adversarial_features=adversarial_features, # shape (B, C, N, H, W)
                    #     device=device,
                    #     sequence_step_to_vis=stage_1_seq_step # Use config step
                    # ) # This function is a placeholder and needs to be adjusted for meaningful feature comparison
                    # batch_metrics.update(feature_perceptual_metrics)
                    pass # Skipping placeholder feature perceptual metrics for now

                # (Optional) Calculate Discriminator scores for Stage 1 features
                if discriminator is not None and original_features is not None and adversarial_features is not None:
                     try:
                         # Discriminator expects (B, C, T, H, W) - need to ensure feature shape matches this
                         # Assuming ATN features are (B, 128, N, H, W), need to match Discriminator input
                         # If Discriminator is 3D (for sequences), it should match (B, C, T, H, W) format
                         # If Discriminator is PatchGAN (3D), it also expects (B, C, T, H, W)
                         # Let's assume the Discriminator in Stage 1 is designed for features (B, 128, N, H, W)
                         # If the Discriminator model definition expects a different shape, adjust here.
                         D_real_output = discriminator(original_features) # Pass original features to D
                         D_fake_output = discriminator(adversarial_features) # Pass adversarial features to D
                         batch_metrics['Discriminator_Score_Real_Avg'] = D_real_output.mean().item()
                         batch_metrics['Discriminator_Score_Fake_Avg'] = D_fake_output.mean().item()
                     except Exception as e:
                          print(f"Error calculating Stage 1 discriminator scores: {e}. Skipping.")
                          # Add NaN placeholders if calculation fails
                          batch_metrics['Discriminator_Score_Real_Avg'] = float('nan')
                          batch_metrics['Discriminator_Score_Fake_Avg'] = float('nan')

            elif current_train_stage == 2:
                # Stage 2 Metrics: Attack Success Rate (TriggerNet), TriggerNet Output Diff, Perceptual Metrics (Images)
                # Ensure original and adversarial TriggerNet outputs are available and valid
                if original_trigger_output is not None and adversarial_trigger_output is not None:
                    # 1. TriggerNet Output Difference Metrics
                    try:
                        # MSE Difference (already calculated in training, but recalculate here for eval consistency)
                        # Flatten outputs for MSE calculation per sample if they are spatial/temporal maps
                        orig_trigger_flat = original_trigger_output.view(original_trigger_output.shape[0], -1)
                        adv_trigger_flat = adversarial_trigger_output.view(adversarial_trigger_output.shape[0], -1)
                        # Calculate MSE per sample
                        trigger_mse_diff_per_sample = torch.mean((adv_trigger_flat - orig_trigger_flat)**2, dim=1) # Shape (B,)
                        batch_metrics['TriggerNet_MSE_Diff_Avg'] = float(torch.mean(trigger_mse_diff_per_sample).item()) # Average over batch

                        # Cosine Similarity
                        cosine_sim_per_sample = F.cosine_similarity(orig_trigger_flat, adv_trigger_flat, dim=1) # Shape (B,)
                        batch_metrics['TriggerNet_Cosine_Sim_Avg'] = float(torch.mean(cosine_sim_per_sample).item()) # Average over batch

                    except Exception as e:
                         print(f"Error calculating TriggerNet output difference metrics: {e}. Skipping.")
                         batch_metrics['TriggerNet_MSE_Diff_Avg'] = float('nan')
                         batch_metrics['TriggerNet_Cosine_Sim_Avg'] = float('nan')

                    # 2. Attack Success Rate (ASR) for TriggerNet
                    try:
                        # Get success criterion and threshold from config for Stage 2 TriggerNet attack
                        trigger_success_criterion = getattr(cfg.evaluation, 'success_criterion', 'mse_diff_threshold')
                        trigger_success_threshold = getattr(cfg.evaluation, 'success_threshold', 0.1)
                        # Calculate ASR using the dedicated function
                        # Ensure inputs to calculate_attack_success_rate are compatible with its expected dimensions
                        # calculate_attack_success_rate expects (B, ...) and will flatten internally if needed.
                        batch_asr = calculate_attack_success_rate(
                            original_trigger_output,
                            adversarial_trigger_output,
                            success_threshold=trigger_success_threshold,
                            success_criterion=trigger_success_criterion,
                            # Add other relevant parameters if needed by the criterion (e.g., topk_k)
                            topk_k=getattr(cfg.losses, 'topk_k', 10) # Pass topk_k from config if needed
                        )
                        batch_metrics['TriggerNet_ASR'] = float(batch_asr) # ASR is already a scalar float

                    except Exception as e:
                         print(f"Error calculating TriggerNet ASR: {e}. Skipping.")
                         batch_metrics['TriggerNet_ASR'] = float('nan')
                elif original_trigger_output is None or adversarial_trigger_output is None:
                     print(f"Warning: Original or adversarial TriggerNet output is None for batch {i}. Cannot calculate Stage 2 TriggerNet metrics.")
                     # Add NaN placeholders if TriggerNet outputs are missing
                     batch_metrics['TriggerNet_MSE_Diff_Avg'] = float('nan')
                     batch_metrics['TriggerNet_Cosine_Sim_Avg'] = float('nan')
                     batch_metrics['TriggerNet_ASR'] = float('nan')


                # 3. Perceptual Metrics (on Images)
                # Ensure original and adversarial images are available and valid (B, 3, T, H, W)
                if real_x_permuted is not None and adversarial_x is not None and real_x_permuted.shape == adversarial_x.shape:
                    # Select a sequence step for 2D perceptual metrics if input is 5D
                    # Use the step specified in config for evaluation visualization/metrics
                    perceptual_eval_step = getattr(cfg.evaluation, 'sequence_step_to_vis', 0)

                    # Extract the specified sequence step for perceptual metrics (B, 3, H, W)
                    if real_x_permuted.ndim == 5:
                         if perceptual_eval_step < real_x_permuted.shape[2]:
                              original_image_step = real_x_permuted[:, :, perceptual_eval_step, :, :]
                              adversarial_image_step = adversarial_x[:, :, perceptual_eval_step, :, :]
                         else:
                              print(f"Warning: Perceptual evaluation step {perceptual_eval_step} out of bounds for image sequence length {real_x_permuted.shape[2]} in batch {i}. Skipping image perceptual metrics.")
                              # Add NaN placeholders if step is out of bounds
                              batch_metrics['psnr'] = float('nan')
                              batch_metrics['lpips'] = float('nan')
                              batch_metrics['ssim'] = float('nan')
                              original_image_step = None # Set to None to skip calculation
                              adversarial_image_step = None # Set to None to skip calculation

                    elif real_x_permuted.ndim == 4: # If input is already 4D (no sequence dim)
                         original_image_step = real_x_permuted
                         adversarial_image_step = adversarial_x

                    else: # Neither 4D nor 5D
                         print(f"Warning: Image tensors are not 4D or 5D ({real_x_permuted.ndim}D) for perceptual metrics in batch {i}. Skipping.")
                         # Add NaN placeholders if dimensions are wrong
                         batch_metrics['psnr'] = float('nan')
                         batch_metrics['lpips'] = float('nan')
                         batch_metrics['ssim'] = float('nan')
                         original_image_step = None # Set to None to skip calculation
                         adversarial_image_step = None # Set to None to skip calculation

                    # Calculate image perceptual metrics if valid image steps were extracted
                    if original_image_step is not None and adversarial_image_step is not None:
                         # Ensure images are 3 channels for SSIM/LPIPS if needed
                         if original_image_step.shape[1] == 3:
                              image_perceptual_metrics = calculate_image_perceptual_metrics(
                                   original_images=original_image_step, # (B, 3, H, W)
                                   adversarial_images=adversarial_image_step, # (B, 3, H, W)
                                   device=device
                              )
                              batch_metrics.update(image_perceptual_metrics) # Add psnr, lpips, ssim
                         else:
                              print(f"Warning: Image tensors for perceptual metrics are not 3 channels ({original_image_step.shape[1]}). Skipping SSIM/LPIPS.")
                              # Calculate PSNR even if not 3 channels, but skip SSIM/LPIPS
                              try:
                                  psnr_vals = calculate_psnr(original_image_step, adversarial_image_step, data_range=1.0)
                                  batch_metrics['psnr'] = float(np.nanmean(psnr_vals.cpu().numpy())) if psnr_vals.numel() > 0 else float('nan')
                              except Exception as e:
                                  print(f"Error calculating PSNR for non-3-channel images: {e}. Skipping.")
                                  batch_metrics['psnr'] = float('nan')
                              batch_metrics['lpips'] = float('nan') # Explicitly NaN
                              batch_metrics['ssim'] = float('nan') # Explicitly NaN


                else:
                     print(f"Warning: Original or adversarial image tensors are None or shape mismatch for perceptual metrics in batch {i}. Skipping.")
                     # Add NaN placeholders if image tensors are missing
                     batch_metrics['psnr'] = float('nan')
                     batch_metrics['lpips'] = float('nan')
                     batch_metrics['ssim'] = float('nan')


            # Collect batch metrics into the lists
            for metric_name, metric_value in batch_metrics.items():
                if metric_name not in eval_metrics_lists:
                    eval_metrics_lists[metric_name] = []
                # Only append finite values to avoid skewing the average with NaNs/Infs
                if np.isfinite(metric_value):
                     eval_metrics_lists[metric_name].append(metric_value)
                # print(f"Debug: Batch {i} - {metric_name}: {metric_value}") # Debug print for batch metrics

    # Calculate average metrics across all batches
    average_metrics: Dict[str, float] = {}
    if eval_metrics_lists:
        print("Calculating average evaluation metrics...")
        for metric_name, metric_values in eval_metrics_lists.items():
            if metric_values: # Ensure list is not empty after filtering NaNs/Infs
                 average_metrics[metric_name] = float(np.mean(metric_values)) # Calculate mean of finite values
            else:
                 average_metrics[metric_name] = float('nan') # If list becomes empty, average is NaN
            # print(f"Debug: Average {metric_name}: {average_metrics[metric_name]}") # Debug print for average metrics

    print("Evaluation finished.")
    # Restore models to training mode (if they were training)
    if generator is not None and current_train_stage in [1, 2]: # Stage 2 also trains generator
        generator.train()
    if discriminator is not None: # Discriminator train in both stages if used
        discriminator.train()

    return average_metrics # Return the dictionary of average metrics
