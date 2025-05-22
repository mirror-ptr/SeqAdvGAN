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
from typing import Optional # Import Optional
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
    try:
        from piq import LPIPS, ssim # type: ignore
        piq_available = True
        # print("piq library found.")
    except ImportError:
        # print("Warning: 'piq' library not found. Skipping LPIPS/SSIM. Install with 'pip install piq'")
        piq_available = False
    except Exception as e:
        print(f"Warning: Error importing piq: {e}. Skipping LPIPS/SSIM.")
        piq_available = False

    if piq_available:
        # print("Attempting to calculate perceptual metrics (LPIPS, SSIM) using piq...")
        try:
            # LPIPS: Input range is typically [0,1] or [-1,1], depending on the pretrained model.
            # AlexNet LPIPS usually requires 3-channel RGB images.
            # features_to_perceptual_input attempts to ensure 3-channel output.
            # The LPIPS class in piq by default normalizes input from [0,1] to [-1,1].
            # So, providing [0,1] should be fine.
            if original_img_perceptual.shape[1] == 3:
                 # Ensure LPIPS metric is on the correct device
                 # LPIPS metric initialization compatible with piq 0.8.0
                 lpips_metric = LPIPS(network='alex').to(device) # Re-initialize or move
                 # lpips_metric expects tensors on the same device
                 lpips_val = lpips_metric(original_img_perceptual * 2 - 1, adversarial_img_perceptual * 2 - 1).mean().item() # Map [0,1] to [-1,1]
                 metrics['lpips'] = lpips_val
            else:
                 print(f"Warning: LPIPS expects 3 channels, got {original_img_perceptual.shape[1]}. Skipping LPIPS.")
                 metrics['lpips'] = float('nan')

            # SSIM: Input range is typically [0,1] or [0, 255] (data_range must match).
            # SSIM can be calculated on single-channel or multi-channel images.
            # perceptual_img is already float [0, 1]
            ssim_val = ssim(original_img_perceptual, adversarial_img_perceptual, data_range=1., reduction='mean').item()
            metrics['ssim'] = ssim_val
        except Exception as e:
            print(f"Error calculating LPIPS/SSIM using piq: {e}")
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
    cfg: object, # Use a more specific type hint if possible, e.g., EasyDict
    current_train_stage: int
) -> dict:
    """
    根据当前训练阶段评估生成器和判别器的性能。

     Args:
        generator (Optional[torch.nn.Module]): 生成器模型。如果评估原始数据，可以为 None。
        discriminator (Optional[torch.nn.Module]): 判别器模型。如果不需要 GAN 指标，可以为 None。
        atn_model (nn.Module): ATN 模型。
        dataloader (DataLoader): 评估数据加载器。
        device (torch.device): 计算设备。
        cfg (object): 包含评估参数的配置对象 (EasyDict 或类似结构)。
        current_train_stage (int): 当前训练阶段 (1 或 2)。

    Returns:
        dict: 包含评估指标的字典。
    """
    # 设置模型为评估模式 (ATN 模型通常在训练期间也保持评估模式)
    if generator is not None:
        generator.eval()
    if discriminator is not None:
        discriminator.eval()
    # ATN 模型通常在训练和评估期间都保持 eval 模式
    atn_model.eval()

    # 初始化指标存储
    total_attack_success_rate_decision = 0.0
    total_attack_success_rate_attention = 0.0
    total_attack_success_rate_feature = 0.0 # 新增：阶段1的特征攻击成功率
    total_linf_norm = 0.0
    total_l2_norm = 0.0
    avg_D_real_score = 0.0
    avg_D_fake_score = 0.0
    total_samples = 0
    all_psnr = []
    all_lpips = []
    all_ssim = []

    # 判断是否应该计算感知指标 (通常在输入数据是图像时相关)
    # 假设 cfg.data.channels == 3 表示输入是图像类型
    calculate_perceptual = getattr(cfg.data, 'channels', 0) == 3 # 安全地获取channels参数

    # 如果需要，初始化感知指标计算器
    if calculate_perceptual:
        try:
            # 尝试导入 piq 库 (延迟导入)
            from piq import LPIPS, ssim # type: ignore 
            # LPIPS 度量初始化兼容 piq 0.8.0
            # 在评估循环外部初始化 LPIPS 度量，以便在多个批次之间重用
            # 使用一个单独的实例 if needed
            lpips_metric_eval = LPIPS(network='alex').to(device) # 移动到正确设备
            # SSIM 不需要像 LPIPS 那样显式初始化
            piq_available = True
        except ImportError:
            # 警告：未找到 piq 库。跳过 LPIPS 和 SSIM 指标。请使用 'pip install piq' 安装。
            print("Warning: piq library not found. Skipping LPIPS and SSIM metrics. Install with 'pip install piq'.")
            calculate_perceptual = False # 如果 piq 缺失则禁用感知指标
        except Exception as e:
            # 警告：初始化感知指标时出错 (评估阶段)：{}。跳过 LPIPS 和 SSIM。
            print(f"Warning: Error initializing perceptual metrics (evaluation): {e}. Skipping LPIPS and SSIM.")
            calculate_perceptual = False

    # 在 torch.no_grad() 上下文中进行评估，不计算梯度
    with torch.no_grad(): # 确保评估期间不计算梯度
        # 获取评估批次数量，如果配置中没有指定，则使用数据加载器的全部批次
        num_eval_batches = cfg.evaluation.get('num_eval_batches', len(dataloader)) # 使用 config 中配置的数量
        # 确保评估批次数量不超过数据加载器中的实际批次数量
        if num_eval_batches > len(dataloader): 
            num_eval_batches = len(dataloader)
        # 处理数据加载器为空的情况
        if num_eval_batches == 0: 
             # 警告：评估期间数据加载器为空。
             print("Warning: Dataloader is empty during evaluation.")
             # 返回包含 NaN 的字典
             return {
                'Attack_Success_Rate_Decision_Stage'+str(current_train_stage): float('nan'),
                'Attack_Success_Rate_Attention_Stage'+str(current_train_stage): float('nan'),
                'Attack_Success_Rate_Feature_Stage'+str(current_train_stage): float('nan'),
                'Linf_Norm_Avg': float('nan'),
                'L2_Norm_Avg': float('nan'),
                'Discriminator_Real_Score_Avg': float('nan'),
                'Discriminator_Fake_Score_Avg': float('nan'),
                'PSNR_Avg': float('nan'),
                'LPIPS_Avg': float('nan'),
                'SSIM_Avg': float('nan'),
            }

        # 遍历评估数据加载器
        for batch_idx, original_input in enumerate(dataloader):
            # 如果已达到指定的评估批次数量，则停止
            if batch_idx >= num_eval_batches:
                break

            # 将原始输入数据移动到计算设备
            original_input = original_input.to(device)
            batch_size = original_input.shape[0]
            # 处理当前批次为空的情况
            if batch_size == 0: continue

            # 在将输入传递给生成器之前，通过 ATN 获取特征 (原始输入是图像数据)
            # 在 torch.no_grad() 上下文外部进行，因为 get_atn_outputs 内部可能有自己的 no_grad()
            # 或者确保 get_atn_outputs 不会计算梯度
            # 这里的 atn_model 应该始终是 .eval() 模式
            # 确保 get_atn_outputs 不会计算梯度
            # 方式1: 在调用 get_atn_outputs 前加 with torch.no_grad():
            # 方式2: 确保 get_atn_outputs 内部有 torch.no_grad()
            # 方式3: 确保 atn_model 在 eval 模式下不会产生需要梯度的操作
            # 鉴于 evaluate_model 整体在 no_grad() 中，这里是安全的。
            
            if atn_model is not None:
                # 通过 ATN 模型获取原始输入的输出，包括特征 (即使在阶段2也获取特征)
                # return_features=True 是为了计算阶段1的特征攻击成功率，以及 GAN 评估指标
                original_atn_outputs = get_atn_outputs(
                    atn_model,
                    original_input, # 传递原始图像数据
                    return_features=True,
                    return_decision=True, # 在评估阶段通常需要决策和注意力
                    return_attention=True
                )
                original_features = original_atn_outputs.get('features')
                original_decision_map = original_atn_outputs.get('decision')
                original_attention_map = original_atn_outputs.get('attention')

                # 如果原始特征为空或维度不正确，无法进行后续评估
                if original_features is None or original_features.numel() == 0:
                    # 错误：ATN 模型在评估期间返回 None 或空特征。无法继续。
                    print("Error: ATN model returned None or empty features during evaluation. Cannot proceed.")
                    return {} # 返回空字典表示评估失败
            else:
                # 错误：评估期间 ATN 模型为None。无法获取特征。
                print("Error: ATN model is None during evaluation. Cannot get features.")
                return {} # 返回空字典表示评估失败

            # 生成器根据当前阶段的输入产生扰动
            # 在阶段1，Generator 输入是特征，输出是特征扰动 delta
            # 在阶段2，Generator 输入是图像，输出是图像扰动 delta
            if generator is not None:
                # 将原始特征传递给生成器 (阶段1)
                if current_train_stage == 1:
                     # delta 是特征扰动 (B, C_feat, N, W_feat, H_feat)
                     delta = generator(original_features)
                # 在阶段2，生成器输入可能是原始图像 (如果 cfg.data.channels == 3) 或其他
                elif current_train_stage == 2:
                     # 假设阶段2的生成器输入是原始图像数据
                     # delta 是图像扰动 (B, C_img, T, H_img, W_img)
                     delta = generator(original_input) # 假设 generator 处理原始图像输入
                else:
                     # 错误：不支持的训练阶段 {} 用于生成器输入确定。
                     print(f"Error: Unsupported training stage {current_train_stage} for generator input determination.")
                     return {} # 返回空字典表示评估失败
            else: # 如果生成器为 None，则扰动为零
                # 如果 generator 为 None (例如，仅评估原始样本)，delta 为零
                if current_train_stage == 1: # 阶段1，delta 是特征扰动
                    delta = torch.zeros_like(original_features)
                elif current_train_stage == 2: # 阶段2，delta 是图像扰动
                    delta = torch.zeros_like(original_input) # 假设原始输入是图像
                else:
                    delta = None # 如果阶段未知或不支持，delta 为 None

            # 应用扰动并获取对抗样本/特征
            adversarial_input = None
            adversarial_features = None
            adversarial_decision_map = None
            adversarial_attention_map = None

            if delta is not None:
                if current_train_stage == 1:
                    # 阶段1：在特征空间应用扰动
                    # 对抗特征 = 原始特征 + 特征扰动
                    adversarial_features = original_features + delta # delta 是特征扰动
                    # 在阶段1评估中，通常不需要将对抗特征通过 ATN 重新获取决策/注意力
                    # 因为攻击目标是特征本身。决策和注意力评估主要在阶段2进行。

                elif current_train_stage == 2:
                    # 阶段2：在图像空间应用扰动
                    # 对抗输入 = 原始输入 + 图像扰动，并进行裁剪
                    # 假设原始输入 original_input 是图像数据 (B, C, T, H, W)
                    adversarial_input = torch.clamp(original_input + delta, 0, 1) # delta 是图像扰动

                    # 在阶段2评估中，将对抗输入通过 ATN 重新获取决策和注意力
                    adversarial_atn_outputs = get_atn_outputs(
                         atn_model,
                         adversarial_input, # 传递对抗图像
                         return_features=True, # 仍然获取特征用于 GAN 评估
                         return_decision=True,
                         return_attention=True
                    )
                    adversarial_features = adversarial_atn_outputs.get('features')
                    adversarial_decision_map = adversarial_atn_outputs.get('decision')
                    adversarial_attention_map = adversarial_atn_outputs.get('attention')
                # else: Unsupported stage handled before

            # --- 计算基于当前阶段的评估指标 ---

            # 攻击成功率 (特征攻击 - 仅在阶段1评估中相关)
            # 此计算需要 original_features (从最开始的 get_atn_outputs 调用获取)
            # 和 adversarial_features (在阶段1内部计算为 original_features + delta)。
            # 确保特征可用且不是 None 或空
            if current_train_stage == 1 and original_features is not None and adversarial_features is not None:
                 # 假设特征攻击成功意味着特征之间存在较大的 MSE 差异
                 # 使用 cfg.evaluation 中的 success_threshold 和 success_criterion
                 # 注意：在 calculate_attack_success_rate 内部，如果 criterion 是 mse_diff_threshold
                 # higher_is_better_orig = False 表示我们希望差异越大越好 (这代表成功的特征攻击)
                 success_rate_feature = calculate_attack_success_rate(
                      original_features.view(batch_size, -1), # 展平特征以兼容标准函数
                      adversarial_features.view(batch_size, -1), # 展平特征
                      cfg.evaluation.success_threshold, # 使用 config 中的阈值
                      success_criterion='mse_diff_threshold', # 特征攻击成功率基于特征差异的 MSE
                      higher_is_better_orig=False # 对于特征差异，差异越大攻击效果越好
                 )
                 total_attack_success_rate_feature += success_rate_feature * batch_size # 累加按样本数加权的成功率
            # 决策图和注意力成功率仅在阶段2计算

            # 攻击成功率 (决策图 - 仅在阶段2评估中相关)
            if current_train_stage == 2 and original_decision_map is not None and adversarial_decision_map is not None:
                # 假设 ATN 决策图的值越高越好 (例如代表置信度)，攻击目标是降低它
                # 使用 cfg.evaluation 中为决策图指定的成功标准和阈值
                # calculate_attack_success_rate 函数会处理 higher_is_better_orig=True 的情况
                success_rate_decision = calculate_attack_success_rate(
                    original_decision_map, # (B, H, W) 或 (B, W, H)
                    adversarial_decision_map,
                    cfg.evaluation.success_threshold, # 使用 config 中的阈值
                    cfg.evaluation.success_criterion, # 使用 config 中为决策图评估指定的标准
                    higher_is_better_orig=True # 假设更高的决策值对于原始 ATN 更好
                )
                total_attack_success_rate_decision += success_rate_decision * batch_size # 累加按样本数加权的成功率

            # 攻击成功率 (注意力图 - 仅在阶段2评估中相关)
            if current_train_stage == 2 and original_attention_map is not None and adversarial_attention_map is not None:
                 # 使用配置中特定于注意力的成功标准/阈值 (如果存在)
                 # 否则，使用默认的评估标准/阈值
                 attention_success_criterion = cfg.evaluation.get('attention_success_criterion', cfg.evaluation.success_criterion)
                 attention_success_threshold = cfg.evaluation.get('attention_success_threshold', cfg.evaluation.success_threshold)
                 # 获取用于 Top-K 评估的 K 值，优先使用 losses.topk_k，否则默认 10
                 eval_topk_k = cfg.losses.get('topk_k', 10) 

                 # calculate_attack_success_rate 函数会根据 criterion 使用 topk_k
                 success_rate_attention = calculate_attack_success_rate(
                     original_attention_map, # (B, head, N, N)
                     adversarial_attention_map,
                     attention_success_threshold,
                     attention_success_criterion, # 使用特定于注意力的标准 (例如 'topk_value_drop', 'topk_position_change')
                     topk_k=eval_topk_k # 使用配置中的 K 值
                 )
                 total_attack_success_rate_attention += success_rate_attention * batch_size # 累加按样本数加权的成功率

            # 扰动范数 (在两个阶段都相关)
            # delta 必须可用且不是 None 或空
            # 在阶段1，delta 是特征扰动 (B, C_feat, N, W_feat, H_feat)
            # 在阶段2，delta 是图像扰动 (B, C_img, T, H_img, W_img)
            if generator is not None and delta is not None and delta.numel() > 0:
                # linf_norm 和 l2_norm 函数需要张量形状类似 (B, C, N, H, W)
                # 确保 delta 具有正确的维度
                if delta.ndim == 5:
                     # 计算 L-inf 和 L2 范数，取批量平均后累加
                     total_linf_norm += linf_norm(delta).mean().item() * batch_size # 对每个样本计算范数，然后取批量平均
                     total_l2_norm += l2_norm(delta).mean().item() * batch_size
                else:
                     # 警告：用于范数计算的 delta 张量不是5D ({})。
                     print(f"Warning: Delta tensor for norm calculation is not 5D ({delta.ndim}D).")

            # GAN 评估指标 (在两个阶段都相关，如果使用了判别器)
            # 需要原始特征和对抗特征，以及判别器
            # 确保特征可用且不是 None 或空，判别器也可用
            if discriminator is not None and original_features is not None and adversarial_features is not None:
                 # 确保特征适合判别器输入 (例如，正确的形状和值范围)
                 # 假设判别器输入是 ATN 特征头输出的特征 (B, C_feat, N, W_feat, H_feat)
                 # 需要确保 original_features 和 adversarial_features 从 get_atn_outputs 获取后是这个形状
                 try:
                     # 通过判别器获取对真实和伪造特征的输出
                     # 使用 detach() 确保不计算判别器评估时的梯度
                     D_real_output = discriminator(original_features.detach()) # (B, ...)
                     D_fake_output = discriminator(adversarial_features.detach()) # (B, ...)

                     # 判别器输出形状通常为 (B, 1) 或 (B, 1, patch_N, patch_H, patch_W)
                     # 展平除批量维度外的所有维度，计算每个样本的平均得分，然后对批量求和
                     # view(batch_size, -1) 将 (B, 1, patch_N, patch_H, patch_W) 展平为 (B, patch_N*patch_H*patch_W)
                     # mean(dim=-1) 计算每个样本展平后的平均得分
                     # sum().item() 计算所有样本的平均得分总和 (稍后除以 total_samples 得到最终平均值)
                     avg_D_real_score += D_real_output.view(batch_size, -1).mean(dim=-1).sum().item()
                     avg_D_fake_score += D_fake_output.view(batch_size, -1).mean(dim=-1).sum().item()
                 except Exception as e:
                     # 警告：计算 GAN 评估指标时出错：{}。跳过此批次。
                     print(f"Warning: Error calculating GAN evaluation metrics: {e}. Skipping for this batch.")

            # 感知指标 (仅在输入是图像且 calculate_perceptual 为 True 时相关)
            # 这些指标理想情况下应该在原始输入空间计算，而不是特征空间
            # 如果生成器输入是图像并输出图像扰动，使用 original_input 和 adversarial_input
            # 如果生成器输入是特征并输出特征扰动 (阶段1)，则在特征上计算这些指标可能没有意义
            # 假设当 calculate_perceptual 为 True 时，生成器输入是图像 (3通道)
            # 也假设 original_input 在阶段2是图像数据 (B, C_img, T, H_img, W_img)
            if calculate_perceptual and original_input is not None and original_input.shape[1] == 3 and adversarial_input is not None:
                 try:
                     # 假设 original_input 和 adversarial_input 是形状为 (B, C, T, H, W) 的图像数据
                     # 选择序列中的特定帧 (例如，第一个，由 cfg.logging.sequence_step_to_vis 指定) 进行2D感知比较
                     # sequence_step_to_vis 必须小于序列长度 T
                     seq_len_input = original_input.shape[2] # 输入图像序列长度
                     vis_step = getattr(cfg.logging, 'sequence_step_to_vis', 0) # 从配置中获取可视化步骤，默认 0
                     if vis_step >= seq_len_input or vis_step < 0:
                          # 警告：配置中的可视化步骤 {} 超出输入图像序列长度 {} 范围。将使用第一个步骤 (0)。
                          print(f"Warning: sequence_step_to_vis ({vis_step}) in config exceeds input image sequence length ({seq_len_input}). Using first step (0).")
                          vis_step = 0

                     # 提取指定序列步骤的帧 (B, C, H, W)
                     original_frame_eval = original_input[:, :, vis_step, :, :].squeeze(2) # squeeze(2) 移除序列维度
                     adversarial_frame_eval = adversarial_input[:, :, vis_step, :, :].squeeze(2)

                     # PSNR
                     # 假设数据归一化到 [0, 1]
                     psnr_vals_batch = calculate_psnr(original_frame_eval, adversarial_frame_eval, data_range=1.0) 
                     # 将当前批次的 PSNR 值添加到列表中
                     all_psnr.extend(psnr_vals_batch.cpu().numpy().tolist()) # 将 tensor 转移到 CPU 并转换为 list

                     # LPIPS 和 SSIM (如果 piq 可用)
                     # 检查 lpips_metric_eval 是否已成功初始化
                     if piq_available and 'lpips_metric_eval' in locals(): 
                          # LPIPS 期望范围 [-1, 1]，SSIM 期望 [0, 1]
                          # 将 [0,1] 映射到 [-1,1] 用于 LPIPS
                          lpips_val_batch = lpips_metric_eval(original_frame_eval * 2 - 1, adversarial_frame_eval * 2 - 1).mean().item()
                          all_lpips.append(lpips_val_batch) # 累加 LPIPS 值
                          # SSIM 计算，data_range=1.0
                          ssim_val_batch = ssim(original_frame_eval, adversarial_frame_eval, data_range=1.0, reduction='mean').item()
                          all_ssim.append(ssim_val_batch) # 累加 SSIM 值

                 except Exception as e:
                      # 警告：计算此批次的感知指标时出错：{}。跳过。
                      print(f"Warning: Error calculating perceptual metrics for this batch: {e}. Skipping.")

            # 累加总样本数
            total_samples += batch_size

    # --- 计算总体平均指标 ---

    # 只报告与当前阶段相关的攻击成功率
    # 如果 total_samples 为 0，则平均值为 NaN
    avg_attack_success_rate_feature = total_attack_success_rate_feature / total_samples if total_samples > 0 else float('nan')
    avg_attack_success_rate_decision = total_attack_success_rate_decision / total_samples if total_samples > 0 else float('nan')
    avg_attack_success_rate_attention = total_attack_success_rate_attention / total_samples if total_samples > 0 else float('nan')

    # 计算平均扰动范数
    avg_linf_norm = total_linf_norm / total_samples if total_samples > 0 else float('nan')
    avg_l2_norm = total_l2_norm / total_samples if total_samples > 0 else float('nan')

    # 计算平均判别器得分
    avg_D_real_score = avg_D_real_score / total_samples if total_samples > 0 else float('nan')
    avg_D_fake_score = avg_D_fake_score / total_samples if total_samples > 0 else float('nan')

    # 处理感知指标列表可能为空或包含 NaN 的情况
    # np.nanmean 会忽略 NaN 值进行平均
    avg_psnr = np.nanmean(all_psnr) if all_psnr else float('nan') # 如果列表为空则为 NaN
    avg_lpips = np.nanmean(all_lpips) if all_lpips else float('nan')
    avg_ssim = np.nanmean(all_ssim) if all_ssim else float('nan')

    # 将结果收集到字典中
    metrics = {
        f'Attack_Success_Rate_Decision_Stage{current_train_stage}': avg_attack_success_rate_decision if current_train_stage == 2 else float('nan'),
        f'Attack_Success_Rate_Attention_Stage{current_train_stage}': avg_attack_success_rate_attention if current_train_stage == 2 else float('nan'),
        f'Attack_Success_Rate_Feature_Stage{current_train_stage}': avg_attack_success_rate_feature if current_train_stage == 1 else float('nan'), # 阶段1报告特征攻击成功率
        'Linf_Norm_Avg': avg_linf_norm,
        'L2_Norm_Avg': avg_l2_norm,
        'Discriminator_Real_Score_Avg': avg_D_real_score,
        'Discriminator_Fake_Score_Avg': avg_D_fake_score,
        'PSNR_Avg': avg_psnr if calculate_perceptual else float('nan'), # 如果不计算感知指标，则为 NaN
        'LPIPS_Avg': avg_lpips if calculate_perceptual else float('nan'),
        'SSIM_Avg': avg_ssim if calculate_perceptual else float('nan'),
    }

    # 将模型设置回训练模式 (判别器通常也应该在训练模式)
    if generator is not None:
        generator.train() 
    if discriminator is not None:
        discriminator.train() 
    # ATN 模型通常在训练期间保持评估模式，所以这里不改回 train()
    # atn_model.eval() # 保持评估模式

    return metrics
