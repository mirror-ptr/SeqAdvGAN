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


# TODO: 定义攻击成功率的衡量标准
def calculate_attack_success_rate(original_map: torch.Tensor,
                                adversarial_map: torch.Tensor,
                                success_threshold: float,
                                success_criterion: str = 'mse_diff_threshold', # Add success criterion parameter
                                higher_is_better_orig: bool = True,
                                topk_k: int = 10): # Add Top-K K value parameter
    """
    计算攻击成功率。
    根据你对 ATN 决策图含义的理解来定义成功标准。

    Args:
        original_decision_map (torch.Tensor): 原始样本下的ATN最终输出 (B, W, H) or (B, H, W).
        adversarial_decision_map (torch.Tensor): 对抗样本下的ATN最终输出 (B, W, H) or (B, H, W).
        success_threshold (float): 判断攻击成功的阈值。其含义取决于 success_criterion。
        success_criterion (str): 攻击成功的衡量标准。可选：
                                 - 'mse_diff_threshold': 对抗样本和原始样本决策图的 MSE 差异大于阈值。
                                 - 'mean_change_threshold': 决策图平均值变化大于阈值 (有方向性，取决于 higher_is_better_orig)。
                                 - 'topk_value_drop': 原始决策图 Top-K 位置的值在对抗样本中平均下降超过阈值。
                                 - 'topk_position_change': 原始决策图 Top-K 位置在对抗样本中的 Top-K 位置是否不同。
                                 - 'classification_change': ATN 最终分类结果发生变化 (需要映射函数或 ATN 输出分类结果)。
                                 - 'decision_map_divergence': 决策图分布变化 (KL/JS 散度) 大于阈值。
        higher_is_better_orig (bool): 原始决策图中是否值越高越好。攻击的目标是使其变差。
                                      如果原始值越高越好，攻击成功意味着对抗值低于原始值一定程度。
                                      如果原始值越低越好，攻击成功意味着对抗值高于原始值一定程度。
        topk_k (int): 对于 Top-K 相关标准，指定 K 的值。

    Returns:
        float: 攻击成功率 (0 到 1 之间的值)。
    """
    if original_map is None or adversarial_map is None:
        print(f"Warning: Original or adversarial map is None for criterion '{success_criterion}'. Returning 0.0 success rate.")
        return 0.0

    # Ensure maps have at least batch dimension
    if original_map.ndim == 0 or adversarial_map.ndim == 0:
         print(f"Warning: Original or adversarial map is scalar for criterion '{success_criterion}'. Cannot calculate success rate on scalars.")
         return 0.0

    assert original_map.shape == adversarial_map.shape
    batch_size = original_map.shape[0]
    if batch_size == 0: return 0.0

    successful_attacks = 0

    if success_criterion == 'mse_diff_threshold':
        # MSE 差异：对抗样本与原始样本的 MSE 差异大于阈值即视为攻击成功
        # Need to handle different input dimensions (B, H, W) or (B, head, N, N)
        # Flatten all dimensions except batch for MSE calculation per sample
        original_flat = original_map.view(batch_size, -1)
        adversarial_flat = adversarial_map.view(batch_size, -1)

        mse_diff = torch.mean((adversarial_flat - original_flat)**2, dim=1) # Mean over flattened dimensions
        successful_attacks = (mse_diff > success_threshold).sum().item()

    elif success_criterion == 'mean_change_threshold':
        # 平均值变化：决策图的平均值变化大于指定阈值
        # This criterion is likely only applicable to decision maps or feature maps, not attention maps
        if original_map.ndim not in [2, 3]: # Expecting (H, W) or (B, H, W) if not batched, or (B, H, W)
            print(f"Warning: Mean change criterion expects 2D/3D input (H, W) or (B, H, W), got {original_map.ndim}D. Skipping.")
            return 0.0

        # Ensure input is (B, X), flatten if needed
        original_flat = original_map.view(batch_size, -1)
        adversarial_flat = adversarial_map.view(batch_size, -1)

        original_mean = original_flat.mean(dim=1)
        adversarial_mean = adversarial_flat.mean(dim=1)

        if higher_is_better_orig: # 原始值越高越好，希望对抗后降低 (original - adversarial > threshold)
            change = original_mean - adversarial_mean
        else: # 原始值越低越好，希望对抗后升高 (adversarial - original > threshold)
            change = adversarial_mean - original_mean

        successful_attacks = (change > success_threshold).sum().item()

    elif success_criterion == 'topk_value_drop':
        # 针对注意力图：原始 top-K 位置的注意力值在对抗样本中显著下降
        # Requires at least 2 spatial/sequence dimensions (N, N) or batched (B, head, N, N)
        if original_map.ndim < 2:
            print(f"Warning: Top-K value drop criterion requires at least 2 dimensions (e.g., flattened attention or spatial), got {original_map.ndim}D. Skipping.")
            return 0.0

        # Flatten to (B, num_features)
        if original_map.ndim == 4: # (B, head, N, N) -> (B, head * N * N)
            original_flat = original_map.view(batch_size, -1)
            adversarial_flat = adversarial_map.view(batch_size, -1)
        elif original_map.ndim == 3: # (B, H, W) -> (B, H*W)
             original_flat = original_map.view(batch_size, -1)
             adversarial_flat = adversarial_map.view(batch_size, -1)
        elif original_map.ndim == 2: # Assuming (B, N*N) or (B, H*W)
             original_flat = original_map
             adversarial_flat = adversarial_map
        else:
             print(f"Warning: Top-K value drop criterion unsupported for {original_map.ndim}D input. Skipping.")
             return 0.0


        # Ensure topk_k 不超过扁平化后的维度大小
        effective_k = min(topk_k, original_flat.shape[-1])
        if effective_k <= 0: return 0.0 # Avoid error if k is invalid after min

        # 在原始 map 的扁平化维度上找到 Top-K 值和索引
        # original_topk_values shape (B, effective_k)
        # original_topk_indices shape (B, effective_k)
        original_topk_values, original_topk_indices = torch.topk(
            original_flat,
            k=effective_k,
            dim=-1,       # 在最后一个维度上找 Top-K (展平后的)
            largest=True, # 找最大的 K 个值
            sorted=False
        )

        # 获取对抗样本在原始 Top-K 索引位置上的值
        # 使用 gather 函数
        # adversarial_values_at_original_topk shape (B, effective_k)
        adversarial_values_at_original_topk = torch.gather(
            adversarial_flat,
            dim=-1,       # 在最后一个维度上根据索引收集
            index=original_topk_indices # 使用原始 Top-K 的索引
        )

        # 计算下降比例 (原始值 - 对抗值) / 原始值
        # 避免除以零，如果原始值为 0，则下降比例也为 0
        value_drop = torch.where(
            original_topk_values > 1e-6, # Avoid division by near-zero
            (original_topk_values - adversarial_values_at_original_topk) / (original_topk_values),
            torch.zeros_like(original_topk_values) # If original value is near zero, drop is zero
        )

        # 计算每个样本的平均下降比例
        mean_value_drop_per_sample = value_drop.mean(dim=-1) # Shape (B,)

        # 攻击成功 if 平均下降比例 > 阈值
        successful_attacks = (mean_value_drop_per_sample > success_threshold).sum().item()

    elif success_criterion == 'topk_position_change':
        # 针对注意力图：原始 top-K 注意力位置与对抗样本 top-K 注意力位置的交集小于阈值
        # Requires at least 2 spatial/sequence dimensions (N, N) or batched (B, head, N, N)
        if original_map.ndim < 2:
            print(f"Warning: Top-K position change criterion requires at least 2 dimensions, got {original_map.ndim}D. Skipping.")
            return 0.0

        # Flatten to (B, num_features)
        if original_map.ndim == 4: # (B, head, N, N) -> (B, head * N * N)
            original_flat = original_map.view(batch_size, -1)
            adversarial_flat = adversarial_map.view(batch_size, -1)
        elif original_map.ndim == 3: # (B, H, W) -> (B, H*W)
             original_flat = original_map.view(batch_size, -1)
             adversarial_flat = adversarial_map.view(batch_size, -1)
        elif original_map.ndim == 2: # Assuming (B, N*N) or (B, H*W)
             original_flat = original_map
             adversarial_flat = adversarial_map
        else:
             print(f"Warning: Top-K position change criterion unsupported for {original_map.ndim}D input. Skipping.")
             return 0.0

        effective_k = min(topk_k, original_flat.shape[-1])
        if effective_k <= 0: return 0.0

        # Find Top-K indices in the original and adversarial maps
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

        # Convert indices to sets for each sample and calculate intersection ratio
        for i in range(batch_size):
            orig_set = set(original_topk_indices[i].cpu().numpy())
            adv_set = set(adversarial_topk_indices[i].cpu().numpy())

            intersection_size = len(orig_set.intersection(adv_set))
            # Calculate ratio of intersection size to effective k
            intersection_ratio = intersection_size / effective_k if effective_k > 0 else 0.0

            # Attack successful if intersection ratio is below the threshold (i.e., positions changed significantly)
            if intersection_ratio < success_threshold:
                successful_attacks += 1

    # elif success_criterion == 'classification_change':
    #      # TODO: Implement classification change criterion if applicable
    #      print("Warning: 'classification_change' criterion is not implemented.")
    #      return 0.0

    else:
        print(f"Warning: Unsupported success criterion: {success_criterion}. Returning 0.0 success rate.")
        return 0.0

    return successful_attacks / batch_size

# Calculate perturbation norms (usually called directly in training script or logged via vis_utils)
# from losses.regularization_losses import linf_norm, l2_norm

# TODO: Convert feature layers to a format suitable for LPIPS or SSIM perceptual evaluation
# This is likely the most challenging part, requiring help from your friend or relevant literature.
def features_to_perceptual_input(features: torch.Tensor, sequence_step=0):
    """
    Converts a (B, C, N, W, H) or (B, C, N, H, W) feature tensor to (B, 3, H', W')
    or (B, 1, H', W') image format suitable for LPIPS or SSIM input.
    This requires deep understanding of how ATN features map back to image space,
    or finding a generic feature visualization method.

    Args:
        features (torch.Tensor): Input feature tensor (B, C, N, W, H) or (B, C, N, H, W).
        sequence_step (int): Which sequence step to convert.

    Returns:
        torch.Tensor: Image tensor suitable for perceptual evaluation, e.g., (B, 3, H', W') or (B, 1, H', W').
                      Note: This is a placeholder implementation; the validity of calculating perceptual metrics directly on ATN features is unknown.
    """
    if features is None or features.numel() == 0:
        print("Warning: Empty sequence dimension in features. Cannot convert for perceptual metrics.")
        # Return an empty tensor for subsequent calculations to handle
        # Return empty tensor matching expected output shape, e.g., (0, 3, H, W)
        # Need to infer H, W from input if possible, or use a default/config value
        # Let's assume the last two dimensions are spatial (H, W)
        # Need at least 3 dimensions (B, C, N, H, W) or (B, C, N, W, H)
        if features is not None and features.ndim >= 4:
             # Assuming last two dims are spatial H, W (or W, H)
             spatial_dims = features.shape[-2:]
             return torch.empty(0, 3, spatial_dims[0], spatial_dims[1], device=features.device, dtype=features.dtype)
        else:
             # If features is None or not enough dimensions, return empty with arbitrary shape
             return torch.empty(0, 3, 64, 64, device='cpu', dtype=torch.float32) # Default shape

    # print("Note: Using a placeholder implementation for features_to_perceptual_input.")
    # Assuming features shape is (B, C, N, H, W) based on Generator/Discriminator definition
    # Need to handle potential (B, C, N, W, H)
    if features.ndim != 5:
         print(f"Warning: Features tensor is not 5D ({features.ndim}D) for perceptual input conversion.")
         # Attempt to handle if it's (B, C, H, W) by adding sequence dim
         if features.ndim == 4: # Assuming (B, C, H, W)
              print("Assuming 4D input is (B, C, H, W), adding sequence dim for compatibility check.")
              features = features.unsqueeze(2) # (B, C, 1, H, W)
              B, C, N, H, W = features.shape # Re-read shape
         else:
              return torch.empty(0, 3, 64, 64, device=features.device, dtype=features.dtype)

    B, C, N, H, W = features.shape
    if N == 0:
        print("Warning: Empty sequence dimension in features. Cannot convert for perceptual metrics.")
        return torch.empty(0, 3, H, W, device=features.device, dtype=features.dtype)


    if sequence_step >= N or sequence_step < 0:
        print(f"Warning: sequence_step {sequence_step} is out of bounds for N={N}. Defaulting to 0.")
        sequence_step = 0 # Default to first step

    # Ensure features are in (B, C, H, W) format for processing
    # If input was (B, C, N, W, H), permute it here
    # We need to know the expected spatial order. Assuming (H, W).
    # If your features are (B, C, N, W, H), uncomment/adjust the following:
    # features = features.permute(0, 1, 2, 4, 3) # (B, C, N, W, H) -> (B, C, N, H, W)

    features_step = features[:, :, sequence_step, :, :] # (B, C, H, W)

    # Simple conversion: if channels >= 3, take first 3; otherwise, take first and repeat to 3 channels
    # (to satisfy LPIPS/SSIM requirement for 3 channels)
    if C >= 3:
        perceptual_img = features_step[:, :3, :, :] # (B, 3, H, W)
    elif C == 1:
        perceptual_img = features_step.repeat(1, 3, 1, 1) # (B, 1, H, W) -> (B, 3, H, W)
    else: # e.g., C=2, take first channel and repeat to 3 channels
        # Ensure channel dimension exists even if C=0 before unsqueeze/repeat
        if C > 0:
            perceptual_img = features_step[:, 0, :, :].unsqueeze(1).repeat(1, 3, 1, 1) # (B, 3, H, W)
        else:
             print("Warning: Feature channels C=0. Cannot create perceptual image.")
             return torch.empty(0, 3, H, W, device=features.device, dtype=features.dtype)


    # Normalize to the range typically expected by perceptual metrics, e.g., [0, 1]
    # This is a simple min-max normalization, may need adjustment based on actual ATN feature distribution.
    min_val = perceptual_img.min() if perceptual_img.numel() > 0 else 0.0
    max_val = perceptual_img.max() if perceptual_img.numel() > 0 else 1.0

    if max_val > min_val: # Avoid division by zero
        perceptual_img = (perceptual_img - min_val) / (max_val - min_val + 1e-8)
    else:
        # Handle case where all values are the same or tensor is empty
        perceptual_img = torch.zeros_like(perceptual_img) # Or fill with 0.5 for mid-gray if appropriate

    # LPIPS/SSIM usually expect float type input
    return perceptual_img.float() # Ensure float type


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, data_range=1.0):
    """
    计算两张图片的 PSNR (Peak Signal-to-Noise Ratio)。
    Args:
        img1 (torch.Tensor): 第一张图片 (B, C, H, W)。
        img2 (torch.Tensor): 第二张图片 (B, C, H, W)。
        data_range (float): 像素值的范围 (例如 1.0 表示 [0, 1] 或 255.0 表示 [0, 255])。
    Returns:
        torch.Tensor: 每张图片的 PSNR 值 (形状 B)。
    """
    if img1 is None or img2 is None or img1.numel() == 0 or img2.numel() == 0:
        print("Warning: Input images for PSNR are None or empty.")
        # Return NaN tensor or handle appropriately
        # Determine a batch size to return a tensor of correct size with NaNs
        batch_size = max(img1.shape[0] if img1 is not None else 0, img2.shape[0] if img2 is not None else 0)
        if batch_size == 0: return torch.tensor([]) # Return empty if no batch dim info
        return torch.full((batch_size,), float('nan'), device=img1.device if img1 is not None else (img2.device if img2 is not None else 'cpu'))


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

def calculate_perceptual_metrics(original_features: torch.Tensor,
                                 adversarial_features: torch.Tensor,
                                 device: torch.device,
                                 sequence_step_to_vis: int = 0): # Add sequence step parameter
    """
    计算 LPIPS, SSIM, PSNR 等感知评估指标。
    Args:
        original_features (torch.Tensor): 原始特征层 (B, C, N, W, H) or (B, C, N, H, W).
        adversarial_features (torch.Tensor): 对抗性特征层 (B, C, N, W, H) or (B, C, N, H, W).
        device (torch.device): 计算设备。
        sequence_step_to_vis (int): 选择序列中的哪个步骤进行可视化和感知评估。
    Returns:
        dict: 包含感知指标的字典。
              注意：这些指标计算在将 ATN 特征转换为图像格式后，其有效性未知。
    """
    metrics = {}

    # Convert features to image format suitable for perceptual evaluation
    # We need to specify which sequence step to use for the 2D image comparison
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
                 lpips_metric = LPIPS(net_type='alex', version='0.1').to(device) # Re-initialize or move
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
def evaluate_model(generator, discriminator, atn_model, dataloader, device, cfg, current_train_stage):
    """
    Evaluate the performance of the generator and discriminator based on the current training stage.

    Args:
        generator (nn.Module or None): Generator model (can be None if evaluating original data).
        discriminator (nn.Module or None): Discriminator model (can be None if GAN metrics are not needed).
        atn_model (nn.Module): ATN model.
        dataloader (DataLoader): Evaluation data loader.
        device (torch.device): Computation device.
        cfg: Configuration object containing evaluation parameters.
        current_train_stage (int): The current training stage (1 or 2).

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    if generator is not None:
        generator.eval()
    if discriminator is not None:
        discriminator.eval()
    atn_model.eval()

    # Initialize metrics storage
    total_attack_success_rate_decision = 0.0
    total_attack_success_rate_attention = 0.0
    total_attack_success_rate_feature = 0.0 # New: Feature attack success rate for stage 1
    total_linf_norm = 0.0
    total_l2_norm = 0.0
    avg_D_real_score = 0.0
    avg_D_fake_score = 0.0
    total_samples = 0
    all_psnr = []
    all_lpips = []
    all_ssim = []

    # Determine if perceptual metrics should be calculated (usually on image input)
    # Assume perceptual metrics are relevant if the input data has 3 channels (like an image)
    calculate_perceptual = cfg.data.channels == 3 # Assuming cfg.data.channels indicates input type

    # Initialize perceptual metrics calculator if needed
    if calculate_perceptual:
        try:
            from piq import LPIPS, ssim # type: ignore # Lazy import
            lpips_metric_eval = LPIPS(net_type='alex', version='0.1').to(device) # Use a separate instance if needed
            # SSIM does not need explicit initialization like LPIPS
            piq_available = True
        except ImportError:
            print("Warning: piq library not found. Skipping LPIPS and SSIM metrics. Install with 'pip install piq'.")
            calculate_perceptual = False # Disable perceptual metrics if piq is missing
        except Exception as e:
            print(f"Error initializing perceptual metrics (evaluation): {e}. Skipping LPIPS and SSIM.")
            calculate_perceptual = False


    with torch.no_grad(): # Ensure no gradients are computed during evaluation
        num_eval_batches = cfg.evaluation.get('num_eval_batches', len(dataloader)) # Use config if available
        if num_eval_batches > len(dataloader): num_eval_batches = len(dataloader)
        if num_eval_batches == 0: # Handle empty dataloader
             print("Warning: Dataloader is empty during evaluation.")
             return {
                'Attack_Success_Rate_Decision': float('nan'),
                'Attack_Success_Rate_Attention': float('nan'),
                'Attack_Success_Rate_Feature': float('nan'),
                'Linf_Norm_Avg': float('nan'),
                'L2_Norm_Avg': float('nan'),
                'Discriminator_Real_Score_Avg': float('nan'),
                'Discriminator_Fake_Score_Avg': float('nan'),
                'PSNR_Avg': float('nan'),
                'LPIPS_Avg': float('nan'),
                'SSIM_Avg': float('nan'),
            }

        for batch_idx, original_input in enumerate(dataloader):
            if batch_idx >= num_eval_batches:
                break

            original_input = original_input.to(device)
            batch_size = original_input.shape[0]
            if batch_size == 0: continue

            # Get ATN outputs based on the current stage
            if current_train_stage == 1:
                 # Stage 1: Need features for evaluation
                 original_atn_outputs = get_atn_outputs(
                     atn_model,
                     original_input,
                     return_features=True, # Request features
                     return_decision=False, # Don't need decision/attention in stage 1 eval
                     return_attention=False
                 )
                 # In stage 1, adversarial is original + delta, we need adversarial features
                 # Pass original_input through the generator first to get delta
                 delta = generator(original_input) if generator is not None else torch.zeros_like(original_input)
                 # Assuming the generator output is a perturbation delta in the same space as original_input
                 # And the ATN model takes original_input space data
                 # We need to pass adversarial_input (original + delta) to ATN to get adversarial features
                 adversarial_input = torch.clamp(original_input + delta, 0, 1) # Clamp to valid range if applicable
                 adversarial_atn_outputs = get_atn_outputs(
                      atn_model,
                      adversarial_input,
                      return_features=True,
                      return_decision=False,
                      return_attention=False
                 )
                 original_features = original_atn_outputs.get('features')
                 adversarial_features = adversarial_atn_outputs.get('features')
                 # Decision and Attention maps are None in stage 1 eval
                 original_decision_map = None
                 adversarial_decision_map = None
                 original_attention_map = None
                 adversarial_attention_map = None

            elif current_train_stage == 2:
                 # Stage 2: Need decision and attention maps for evaluation
                 # Still need features if discriminator/GAN eval uses them
                 # And need original_input and adversarial_input for perturbation norms
                 delta = generator(original_input) if generator is not None else torch.zeros_like(original_input)
                 adversarial_input = torch.clamp(original_input + delta, 0, 1)

                 original_atn_outputs = get_atn_outputs(
                      atn_model,
                      original_input,
                      return_features=True, # Needed for GAN eval if applicable
                      return_decision=True,  # Need decision map in stage 2
                      return_attention=True # Need attention map in stage 2
                 )
                 adversarial_atn_outputs = get_atn_outputs(
                      atn_model,
                      adversarial_input,
                      return_features=True,
                      return_decision=True,
                      return_attention=True
                 )
                 original_features = original_atn_outputs.get('features')
                 adversarial_features = adversarial_atn_outputs.get('features')
                 original_decision_map = original_atn_outputs.get('decision')
                 adversarial_decision_map = adversarial_atn_outputs.get('decision')
                 original_attention_map = original_atn_outputs.get('attention')
                 adversarial_attention_map = adversarial_atn_outputs.get('attention')

            else:
                 raise ValueError(f"Unsupported training stage for evaluation: {current_train_stage}")

            # Calculate metrics based on available outputs

            # Attack Success Rate (Feature - only in stage 1 eval relevantly)
            if current_train_stage == 1 and original_features is not None and adversarial_features is not None:
                 # Assuming feature attack success means a large MSE difference between features
                 # success_threshold and success_criterion from cfg.evaluation will be used
                 success_rate_feature = calculate_attack_success_rate(
                      original_features.view(batch_size, -1), # Flatten features for criterion compatibility
                      adversarial_features.view(batch_size, -1), # Flatten features
                      cfg.evaluation.success_threshold,
                      success_criterion='mse_diff_threshold', # Feature attack success based on feature difference
                      higher_is_better_orig=False # For feature difference, higher is better for attack
                 )
                 total_attack_success_rate_feature += success_rate_feature * batch_size

            # Attack Success Rate (Decision Map - only in stage 2 eval relevantly)
            if current_train_stage == 2 and original_decision_map is not None and adversarial_decision_map is not None:
                # Assuming ATN decision map is higher is better, goal is to lower it
                # Use success criterion from cfg.evaluation for decision map
                success_rate_decision = calculate_attack_success_rate(
                    original_decision_map,
                    adversarial_decision_map,
                    cfg.evaluation.success_threshold,
                    cfg.evaluation.success_criterion, # Use config criterion for decision map eval
                    higher_is_better_orig=True # Assuming higher decision value is better for original ATN
                )
                total_attack_success_rate_decision += success_rate_decision * batch_size

            # Attack Success Rate (Attention Map - only in stage 2 eval relevantly)
            if current_train_stage == 2 and original_attention_map is not None and adversarial_attention_map is not None:
                 # Use specific attention success criterion/threshold if available in config
                 # Otherwise, use the default evaluation criterion/threshold
                 attention_success_criterion = cfg.evaluation.get('attention_success_criterion', cfg.evaluation.success_criterion)
                 attention_success_threshold = cfg.evaluation.get('attention_success_threshold', cfg.evaluation.success_threshold)
                 eval_topk_k = cfg.losses.get('topk_k', 10) # Get topk_k from losses or default

                 success_rate_attention = calculate_attack_success_rate(
                     original_attention_map,
                     adversarial_attention_map,
                     attention_success_threshold,
                     attention_success_criterion, # Use specific criterion for attention
                     topk_k=eval_topk_k # Use topk_k for relevant criteria
                 )
                 total_attack_success_rate_attention += success_rate_attention * batch_size

            # Perturbation Norms (relevant in both stages)
            if generator is not None and delta is not None and delta.numel() > 0:
                # linf_norm and l2_norm expect (B, C, N, H, W) or similar. delta should be in this format.
                if delta.ndim == 5:
                     total_linf_norm += linf_norm(delta).mean().item() * batch_size # mean over batch, sum over batches
                     total_l2_norm += l2_norm(delta).mean().item() * batch_size
                else:
                     print(f"Warning: Delta tensor for norm calculation is not 5D ({delta.ndim}D).")

            # GAN Evaluation Metrics (relevant in both stages if discriminator is used)
            if discriminator is not None and original_features is not None and adversarial_features is not None:
                 # Ensure features are suitable for discriminator (e.g., correct shape and values)
                 # Assuming discriminator input is features from ATN feature head (B, C_feat, N, W_feat, H_feat)
                 # Need to ensure original_features and adversarial_features from get_atn_outputs are in this format
                 try:
                     D_real_output = discriminator(original_features.detach())
                     D_fake_output = discriminator(adversarial_features.detach())

                     # Discriminator output shape is typically (B, 1) or (B, 1, patch_N, patch_H, patch_W)
                     # Flatten all dimensions except batch and take mean per sample, then sum over batch
                     avg_D_real_score += D_real_output.view(batch_size, -1).mean(dim=-1).sum().item()
                     avg_D_fake_score += D_fake_output.view(batch_size, -1).mean(dim=-1).sum().item()
                 except Exception as e:
                     print(f"Warning: Error calculating GAN evaluation metrics: {e}. Skipping for this batch.")

            # Perceptual Metrics (relevant if input is image and calculate_perceptual is True)
            # These should ideally be calculated on the original input space, not feature space
            # If the generator takes image input and outputs image perturbation, use original_input and adversarial_input
            # If the generator takes feature input and outputs feature perturbation, these metrics on features may not be meaningful
            # Assuming generator takes image-like input (3 channels) when calculate_perceptual is True
            if calculate_perceptual and original_input.shape[1] == 3:
                 try:
                     # Assuming original_input and adversarial_input are (B, C, T, H, W) image data
                     # Select a specific frame (e.g., the first one) for 2D perceptual comparison
                     original_frame_eval = original_input[:, :, cfg.logging.sequence_step_to_vis, :, :].squeeze(2) # (B, C, H, W)
                     adversarial_frame_eval = adversarial_input[:, :, cfg.logging.sequence_step_to_vis, :, :].squeeze(2) # (B, C, H, W)

                     # PSNR
                     psnr_vals_batch = calculate_psnr(original_frame_eval, adversarial_frame_eval, data_range=1.0) # data_range=1.0 if normalized [0,1]
                     all_psnr.extend(psnr_vals_batch.cpu().numpy().tolist())

                     # LPIPS and SSIM (if piq available)
                     if 'lpips_metric_eval' in locals(): # Check if piq and metric were initialized
                          # LPIPS expects [-1, 1], SSIM expects [0, 1]
                          lpips_val_batch = lpips_metric_eval(original_frame_eval * 2 - 1, adversarial_frame_eval * 2 - 1).mean().item()
                          all_lpips.append(lpips_val_batch)
                          ssim_val_batch = ssim(original_frame_eval, adversarial_frame_eval, data_range=1.0, reduction='mean').item()
                          all_ssim.append(ssim_val_batch)

                 except Exception as e:
                      print(f"Warning: Error calculating perceptual metrics for this batch: {e}. Skipping.")

            total_samples += batch_size

    # Calculate overall averages
    # Only report relevant attack success rates based on the stage
    avg_attack_success_rate_feature = total_attack_success_rate_feature / total_samples if total_samples > 0 else float('nan')
    avg_attack_success_rate_decision = total_attack_success_rate_decision / total_samples if total_samples > 0 else float('nan')
    avg_attack_success_rate_attention = total_attack_success_rate_attention / total_samples if total_samples > 0 else float('nan')

    avg_linf_norm = total_linf_norm / total_samples if total_samples > 0 else float('nan')
    avg_l2_norm = total_l2_norm / total_samples if total_samples > 0 else float('nan')

    avg_D_real_score = avg_D_real_score / total_samples if total_samples > 0 else float('nan')
    avg_D_fake_score = avg_D_fake_score / total_samples if total_samples > 0 else float('nan')

    # Handle cases where perceptual metrics lists might be empty or full of NaNs
    avg_psnr = np.nanmean(all_psnr) if all_psnr else float('nan') # nanmean ignores NaNs
    avg_lpips = np.nanmean(all_lpips) if all_lpips else float('nan')
    avg_ssim = np.nanmean(all_ssim) if all_ssim else float('nan')

    # Collect results in a dictionary
    metrics = {
        f'Attack_Success_Rate_Decision_Stage{current_train_stage}': avg_attack_success_rate_decision if current_train_stage == 2 else float('nan'),
        f'Attack_Success_Rate_Attention_Stage{current_train_stage}': avg_attack_success_rate_attention if current_train_stage == 2 else float('nan'),
        f'Attack_Success_Rate_Feature_Stage{current_train_stage}': avg_attack_success_rate_feature if current_train_stage == 1 else float('nan'), # Report feature success rate for stage 1
        'Linf_Norm_Avg': avg_linf_norm,
        'L2_Norm_Avg': avg_l2_norm,
        'Discriminator_Real_Score_Avg': avg_D_real_score,
        'Discriminator_Fake_Score_Avg': avg_D_fake_score,
        'PSNR_Avg': avg_psnr if calculate_perceptual else float('nan'),
        'LPIPS_Avg': avg_lpips if calculate_perceptual else float('nan'),
        'SSIM_Avg': avg_ssim if calculate_perceptual else float('nan'),
    }

    generator.train() # Set models back to training mode
    discriminator.train() # Discriminator should also be in train mode usually
    # atn_model.eval() # ATN model stays in eval mode during training

    return metrics
