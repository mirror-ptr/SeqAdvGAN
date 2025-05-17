import sys
import os
# 将项目根目录添加到 Python 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
# 可能需要安装 piq 库来计算 LPIPS 或 SSIM: pip install piq
# from piq import LPIPS, ssim

# 扰动范数
from losses.regularization_losses import linf_norm, l2_norm

# 导入 ATN 工具
from utils.atn_utils import get_atn_outputs

# TODO: 定义攻击成功率的衡量标准
def calculate_attack_success_rate(original_decision_map: torch.Tensor,
                                adversarial_decision_map: torch.Tensor,
                                success_threshold: float,
                                success_criterion: str = 'mse_diff_threshold', # 添加成功标准参数
                                higher_is_better_orig: bool = True,
                                topk_k: int = 10): # 添加 Top-K 的 K 值参数
    """
    计算攻击成功率。
    根据你对 ATN 决策图含义的理解来定义成功标准。

    Args:
        original_decision_map (torch.Tensor): 原始样本下的ATN最终输出 (B, W, H)。
        adversarial_decision_map (torch.Tensor): 对抗样本下的ATN最终输出 (B, W, H)。
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
    if original_decision_map is None or adversarial_decision_map is None:
        print("Warning: Decision maps are None in calculate_attack_success_rate. Returning 0.0 success rate.")
        return 0.0

    B, W, H = original_decision_map.shape
    if B == 0: return 0.0
    successful_attacks = 0

    # 展平 W x H 维度，方便 Top-K 操作
    original_decision_flat = original_decision_map.view(B, -1) # Shape (B, W*H)
    adversarial_decision_flat = adversarial_decision_map.view(B, -1) # Shape (B, W*H)
    spatial_size = W * H

    if success_criterion == 'mse_diff_threshold':
        # 示例1: 基于 MSE 差异
        # 攻击成功：MSE 差异大于阈值
        decision_mse = torch.mean((adversarial_decision_map - original_decision_map)**2, dim=(-1,-2))
        successful_attacks = torch.sum(decision_mse > success_threshold).item()

    elif success_criterion == 'mean_change_threshold':
        # 示例2: 基于平均值的变化
        original_mean = original_decision_flat.mean(dim=1)
        adversarial_mean = adversarial_decision_flat.mean(dim=1)
        if higher_is_better_orig:
            # 攻击成功：原始均值 - 对抗均值 > 阈值
            successful_attacks = torch.sum(original_mean - adversarial_mean > success_threshold).item()
        else:
            # 攻击成功：对抗均值 - 原始均值 > 阈值
            successful_attacks = torch.sum(adversarial_mean - original_mean > success_threshold).item()

    elif success_criterion == 'topk_value_drop':
        # 攻击成功：原始决策图 Top-K 位置的值在对抗样本中平均下降超过阈值
        if topk_k <= 0 or topk_k > spatial_size:
            print(f"Warning: Invalid topk_k value {topk_k} for topk_value_drop criterion. Must be between 1 and {spatial_size}. Returning 0.0 success rate.")
            return 0.0

        # 找到原始决策图的 Top-K 值和索引
        topk_values_orig, topk_indices_orig = torch.topk(
            original_decision_flat,
            k=topk_k,
            dim=-1,
            largest=True # 假设原始决策图高值更重要
        )

        # 获取对抗样本在原始 Top-K 索引位置上的值
        adversarial_values_at_topk_indices = torch.gather(
            adversarial_decision_flat,
            dim=-1,
            index=topk_indices_orig
        )

        # 计算这些位置的值下降量 (原始值 - 对抗值)
        value_drop = topk_values_orig - adversarial_values_at_topk_indices

        # 计算每个样本 Top-K 位置的平均值下降
        mean_value_drop_per_sample = value_drop.mean(dim=-1) # Shape (B,)

        # 攻击成功：平均值下降超过阈值
        successful_attacks = torch.sum(mean_value_drop_per_sample > success_threshold).item()

    elif success_criterion == 'topk_position_change':
        # 攻击成功：原始决策图 Top-K 位置在对抗样本中的 Top-K 位置至少有一个不同
        if topk_k <= 0 or topk_k > spatial_size:
             print(f"Warning: Invalid topk_k value {topk_k} for topk_position_change criterion. Must be between 1 and {spatial_size}. Returning 0.0 success rate.")
             return 0.0

        # 找到原始决策图的 Top-K 索引
        _, topk_indices_orig = torch.topk(
             original_decision_flat,
             k=topk_k,
             dim=-1,
             largest=True
        )

        # 找到对抗样本的 Top-K 索引
        _, topk_indices_adv = torch.topk(
             adversarial_decision_flat,
             k=topk_k,
             dim=-1,
             largest=True
        )

        # 检查原始 Top-K 索引集合与对抗 Top-K 索引集合是否有差异
        # 将索引转换为集合进行比较 (需要注意批次维度)
        # 攻击成功：如果对于任一样本，其原始 Top-K 索引集合与对抗 Top-K 索引集合不同
        successful_attacks = 0
        for i in range(B):
             set_orig = set(topk_indices_orig[i].tolist())
             set_adv = set(topk_indices_adv[i].tolist())
             if set_orig != set_adv:
                  successful_attacks += 1


    elif success_criterion == 'region_mean_change_threshold':
         # TODO: 实现基于特定区域平均值变化的成功标准
         # 需要额外的 region_mask 参数传入此函数
         print("Warning: 'region_mean_change_threshold' criterion requires region mask and is not fully implemented.")
         return 0.0 # Placeholder

    elif success_criterion == 'classification_change':
         # TODO: 实现基于最终分类结果变化的成功标准
         # 这需要 ATN 输出分类结果，或者有一个从决策图到分类结果的映射函数。
         print("Warning: 'classification_change' criterion requires ATN classification output and is not implemented.")
         return 0.0 # Placeholder

    elif success_criterion == 'decision_map_divergence':
         # TODO: 实现基于决策图分布变化的成功标准 (KL/JS 散度)
         # 需要将决策图转换为概率分布 (例如，Softmax)，然后计算散度。
         print("Warning: 'decision_map_divergence' criterion requires distribution conversion and is not implemented.")
         return 0.0 # Placeholder

    else:
        raise ValueError(f"Unsupported success criterion: {success_criterion}")


    return successful_attacks / B

# 计算扰动范数 (通常在训练脚本中直接调用或通过 vis_utils 记录)
# from losses.regularization_losses import linf_norm, l2_norm

# TODO: 将特征层转换为可进行 LPIPS 或 SSIM 感知评估的形式
# 这可能是最挑战的部分，需要朋友的帮助或查阅相关文献
def features_to_perceptual_input(features: torch.Tensor, sequence_step=0):
    """
    将 (B, C, N, W, H) 特征张量转换为 (B, 3, H, W) 或 (B, 1, H, W)
    等适合 LPIPS 或 SSIM 输入的图像格式。
    这需要深入理解 ATN 的特征如何映射回图像空间，或者找到一种通用的特征可视化方法。

    Args:
        features (torch.Tensor): 输入特征张量 (B, C, N, W, H)。
        sequence_step (int): 选择哪个序列步骤进行转换。

    Returns:
        torch.Tensor: 适合感知评估的图像张量，例如 (B, 3, H, W) 或 (B, 1, H, W)。
                      注意：这是一个占位符实现，直接使用 ATN 特征计算感知指标的有效性未知。
    """
    print("Note: Using a placeholder implementation for features_to_perceptual_input.")
    B, C, N, W, H = features.shape
    if N == 0:
        print("Warning: Empty sequence dimension in features. Cannot convert for perceptual metrics.")
        # 返回一个空张量，以便后续计算能处理
        # 根据期望的输出形状返回空张量，例如 (0, 3, H, W)
        return torch.empty(0, 3, H, W, device=features.device, dtype=features.dtype)

    if sequence_step >= N:
        print(f"Warning: sequence_step {sequence_step} is out of bounds for N={N}. Defaulting to 0.")
        sequence_step = 0 # 默认可视化第一个步骤

    features_step = features[:, :, sequence_step, :, :] # (B, C, W, H)

    # 简单转换：如果通道数大于等于3，取前3个；否则取第一个并复制到3通道 (以满足LPIPS/SSIM对3通道的要求)
    if C >= 3:
        perceptual_img = features_step[:, :3, :, :] # (B, 3, W, H)
    elif C == 1:
        perceptual_img = features_step.repeat(1, 3, 1, 1) # (B, 1, W, H) -> (B, 3, W, H)
    else: # e.g., C=2, take first channel and repeat to 3 channels
        perceptual_img = features_step[:, 0, :, :].unsqueeze(1).repeat(1, 3, 1, 1) # (B, 3, W, H)


    # 归一化到感知指标通常期望的范围，例如 [0, 1]
    # 这是一个简单的 min-max 归一化，可能需要根据 ATN 特征的实际分布进行调整
    min_val = perceptual_img.min() if perceptual_img.numel() > 0 else 0.0
    max_val = perceptual_img.max() if perceptual_img.numel() > 0 else 1.0

    if max_val > min_val:
        perceptual_img = (perceptual_img - min_val) / (max_val - min_val + 1e-8)
    else:
        # Handle case where all values are the same or tensor is empty
        perceptual_img = torch.zeros_like(perceptual_img) # Or fill with 0.5 for mid-gray if appropriate

    # LPIPS/SSIM 通常期望 float 类型的输入
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
    mse = torch.mean((img1 - img2) ** 2, dim=(-3, -2, -1)) # 对 C, H, W 维度求平均
    # 避免 log(0)
    mse = torch.clamp(mse, min=1e-10)
    psnr = 20 * torch.log10(data_range / torch.sqrt(mse))
    return psnr

def calculate_perceptual_metrics(original_features: torch.Tensor,
                                 adversarial_features: torch.Tensor,
                                 device: torch.device):
    """
    计算 LPIPS, SSIM, PSNR 等感知评估指标。
    Args:
        original_features (torch.Tensor): 原始特征层 (B, C, N, W, H)。
        adversarial_features (torch.Tensor): 对抗性特征层 (B, C, N, W, H)。
        device (torch.device): 计算设备。
    Returns:
        dict: 包含感知指标的字典。
              注意：这些指标计算在将 ATN 特征转换为图像格式后，其有效性未知。
    """
    metrics = {}

    # 将特征转换为适合感知评估的图像 (选择第一个序列步骤为例)
    # 将特征移动到 CPU 进行转换（如果转换函数不支持 GPU 操作），再移回设备计算指标
    # perceptual_input_orig 和 perceptual_input_adv 形状应为 (B, 3, H, W) 或 (B, 1, H, W)
    original_img_perceptual = features_to_perceptual_input(original_features, sequence_step=0).to(device)
    adversarial_img_perceptual = features_to_perceptual_input(adversarial_features, sequence_step=0).to(device)

    # 检查转换后的张量是否为空
    if original_img_perceptual.shape[0] == 0 or adversarial_img_perceptual.shape[0] == 0:
        print("Warning: Converted perceptual input tensors are empty. Skipping perceptual metric calculations.")
        metrics['psnr'] = float('nan')
        metrics['lpips'] = float('nan')
        metrics['ssim'] = float('nan')
        return metrics


    # PSNR Calculation
    try:
        # PSNR 可以在单通道或多通道图像上计算
        psnr_vals = calculate_psnr(original_img_perceptual, adversarial_img_perceptual, data_range=1.0) # Assuming data is normalized to [0, 1]
        metrics['psnr'] = psnr_vals.mean().item()
    except Exception as e:
        print(f"Error calculating PSNR: {e}")
        metrics['psnr'] = float('nan')


    # LPIPS and SSIM Calculation (requires piq)
    try:
        from piq import LPIPS, ssim # type: ignore
        piq_available = True
    except ImportError:
        print("Warning: 'piq' library not found. Skipping LPIPS/SSIM. Install with 'pip install piq'")
        piq_available = False
    except Exception as e:
        print(f"Warning: Error importing piq: {e}. Skipping LPIPS/SSIM.")
        piq_available = False

    if piq_available:
        print("Attempting to calculate perceptual metrics (LPIPS, SSIM) using piq...")
        try:
            # LPIPS: 通常输入范围是 [0,1] 或 [-1,1], 取决于预训练模型
            # AlexNet LPIPS 通常需要3通道RGB图像
            # features_to_perceptual_input 已经尝试确保输出是3通道
            # The LPIPS class in piq by default normalizes input from [0,1] to [-1,1].
            # So, providing [0,1] should be fine.
            if original_img_perceptual.shape[1] == 3:
                 lpips_metric = LPIPS(net_type='alex', version='0.1').to(device)
                 lpips_val = lpips_metric(original_img_perceptual, adversarial_img_perceptual).mean().item()
                 metrics['lpips'] = lpips_val
            else:
                 print(f"Warning: LPIPS expects 3 channels, got {original_img_perceptual.shape[1]}. Skipping LPIPS.")
                 metrics['lpips'] = float('nan')

            # SSIM: 通常输入范围是 [0,1] 或 [0, 255] (data_range 要匹配)
            # SSIM 可以在单通道或多通道图像上计算
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

# --- 示例如何在训练/评估脚本中使用 ---
def evaluate_model(generator, discriminator, atn_model, dataloader, device, args):
    """
    评估生成器和判别器的性能。

    Args:
        generator (nn.Module): 生成器模型。
        discriminator (nn.Module): 判别器模型。
        atn_model (nn.Module): ATN 模型。
        dataloader (DataLoader): 评估数据加载器。
        device (torch.device): 计算设备。
        args: 命令行参数 (应包含 eval_success_threshold, eval_success_criterion, topk_k 等)。

    Returns:
        dict: 包含评估指标的字典。
    """
    generator.eval()
    discriminator.eval()
    atn_model.eval()

    total_attack_success_rate = 0.0
    total_samples = 0
    all_linf_norms = []
    all_l2_norms = []
    all_psnr = [] # 添加 PSNR 列表
    all_lpips = []
    all_ssim = []
    # TODO: 添加 GAN 评估指标，例如判别器对真实/伪造样本的平均输出
    avg_D_real = 0.0
    avg_D_fake = 0.0

    # TODO: 如果数据加载器返回掩码，需要在这里接收
    # for i, (original_features_batch, decision_mask_batch, attention_mask_batch) in enumerate(dataloader):
    for i, original_features_batch in enumerate(dataloader):
        original_features = original_features_batch.to(device)
        batch_size = original_features.shape[0]

        if batch_size == 0: continue # Skip empty batches

        # TODO: 如果使用掩码，将掩码也移动到设备
        # decision_mask = decision_mask_batch.to(device) if decision_mask_batch is not None else None
        # attention_mask = attention_mask_batch.to(device) if attention_mask_batch is not None else None
        decision_mask = None # Placeholder for now
        attention_mask = None # Placeholder for now


        with torch.no_grad():
            # 生成对抗扰动
            delta = generator(original_features)
            adversarial_features = original_features + delta

            # 获取 ATN 输出
            original_atn_outputs = get_atn_outputs(atn_model, original_features)
            adversarial_atn_outputs = get_atn_outputs(atn_model, adversarial_features)

            # 获取判别器输出
            # 注意：这里的判别器输出是用于评估，不是训练，所以不需要 Sigmoid/Logits 的特定要求
            # 但为了保持一致性，使用训练时的 gan_loss_type 对应的输出方式可能更好。
            # 假设 evaluate_model 直接使用判别器原始输出。
            D_real_output = discriminator(original_features)
            D_fake_output = discriminator(adversarial_features)


            # 1. 攻击成功率
            if original_atn_outputs.get('decision') is not None and adversarial_atn_outputs.get('decision') is not None:
                 success_rate_batch = calculate_attack_success_rate(
                     original_atn_outputs.get('decision'),
                     adversarial_atn_outputs.get('decision'),
                     success_threshold=getattr(args, 'eval_success_threshold', 0.1), # 获取评估阈值，提供默认值
                     success_criterion=getattr(args, 'eval_success_criterion', 'mse_diff_threshold'), # 获取评估标准
                     topk_k=getattr(args, 'topk_k', 10) # 获取 Top-K 参数
                 )
                 total_attack_success_rate += success_rate_batch * batch_size
            else:
                print("Warning: Decision maps not found for success rate calculation in evaluation batch.")


            # 2. 扰动范数
            all_linf_norms.extend(linf_norm(delta).cpu().numpy())
            all_l2_norms.extend(l2_norm(delta).cpu().numpy())

            # 3. 感知指标 (PSNR, LPIPS, SSIM)
            perceptual_metrics_batch = calculate_perceptual_metrics(original_features, adversarial_features, device)
            if 'psnr' in perceptual_metrics_batch and not np.isnan(perceptual_metrics_batch['psnr']):
                 all_psnr.append(perceptual_metrics_batch['psnr']) # 记录 PSNR
            if 'lpips' in perceptual_metrics_batch and not np.isnan(perceptual_metrics_batch['lpips']):
                all_lpips.append(perceptual_metrics_batch['lpips'])
            if 'ssim' in perceptual_metrics_batch and not np.isnan(perceptual_metrics_batch['ssim']):
                 all_ssim.append(perceptual_metrics_batch['ssim'])

            # 4. GAN 评估指标
            # 对于 BCE Loss，输出是概率；对于 LSGAN，输出是 logits。
            # 直接取均值作为评估指标是合理的。
            # 展平判别器输出并计算均值
            avg_D_real += D_real_output.view(batch_size, -1).mean(dim=-1).sum().item() # 累加批次总和
            avg_D_fake += D_fake_output.view(batch_size, -1).mean(dim=-1).sum().item() # 累加批次总和

            total_samples += batch_size

    avg_attack_success_rate = total_attack_success_rate / total_samples if total_samples > 0 else 0.0
    avg_linf_norm = np.mean(all_linf_norms) if all_linf_norms else 0.0
    avg_l2_norm = np.mean(all_l2_norms) if all_l2_norms else 0.0
    avg_psnr = np.mean(all_psnr) if all_psnr else float('nan') # 计算 PSNR 平均值
    avg_lpips = np.mean(all_lpips) if all_lpips else float('nan')
    avg_ssim = np.mean(all_ssim) if all_ssim else float('nan')
    avg_D_real = avg_D_real / total_samples if total_samples > 0 else 0.0
    avg_D_fake = avg_D_fake / total_samples if total_samples > 0 else 0.0


    print(f"\nEvaluation Results ({total_samples} samples):")
    print(f"  Avg. Attack Success Rate: {avg_attack_success_rate:.4f}")
    print(f"  Avg. L-inf Norm: {avg_linf_norm:.4f}")
    print(f"  Avg. L2 Norm: {avg_l2_norm:.4f}")
    print(f"  Avg. PSNR: {avg_psnr:.4f}") # 打印 PSNR
    print(f"  Avg. LPIPS: {avg_lpips:.4f}")
    print(f"  Avg. SSIM: {avg_ssim:.4f}")
    print(f"  Avg. Discriminator Real Output: {avg_D_real:.4f}") # 记录判别器对真实样本的平均输出
    print(f"  Avg. Discriminator Fake Output: {avg_D_fake:.4f}") # 记录判别器对伪造样本的平均输出


    results = {
        "avg_attack_success_rate": avg_attack_success_rate,
        "avg_linf_norm": avg_linf_norm,
        "avg_l2_norm": avg_l2_norm,
        "avg_psnr": avg_psnr, # 添加 PSNR 到结果字典
        "avg_lpips": avg_lpips,
        "avg_ssim": avg_ssim,
        "avg_D_real": avg_D_real, # 添加到结果字典
        "avg_D_fake": avg_D_fake # 添加到结果字典
    }
    return results

# 在 train.py 中, 你可以添加一个评估阶段:
# parser.add_argument('--eval_interval', type=int, default=5, help='how many epochs to wait before evaluating')
# parser.add_argument('--eval_success_threshold', type=float, default=0.1, help='threshold for attack success in evaluation')
# parser.add_argument('--eval_success_criterion', type=str, default='mse_diff_threshold', help='criterion for attack success in evaluation')
# parser.add_argument('--eval_topk_k', type=int, default=10, help='K value for Top-K attack success criterion in evaluation') # 添加评估 Top-K 参数
# ...
# if (epoch + 1) % args.eval_interval == 0:
#     print(f"\nEvaluating model at epoch {epoch + 1}...")
#     # 假设你有一个 eval_dataloader
#     # eval_dataloader = create_real_dataloader(...) # 或者 create_mock_dataloader(...)
#     # eval_results = evaluate_model(generator, discriminator, atn_model, eval_dataloader, device, args) # 将 args 传入评估函数
#     # for key, value in eval_results.items():
#     #     writer.add_scalar(f'Evaluation/{key}', value, epoch + 1)
#     # generator.train() # Switch back to training mode
#     # discriminator.train() # Switch back to training mode
#     print("Evaluation placeholder executed in eval_utils.") 