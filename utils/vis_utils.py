import sys
import os
# 将项目根目录添加到 Python 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

# 导入正则化损失中的范数计算函数
# 假设 regularization_losses.py 在 losses 文件夹下
from losses.regularization_losses import linf_norm, l2_norm

# TODO: 辅助函数：将特征层转换为可可视化格式
# 这需要根据 ATN 的特征层含义来决定如何可视化 128 个通道和 N 个序列步骤。
# 简单示例：选取一个序列步骤，对通道进行平均或选取某个通道子集
def feature_to_visualizable(features: torch.Tensor, sequence_step=0):
    """
    将 (B, C, N, W, H) 特征张量转换为 (B, W, H) 或 (B, C', W, H) 的可视化格式。
    当前实现是简单示例，仅取某个序列步骤，并在通道维度进行简单处理。
    实际可视化需要根据 ATN 特征的语义来确定如何有效转换。

    Args:
        features (torch.Tensor): 输入特征张量。
        sequence_step (int): 选择可视化哪个序列步骤。
    Returns:
        torch.Tensor: 可视化格式的张量 (B, C', W, H) 或 (B, W, H)。
    """
    B, C, N, W, H = features.shape
    if sequence_step >= N:
        print(f"Warning: sequence_step {sequence_step} is out of bounds for N={N}. Defaulting to 0.")
        sequence_step = 0 # 默认可视化第一个步骤

    # 选取指定序列步骤
    features_step = features[:, :, sequence_step, :, :] # (B, C, W, H)

    # 简单示例：对通道进行平均
    # visualizable_features = features_step.mean(dim=1) # (B, W, H)

    # 或者：选取前3个通道作为RGB (如果特征有图像语义的话)
    if C >= 3:
        visualizable_features = features_step[:, :3, :, :] # (B, 3, W, H)
    elif C == 1:
        visualizable_features = features_step # (B, 1, W, H)
    else: # C == 2 or other cases, take the first channel and unsqueeze
        visualizable_features = features_step[:, 0, :, :].unsqueeze(1) # (B, 1, W, H)

    # 将值缩放到 [0, 1] 范围以便显示 (对于特征图，可能需要更合适的归一化)
    min_val = visualizable_features.min() if visualizable_features.numel() > 0 else 0.0
    max_val = visualizable_features.max() if visualizable_features.numel() > 0 else 1.0

    if max_val > min_val:
        visualizable_features = (visualizable_features - min_val) / (max_val - min_val + 1e-8)
    else:
        visualizable_features = torch.zeros_like(visualizable_features)

    return visualizable_features

# 可视化函数
def visualize_training_losses(writer: SummaryWriter, loss_dict: dict, step: int):
    """
    将训练损失记录到 TensorBoard。
    Args:
        writer (SummaryWriter): TensorBoard SummaryWriter 对象。
        loss_dict (dict): 包含各种损失的字典。
        step (int): 当前训练步数。
    """
    for name, loss_val in loss_dict.items(): # loss_val to avoid conflict
        if isinstance(loss_val, torch.Tensor):
            writer.add_scalar(f'Loss/{name}', loss_val.item(), step)
        else:
            writer.add_scalar(f'Loss/{name}', loss_val, step)

def visualize_perturbation_norms(writer: SummaryWriter, delta: torch.Tensor, step: int):
    """
    将扰动范数记录到 TensorBoard。
    Args:
        writer (SummaryWriter): TensorBoard SummaryWriter 对象。
        delta (torch.Tensor): 生成的扰动。
        step (int): 当前训练步数。
    """
    with torch.no_grad():
        linf = linf_norm(delta).mean().item()
        l2 = l2_norm(delta).mean().item()
        writer.add_scalar('Perturbation/Linf_Norm', linf, step)
        writer.add_scalar('Perturbation/L2_Norm', l2, step)

def visualize_samples_and_outputs(writer: SummaryWriter,
                                  original_features: torch.Tensor,
                                  delta: torch.Tensor,
                                  adversarial_features: torch.Tensor,
                                  original_decision_map: torch.Tensor,
                                  adversarial_decision_map: torch.Tensor,
                                  step: int,
                                  num_samples=4,
                                  sequence_step_to_vis=0,
                                  visualize_decision_diff=True): # 添加可视化决策图差异的参数
    """
    可视化原始特征、扰动、对抗特征、原始决策图和对抗决策图。
    将图像记录到 TensorBoard。
    Args:
        writer (SummaryWriter): TensorBoard SummaryWriter 对象。
        original_features (torch.Tensor): 原始特征层 (B, C, N, W, H)。
        delta (torch.Tensor): 生成的扰动 (B, C, N, W, H)。
        adversarial_features (torch.Tensor): 对抗性特征层 (B, C, N, W, H)。
        original_decision_map (torch.Tensor): 原始样本下的 ATN 决策图 (B, W, H)。
        adversarial_decision_map (torch.Tensor): 对抗样本下的 ATN 决策图 (B, W, H)。
        step (int): 当前训练步数。
        num_samples (int): 可视化多少个样本。
        sequence_step_to_vis (int): 可视化特征/扰动的哪个序列步骤。
        visualize_decision_diff (bool): 是否可视化决策图的差异。默认为 True。
    """
    num_samples_to_show = min(num_samples, original_features.shape[0])
    if num_samples_to_show == 0: return

    # 将张量移动到 CPU
    original_features_cpu = original_features[:num_samples_to_show].cpu()
    delta_cpu = delta[:num_samples_to_show].cpu()
    adversarial_features_cpu = adversarial_features[:num_samples_to_show].cpu()

    # 可视化特征层 (选择一个序列步骤)
    vis_orig_feat = feature_to_visualizable(original_features_cpu, sequence_step=sequence_step_to_vis) # (B, C', W, H) or (B, W, H)
    vis_delta = feature_to_visualizable(delta_cpu, sequence_step=sequence_step_to_vis)
    vis_adv_feat = feature_to_visualizable(adversarial_features_cpu, sequence_step=sequence_step_to_vis)

    # 确保特征可视化输出是 4D (B, C', H, W) 或 (B, 1, H, W) 以便 make_grid 处理
    if vis_orig_feat.ndim == 3: # (B, H, W)
        vis_orig_feat = vis_orig_feat.unsqueeze(1)
    if vis_delta.ndim == 3:
        vis_delta = vis_delta.unsqueeze(1)
    if vis_adv_feat.ndim == 3:
        vis_adv_feat = vis_adv_feat.unsqueeze(1)

    grid_original_features = make_grid(vis_orig_feat, nrow=num_samples_to_show, normalize=True)
    writer.add_image(f'Samples/Original_Features_Seq{sequence_step_to_vis}', grid_original_features, step)

    grid_delta = make_grid(vis_delta, nrow=num_samples_to_show, normalize=True)
    writer.add_image(f'Samples/Perturbation_Delta_Seq{sequence_step_to_vis}', grid_delta, step)

    grid_adversarial_features = make_grid(vis_adv_feat, nrow=num_samples_to_show, normalize=True)
    writer.add_image(f'Samples/Adversarial_Features_Seq{sequence_step_to_vis}', grid_adversarial_features, step)

    # 可视化 ATN 决策图 (B, W, H)
    if original_decision_map is not None and adversarial_decision_map is not None:
        original_decision_map_cpu = original_decision_map[:num_samples_to_show].cpu()
        adversarial_decision_map_cpu = adversarial_decision_map[:num_samples_to_show].cpu()

        # 将决策图转换为适合可视化的格式 (B, 1, W, H)
        if original_decision_map_cpu.ndim == 3: # (B,W,H)
            original_decision_img = original_decision_map_cpu.unsqueeze(1)
        else: # Already (B,1,W,H) or some other error
            original_decision_img = original_decision_map_cpu

        if adversarial_decision_map_cpu.ndim == 3:
            adversarial_decision_img = adversarial_decision_map_cpu.unsqueeze(1)
        else:
            adversarial_decision_img = adversarial_decision_map_cpu

        # 确保它们形状兼容
        if original_decision_img is not None and adversarial_decision_img is not None and original_decision_img.shape == adversarial_decision_img.shape:
             # 可视化原始和对抗决策图
            combined_decision_maps = torch.cat((original_decision_img, adversarial_decision_img), dim=0)
            grid_decision = make_grid(combined_decision_maps, nrow=num_samples_to_show, normalize=True)
            writer.add_image('ATN_Outputs/Decision_Maps (Original_Top vs Adversarial_Bottom)', grid_decision, step)

            # 可视化决策图差异
            if visualize_decision_diff:
                 decision_diff = adversarial_decision_map_cpu - original_decision_map_cpu # (B, W, H)
                 # 将差异归一化到 [0, 1] 或 [-1, 1] 进行可视化
                 # 可以使用 symmetric normalization around 0 for differences
                 max_abs_diff = torch.max(torch.abs(decision_diff)) if decision_diff.numel() > 0 else 1.0
                 # 避免除以零
                 if max_abs_diff > 1e-8:
                      normalized_diff = decision_diff / max_abs_diff # Range [-1, 1]
                      # 将 [-1, 1] 映射到 [0, 1] 进行 make_grid 默认的 normalize=True 处理
                      normalized_diff = (normalized_diff + 1.0) / 2.0 # Range [0, 1]
                 else:
                      normalized_diff = torch.zeros_like(decision_diff)

                 normalized_diff_img = normalized_diff.unsqueeze(1) # (B, 1, W, H)
                 grid_decision_diff = make_grid(normalized_diff_img, nrow=num_samples_to_show, normalize=True) # normalize=True 会再次归一化到 [0, 1]
                 writer.add_image('ATN_Outputs/Decision_Map_Difference (Adversarial - Original)', grid_decision_diff, step)

        elif original_decision_img is not None:
            grid_orig_decision = make_grid(original_decision_img, nrow=num_samples_to_show, normalize=True)
            writer.add_image('ATN_Outputs/Original_Decision_Map', grid_orig_decision, step)
        elif adversarial_decision_img is not None:
            grid_adv_decision = make_grid(adversarial_decision_img, nrow=num_samples_to_show, normalize=True)
            writer.add_image('ATN_Outputs/Adversarial_Decision_Map', grid_adv_decision, step)
    elif original_decision_map is not None: # Only original decision map available
         original_decision_map_cpu = original_decision_map[:num_samples_to_show].cpu()
         if original_decision_map_cpu.ndim == 3: # (B,W,H)
              original_decision_img = original_decision_map_cpu.unsqueeze(1)
         else:
              original_decision_img = original_decision_map_cpu
         grid_orig_decision = make_grid(original_decision_img, nrow=num_samples_to_show, normalize=True)
         writer.add_image('ATN_Outputs/Original_Decision_Map', grid_orig_decision, step)

# 可视化 ATN 注意力矩阵热力图
def visualize_attention_maps(writer: SummaryWriter,
                             attention_matrix_orig: torch.Tensor,
                             attention_matrix_adv: torch.Tensor,
                             step: int,
                             num_samples=1,
                             num_heads_to_vis=1):
    """
    可视化 ATN 注意力矩阵的热力图。

    Args:
        writer (SummaryWriter): TensorBoard SummaryWriter 对象。
        attention_matrix_orig (torch.Tensor): 原始样本下的ATN注意力矩阵 (B, head, N, N)。
        attention_matrix_adv (torch.Tensor): 对抗样本下的ATN注意力矩阵 (B, head, N, N)。
        step (int): 当前训练步数。
        num_samples (int): 可视化多少个样本。
        num_heads_to_vis (int): 可视化多少个注意力头。
    """
    print("Generating attention map visualizations...")

    if attention_matrix_orig is None and attention_matrix_adv is None:
        print("Warning: Attention matrices are None. Skipping attention map visualization.")
        return

    # 确保至少有一个注意力矩阵不是 None
    if attention_matrix_orig is not None:
        B, head, N1, N2 = attention_matrix_orig.shape
        attention_matrix_orig_cpu = attention_matrix_orig.cpu().detach().numpy()
    elif attention_matrix_adv is not None:
        B, head, N1, N2 = attention_matrix_adv.shape
        attention_matrix_adv_cpu = attention_matrix_adv.cpu().detach().numpy()

    assert N1 == N2, "Attention matrix is expected to be square (N x N)"
    num_samples_to_show_attn = min(num_samples, B)
    num_heads_total = head

    for i in range(num_samples_to_show_attn):
        for h_idx in range(min(num_heads_to_vis, num_heads_total)):
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Original Attention Heatmap
            if attention_matrix_orig is not None:
                attn_map_orig = attention_matrix_orig_cpu[i, h_idx, :, :]
                # Use a consistent color scale range if possible, e.g., [0, 1] for softmax outputs
                im_orig = axes[0].imshow(attn_map_orig, cmap='hot', interpolation='nearest', vmin=0, vmax=1) # Assume attention is in [0, 1]
                axes[0].set_title(f'Sample {i} Head {h_idx} Orig Attn')
                axes[0].set_xlabel('Key Sequence Step')
                axes[0].set_ylabel('Query Sequence Step')
                fig.colorbar(im_orig, ax=axes[0])
            else:
                axes[0].text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=12)
                axes[0].set_title(f'Sample {i} Head {h_idx} Orig Attn')
                axes[0].axis('off') # Hide axes for N/A plot

            # Adversarial Attention Heatmap
            if attention_matrix_adv is not None:
                attn_map_adv = attention_matrix_adv_cpu[i, h_idx, :, :]
                 # Use the same color scale range as original for comparison
                im_adv = axes[1].imshow(attn_map_adv, cmap='hot', interpolation='nearest', vmin=0, vmax=1) # Assume attention is in [0, 1]
                axes[1].set_title(f'Sample {i} Head {h_idx} Adv Attn')
                axes[1].set_xlabel('Key Sequence Step')
                axes[1].set_ylabel('Query Sequence Step')
                fig.colorbar(im_adv, ax=axes[1])
            else:
                axes[1].text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=12)
                axes[1].set_title(f'Sample {i} Head {h_idx} Adv Attn')
                axes[1].axis('off') # Hide axes for N/A plot

            # 可选：可视化注意力差异热力图
            if attention_matrix_orig is not None and attention_matrix_adv is not None:
                 attention_diff = attention_matrix_adv_cpu[i, h_idx, :, :] - attention_matrix_orig_cpu[i, h_idx, :, :]
                 fig_diff, ax_diff = plt.subplots(figsize=(6, 5))
                 # Use a diverging colormap for differences, centered at 0
                 max_abs_diff = np.max(np.abs(attention_diff)) if attention_diff.size > 0 else 1.0
                 im_diff = ax_diff.imshow(attention_diff, cmap='coolwarm', interpolation='nearest', vmin=-max_abs_diff, vmax=max_abs_diff)
                 ax_diff.set_title(f'Sample {i} Head {h_idx} Attn Diff (Adv - Orig)')
                 ax_diff.set_xlabel('Key Sequence Step')
                 ax_diff.set_ylabel('Query Sequence Step')
                 fig_diff.colorbar(im_diff, ax=ax_diff)
                 writer.add_figure(f'ATN_Outputs/Attention_Difference/Sample{i}_Head{h_idx}', fig_diff, step)
                 plt.close(fig_diff) # Close the figure to free memory

            writer.add_figure(f'ATN_Outputs/Attention/Sample{i}_Head{h_idx}', fig, step)
            plt.close(fig) # Close the figure to free memory

# 示例：如何在 train.py 中调用 (假设已获取相关张量) - 更新注释
# from utils.vis_utils import visualize_samples_and_outputs, visualize_attention_maps
# ... 在训练循环中 ...
# if global_step % args.vis_interval == 0:
#     with torch.no_grad():
#         # 确保这些张量在调用前已经存在并且在CPU上（或vis函数内部处理）
#         # original_features, delta, adversarial_features
#         # original_atn_outputs.get('decision'), adversarial_atn_outputs.get('decision')
#         # original_atn_outputs.get('attention'), adversarial_atn_outputs.get('attention')

#         visualize_samples_and_outputs(
#             writer,
#             original_features, # (B, C, N, W, H)
#             current_delta,     # (B, C, N, W, H)
#             current_adversarial_features, # (B, C, N, W, H)
#             original_atn_outputs.get('decision'), # (B, W, H)
#             adversarial_atn_outputs.get('decision'),# (B, W, H)
#             global_step,
#             num_samples=args.num_vis_samples, # e.g., 4
#             sequence_step_to_vis=args.sequence_step_to_vis, # e.g., visualize the first sequence step
#             visualize_decision_diff=True # Enable decision map difference visualization
#         )

#         if original_atn_outputs.get('attention') is not None or adversarial_atn_outputs.get('attention') is not None:
#             visualize_attention_maps(
#                 writer,
#                 original_atn_outputs.get('attention'), # (B, head, N, N)
#                 adversarial_atn_outputs.get('attention'),# (B, head, N, N)
#                 global_step,
#                 num_samples=args.num_vis_samples, # 可以控制可视化样本数量
#                 num_heads_to_vis=getattr(args, 'num_vis_heads', 1) # Use args.num_vis_heads, default to 1
#             ) 