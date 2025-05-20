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
    将 (B, C, N, W, H) 特征张量转换为 (B, C', H, W) 或 (B, 1, H, W) 的可视化格式。
    当前实现是简单示例，仅取某个序列步骤，并在通道维度进行简单处理。
    实际可视化需要根据 ATN 特征的语义来确定如何有效转换。

    Args:
        features (torch.Tensor): 输入特征张量 (B, C, N, W, H)。
        sequence_step (int): 选择可视化哪个序列步骤。
    Returns:
        torch.Tensor: 可视化格式的张量 (B, C', H, W) 或 (B, 1, H, W)。
                       返回的张量已归一化到 [0, 1]。
    """
    if features is None or features.numel() == 0:
        print("Warning: Input features are None or empty. Cannot visualize features.")
        return torch.empty(0) # Return empty tensor

    assert features.ndim == 5, f"Features tensor must be 5D (B, C, N, W, H), got {features.ndim}D"
    B, C, N, W, H = features.shape
    
    if N == 0:
        print("Warning: Sequence dimension N is 0. Cannot visualize features.")
        return torch.empty(0)

    if sequence_step >= N or sequence_step < 0:
        print(f"Warning: sequence_step {sequence_step} is out of bounds for N={N}. Defaulting to 0.")
        sequence_step = 0 # 默认可视化第一个步骤

    # 选取指定序列步骤
    features_step = features[:, :, sequence_step, :, :] # (B, C, W, H) -> Expected (B, C, H, W) based on common CNN practice?
    # Let's assume the input features tensor is (B, C, N, H, W) as per common convention
    # If your data_utils or models use (B, C, N, W, H), adjust here or in the caller
    # Assuming (B, C, N, H, W) based on Generator/Discriminator kernel/stride definitions
    features_step = features[:, :, sequence_step, :, :] # (B, C, H, W)

    # 简单示例：对通道进行平均 或 选取前3个通道
    if C >= 3:
        # 选取前3个通道作为RGB
        visualizable_features = features_step[:, :3, :, :] # (B, 3, H, W)
    elif C == 1:
        # 如果是单通道，直接返回
        visualizable_features = features_step # (B, 1, H, W)
    else: # C == 2 or other cases, take the first channel and unsqueeze, then repeat to 3 channels for make_grid
        print(f"Warning: Feature channels C={C}. Taking the first channel and repeating to 3 for visualization.")
        visualizable_features = features_step[:, 0, :, :].unsqueeze(1).repeat(1, 3, 1, 1) # (B, 3, H, W)


    # 将值缩放到 [0, 1] 范围以便显示
    # For feature maps, simple min-max normalization might not be ideal if distribution is skewed
    # But for visualization, it's usually acceptable to see relative patterns
    min_val = visualizable_features.min() if visualizable_features.numel() > 0 else 0.0
    max_val = visualizable_features.max() if visualizable_features.numel() > 0 else 1.0

    if max_val > min_val:
        visualizable_features = (visualizable_features - min_val) / (max_val - min_val + 1e-8)
    else:
        # Handle case where all values are the same (e.g., zeros)
        visualizable_features = torch.zeros_like(visualizable_features)

    return visualizable_features.float() # Ensure float type for make_grid

# 可视化函数
def visualize_training_losses(writer: SummaryWriter, loss_dict: dict, step: int):
    """
    将训练损失记录到 TensorBoard。
    Args:
        writer (SummaryWriter): TensorBoard SummaryWriter 对象。
        loss_dict (dict): 包含各种损失的字典。
        step (int): 当前训练步数。
    """
    for name, loss_val in loss_dict.items():
        if isinstance(loss_val, torch.Tensor):
            # Check if loss_val is a scalar tensor
            if loss_val.ndim == 0:
                writer.add_scalar(f'Loss/{name}', loss_val.item(), step)
            else:
                 # If it's not a scalar, maybe log its mean or a warning
                 # print(f"Warning: Loss tensor '{name}' is not scalar ({loss_val.shape}). Logging mean.")
                 writer.add_scalar(f'Loss/{name}', loss_val.mean().item(), step) # Log mean if not scalar

        else:
            writer.add_scalar(f'Loss/{name}', loss_val, step)

def visualize_perturbation_norms(writer: SummaryWriter, delta: torch.Tensor, step: int):
    """
    将扰动范数记录到 TensorBoard。
    Args:
        writer (SummaryWriter): TensorBoard SummaryWriter 对象。
        delta (torch.Tensor): 生成的扰动 (B, C, N, W, H) 或 (B, C, N, H, W)。
        step (int): 当前训练步数。
    """
    if delta is None or delta.numel() == 0:
        print("Warning: Delta tensor is None or empty. Skipping perturbation norm visualization.")
        return

    with torch.no_grad():
        # Assuming delta is (B, C, N, H, W) or (B, C, N, W, H)
        if delta.ndim != 5:
             print(f"Warning: Delta tensor is not 5D ({delta.ndim}D). Cannot calculate perturbation norms.")
             return

        linf = linf_norm(delta).mean().item() # linf_norm returns (B,), take mean
        l2 = l2_norm(delta).mean().item() # l2_norm returns (B,), take mean
        writer.add_scalar('Perturbation/Linf_Norm', linf, step)
        writer.add_scalar('Perturbation/L2_Norm', l2, step)

def visualize_samples_and_outputs(writer: SummaryWriter,
                                  original_features: torch.Tensor, # (B, C, N, W, H) or (B, C, N, H, W)
                                  delta: torch.Tensor,             # (B, C, N, W, H) or (B, C, N, H, W)
                                  adversarial_features: torch.Tensor, # (B, C, N, W, H) or (B, C, N, H, W)
                                  original_decision_map: torch.Tensor, # (B, W, H) or (B, H, W)
                                  adversarial_decision_map: torch.Tensor, # (B, W, H) or (B, H, W)
                                  step: int,
                                  num_samples=4,
                                  sequence_step_to_vis=0,
                                  visualize_decision_diff=True):
    """
    可视化原始特征、扰动、对抗特征、原始决策图和对抗决策图。
    将图像记录到 TensorBoard。
    Args:
        writer (SummaryWriter): TensorBoard SummaryWriter 对象。
        original_features (torch.Tensor): 原始特征层 (B, C, N, W, H) 或 (B, C, N, H, W)。
        delta (torch.Tensor): 生成的扰动 (B, C, N, W, H) 或 (B, C, N, H, W)。
        adversarial_features (torch.Tensor): 对抗性特征层 (B, C, N, W, H) 或 (B, C, N, H, W)。
        original_decision_map (torch.Tensor): 原始样本下的 ATN 决策图 (B, W, H) 或 (B, H, W)。
        adversarial_decision_map (torch.Tensor): 对抗样本下的 ATN 决策图 (B, W, H) 或 (B, H, W)。
        step (int): 当前训练步数。
        num_samples (int): 可视化多少个样本。
        sequence_step_to_vis (int): 可视化特征/扰动的哪个序列步骤。
        visualize_decision_diff (bool): 是否可视化决策图的差异。默认为 True。
    """
    if original_features is None or original_features.numel() == 0:
        print("Warning: Original features are None or empty. Skipping sample visualization.")
        return

    num_samples_to_show = min(num_samples, original_features.shape[0])
    if num_samples_to_show == 0: return

    # 将张量移动到 CPU (如果不在 CPU 上) 并选取要显示的样本
    original_features_cpu = original_features[:num_samples_to_show].cpu()
    delta_cpu = delta[:num_samples_to_show].cpu() if delta is not None else None
    adversarial_features_cpu = adversarial_features[:num_samples_to_show].cpu()

    # 可视化特征层 (选择一个序列步骤)
    # feature_to_visualizable expects (B, C, N, H, W) or (B, C, N, W, H)
    vis_orig_feat = feature_to_visualizable(original_features_cpu, sequence_step=sequence_step_to_vis) # (B, C', H, W) or (B, 1, H, W)
    vis_delta = feature_to_visualizable(delta_cpu, sequence_step=sequence_step_to_vis) if delta_cpu is not None else torch.empty(0)
    vis_adv_feat = feature_to_visualizable(adversarial_features_cpu, sequence_step=sequence_step_to_vis)

    # Ensure feature visualizations are not empty before creating grids
    if vis_orig_feat.numel() > 0:
         # Ensure it's 4D (B, C', H, W) or (B, 1, H, W) for make_grid
        if vis_orig_feat.ndim == 3: # (B, H, W) - should not happen if feature_to_visualizable works as expected
            vis_orig_feat = vis_orig_feat.unsqueeze(1)
    grid_original_features = make_grid(vis_orig_feat, nrow=num_samples_to_show, normalize=True)
    writer.add_image(f'Samples/Original_Features_Seq{sequence_step_to_vis}', grid_original_features, step)

    if vis_delta.numel() > 0:
         if vis_delta.ndim == 3:
              vis_delta = vis_delta.unsqueeze(1)
    grid_delta = make_grid(vis_delta, nrow=num_samples_to_show, normalize=True)
    writer.add_image(f'Samples/Perturbation_Delta_Seq{sequence_step_to_vis}', grid_delta, step)

    if vis_adv_feat.numel() > 0:
         if vis_adv_feat.ndim == 3:
              vis_adv_feat = vis_adv_feat.unsqueeze(1)
    grid_adversarial_features = make_grid(vis_adv_feat, nrow=num_samples_to_show, normalize=True)
    writer.add_image(f'Samples/Adversarial_Features_Seq{sequence_step_to_vis}', grid_adversarial_features, step)


    # 可视化 ATN 决策图 (B, W, H) 或 (B, H, W)
    if original_decision_map is not None and adversarial_decision_map is not None:
        assert original_decision_map.shape == adversarial_decision_map.shape, "Decision map shapes mismatch"
        # Assuming decision maps are (B, H, W) based on decision_layer output in mock ATN
        original_decision_map_cpu = original_decision_map[:num_samples_to_show].cpu()
        adversarial_decision_map_cpu = adversarial_decision_map[:num_samples_to_show].cpu()

        # 将决策图转换为适合可视化的格式 (B, 1, H, W)
        if original_decision_map_cpu.ndim == 3: # (B, H, W)
            original_decision_img = original_decision_map_cpu.unsqueeze(1)
            adversarial_decision_img = adversarial_decision_map_cpu.unsqueeze(1)
        elif original_decision_map_cpu.ndim == 4 and original_decision_map_cpu.shape[1] == 1: # Already (B, 1, H, W)
            original_decision_img = original_decision_map_cpu
            adversarial_decision_img = adversarial_decision_map_cpu
        else:
            print(f"Warning: Unexpected decision map shape {original_decision_map_cpu.shape}. Skipping decision map visualization.")
            original_decision_img = None
            adversarial_decision_img = None


        if original_decision_img is not None and adversarial_decision_img is not None:
             # 可视化原始和对抗决策图
            combined_decision_maps = torch.cat((original_decision_img, adversarial_decision_img), dim=0)
            # Normalize across the combined set for consistent color scale
            grid_decision = make_grid(combined_decision_maps, nrow=num_samples_to_show, normalize=True)
            writer.add_image('ATN_Outputs/Decision_Maps (Original_Top vs Adversarial_Bottom)', grid_decision, step)

            # 可视化决策图差异
            if visualize_decision_diff:
                 decision_diff = adversarial_decision_map_cpu - original_decision_map_cpu # (B, H, W)
                 # Ensure diff is 4D for make_grid
                 if decision_diff.ndim == 3:
                      decision_diff = decision_diff.unsqueeze(1) # (B, 1, H, W)

                 # Use a diverging colormap for differences, centered at 0
                 # Need to normalize manually for a centered colormap, make_grid normalize=True is [0, 1]
                 # Convert to numpy for matplotlib visualization, or normalize manually for make_grid
                 # Let's normalize manually to [-1, 1] then map to [0, 1] for make_grid
                 max_abs_diff = torch.max(torch.abs(decision_diff)) if decision_diff.numel() > 0 else 1.0
                 # Avoid division by zero
                 if max_abs_diff > 1e-8:
                      normalized_diff = decision_diff / max_abs_diff # Range [-1, 1]
                      normalized_diff = (normalized_diff + 1.0) / 2.0 # Map [-1, 1] to [0, 1]
                 else:
                      normalized_diff = torch.zeros_like(decision_diff) # All diffs are zero

                 grid_decision_diff = make_grid(normalized_diff, nrow=num_samples_to_show, normalize=True) # make_grid normalize=True will rescale to [0, 1] again
                 writer.add_image('ATN_Outputs/Decision_Map_Difference (Adversarial - Original)', grid_decision_diff, step)

        elif original_decision_map is not None: # Only original decision map available
             original_decision_map_cpu = original_decision_map[:num_samples_to_show].cpu()
             if original_decision_map_cpu.ndim == 3:
                  original_decision_img = original_decision_map_cpu.unsqueeze(1)
             else:
                  original_decision_img = original_decision_map_cpu
        grid_orig_decision = make_grid(original_decision_img, nrow=num_samples_to_show, normalize=True)
        writer.add_image('ATN_Outputs/Original_Decision_Map', grid_orig_decision, step)
    elif adversarial_decision_map is not None: # Only adversarial decision map available
            adversarial_decision_map_cpu = adversarial_decision_map[:num_samples_to_show].cpu()
            if adversarial_decision_map_cpu.ndim == 3:
                 adversarial_decision_img = adversarial_decision_map_cpu.unsqueeze(1)
            else:
                 adversarial_decision_img = adversarial_decision_map_cpu
            grid_adv_decision = make_grid(adversarial_decision_img, nrow=num_samples_to_show, normalize=True)
            writer.add_image('ATN_Outputs/Adversarial_Decision_Map', grid_adv_decision, step)


# 可视化 ATN 注意力矩阵热力图
def visualize_attention_maps(writer: SummaryWriter,
                             attention_matrix_orig: torch.Tensor, # (B, head, N, N)
                             attention_matrix_adv: torch.Tensor, # (B, head, N, N)
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

    # Ensure at least one attention matrix is not None and not empty
    if (attention_matrix_orig is None or attention_matrix_orig.numel() == 0) and \
       (attention_matrix_adv is None or attention_matrix_adv.numel() == 0):
        print("Warning: Attention matrices are None or empty. Skipping attention map visualization.")
        return

    # Determine batch size and number of heads from available tensors
    B, head, N1, N2 = 0, 0, 0, 0
    if attention_matrix_orig is not None and attention_matrix_orig.numel() > 0:
        assert attention_matrix_orig.ndim == 4, f"Original attention matrix must be 4D (B, head, N, N), got {attention_matrix_orig.ndim}D"
        B, head, N1, N2 = attention_matrix_orig.shape
    elif attention_matrix_adv is not None and attention_matrix_adv.numel() > 0:
        assert attention_matrix_adv.ndim == 4, f"Adversarial attention matrix must be 4D (B, head, N, N), got {attention_matrix_adv.ndim}D"
        B, head, N1, N2 = attention_matrix_adv.shape

    if B == 0 or head == 0 or N1 == 0 or N2 == 0:
         print("Warning: Attention matrix dimensions are zero. Skipping attention map visualization.")
         return

    assert N1 == N2, "Attention matrix is expected to be square (N x N)"
    num_samples_to_show_attn = min(num_samples, B)
    num_heads_total = head
    num_heads_to_show = min(num_heads_to_vis, num_heads_total)

    # Convert to numpy for matplotlib
    attention_matrix_orig_np = attention_matrix_orig.cpu().detach().numpy() if attention_matrix_orig is not None else None
    attention_matrix_adv_np = attention_matrix_adv.cpu().detach().numpy() if attention_matrix_adv is not None else None


    for i in range(num_samples_to_show_attn):
        for h_idx in range(num_heads_to_show):
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Determine a common color scale range
            vmin, vmax = 0, 1 # Assume attention is normalized between 0 and 1
            # If you want to dynamically determine range from data:
            # all_attn_values = []
            # if attention_matrix_orig_np is not None: all_attn_values.append(attention_matrix_orig_np[i, h_idx])
            # if attention_matrix_adv_np is not None: all_attn_values.append(attention_matrix_adv_np[i, h_idx])
            # if all_attn_values:
            #     all_attn_values = np.concatenate([arr.flatten() for arr in all_attn_values])
            #     vmin, vmax = np.min(all_attn_values), np.max(all_attn_values)
            # if vmax - vmin < 1e-8: vmin, vmax = 0, 1 # Handle flat data

            # Original Attention Heatmap
            if attention_matrix_orig_np is not None:
                attn_map_orig = attention_matrix_orig_np[i, h_idx, :, :]
                im_orig = axes[0].imshow(attn_map_orig, cmap='hot', interpolation='nearest', vmin=vmin, vmax=vmax)
                axes[0].set_title(f'Sample {i} Head {h_idx} Orig Attn')
                axes[0].set_xlabel('Key Sequence Step')
                axes[0].set_ylabel('Query Sequence Step')
                fig.colorbar(im_orig, ax=axes[0])
            else:
                axes[0].text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=12)
                axes[0].set_title(f'Sample {i} Head {h_idx} Orig Attn')
                axes[0].axis('off') # Hide axes for N/A plot

            # Adversarial Attention Heatmap
            if attention_matrix_adv_np is not None:
                attn_map_adv = attention_matrix_adv_np[i, h_idx, :, :]
                im_adv = axes[1].imshow(attn_map_adv, cmap='hot', interpolation='nearest', vmin=vmin, vmax=vmax)
                axes[1].set_title(f'Sample {i} Head {h_idx} Adv Attn')
                axes[1].set_xlabel('Key Sequence Step')
                axes[1].set_ylabel('Query Sequence Step')
                fig.colorbar(im_adv, ax=axes[1])
            else:
                axes[1].text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=12)
                axes[1].set_title(f'Sample {i} Head {h_idx} Adv Attn')
                axes[1].axis('off') # Hide axes for N/A plot

            # Optional: Visualize Attention Difference Heatmap
            if attention_matrix_orig_np is not None and attention_matrix_adv_np is not None:
                 attention_diff = attn_map_adv - attn_map_orig
                 fig_diff, ax_diff = plt.subplots(figsize=(6, 5))
                 # Use a diverging colormap for differences, centered at 0
                 max_abs_diff = np.max(np.abs(attention_diff)) if attention_diff.size > 0 else 1.0
                 # Avoid division by zero for color scale
                 vmax_diff = max(max_abs_diff, 1e-8)
                 im_diff = ax_diff.imshow(attention_diff, cmap='coolwarm', interpolation='nearest', vmin=-vmax_diff, vmax=vmax_diff)
                 ax_diff.set_title(f'Sample {i} Head {h_idx} Attn Diff (Adv - Orig)')
                 ax_diff.set_xlabel('Key Sequence Step')
                 ax_diff.set_ylabel('Query Sequence Step')
                 fig_diff.colorbar(im_diff, ax=ax_diff)
                 writer.add_figure(f'ATN_Outputs/Attention_Difference/Sample{i}_Head{h_idx}', fig_diff, step)
                 plt.close(fig_diff) # Close the figure to free memory

            writer.add_figure(f'ATN_Outputs/Attention/Sample{i}_Head{h_idx}', fig, step)
            plt.close(fig) # Close the figure to free memory

# 可视化函数
def visualize_training_losses(writer: SummaryWriter, loss_dict: dict, step: int):
    """
    将训练损失记录到 TensorBoard。
    Args:
        writer (SummaryWriter): TensorBoard SummaryWriter 对象。
        loss_dict (dict): 包含各种损失的字典。
        step (int): 当前训练步数。
    """
    for name, loss_val in loss_dict.items():
        if isinstance(loss_val, torch.Tensor):
            # Check if loss_val is a scalar tensor
            if loss_val.ndim == 0:
                writer.add_scalar(f'Loss/{name}', loss_val.item(), step)
            else:
                 # If it's not a scalar, maybe log its mean or a warning
                 # print(f"Warning: Loss tensor '{name}' is not scalar ({loss_val.shape}). Logging mean.")
                 writer.add_scalar(f'Loss/{name}', loss_val.mean().item(), step) # Log mean if not scalar

        else:
            writer.add_scalar(f'Loss/{name}', loss_val, step)

def visualize_perturbation_norms(writer: SummaryWriter, delta: torch.Tensor, step: int):
    """
    将扰动范数记录到 TensorBoard。
    Args:
        writer (SummaryWriter): TensorBoard SummaryWriter 对象。
        delta (torch.Tensor): 生成的扰动 (B, C, N, W, H) 或 (B, C, N, H, W)。
        step (int): 当前训练步数。
    """
    if delta is None or delta.numel() == 0:
        print("Warning: Delta tensor is None or empty. Skipping perturbation norm visualization.")
        return

    with torch.no_grad():
        # Assuming delta is (B, C, N, H, W) or (B, C, N, W, H)
        if delta.ndim != 5:
             print(f"Warning: Delta tensor is not 5D ({delta.ndim}D). Cannot calculate perturbation norms.")
             return

        linf = linf_norm(delta).mean().item() # linf_norm returns (B,), take mean
        l2 = l2_norm(delta).mean().item() # l2_norm returns (B,), take mean
        writer.add_scalar('Perturbation/Linf_Norm', linf, step)
        writer.add_scalar('Perturbation/L2_Norm', l2, step)

def visualize_samples_and_outputs(writer: SummaryWriter,
                                  original_features: torch.Tensor, # (B, C, N, W, H) or (B, C, N, H, W)
                                  delta: torch.Tensor,             # (B, C, N, W, H) or (B, C, N, H, W)
                                  adversarial_features: torch.Tensor, # (B, C, N, W, H) or (B, C, N, H, W)
                                  original_decision_map: torch.Tensor, # (B, W, H) or (B, H, W)
                                  adversarial_decision_map: torch.Tensor, # (B, W, H) or (B, H, W)
                                  step: int,
                                  num_samples=4,
                                  sequence_step_to_vis=0,
                                  visualize_decision_diff=True):
    """
    可视化原始特征、扰动、对抗特征、原始决策图和对抗决策图。
    将图像记录到 TensorBoard。
    Args:
        writer (SummaryWriter): TensorBoard SummaryWriter 对象。
        original_features (torch.Tensor): 原始特征层 (B, C, N, W, H) 或 (B, C, N, H, W)。
        delta (torch.Tensor): 生成的扰动 (B, C, N, W, H) 或 (B, C, N, H, W)。
        adversarial_features (torch.Tensor): 对抗性特征层 (B, C, N, W, H) 或 (B, C, N, H, W)。
        original_decision_map (torch.Tensor): 原始样本下的 ATN 决策图 (B, W, H) 或 (B, H, W)。
        adversarial_decision_map (torch.Tensor): 对抗样本下的 ATN 决策图 (B, W, H) 或 (B, H, W)。
        step (int): 当前训练步数。
        num_samples (int): 可视化多少个样本。
        sequence_step_to_vis (int): 可视化特征/扰动的哪个序列步骤。
        visualize_decision_diff (bool): 是否可视化决策图的差异。默认为 True。
    """
    if original_features is None or original_features.numel() == 0:
        print("Warning: Original features are None or empty. Skipping sample visualization.")
        return

    num_samples_to_show = min(num_samples, original_features.shape[0])
    if num_samples_to_show == 0: return

    # 将张量移动到 CPU (如果不在 CPU 上) 并选取要显示的样本
    original_features_cpu = original_features[:num_samples_to_show].cpu()
    delta_cpu = delta[:num_samples_to_show].cpu() if delta is not None else None
    adversarial_features_cpu = adversarial_features[:num_samples_to_show].cpu()

    # 可视化特征层 (选择一个序列步骤)
    # feature_to_visualizable expects (B, C, N, H, W) or (B, C, N, W, H)
    vis_orig_feat = feature_to_visualizable(original_features_cpu, sequence_step=sequence_step_to_vis) # (B, C', H, W) or (B, 1, H, W)
    vis_delta = feature_to_visualizable(delta_cpu, sequence_step=sequence_step_to_vis) if delta_cpu is not None else torch.empty(0)
    vis_adv_feat = feature_to_visualizable(adversarial_features_cpu, sequence_step=sequence_step_to_vis)

    # Ensure feature visualizations are not empty before creating grids
    if vis_orig_feat.numel() > 0:
         # Ensure it's 4D (B, C', H, W) or (B, 1, H, W) for make_grid
         if vis_orig_feat.ndim == 3: # (B, H, W) - should not happen if feature_to_visualizable works as expected
              vis_orig_feat = vis_orig_feat.unsqueeze(1)
         grid_original_features = make_grid(vis_orig_feat, nrow=num_samples_to_show, normalize=True)
         writer.add_image(f'Samples/Original_Features_Seq{sequence_step_to_vis}', grid_original_features, step)

    if vis_delta.numel() > 0:
         if vis_delta.ndim == 3:
              vis_delta = vis_delta.unsqueeze(1)
         grid_delta = make_grid(vis_delta, nrow=num_samples_to_show, normalize=True)
         writer.add_image(f'Samples/Perturbation_Delta_Seq{sequence_step_to_vis}', grid_delta, step)

    if vis_adv_feat.numel() > 0:
         if vis_adv_feat.ndim == 3:
              vis_adv_feat = vis_adv_feat.unsqueeze(1)
         grid_adversarial_features = make_grid(vis_adv_feat, nrow=num_samples_to_show, normalize=True)
         writer.add_image(f'Samples/Adversarial_Features_Seq{sequence_step_to_vis}', grid_adversarial_features, step)


    # 可视化 ATN 决策图 (B, W, H) 或 (B, H, W)
    if original_decision_map is not None and adversarial_decision_map is not None:
        assert original_decision_map.shape == adversarial_decision_map.shape, "Decision map shapes mismatch"
        # Assuming decision maps are (B, H, W) based on decision_layer output in mock ATN
        original_decision_map_cpu = original_decision_map[:num_samples_to_show].cpu()
        adversarial_decision_map_cpu = adversarial_decision_map[:num_samples_to_show].cpu()

        # 将决策图转换为适合可视化的格式 (B, 1, H, W)
        if original_decision_map_cpu.ndim == 3: # (B, H, W)
            original_decision_img = original_decision_map_cpu.unsqueeze(1)
            adversarial_decision_img = adversarial_decision_map_cpu.unsqueeze(1)
        elif original_decision_map_cpu.ndim == 4 and original_decision_map_cpu.shape[1] == 1: # Already (B, 1, H, W)
            original_decision_img = original_decision_map_cpu
            adversarial_decision_img = adversarial_decision_map_cpu
        else:
            print(f"Warning: Unexpected decision map shape {original_decision_map_cpu.shape}. Skipping decision map visualization.")
            original_decision_img = None
            adversarial_decision_img = None


        if original_decision_img is not None and adversarial_decision_img is not None:
             # 可视化原始和对抗决策图
            combined_decision_maps = torch.cat((original_decision_img, adversarial_decision_img), dim=0)
            # Normalize across the combined set for consistent color scale
            grid_decision = make_grid(combined_decision_maps, nrow=num_samples_to_show, normalize=True)
            writer.add_image('ATN_Outputs/Decision_Maps (Original_Top vs Adversarial_Bottom)', grid_decision, step)

            # 可视化决策图差异
            if visualize_decision_diff:
                 decision_diff = adversarial_decision_map_cpu - original_decision_map_cpu # (B, H, W)
                 # Ensure diff is 4D for make_grid
                 if decision_diff.ndim == 3:
                      decision_diff = decision_diff.unsqueeze(1) # (B, 1, H, W)

                 # Use a diverging colormap for differences, centered at 0
                 # Need to normalize manually for a centered colormap, make_grid normalize=True is [0, 1]
                 # Convert to numpy for matplotlib visualization, or normalize manually for make_grid
                 # Let's normalize manually to [-1, 1] then map to [0, 1] for make_grid
                 max_abs_diff = torch.max(torch.abs(decision_diff)) if decision_diff.numel() > 0 else 1.0
                 # Avoid division by zero
                 if max_abs_diff > 1e-8:
                      normalized_diff = decision_diff / max_abs_diff # Range [-1, 1]
                      normalized_diff = (normalized_diff + 1.0) / 2.0 # Map [-1, 1] to [0, 1]
                 else:
                      normalized_diff = torch.zeros_like(decision_diff) # All diffs are zero

                 grid_decision_diff = make_grid(normalized_diff, nrow=num_samples_to_show, normalize=True) # make_grid normalize=True will rescale to [0, 1] again
                 writer.add_image('ATN_Outputs/Decision_Map_Difference (Adversarial - Original)', grid_decision_diff, step)

    elif original_decision_map is not None: # Only original decision map available
            original_decision_map_cpu = original_decision_map[:num_samples_to_show].cpu()
            if original_decision_map_cpu.ndim == 3:
              original_decision_img = original_decision_map_cpu.unsqueeze(1)
            else:
              original_decision_img = original_decision_map_cpu
            grid_orig_decision = make_grid(original_decision_img, nrow=num_samples_to_show, normalize=True)
            writer.add_image('ATN_Outputs/Original_Decision_Map', grid_orig_decision, step)
    elif adversarial_decision_img is not None: # Only adversarial decision map available
            adversarial_decision_map_cpu = adversarial_decision_map[:num_samples_to_show].cpu()
            if adversarial_decision_map_cpu.ndim == 3:
                 adversarial_decision_img = adversarial_decision_map_cpu.unsqueeze(1)
            else:
                 adversarial_decision_img = adversarial_decision_map_cpu
            grid_adv_decision = make_grid(adversarial_decision_img, nrow=num_samples_to_show, normalize=True)
            writer.add_image('ATN_Outputs/Adversarial_Decision_Map', grid_adv_decision, step)


# 可视化 ATN 注意力矩阵热力图
def visualize_attention_maps(writer: SummaryWriter,
                             attention_matrix_orig: torch.Tensor, # (B, head, N, N)
                             attention_matrix_adv: torch.Tensor, # (B, head, N, N)
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

    # Ensure at least one attention matrix is not None and not empty
    if (attention_matrix_orig is None or attention_matrix_orig.numel() == 0) and \
       (attention_matrix_adv is None or attention_matrix_adv.numel() == 0):
        print("Warning: Attention matrices are None or empty. Skipping attention map visualization.")
        return

    # Determine batch size and number of heads from available tensors
    B, head, N1, N2 = 0, 0, 0, 0
    if attention_matrix_orig is not None and attention_matrix_orig.numel() > 0:
        assert attention_matrix_orig.ndim == 4, f"Original attention matrix must be 4D (B, head, N, N), got {attention_matrix_orig.ndim}D"
        B, head, N1, N2 = attention_matrix_orig.shape
    elif attention_matrix_adv is not None and attention_matrix_adv.numel() > 0:
        assert attention_matrix_adv.ndim == 4, f"Adversarial attention matrix must be 4D (B, head, N, N), got {attention_matrix_adv.ndim}D"
        B, head, N1, N2 = attention_matrix_adv.shape

    if B == 0 or head == 0 or N1 == 0 or N2 == 0:
         print("Warning: Attention matrix dimensions are zero. Skipping attention map visualization.")
         return

    assert N1 == N2, "Attention matrix is expected to be square (N x N)"
    num_samples_to_show_attn = min(num_samples, B)
    num_heads_total = head
    num_heads_to_show = min(num_heads_to_vis, num_heads_total)

    # Convert to numpy for matplotlib
    attention_matrix_orig_np = attention_matrix_orig.cpu().detach().numpy() if attention_matrix_orig is not None else None
    attention_matrix_adv_np = attention_matrix_adv.cpu().detach().numpy() if attention_matrix_adv is not None else None


    for i in range(num_samples_to_show_attn):
        for h_idx in range(num_heads_to_show):
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Determine a common color scale range
            vmin, vmax = 0, 1 # Assume attention is normalized between 0 and 1
            # If you want to dynamically determine range from data:
            # all_attn_values = []
            # if attention_matrix_orig_np is not None: all_attn_values.append(attention_matrix_orig_np[i, h_idx])
            # if attention_matrix_adv_np is not None: all_attn_values.append(attention_matrix_adv_np[i, h_idx])
            # if all_attn_values:
            #     all_attn_values = np.concatenate([arr.flatten() for arr in all_attn_values])
            #     vmin, vmax = np.min(all_attn_values), np.max(all_attn_values)
            # if vmax - vmin < 1e-8: vmin, vmax = 0, 1 # Handle flat data

            # Original Attention Heatmap
            if attention_matrix_orig_np is not None:
                attn_map_orig = attention_matrix_orig_np[i, h_idx, :, :]
                im_orig = axes[0].imshow(attn_map_orig, cmap='hot', interpolation='nearest', vmin=vmin, vmax=vmax)
                axes[0].set_title(f'Sample {i} Head {h_idx} Orig Attn')
                axes[0].set_xlabel('Key Sequence Step')
                axes[0].set_ylabel('Query Sequence Step')
                fig.colorbar(im_orig, ax=axes[0])
            else:
                axes[0].text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=12)
                axes[0].set_title(f'Sample {i} Head {h_idx} Orig Attn')
                axes[0].axis('off') # Hide axes for N/A plot

            # Adversarial Attention Heatmap
            if attention_matrix_adv_np is not None:
                attn_map_adv = attention_matrix_adv_np[i, h_idx, :, :]
                im_adv = axes[1].imshow(attn_map_adv, cmap='hot', interpolation='nearest', vmin=vmin, vmax=vmax)
                axes[1].set_title(f'Sample {i} Head {h_idx} Adv Attn')
                axes[1].set_xlabel('Key Sequence Step')
                axes[1].set_ylabel('Query Sequence Step')
                fig.colorbar(im_adv, ax=axes[1])
            else:
                axes[1].text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=12)
                axes[1].set_title(f'Sample {i} Head {h_idx} Adv Attn')
                axes[1].axis('off') # Hide axes for N/A plot

            # Optional: Visualize Attention Difference Heatmap
            if attention_matrix_orig_np is not None and attention_matrix_adv_np is not None:
                 attention_diff = attn_map_adv - attn_map_orig
                 fig_diff, ax_diff = plt.subplots(figsize=(6, 5))
                 # Use a diverging colormap for differences, centered at 0
                 max_abs_diff = np.max(np.abs(attention_diff)) if attention_diff.size > 0 else 1.0
                 # Avoid division by zero for color scale
                 vmax_diff = max(max_abs_diff, 1e-8)
                 im_diff = ax_diff.imshow(attention_diff, cmap='coolwarm', interpolation='nearest', vmin=-vmax_diff, vmax=vmax_diff)
                 ax_diff.set_title(f'Sample {i} Head {h_idx} Attn Diff (Adv - Orig)')
                 ax_diff.set_xlabel('Key Sequence Step')
                 ax_diff.set_ylabel('Query Sequence Step')
                 fig_diff.colorbar(im_diff, ax=ax_diff)
                 writer.add_figure(f'ATN_Outputs/Attention_Difference/Sample{i}_Head{h_idx}', fig_diff, step)
                 plt.close(fig_diff) # Close the figure to free memory

            writer.add_figure(f'ATN_Outputs/Attention/Sample{i}_Head{h_idx}', fig, step)
            plt.close(fig) # Close the figure to free memory

# 可视化函数
def visualize_training_losses(writer: SummaryWriter, loss_dict: dict, step: int):
    """
    将训练损失记录到 TensorBoard。
    Args:
        writer (SummaryWriter): TensorBoard SummaryWriter 对象。
        loss_dict (dict): 包含各种损失的字典。
        step (int): 当前训练步数。
    """
    for name, loss_val in loss_dict.items():
        if isinstance(loss_val, torch.Tensor):
            # Check if loss_val is a scalar tensor
            if loss_val.ndim == 0:
                writer.add_scalar(f'Loss/{name}', loss_val.item(), step)
            else:
                 # If it's not a scalar, maybe log its mean or a warning
                 # print(f"Warning: Loss tensor '{name}' is not scalar ({loss_val.shape}). Logging mean.")
                 writer.add_scalar(f'Loss/{name}', loss_val.mean().item(), step) # Log mean if not scalar

        else:
            writer.add_scalar(f'Loss/{name}', loss_val, step)

def visualize_perturbation_norms(writer: SummaryWriter, delta: torch.Tensor, step: int):
    """
    将扰动范数记录到 TensorBoard。
    Args:
        writer (SummaryWriter): TensorBoard SummaryWriter 对象。
        delta (torch.Tensor): 生成的扰动 (B, C, N, W, H) 或 (B, C, N, H, W)。
        step (int): 当前训练步数。
    """
    if delta is None or delta.numel() == 0:
        print("Warning: Delta tensor is None or empty. Skipping perturbation norm visualization.")
        return

    with torch.no_grad():
        # Assuming delta is (B, C, N, H, W) or (B, C, N, W, H)
        if delta.ndim != 5:
             print(f"Warning: Delta tensor is not 5D ({delta.ndim}D). Cannot calculate perturbation norms.")
             return

        linf = linf_norm(delta).mean().item() # linf_norm returns (B,), take mean
        l2 = l2_norm(delta).mean().item() # l2_norm returns (B,), take mean
        writer.add_scalar('Perturbation/Linf_Norm', linf, step)
        writer.add_scalar('Perturbation/L2_Norm', l2, step)

def visualize_samples_and_outputs(writer: SummaryWriter,
                                  original_features: torch.Tensor, # (B, C, N, W, H) or (B, C, N, H, W)
                                  delta: torch.Tensor,             # (B, C, N, W, H) or (B, C, N, H, W)
                                  adversarial_features: torch.Tensor, # (B, C, N, W, H) or (B, C, N, H, W)
                                  original_decision_map: torch.Tensor, # (B, W, H) or (B, H, W)
                                  adversarial_decision_map: torch.Tensor, # (B, W, H) or (B, H, W)
                                  step: int,
                                  num_samples=4,
                                  sequence_step_to_vis=0,
                                  visualize_decision_diff=True):
    """
    可视化原始特征、扰动、对抗特征、原始决策图和对抗决策图。
    将图像记录到 TensorBoard。
    Args:
        writer (SummaryWriter): TensorBoard SummaryWriter 对象。
        original_features (torch.Tensor): 原始特征层 (B, C, N, W, H) 或 (B, C, N, H, W)。
        delta (torch.Tensor): 生成的扰动 (B, C, N, W, H) 或 (B, C, N, H, W)。
        adversarial_features (torch.Tensor): 对抗性特征层 (B, C, N, W, H) 或 (B, C, N, H, W)。
        original_decision_map (torch.Tensor): 原始样本下的 ATN 决策图 (B, W, H) 或 (B, H, W)。
        adversarial_decision_map (torch.Tensor): 对抗样本下的 ATN 决策图 (B, W, H) 或 (B, H, W)。
        step (int): 当前训练步数。
        num_samples (int): 可视化多少个样本。
        sequence_step_to_vis (int): 可视化特征/扰动的哪个序列步骤。
        visualize_decision_diff (bool): 是否可视化决策图的差异。默认为 True。
    """
    if original_features is None or original_features.numel() == 0:
        print("Warning: Original features are None or empty. Skipping sample visualization.")
        return

    num_samples_to_show = min(num_samples, original_features.shape[0])
    if num_samples_to_show == 0: return

    # 将张量移动到 CPU (如果不在 CPU 上) 并选取要显示的样本
    original_features_cpu = original_features[:num_samples_to_show].cpu()
    delta_cpu = delta[:num_samples_to_show].cpu() if delta is not None else None
    adversarial_features_cpu = adversarial_features[:num_samples_to_show].cpu()

    # 可视化特征层 (选择一个序列步骤)
    # feature_to_visualizable expects (B, C, N, H, W) or (B, C, N, W, H)
    vis_orig_feat = feature_to_visualizable(original_features_cpu, sequence_step=sequence_step_to_vis) # (B, C', H, W) or (B, 1, H, W)
    vis_delta = feature_to_visualizable(delta_cpu, sequence_step=sequence_step_to_vis) if delta_cpu is not None else torch.empty(0)
    vis_adv_feat = feature_to_visualizable(adversarial_features_cpu, sequence_step=sequence_step_to_vis)

    # Ensure feature visualizations are not empty before creating grids
    if vis_orig_feat.numel() > 0:
         # Ensure it's 4D (B, C', H, W) or (B, 1, H, W) for make_grid
         if vis_orig_feat.ndim == 3: # (B, H, W) - should not happen if feature_to_visualizable works as expected
              vis_orig_feat = vis_orig_feat.unsqueeze(1)
         grid_original_features = make_grid(vis_orig_feat, nrow=num_samples_to_show, normalize=True)
         writer.add_image(f'Samples/Original_Features_Seq{sequence_step_to_vis}', grid_original_features, step)

    if vis_delta.numel() > 0:
         if vis_delta.ndim == 3:
              vis_delta = vis_delta.unsqueeze(1)
         grid_delta = make_grid(vis_delta, nrow=num_samples_to_show, normalize=True)
         writer.add_image(f'Samples/Perturbation_Delta_Seq{sequence_step_to_vis}', grid_delta, step)

    if vis_adv_feat.numel() > 0:
         if vis_adv_feat.ndim == 3:
              vis_adv_feat = vis_adv_feat.unsqueeze(1)
         grid_adversarial_features = make_grid(vis_adv_feat, nrow=num_samples_to_show, normalize=True)
         writer.add_image(f'Samples/Adversarial_Features_Seq{sequence_step_to_vis}', grid_adversarial_features, step)


    # 可视化 ATN 决策图 (B, W, H) 或 (B, H, W)
    if original_decision_map is not None and adversarial_decision_map is not None:
        assert original_decision_map.shape == adversarial_decision_map.shape, "Decision map shapes mismatch"
        # Assuming decision maps are (B, H, W) based on decision_layer output in mock ATN
        original_decision_map_cpu = original_decision_map[:num_samples_to_show].cpu()
        adversarial_decision_map_cpu = adversarial_decision_map[:num_samples_to_show].cpu()

        # 将决策图转换为适合可视化的格式 (B, 1, H, W)
        if original_decision_map_cpu.ndim == 3: # (B, H, W)
            original_decision_img = original_decision_map_cpu.unsqueeze(1)
            adversarial_decision_img = adversarial_decision_map_cpu.unsqueeze(1)
        elif original_decision_map_cpu.ndim == 4 and original_decision_map_cpu.shape[1] == 1: # Already (B, 1, H, W)
            original_decision_img = original_decision_map_cpu
            adversarial_decision_img = adversarial_decision_map_cpu
        else:
            print(f"Warning: Unexpected decision map shape {original_decision_map_cpu.shape}. Skipping decision map visualization.")
            original_decision_img = None
            adversarial_decision_img = None


        if original_decision_img is not None and adversarial_decision_img is not None:
             # 可视化原始和对抗决策图
            combined_decision_maps = torch.cat((original_decision_img, adversarial_decision_img), dim=0)
            # Normalize across the combined set for consistent color scale
            grid_decision = make_grid(combined_decision_maps, nrow=num_samples_to_show, normalize=True)
            writer.add_image('ATN_Outputs/Decision_Maps (Original_Top vs Adversarial_Bottom)', grid_decision, step)

            # 可视化决策图差异
            if visualize_decision_diff:
                 decision_diff = adversarial_decision_map_cpu - original_decision_map_cpu # (B, H, W)
                 # Ensure diff is 4D for make_grid
                 if decision_diff.ndim == 3:
                      decision_diff = decision_diff.unsqueeze(1) # (B, 1, H, W)

                 # Use a diverging colormap for differences, centered at 0
                 # Need to normalize manually for a centered colormap, make_grid normalize=True is [0, 1]
                 # Convert to numpy for matplotlib visualization, or normalize manually for make_grid
                 # Let's normalize manually to [-1, 1] then map to [0, 1] for make_grid
                 max_abs_diff = torch.max(torch.abs(decision_diff)) if decision_diff.numel() > 0 else 1.0
                 # Avoid division by zero
                 if max_abs_diff > 1e-8:
                      normalized_diff = decision_diff / max_abs_diff # Range [-1, 1]
                      normalized_diff = (normalized_diff + 1.0) / 2.0 # Map [-1, 1] to [0, 1]
                 else:
                      normalized_diff = torch.zeros_like(decision_diff) # All diffs are zero

                 grid_decision_diff = make_grid(normalized_diff, nrow=num_samples_to_show, normalize=True) # make_grid normalize=True will rescale to [0, 1] again
                 writer.add_image('ATN_Outputs/Decision_Maps_Difference (Adversarial - Original)', grid_decision_diff, step)

        elif original_decision_map is not None: # Only original decision map available
             original_decision_map_cpu = original_decision_map[:num_samples_to_show].cpu()
             if original_decision_map_cpu.ndim == 3:
                  original_decision_img = original_decision_map_cpu.unsqueeze(1)
             else:
                  original_decision_img = original_decision_map_cpu
             grid_orig_decision = make_grid(original_decision_img, nrow=num_samples_to_show, normalize=True)
             writer.add_image('ATN_Outputs/Original_Decision_Map', grid_orig_decision, step)
        elif adversarial_decision_img is not None: # Only adversarial decision map available
            adversarial_decision_map_cpu = adversarial_decision_map[:num_samples_to_show].cpu()
            if adversarial_decision_map_cpu.ndim == 3:
                 adversarial_decision_img = adversarial_decision_map_cpu.unsqueeze(1)
            else:
                 adversarial_decision_img = adversarial_decision_map_cpu
            grid_adv_decision = make_grid(adversarial_decision_img, nrow=num_samples_to_show, normalize=True)
            writer.add_image('ATN_Outputs/Adversarial_Decision_Map', grid_adv_decision, step)


# 可视化 ATN 注意力矩阵热力图
def visualize_attention_maps(writer: SummaryWriter,
                             attention_matrix_orig: torch.Tensor, # (B, head, N, N)
                             attention_matrix_adv: torch.Tensor, # (B, head, N, N)
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

    # Ensure at least one attention matrix is not None and not empty
    if (attention_matrix_orig is None or attention_matrix_orig.numel() == 0) and \
       (attention_matrix_adv is None or attention_matrix_adv.numel() == 0):
        print("Warning: Attention matrices are None or empty. Skipping attention map visualization.")
        return

    # Determine batch size and number of heads from available tensors
    B, head, N1, N2 = 0, 0, 0, 0
    if attention_matrix_orig is not None and attention_matrix_orig.numel() > 0:
        assert attention_matrix_orig.ndim == 4, f"Original attention matrix must be 4D (B, head, N, N), got {attention_matrix_orig.ndim}D"
        B, head, N1, N2 = attention_matrix_orig.shape
    elif attention_matrix_adv is not None and attention_matrix_adv.numel() > 0:
        assert attention_matrix_adv.ndim == 4, f"Adversarial attention matrix must be 4D (B, head, N, N), got {attention_matrix_adv.ndim}D"
        B, head, N1, N2 = attention_matrix_adv.shape

    if B == 0 or head == 0 or N1 == 0 or N2 == 0:
         print("Warning: Attention matrix dimensions are zero. Skipping attention map visualization.")
         return

    assert N1 == N2, "Attention matrix is expected to be square (N x N)"
    num_samples_to_show_attn = min(num_samples, B)
    num_heads_total = head
    num_heads_to_show = min(num_heads_to_vis, num_heads_total)

    # Convert to numpy for matplotlib
    attention_matrix_orig_np = attention_matrix_orig.cpu().detach().numpy() if attention_matrix_orig is not None else None
    attention_matrix_adv_np = attention_matrix_adv.cpu().detach().numpy() if attention_matrix_adv is not None else None


    for i in range(num_samples_to_show_attn):
        for h_idx in range(num_heads_to_show):
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Determine a common color scale range
            vmin, vmax = 0, 1 # Assume attention is normalized between 0 and 1
            # If you want to dynamically determine range from data:
            # all_attn_values = []
            # if attention_matrix_orig_np is not None: all_attn_values.append(attention_matrix_orig_np[i, h_idx])
            # if attention_matrix_adv_np is not None: all_attn_values.append(attention_matrix_adv_np[i, h_idx])
            # if all_attn_values:
            #     all_attn_values = np.concatenate([arr.flatten() for arr in all_attn_values])
            #     vmin, vmax = np.min(all_attn_values), np.max(all_attn_values)
            # if vmax - vmin < 1e-8: vmin, vmax = 0, 1 # Handle flat data

            # Original Attention Heatmap
            if attention_matrix_orig_np is not None:
                attn_map_orig = attention_matrix_orig_np[i, h_idx, :, :]
                im_orig = axes[0].imshow(attn_map_orig, cmap='hot', interpolation='nearest', vmin=vmin, vmax=vmax)
                axes[0].set_title(f'Sample {i} Head {h_idx} Orig Attn')
                axes[0].set_xlabel('Key Sequence Step')
                axes[0].set_ylabel('Query Sequence Step')
                fig.colorbar(im_orig, ax=axes[0])
            else:
                axes[0].text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=12)
                axes[0].set_title(f'Sample {i} Head {h_idx} Orig Attn')
                axes[0].axis('off') # Hide axes for N/A plot

            # Adversarial Attention Heatmap
            if attention_matrix_adv_np is not None:
                attn_map_adv = attention_matrix_adv_np[i, h_idx, :, :]
                im_adv = axes[1].imshow(attn_map_adv, cmap='hot', interpolation='nearest', vmin=vmin, vmax=vmax)
                axes[1].set_title(f'Sample {i} Head {h_idx} Adv Attn')
                axes[1].set_xlabel('Key Sequence Step')
                axes[1].set_ylabel('Query Sequence Step')
                fig.colorbar(im_adv, ax=axes[1])
            else:
                axes[1].text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=12)
                axes[1].set_title(f'Sample {i} Head {h_idx} Adv Attn')
                axes[1].axis('off') # Hide axes for N/A plot

            # Optional: Visualize Attention Difference Heatmap
            if attention_matrix_orig_np is not None and attention_matrix_adv_np is not None:
                 attention_diff = attn_map_adv - attn_map_orig
                 fig_diff, ax_diff = plt.subplots(figsize=(6, 5))
                 # Use a diverging colormap for differences, centered at 0
                 max_abs_diff = np.max(np.abs(attention_diff)) if decision_diff.size > 0 else 1.0
                 # Avoid division by zero for color scale
                 vmax_diff = max(max_abs_diff, 1e-8)
                 im_diff = ax_diff.imshow(attention_diff, cmap='coolwarm', interpolation='nearest', vmin=-vmax_diff, vmax=vmax_diff)
                 ax_diff.set_title(f'Sample {i} Head {h_idx} Attn Diff (Adv - Orig)')
                 ax_diff.set_xlabel('Key Sequence Step')
                 ax_diff.set_ylabel('Query Sequence Step')
                 fig_diff.colorbar(im_diff, ax=ax_diff)
                 writer.add_figure(f'ATN_Outputs/Attention_Difference/Sample{i}_Head{h_idx}', fig_diff, step)
                 plt.close(fig_diff) # Close the figure to free memory

            writer.add_figure(f'ATN_Outputs/Attention/Sample{i}_Head{h_idx}', fig, step)
            plt.close(fig) # Close the figure to free memory

# 可视化函数
def visualize_training_losses(writer: SummaryWriter, loss_dict: dict, step: int):
    """
    将训练损失记录到 TensorBoard。
    Args:
        writer (SummaryWriter): TensorBoard SummaryWriter 对象。
        loss_dict (dict): 包含各种损失的字典。
        step (int): 当前训练步数。
    """
    for name, loss_val in loss_dict.items():
        if isinstance(loss_val, torch.Tensor):
            # Check if loss_val is a scalar tensor
            if loss_val.ndim == 0:
                writer.add_scalar(f'Loss/{name}', loss_val.item(), step)
            else:
                 # If it's not a scalar, maybe log its mean or a warning
                 # print(f"Warning: Loss tensor '{name}' is not scalar ({loss_val.shape}). Logging mean.")
                 writer.add_scalar(f'Loss/{name}', loss_val.mean().item(), step) # Log mean if not scalar

        else:
            writer.add_scalar(f'Loss/{name}', loss_val, step)

def visualize_perturbation_norms(writer: SummaryWriter, delta: torch.Tensor, step: int):
    """
    将扰动范数记录到 TensorBoard。
    Args:
        writer (SummaryWriter): TensorBoard SummaryWriter 对象。
        delta (torch.Tensor): 生成的扰动 (B, C, N, W, H) 或 (B, C, N, H, W)。
        step (int): 当前训练步数。
    """
    if delta is None or delta.numel() == 0:
        print("Warning: Delta tensor is None or empty. Skipping perturbation norm visualization.")
        return

    with torch.no_grad():
        # Assuming delta is (B, C, N, H, W) or (B, C, N, W, H)
        if delta.ndim != 5:
             print(f"Warning: Delta tensor is not 5D ({delta.ndim}D). Cannot calculate perturbation norms.")
             return

        linf = linf_norm(delta).mean().item() # linf_norm returns (B,), take mean
        l2 = l2_norm(delta).mean().item() # l2_norm returns (B,), take mean
        writer.add_scalar('Perturbation/Linf_Norm', linf, step)
        writer.add_scalar('Perturbation/L2_Norm', l2, step)

def visualize_samples_and_outputs(writer: SummaryWriter,
                                  original_features: torch.Tensor, # (B, C, N, W, H) or (B, C, N, H, W)
                                  delta: torch.Tensor,             # (B, C, N, W, H) or (B, C, N, H, W)
                                  adversarial_features: torch.Tensor, # (B, C, N, W, H) or (B, C, N, H, W)
                                  original_decision_map: torch.Tensor, # (B, W, H) or (B, H, W)
                                  adversarial_decision_map: torch.Tensor, # (B, W, H) or (B, H, W)
                                  step: int,
                                  num_samples=4,
                                  sequence_step_to_vis=0,
                                  visualize_decision_diff=True):
    """
    可视化原始特征、扰动、对抗特征、原始决策图和对抗决策图。
    将图像记录到 TensorBoard。
    Args:
        writer (SummaryWriter): TensorBoard SummaryWriter 对象。
        original_features (torch.Tensor): 原始特征层 (B, C, N, W, H) 或 (B, C, N, H, W)。
        delta (torch.Tensor): 生成的扰动 (B, C, N, W, H) 或 (B, C, N, H, W)。
        adversarial_features (torch.Tensor): 对抗性特征层 (B, C, N, W, H) 或 (B, C, N, H, W)。
        original_decision_map (torch.Tensor): 原始样本下的 ATN 决策图 (B, W, H) 或 (B, H, W)。
        adversarial_decision_map (torch.Tensor): 对抗样本下的 ATN 决策图 (B, W, H) 或 (B, H, W)。
        step (int): 当前训练步数。
        num_samples (int): 可视化多少个样本。
        sequence_step_to_vis (int): 可视化特征/扰动的哪个序列步骤。
        visualize_decision_diff (bool): 是否可视化决策图的差异。默认为 True。
    """
    if original_features is None or original_features.numel() == 0:
        print("Warning: Original features are None or empty. Skipping sample visualization.")
        return

    num_samples_to_show = min(num_samples, original_features.shape[0])