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
def feature_to_visualizable(features: torch.Tensor, sequence_step=0, mode='first_3_channels', channel_subset=None):
    """
    将 (B, C, N, H, W) 特征张量转换为可可视化格式 (B, C', H, W)。
    可以根据 mode 选择可视化方式。

    Args:
        features (torch.Tensor): 输入特征张量 (B, C, N, H, W)。
        sequence_step (int): 选择可视化哪个序列步骤。
        mode (str): 可视化模式。可选：
                    'first_3_channels': 选取前 3 个通道 (如果 C >= 3)，不足则重复第一通道。
                    'l2_norm': 计算空间维度上每个点的 L2 范数 (B, 1, H, W)。
                    'mean': 计算空间维度上每个点的通道平均值 (B, 1, H, W)。
                    'channel_subset': 可视化指定通道子集 (B, num_channels, H, W)。
        channel_subset (list, optional): 如果 mode 为 'channel_subset'，指定要可视化的通道索引列表。
    Returns:
        torch.Tensor: 可视化格式的张量 (B, C', H, W)。
                       返回的张量已归一化到 [0, 1]。
    """
    if features is None or features.numel() == 0:
        # print("Warning: Input features are None or empty. Cannot visualize features.") # Avoid excessive prints
        return torch.empty(0) # Return empty tensor

    # Assuming (B, C, N, H, W)
    if features.ndim != 5:
         print(f"Warning: Features tensor must be 5D (B, C, N, H, W), got {features.ndim}D. Cannot visualize features.")
         return torch.empty(0)

    B, C, N, H, W = features.shape
    
    if N == 0:
        # print("Warning: Sequence dimension N is 0. Cannot visualize features.") # Avoid excessive prints
        return torch.empty(0)

    if sequence_step >= N or sequence_step < 0:
        # print(f"Warning: sequence_step {sequence_step} is out of bounds for N={N}. Defaulting to 0.") # Avoid excessive prints
        sequence_step = 0 # 默认可视化第一个步骤

    # 选取指定序列步骤
    features_step = features[:, :, sequence_step, :, :] # (B, C, H, W)

    if mode == 'first_3_channels':
        if C >= 3:
            visualizable_features = features_step[:, :3, :, :] # (B, 3, H, W)
        elif C == 1:
            visualizable_features = features_step.repeat(1, 3, 1, 1) # (B, 1, H, W) -> (B, 3, H, W)
        else:
            # C==2 or other cases < 3, repeat first channel to 3
            if C > 0:
                visualizable_features = features_step[:, 0, :, :].unsqueeze(1).repeat(1, 3, 1, 1) # (B, 3, H, W)
            else: # C == 0
                print("Warning: Feature channels C=0. Cannot create visualizable features for 'first_3_channels'.")
                return torch.empty(0)

    elif mode == 'l2_norm':
        # 计算空间维度上每个点的 L2 范数 across channels
        l2_norm_map = torch.linalg.norm(features_step, dim=1, keepdim=True) # (B, 1, H, W)
        visualizable_features = l2_norm_map

    elif mode == 'mean':
        # 计算空间维度上每个点的通道平均值 across channels
        mean_map = torch.mean(features_step, dim=1, keepdim=True) # (B, 1, H, W)
        visualizable_features = mean_map

    elif mode == 'channel_subset':
        if channel_subset is None or not channel_subset:
            print("Warning: 'channel_subset' mode requires 'channel_subset' list. Using 'l2_norm' mode instead.")
            return feature_to_visualizable(features, sequence_step, mode='l2_norm')

        valid_channels = [c for c in channel_subset if 0 <= c < C]
        if not valid_channels:
            print(f"Warning: None of the specified channel subset {channel_subset} are valid for C={C}. Cannot visualize channel subset.")
            return torch.empty(0)

        visualizable_features = features_step[:, valid_channels, :, :] # (B, num_subset_channels, H, W)

        # 如果子集只有一个通道，unsqueeze使其成为 (B, 1, H, W)
        if visualizable_features.shape[1] == 1:
            pass # Already (B, 1, H, W) if keepdim=True in a single-channel operation, but here we sliced.
            # Ensure it's 4D for make_grid if it becomes 3D after slicing a single channel batch
            if visualizable_features.ndim == 3:
                visualizable_features = visualizable_features.unsqueeze(1) # (B, 1, H, W)
        # 如果子集通道数是2，重复第一通道使其成为3通道
        elif visualizable_features.shape[1] == 2:
            print("Warning: Channel subset has 2 channels. Repeating the first channel to make it 3 for visualization.")
            visualizable_features = torch.cat([visualizable_features, visualizable_features[:, :1, :, :]], dim=1) # (B, 3, H, W)
        # 如果子集通道数 >= 3，取前三个进行RGB可视化
        elif visualizable_features.shape[1] > 3:
            print(f"Warning: Channel subset has {visualizable_features.shape[1]} channels. Taking the first 3 for visualization.")
            visualizable_features = visualizable_features[:, :3, :, :] # (B, 3, H, W)


    else:
        print(f"Warning: Unsupported visualization mode: {mode}. Choose 'first_3_channels', 'l2_norm', 'mean', or 'channel_subset'. Using 'l2_norm' instead.")
        return feature_to_visualizable(features, sequence_step, mode='l2_norm')


    # 将值缩放到 [0, 1] 范围以便显示
    if visualizable_features.numel() == 0:
        return torch.empty(0)

    # For 'l2_norm' and 'mean' modes, it's often better to normalize across the *batch*
    # to show relative intensity within the batch.
    # For 'first_3_channels' or 'channel_subset', batch normalization might distort color perception.
    # Let's normalize each sample independently for 'first_3_channels' and 'channel_subset'.
    # Normalize across the batch for 'l2_norm' and 'mean'.

    if mode in ['l2_norm', 'mean']:
        min_val = visualizable_features.min()
        max_val = visualizable_features.max()
        if max_val > min_val:
            visualizable_features = (visualizable_features - min_val) / (max_val - min_val + 1e-8)
        else:
            visualizable_features = torch.zeros_like(visualizable_features)
    else: # 'first_3_channels', 'channel_subset'
        # Normalize each sample independently
        normalized_batch = []
        for img in visualizable_features: # img shape (C', H, W)
            min_val = img.min()
            max_val = img.max()
            if max_val > min_val:
                normalized_img = (img - min_val) / (max_val - min_val + 1e-8)
            else:
                normalized_img = torch.zeros_like(img)
            normalized_batch.append(normalized_img)
        visualizable_features = torch.stack(normalized_batch, dim=0)


    return visualizable_features.float()


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
                 # print(f"Warning: Loss tensor '{name}' is not scalar ({loss_val.shape}). Logging mean.") # Avoid excessive prints
                 writer.add_scalar(f'Loss/{name}', loss_val.mean().item(), step) # Log mean if not scalar

        else:
            writer.add_scalar(f'Loss/{name}', loss_val, step)

def visualize_perturbation_norms(writer: SummaryWriter, delta: torch.Tensor, step: int):
    """
    将扰动范数记录到 TensorBoard。
    Args:
        writer (SummaryWriter): TensorBoard SummaryWriter 对象。
        delta (torch.Tensor): 生成的扰动 (B, C, N, H, W)。
        step (int): 当前训练步数。
    """
    if delta is None or delta.numel() == 0:
        # print("Warning: Delta tensor is None or empty. Skipping perturbation norm visualization.") # Avoid excessive prints
        return

    with torch.no_grad():
        # Assuming delta is (B, C, N, H, W)
        if delta.ndim != 5:
             print(f"Warning: Delta tensor is not 5D ({delta.ndim}D). Cannot calculate perturbation norms.")
             return

        # Norms are calculated over C, N, H, W dimensions, leaving batch dimension
        # linf_norm returns (B,), l2_norm returns (B,)
        # We take the mean over the batch dimension for logging a single scalar
        linf = linf_norm(delta).mean().item()
        l2 = l2_norm(delta).mean().item()

        writer.add_scalar('Perturbation/Linf_Norm', linf, step)
        writer.add_scalar('Perturbation/L2_Norm', l2, step)

def visualize_samples_and_outputs(writer: SummaryWriter,
                                  original_image: torch.Tensor,   # 原始图像 (B, C, T, H, W)
                                  original_features: torch.Tensor, # 原始特征 (B, C, N, H, W)
                                  feature_delta: torch.Tensor,    # 特征扰动 (B, C, N, H, W)
                                  adversarial_features: torch.Tensor, # 对抗特征 (B, C, N, H, W)
                                  original_decision_map: torch.Tensor, # 原始决策图 (B, H, W) 或 (B, W, H)
                                  adversarial_decision_map: torch.Tensor, # 对抗决策图 (B, H, W) 或 (B, W, H)
                                  step: int,
                                  num_samples=4,
                                  sequence_step_to_vis=0,
                                  visualize_decision_diff=True):
    """
    可视化原始图像、原始特征、特征扰动、对抗特征、原始决策图和对抗决策图。
    将图像记录到 TensorBoard。

    在 Stage 1 (特征攻击) 中，我们主要可视化：
    - 原始图像 (可选)
    - 原始特征 (L2 Norm, Histograms, Channel Subsets)
    - 特征扰动 Delta (L2 Norm, Histograms, Channel Subsets)
    - 对抗性特征 (L2 Norm, Histograms, Channel Subsets)
    - 对抗性特征 与 原始特征 的差异 (L2 Norm)

    Args:
        writer (SummaryWriter): TensorBoard SummaryWriter 对象。
        original_image (torch.Tensor): 原始图像张量 (B, C, T, H, W)。
        original_features (torch.Tensor): 原始特征张量 (B, C, N, H, W)。
        feature_delta (torch.Tensor): 生成的特征扰动 (B, C, N, H, W)。
        adversarial_features (torch.Tensor): 对抗性特征层 (B, C, N, H, W)。
        original_decision_map (torch.Tensor): 原始样本下的 ATN 决策图 (B, H, W) 或 (B, W, H)。
        adversarial_decision_map (torch.Tensor): 对抗样本下的 ATN 决策图 (B, H, W) 或 (B, W, H)。
        step (int): 当前训练步数。
        num_samples (int): 可视化多少个样本。
        sequence_step_to_vis (int): 可视化哪个序列步骤。
        visualize_decision_diff (bool): 是否可视化决策图的差异。默认为 True (主要用于 Stage 2)。
    """
    # print(f"Debug: visualize_samples_and_outputs called at step {step}") # Debug print

    # Determine batch size from available tensors
    batch_sizes = [t.shape[0] for t in [original_image, original_features, feature_delta, adversarial_features, original_decision_map, adversarial_decision_map] if t is not None and t.numel() > 0]
    num_samples_to_show = min(num_samples, batch_sizes[0] if batch_sizes else 0)
    if num_samples_to_show == 0:
        # print("Warning: No valid input tensors with batch dimension. Skipping sample visualization.") # Avoid excessive prints
        return

    # 将张量移动到 CPU (如果不在 CPU 上) 并选取要显示的样本
    original_image_cpu = original_image[:num_samples_to_show].cpu().detach() if original_image is not None else None
    original_features_cpu = original_features[:num_samples_to_show].cpu().detach() if original_features is not None and original_features.numel() > 0 else torch.empty(0, device='cpu')
    feature_delta_cpu = feature_delta[:num_samples_to_show].cpu().detach() if feature_delta is not None and feature_delta.numel() > 0 else torch.empty(0, device='cpu')
    adversarial_features_cpu = adversarial_features[:num_samples_to_show].cpu().detach() if adversarial_features is not None and adversarial_features.numel() > 0 else torch.empty(0, device='cpu')
    original_decision_map_cpu = original_decision_map[:num_samples_to_show].cpu().detach() if original_decision_map is not None else None
    adversarial_decision_map_cpu = adversarial_decision_map[:num_samples_to_show].cpu().detach() if adversarial_decision_map is not None else None


    # --- Visualize Original Image ---
    # original_image_cpu shape: (B, C, T, H, W)
    if original_image_cpu is not None and original_image_cpu.ndim == 5 and original_image_cpu.shape[2] > 0:
        B_img, C_img, T_img, H_img, W_img = original_image_cpu.shape
        vis_image_step = min(sequence_step_to_vis, T_img - 1)
        img_to_vis = original_image_cpu[:, :, vis_image_step, :, :] # (B, C, H, W)

        # Ensure image has 3 channels (repeat if grayscale)
        if C_img == 1:
            img_to_vis = img_to_vis.repeat(1, 3, 1, 1)
        elif C_img > 3:
             img_to_vis = img_to_vis[:, :3, :, :]

        # Ensure image is in [0, 1] range (assuming it was normalized)
        grid_original_image = make_grid(img_to_vis, nrow=num_samples_to_show, normalize=True)
        writer.add_image(f'Samples/Original_Image_Seq{vis_image_step}', grid_original_image, step)
    # else: print("Warning: Original image tensor not available or in unexpected format. Skipping image visualization.") # Avoid excessive prints


    # --- Log Histograms ---
    if original_features_cpu.numel() > 0:
        writer.add_histogram(f'Histograms/Original_Features_Seq{sequence_step_to_vis}', original_features_cpu[:, :, sequence_step_to_vis, :, :].flatten(), step)
    if feature_delta_cpu.numel() > 0:
        writer.add_histogram(f'Histograms/Feature_Perturbation_Delta_Seq{sequence_step_to_vis}', feature_delta_cpu[:, :, sequence_step_to_vis, :, :].flatten(), step)
    if adversarial_features_cpu.numel() > 0:
        writer.add_histogram(f'Histograms/Adversarial_Features_Seq{sequence_step_to_vis}', adversarial_features_cpu[:, :, sequence_step_to_vis, :, :].flatten(), step)


    # --- Visualize Original Features (L2 Norm) ---
    if original_features_cpu.numel() > 0:
        vis_orig_feat = feature_to_visualizable(original_features_cpu, # Removed unsqueeze(2)
                                                sequence_step=sequence_step_to_vis, # Use sequence_step_to_vis
                                                mode='l2_norm') # (B, 1, H', W')

        if vis_orig_feat.numel() > 0:
             if vis_orig_feat.ndim == 3: vis_orig_feat = vis_orig_feat.unsqueeze(1)
             grid_orig_feat = make_grid(vis_orig_feat, nrow=num_samples_to_show, normalize=True)
             writer.add_image(f'Samples/Original_Features_L2Norm_Seq{sequence_step_to_vis}', grid_orig_feat, step)
        # else: print("Warning: feature_to_visualizable returned empty tensor for original features L2 Norm.") # Avoid excessive prints


    # --- Visualize Feature Perturbation (Delta) using L2 Norm ---
    if feature_delta_cpu.numel() > 0:
        vis_delta = feature_to_visualizable(feature_delta_cpu,
                                            sequence_step=sequence_step_to_vis,
                                            mode='l2_norm') # (B, 1, H', W')

        if vis_delta.numel() > 0:
             if vis_delta.ndim == 3: vis_delta = vis_delta.unsqueeze(1)
             grid_delta = make_grid(vis_delta, nrow=num_samples_to_show, normalize=True)
             writer.add_image(f'Samples/Feature_Perturbation_Delta_L2Norm_Seq{sequence_step_to_vis}', grid_delta, step)
        # else: print("Warning: feature_to_visualizable returned empty tensor for feature delta L2 Norm.") # Avoid excessive prints


    # --- Visualize Adversarial Features using L2 Norm ---
    if adversarial_features_cpu.numel() > 0:
        vis_adv_feat = feature_to_visualizable(adversarial_features_cpu,
                                             sequence_step=sequence_step_to_vis,
                                             mode='l2_norm') # (B, 1, H', W')

        if vis_adv_feat.numel() > 0:
             if vis_adv_feat.ndim == 3: vis_adv_feat = vis_adv_feat.unsqueeze(1)
             grid_adversarial_features = make_grid(vis_adv_feat, nrow=num_samples_to_show, normalize=True)
             writer.add_image(f'Samples/Adversarial_Features_L2Norm_Seq{sequence_step_to_vis}', grid_adversarial_features, step)
        # else: print("Warning: feature_to_visualizable returned empty tensor for adversarial features L2 Norm.") # Avoid excessive prints


    # --- Visualize Feature Difference L2 Norm ---
    # original_features_cpu and adversarial_features_cpu must be available and have compatible shapes
    if original_features_cpu.numel() > 0 and adversarial_features_cpu.numel() > 0 and \
       original_features_cpu.shape == adversarial_features_cpu.shape:

       # Select the specified sequence step for difference calculation
       # features_step shape (B, C, H, W)
       # Ensure sequence_step_to_vis is valid for feature tensors (shape (B, C, N, H, W))
       feat_sequence_length = original_features_cpu.shape[2]
       vis_feat_step = min(sequence_step_to_vis, feat_sequence_length - 1)
       if feat_sequence_length == 0: vis_feat_step = 0 # Handle empty sequence


       original_features_step = original_features_cpu[:, :, vis_feat_step, :, :] # (B, C, H, W)
       adversarial_features_step = adversarial_features_cpu[:, :, vis_feat_step, :, :] # (B, C, H, W)

       # Calculate difference
       feature_difference = adversarial_features_step - original_features_step # (B, C, H, W)

       # Calculate L2 norm of the difference across channels
       feature_difference_l2_norm = torch.linalg.norm(feature_difference, dim=1, keepdim=True) # (B, 1, H, W)

       # Normalize for visualization
       min_val = feature_difference_l2_norm.min()
       max_val = feature_difference_l2_norm.max()

       if max_val > min_val:
           vis_feature_diff_l2_norm = (feature_difference_l2_norm - min_val) / (max_val - min_val + 1e-8)
       else:
           vis_feature_diff_l2_norm = torch.zeros_like(feature_difference_l2_norm)

       grid_feature_diff_l2_norm = make_grid(vis_feature_diff_l2_norm, nrow=num_samples_to_show, normalize=True)
       writer.add_image(f'Samples/Feature_Difference_L2Norm_Seq{vis_feat_step}', grid_feature_diff_l2_norm, step)
    # else: print("Warning: Original or adversarial features not available or shapes mismatch. Skipping feature difference visualization.") # Avoid excessive prints


    # --- Visualize Important Channel Subsets ---
    if original_features_cpu.numel() > 0 and adversarial_features_cpu.numel() > 0:
        # Assuming 128 channels
        all_channels = original_features_cpu.shape[1]
        if all_channels > 0:
            # Define channel subsets to visualize
            # You can adjust these lists based on your observations
            channel_subsets_to_vis = {
                'First_3_Channels': [0, 1, 2],
                'Mid_Channels_10_50_100': [10, 50, 100] if all_channels > 100 else ([10, 50] if all_channels > 50 else ([10] if all_channels > 10 else [0])),
                'Last_3_Channels': [all_channels - 3, all_channels - 2, all_channels - 1] if all_channels >= 3 else ([all_channels - 2, all_channels - 1] if all_channels == 2 else ([all_channels - 1] if all_channels == 1 else []))
            }
            # Filter out empty lists
            channel_subsets_to_vis = {name: subset for name, subset in channel_subsets_to_vis.items() if subset}

            for subset_name, channel_subset in channel_subsets_to_vis.items():
                # Visualize original features subset
                vis_orig_subset = feature_to_visualizable(original_features_cpu,
                                                          sequence_step=sequence_step_to_vis,
                                                          mode='channel_subset',
                                                          channel_subset=channel_subset)
                if vis_orig_subset.numel() > 0:
                     grid_orig_subset = make_grid(vis_orig_subset, nrow=num_samples_to_show, normalize=True)
                     writer.add_image(f'Channel_Subsets/Original_Features/{subset_name}_Seq{sequence_step_to_vis}', grid_orig_subset, step)

                # Visualize adversarial features subset
                vis_adv_subset = feature_to_visualizable(adversarial_features_cpu,
                                                         sequence_step=sequence_step_to_vis,
                                                         mode='channel_subset',
                                                         channel_subset=channel_subset)
                if vis_adv_subset.numel() > 0:
                     grid_adv_subset = make_grid(vis_adv_subset, nrow=num_samples_to_show, normalize=True)
                     writer.add_image(f'Channel_Subsets/Adversarial_Features/{subset_name}_Seq{sequence_step_to_vis}', grid_adv_subset, step)

    # --- Visualize ATN Decision Maps (only in Stage 2) ---
    # This part remains largely the same, and should only run if decision maps are provided (e.g., in Stage 2)
    if original_decision_map_cpu is not None and adversarial_decision_map_cpu is not None:
        # print("Visualizing decision maps...")
        assert original_decision_map_cpu.shape == adversarial_decision_map_cpu.shape, "Decision map shapes mismatch"
        # Assuming decision maps are (B, H, W) or (B, W, H)
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
            grid_decision = make_grid(combined_decision_maps, nrow=num_samples_to_show, normalize=True)
            writer.add_image('ATN_Outputs/Decision_Maps (Original_Top vs Adversarial_Bottom)', grid_decision, step)

            # 可视化决策图差异
            if visualize_decision_diff:
                 decision_diff = adversarial_decision_map_cpu - original_decision_map_cpu # (B, H, W) or (B, 1, H, W)
                 if decision_diff.ndim == 3: decision_diff = decision_diff.unsqueeze(1) # (B, 1, H, W)

                 max_abs_diff = torch.max(torch.abs(decision_diff)) if decision_diff.numel() > 0 else 1.0
                 if max_abs_diff > 1e-8:
                      normalized_diff = decision_diff / max_abs_diff # Range [-1, 1]
                      normalized_diff = (normalized_diff + 1.0) / 2.0 # Map [-1, 1] to [0, 1]
                 else:
                      normalized_diff = torch.zeros_like(decision_diff) # All diffs are zero

                 grid_decision_diff = make_grid(normalized_diff, nrow=num_samples_to_show, normalize=True)
                 writer.add_image('ATN_Outputs/Decision_Map_Difference (Adversarial - Original)', grid_decision_diff, step)

        # else: print("Warning: Decision maps not available or in unexpected format. Skipping decision map visualization.") # Avoid excessive prints

    # visualize_attention_maps (not called here in Stage 1 training)


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
    # Ensure at least one attention matrix is not None and not empty
    if (attention_matrix_orig is None or attention_matrix_orig.numel() == 0) and \
       (attention_matrix_adv is None or attention_matrix_adv.numel() == 0):
        # print("Warning: Attention matrices are None or empty. Skipping attention map visualization.") # Avoid excessive prints
        return
    # print("Generating attention map visualizations...") # Only print if actually generating

    # Determine batch size and number of heads from available tensors
    B, head, N1, N2 = 0, 0, 0, 0
    if attention_matrix_orig is not None and attention_matrix_orig.numel() > 0:
        assert attention_matrix_orig.ndim == 4, f"Original attention matrix must be 4D (B, head, N, N), got {attention_matrix_orig.ndim}D"
        B, head, N1, N2 = attention_matrix_orig.shape
    elif attention_matrix_adv is not None and attention_matrix_adv.numel() > 0:
        assert attention_matrix_adv.ndim == 4, f"Adversarial attention matrix must be 4D (B, head, N, N), got {attention_matrix_adv.ndim}D"
        B, head, N1, N2 = attention_matrix_adv.shape

    if B == 0 or head == 0 or N1 == 0 or N2 == 0:
         # print("Warning: Attention matrix dimensions are zero after checking availability. Skipping attention map visualization.") # Avoid excessive prints
         return

    assert N1 == N2, "Attention matrix is expected to be square (N x N)"
    num_samples_to_show_attn = min(num_samples, B)
    num_heads_total = head
    num_heads_to_show = min(num_heads_to_vis, num_heads_total)

    # Convert to numpy for matplotlib
    # Use .detach().cpu() to handle tensors from GPU and ensure no gradient tracking
    attention_matrix_orig_np = attention_matrix_orig.cpu().detach().numpy() if attention_matrix_orig is not None else None
    attention_matrix_adv_np = attention_matrix_adv.cpu().detach().numpy() if attention_matrix_adv is not None else None


    for i in range(num_samples_to_show_attn):
        for h_idx in range(num_heads_to_show):
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Determine a common color scale range
            vmin, vmax = 0, 1 # Assume attention is normalized between 0 and 1
            # Optional: dynamically determine range from data
            # all_attn_values = []
            # if attention_matrix_orig_np is not None: all_attn_values.extend(attention_matrix_orig_np[i, h_idx].flatten())
            # if attention_matrix_adv_np is not None: all_attn_values.extend(attention_matrix_adv_np[i, h_idx].flatten())
            # if all_attn_values:
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

# Keep other visualization functions if they exist and are used elsewhere, or remove if unused.
# visualize_training_losses(writer: SummaryWriter, loss_dict: dict, step: int): ...
# visualize_perturbation_norms(writer: SummaryWriter, delta: torch.Tensor, step: int): ...