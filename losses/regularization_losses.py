import torch

# 注意：由于 Generator 的设计（Tanh + epsilon 缩放）,
# L-infinity 约束已经通过网络结构强制执行。
# 因此，下面的 L-infinity 范数计算主要用于监控或可能的未来扩展，
# 不一定需要作为惩罚项添加到生成器的总损失中。

def linf_norm(delta):
    """
    计算扰动张量的 L-infinity 范数。
    Args:
        delta (torch.Tensor): 扰动张量 (例如, G(x))。
            可以是任意形状，通常是 (B, C, N, H, W)。
    Returns:
        torch.Tensor: 该批次中每个样本的 L-infinity 范数 (形状 B)。
                       或者整个批次的平均 L-infinity 范数 (标量)。
                       这里返回每个样本的最大值。
    """
    # 在除了 Batch 之外的所有维度上计算绝对值的最大值
    # 保持 batch 维度
    # .view(delta.shape[0], -1) 将 B, C, N, H, W 展平成 B, C*N*H*W
    # .abs() 计算绝对值
    # .max(dim=1)[0] 找到每个样本（在 dim=1 上）的最大值
    return delta.view(delta.shape[0], -1).abs().max(dim=1)[0]

def l2_norm(delta):
    """
    计算扰动张量的 L2 范数。
    Args:
        delta (torch.Tensor): 扰动张量 (例如, G(x))。
            可以是任意形状，通常是 (B, C, N, H, W)。
    Returns:
        torch.Tensor: 该批次中每个样本的 L2 范数 (形状 B)。
    """
    # 在除了 Batch 之外的所有维度上计算平方和的平方根
    return (delta.view(delta.shape[0], -1)**2).sum(dim=1).sqrt()

# --- 可能的损失函数 (如果需要明确惩罚范数) ---

def linf_penalty(delta):
    """
    计算 L-infinity 范数的平均值，可作为损失项。
    """
    return linf_norm(delta).mean()

def l2_penalty(delta):
    """
    计算 L2 范数的平均值，可作为损失项。
    """
    return l2_norm(delta).mean()

def total_variation_loss(delta: torch.Tensor):
    """
    计算扰动张量的 Total Variation (TV) Loss。
    对于 5D 张量 (B, C, N, H, W)，我们通常在空间维度 (H, W) 上计算 TV Loss。
    可以选择是否在通道 (C) 和序列 (N) 维度上也计算。
    这里实现一个在空间维度上计算的简易版本。

    Args:
        delta (torch.Tensor): 扰动张量 (B, C, N, H, W)。

    Returns:
        torch.Tensor: TV Loss 的标量值。
    """
    # 在 H 和 W 方向计算差异并取绝对值
    tv_h = torch.abs(delta[:, :, :, 1:, :] - delta[:, :, :, :-1, :]).sum()
    tv_w = torch.abs(delta[:, :, :, :, 1:] - delta[:, :, :, :, :-1]).sum()

    # 可以选择是否加入序列维度和通道维度的TV Loss
    # tv_n = torch.abs(delta[:, :, 1:, :, :] - delta[:, :, :-1, :, :]).sum()
    # tv_c = torch.abs(delta[:, 1:, :, :, :] - delta[:, :-1, :, :, :]).sum()

    total_tv = tv_h + tv_w # + tv_n + tv_c # 根据需求决定是否包含序列和通道TV

    # 通常会对 TV Loss 进行归一化，例如除以总像素数量或非零像素数量
    # 这里简单除以批次大小和通道数及序列长度，以获得相对平均值
    # 如果追求更精确的归一化，可以除以 delta 中元素的总数 (B*C*N*H*W)
    B, C, N, H, W = delta.shape
    num_elements = B * C * N * H * W
    # 避免除以零
    if num_elements == 0:
        return torch.tensor(0.0, device=delta.device)

    return total_tv / num_elements # 简单的平均化

# 示例用法 (在训练脚本中):
# from losses.regularization_losses import linf_norm, l2_penalty, total_variation_loss
#
# delta = generator(input_batch)
# max_linf = linf_norm(delta) # 监控每张图片的最大扰动, shape (B,)
# print(f"Max L-inf norm in batch: {max_linf.max().item()}")
#
# # 如果需要将 L2 范数加入总损失:
# loss_perturb_l2 = l2_penalty(delta)
#
# # 如果需要将 TV loss 加入总损失:
# loss_perturb_tv = total_variation_loss(delta)
#
# # 总的正则化损失可以是多个项的加权和
# # total_generator_loss += gamma_l2 * loss_perturb_l2 + gamma_tv * loss_perturb_tv