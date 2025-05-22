import torch

# 注意：由于 Generator 的设计（Tanh + epsilon 缩放）,
# L-infinity 约束已经通过网络结构强制执行。
# 因此，下面的 L-infinity 范数计算主要用于监控或可能的未来扩展，
# 不一定需要作为惩罚项添加到生成器的总损失中。

def linf_norm(delta: torch.Tensor) -> torch.Tensor:
    """
    计算扰动张量在除批量维度外所有维度上的 L-infinity 范数。
    
    L-infinity 范数是向量或张量中元素绝对值的最大值，这里计算的是每个样本（批次中的一项）的 L-infinity 范数。

    Args:
        delta (torch.Tensor): 输入的扰动张量。
                             形状通常为 (B, C, N, H, W)，其中 B 是批量大小。
                             支持任意形状，但会展平除第一个维度外的所有维度。

    Returns:
        torch.Tensor: 一个形状为 (B,) 的张量，包含该批次中每个样本的 L-infinity 范数。
                      如果输入张量为空，返回一个形状为 (0,) 的张量。
    """
    # 检查输入张量是否为空
    if delta is None or delta.numel() == 0:
        # 警告：输入张量为空，无法计算 L-infinity 范数。
        print("Warning: Input tensor is None or empty, cannot calculate L-infinity norm.")
        return torch.empty(0, dtype=torch.float32, device=delta.device if delta is not None else 'cpu')

    # 在除了 Batch (dim 0) 之外的所有维度上计算绝对值的最大值
    # reshape(delta.shape[0], -1) 将 B, C, N, H, W 展平成 B, C*N*H*W
    # abs() 计算绝对值
    # max(dim=1)[0] 找到每个样本（在 dim=1 上）的最大值，[0] 获取最大值张量本身
    return delta.reshape(delta.shape[0], -1).abs().max(dim=1)[0]

def l2_norm(delta: torch.Tensor) -> torch.Tensor:
    """
    计算扰动张量在除批量维度外所有维度上的 L2 范数。
    
    L2 范数是向量或张量中元素平方和的平方根，这里计算的是每个样本（批次中的一项）的 L2 范数。

    Args:
        delta (torch.Tensor): 输入的扰动张量。
                             形状通常为 (B, C, N, H, W)，其中 B 是批量大小。
                             支持任意形状，但会展平除第一个维度外的所有维度。

    Returns:
        torch.Tensor: 一个形状为 (B,) 的张量，包含该批次中每个样本的 L2 范数。
                      如果输入张量为空，返回一个形状为 (0,) 的张量。
    """
    # 检查输入张量是否为空
    if delta is None or delta.numel() == 0:
        # 警告：输入张量为空，无法计算 L2 范数。
        print("Warning: Input tensor is None or empty, cannot calculate L2 norm.")
        return torch.empty(0, dtype=torch.float32, device=delta.device if delta is not None else 'cpu')

    # 在除了 Batch (dim 0) 之外的所有维度上计算平方和的平方根
    # reshape(delta.shape[0], -1) 将 B, C, N, H, W 展平成 B, C*N*H*W
    # **2 计算平方
    # sum(dim=1) 计算每个样本（在 dim=1 上）的平方和
    # sqrt() 计算平方根
    return (delta.reshape(delta.shape[0], -1)**2).sum(dim=1).sqrt()

# --- 可能的损失函数 (如果需要明确惩罚范数) ---

def linf_penalty(delta: torch.Tensor) -> torch.Tensor:
    """
    计算扰动张量 L-infinity 范数在批次维度上的平均值，用作正则化损失项。

    Args:
        delta (torch.Tensor): 输入的扰动张量，形状通常为 (B, C, N, H, W)。

    Returns:
        torch.Tensor: L-infinity 范数的平均值，形状为 () 的标量张量。
                      如果输入张量为空或批次大小为零，返回 NaN。
    """
    linf_norms = linf_norm(delta) # 获取每个样本的 L-infinity 范数 (B,)
    # 如果范数张量为空，返回 NaN
    if linf_norms.numel() == 0:
         return torch.tensor(float('nan'), device=linf_norms.device if delta is not None else 'cpu')
    return linf_norms.mean() # 计算批次平均值

def l2_penalty(delta: torch.Tensor) -> torch.Tensor:
    """
    计算扰动张量 L2 范数在批次维度上的平均值，用作正则化损失项。

    Args:
        delta (torch.Tensor): 输入的扰动张量，形状通常为 (B, C, N, H, W)。

    Returns:
        torch.Tensor: L2 范数的平均值，形状为 () 的标量张量。
                      如果输入张量为空或批次大小为零，返回 NaN。
    """
    l2_norms = l2_norm(delta) # 获取每个样本的 L2 范数 (B,)
     # 如果范数张量为空，返回 NaN
    if l2_norms.numel() == 0:
         return torch.tensor(float('nan'), device=l2_norms.device if delta is not None else 'cpu')
    return l2_norms.mean() # 计算批次平均值

def total_variation_loss(delta: torch.Tensor) -> torch.Tensor:
    """
    计算扰动张量在空间维度 (H, W) 上的 Total Variation (TV) Loss。
    TV Loss 鼓励生成的扰动在空间上是平滑的，减少高频噪声。
    对于 5D 张量 (B, C, N, H, W)，通常在空间维度 (H, W) 上计算 TV Loss。
    此实现计算 H 和 W 方向的绝对差异之和，并按元素总数进行简单归一化。

    Args:
        delta (torch.Tensor): 输入的扰动张量，期望形状为 (B, C, N, H, W)。
                             输入维度不足 4D (Batch, Spatial, Spatial) 时，TV Loss 将为 0。

    Returns:
        torch.Tensor: TV Loss 的标量值，形状为 ()。
                      如果输入张量为空或维度不足，返回 0.0。
    """
    # 检查输入张量是否为 None 或维度不足
    if delta is None or delta.numel() == 0 or delta.ndim < 4:
         # 警告：输入张量为空或维度不足 {} (<4D)，无法计算 TV Loss。返回 0.0。
         print(f"Warning: Input tensor is None/empty or has insufficient dimensions ({delta.ndim}D) for TV Loss. Returning 0.0.")
         return torch.tensor(0.0, device=delta.device if delta is not None else 'cpu')

    # 在 H (倒数第二个) 和 W (倒数第一个) 方向计算差异并取绝对值
    # 形状 (B, C, N, H-1, W) 和 (B, C, N, H, W-1)
    tv_h = torch.abs(delta[:, :, :, 1:, :] - delta[:, :, :, :-1, :]).sum()
    tv_w = torch.abs(delta[:, :, :, :, 1:] - delta[:, :, :, :, :-1]).sum()

    # 可以选择是否加入序列维度 (N) 和通道维度 (C) 的TV Loss
    # tv_n = torch.abs(delta[:, :, 1:, :, :] - delta[:, :, :-1, :, :]).sum()
    # tv_c = torch.abs(delta[:, 1:, :, :, :] - delta[:, :-1, :, :, :]).sum()

    total_tv = tv_h + tv_w # + tv_n + tv_c # 根据需求决定是否包含序列和通道TV

    # 通常会对 TV Loss 进行归一化，例如除以总像素数量或非零像素数量
    # 这里简单除以 delta 中元素的总数 (B*C*N*H*W) 以获得平均值
    num_elements = delta.numel()
    # 避免除以零
    if num_elements == 0:
        return torch.tensor(0.0, device=delta.device)

    # 返回平均 TV Loss
    return total_tv / num_elements

# 示例用法 (在训练脚本中): # 添加注释以说明示例用途
# from losses.regularization_losses import linf_norm, l2_penalty, total_variation_loss
#
# # 假设 delta 是生成器输出的扰动，形状例如 (B, C, N, H, W)
# delta = generator(input_batch)
#
# # 监控每个样本的最大 L-inf 范数 (不需要加入损失)
# max_linf_per_sample = linf_norm(delta) # 形状 (B,)
# if max_linf_per_sample.numel() > 0:
#      # 打印当前批次中的最大 L-inf 范数
#      print(f"Max L-inf norm in batch: {max_linf_per_sample.max().item()}")
#
# # 如果需要将 L2 范数作为正则化项加入生成器总损失:
# # 权重 lambda_l2 应该从配置中获取
# # 例如: lambda_l2 = cfg.regularization.lambda_l2
# lambda_l2 = 0.01 # 示例权重
# loss_perturb_l2 = lambda_l2 * l2_penalty(delta) if lambda_l2 > 0 else torch.tensor(0.0, device=delta.device)
#
# # 如果需要将 TV loss 作为正则化项加入生成器总损失:
# # 权重 lambda_tv 应该从配置中获取
# # 例如: lambda_tv = cfg.regularization.lambda_tv
# lambda_tv = 0.001 # 示例权重
# loss_perturb_tv = lambda_tv * total_variation_loss(delta) if lambda_tv > 0 else torch.tensor(0.0, device=delta.device)
#
# # 如果需要将生成器参数的 L2 惩罚加入总损失 (这里只是示例函数签名，实际需要遍历模型参数)
# # 权重 lambda_l2_penalty 应该从配置中获取
# # 例如: lambda_l2_penalty = cfg.regularization.lambda_l2_penalty
# lambda_l2_penalty = 0.0001 # 示例权重
# # loss_param_l2_penalty = lambda_l2_penalty * calculate_model_l2_penalty(generator) if lambda_l2_penalty > 0 else torch.tensor(0.0, device=delta.device)
# # 注意：calculate_model_l2_penalty 需要单独实现，遍历 model.parameters() 计算平方和
#
# # 总的生成器正则化损失可以是多个项的加权和
# # regularization_loss_G = loss_perturb_l2 + loss_perturb_tv # + loss_param_l2_penalty
