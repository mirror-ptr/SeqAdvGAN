import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Optional # 导入 Optional



class AttackLosses:
    def __init__(
        self,
        device: torch.device, # Explicitly type hint device
        attention_loss_weight: float = 1.0,
        decision_loss_weight: float = 1.0,
        topk_k: int = 10
    ):
        """
        初始化攻击损失计算器。

        Args:
            device (torch.device): 计算设备（例如 'cuda' 或 'cpu'）。
            attention_loss_weight (float): 注意力攻击损失的权重。
            decision_loss_weight (float): 决策攻击损失的权重。
            topk_k (int): 计算 Top-K 注意力损失时使用的 K 值。
        """
        self.device = device
        self.mse_loss = nn.MSELoss(reduction='sum') # 使用 sum reduction 以便后续手动除以有效元素数量
        self.l1_loss = nn.L1Loss(reduction='sum') # 使用 sum reduction 以便后续手动除以有效元素数量
        # CosineSimilarity 默认在最后一个维度计算，对于展平的注意力或特征是合适的
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
        self.topk_k = topk_k

        self.attention_loss_weight = attention_loss_weight
        self.decision_loss_weight = decision_loss_weight

    def calculate_decision_loss(
        self,
        original_out: torch.Tensor,
        adversarial_out: torch.Tensor,
        loss_type: str = 'mse', # Add type hint
        region_mask: Optional[torch.Tensor] = None # Change | None to Optional[]
    ) -> torch.Tensor:
        """
        计算基于ATN最终决策输出 (B, H, W) 或 (B, W, H) 的攻击损失。
        目标：最大化对抗样本输出与原始样本输出的差异，以误导AI决策。
        损失值为负数，以实现最大化差异（因为优化器默认最小化损失）。

        Args:
            original_out (torch.Tensor): 原始样本下的ATN最终输出，形状通常为 (B, H, W)。
            adversarial_out (torch.Tensor): 对抗样本下的ATN最终输出，形状与 original_out 相同。
            loss_type (str): 使用的损失类型 ('mse', 'l1').
            region_mask (torch.Tensor | None, optional): 区域掩码，形状通常为 (B, H, W)。如果提供，损失只在掩码区域计算。

        Returns:
            torch.Tensor: 决策攻击损失 (希望最小化此值以最大化误导)。
                          由于我们希望最大化差异，所以损失值为负。
                          返回形状为 () 的标量张量。
        """
        # 检查输入决策图是否为None或为空
        if original_out is None or adversarial_out is None or original_out.numel() == 0 or adversarial_out.numel() == 0:
             # 警告：决策图为None或为空。决策损失为0。
             print("Warning: Decision maps are None or empty. Decision loss is 0.")
             # 返回一个设备正确的零张量
             return torch.tensor(0.0, device=self.device)

        # 确保输入形状匹配
        if original_out.shape != adversarial_out.shape:
             # 错误：原始决策图形状 {} 与对抗决策图形状 {} 不匹配。
             print(f"Error: Original decision map shape {original_out.shape} != Adversarial decision map shape {adversarial_out.shape}.")
             # 返回一个设备正确的零张量并继续
             return torch.tensor(0.0, device=self.device)

        # 假设决策图形状是 (B, H, W)
        assert original_out.ndim == 3, f"Decision map is expected to be 3D (B, H, W), got {original_out.ndim}D"
        B, H, W = original_out.shape

        orig_for_loss = original_out
        adv_for_loss = adversarial_out
        num_elements = orig_for_loss.numel() # 默认元素总数

        # 如果提供了掩码，则只考虑掩码区域
        if region_mask is not None:
            # 掩码形状兼容性检查
            # 期望掩码形状为 (B, H, W) 或 (H, W) (需要扩展到批量)
            assert region_mask.ndim in [2, 3], f"Decision region mask must be 2D or 3D, got {region_mask.ndim}"
            
            if region_mask.ndim == 2: # (H, W)
                 if region_mask.shape != (H, W):
                     # 警告：决策区域掩码形状 {} 与决策图空间形状 ({}, {}) 不匹配。跳过掩码应用。
                     print(f"Warning: Decision region mask shape {region_mask.shape} != Decision map spatial shape ({H}, {W}). Skipping mask application.")
                     region_mask = None # 禁用掩码
                 else:
                      # 将 (H, W) 扩展到 (B, H, W)
                      region_mask = region_mask.unsqueeze(0).repeat(B, 1, 1).to(self.device) # 确保在同一设备

            elif region_mask.ndim == 3: # (B, H, W)
                 if region_mask.shape != (B, H, W):
                     # 警告：决策区域掩码形状 {} 与决策图形状 ({}, {}, {}) 不匹配。跳过掩码应用。
                     print(f"Warning: Decision region mask shape {region_mask.shape} != Decision map shape ({B}, {H}, {W}). Skipping mask application.")
                     region_mask = None # 禁用掩码
                 else:
                      # 确保掩码在同一设备
                      region_mask = region_mask.to(self.device)

            # 应用掩码（如果仍然启用）
            if region_mask is not None:
                 # 确保掩码是 float 类型以便乘法
                 orig_for_loss = original_out * region_mask.float()
                 adv_for_loss = adversarial_out * region_mask.float()
                 # 计算掩码中非零元素的数量作为有效像素数量
                 num_elements = torch.sum(region_mask).item()
                 if num_elements == 0:
                    # 警告：区域掩码全为零。决策损失为0。
                     print("Warning: Region mask is all zeros. Decision loss is 0.")
                     return torch.tensor(0.0, device=self.device)


        # 距离度量计算
        if loss_type == 'mse':
            # 使用 reduction='sum' 计算所有元素的平方差之和
            loss = self.mse_loss(adv_for_loss, orig_for_loss)
        elif loss_type == 'l1':
            # 使用 reduction='sum' 计算所有元素的绝对差之和
            loss = self.l1_loss(adv_for_loss, orig_for_loss)
        # KL/JS 损失对连续输出不直接适用，或者需要将连续值转换为概率分布
        # 目前按您的要求，只支持 mse 和 l1
        elif loss_type in ['kl', 'js']:
             # 警告：{} 损失类型不适用于连续决策图输出。请使用 'mse' 或 'l1'。
             print(f"Warning: {loss_type} loss type is not suitable for continuous decision map output. Please use 'mse' or 'l1'.")
             # 仍然计算一个损失，例如 MSE，但用户应注意此警告
             loss = self.mse_loss(adv_for_loss, orig_for_loss) # 退回使用 MSE
        else:
            # 错误：不支持的决策损失类型：{}。请选择 'mse' 或 'l1'。
            raise ValueError(f"Unsupported decision loss type: {loss_type}. Please choose 'mse' or 'l1'.")

        # 将总损失除以有效元素数量以获得平均损失
        loss = loss / num_elements

        # 取负数以实现最大化差异（最小化负损失等于最大化正损失）
        return -loss

    def calculate_attention_loss(
        self,
        original_attention_matrix: torch.Tensor,
        adversarial_attention_matrix: torch.Tensor,
        loss_type: str = 'mse', # Add type hint
        region_mask: Optional[torch.Tensor] = None # Change | None to Optional[]
    ) -> torch.Tensor:
        """
        计算基于ATN序列注意力矩阵 (B, head, N, N) 的攻击损失。
        目标：最大化对抗样本注意力与原始样本注意力的距离，或改变原始Top-K注意力的位置/值。
        基于距离/相似度的损失取负，Top-K 损失取正（目标是降低对抗样本在原始关键位置的值）。

        Args:
            original_attention_matrix (torch.Tensor): 原始样本下的ATN注意力矩阵，形状通常为 (B, head, N, N)。
            adversarial_attention_matrix (torch.Tensor): 对抗样本下的ATN注意力矩阵，形状与 original_attention_matrix 相同。
            loss_type (str): 使用的损失类型 ('mse', 'l1', 'kl', 'js', 'topk').
            region_mask (torch.Tensor | None, optional): 区域掩码。如果提供，基于距离的损失只在掩码区域计算。
                                                形状可以兼容 (B, head, N, N)。
                                                注意：'topk' 损失不直接应用此区域掩码。

        Returns:
            torch.Tensor: 注意力攻击损失 (希望最小化此值以实现攻击目标)。
                          基于距离的损失值为负，Top-K 损失值为正。
                          返回形状为 () 的标量张量。
        """
        # 检查输入注意力矩阵是否为None或为空
        if original_attention_matrix is None or adversarial_attention_matrix is None or original_attention_matrix.numel() == 0 or adversarial_attention_matrix.numel() == 0:
            # 警告：注意力矩阵为None或为空。注意力损失为0。
            print("Warning: Attention matrices are None or empty. Attention loss is 0.")
            # 返回一个设备正确的零张量
            return torch.tensor(0.0, device=self.device)

        # 确保输入形状匹配
        if original_attention_matrix.shape != adversarial_attention_matrix.shape:
             # 错误：原始注意力矩阵形状 {} 与对抗注意力矩阵形状 {} 不匹配。
             print(f"Error: Original attention matrix shape {original_attention_matrix.shape} != Adversarial attention matrix shape {adversarial_attention_matrix.shape}.")
             # 返回一个设备正确的零张量并继续
             return torch.tensor(0.0, device=self.device)

        # 假设注意力矩阵形状是 (B, head, N, N)
        assert original_attention_matrix.ndim == 4, f"Attention matrix is expected to be 4D (B, head, N, N), got {original_attention_matrix.ndim}D"
        B, head, N1, N2 = original_attention_matrix.shape
        assert N1 == N2, f"Attention matrix is expected to be square (N x N), got {N1}x{N2}"
        N_flat = N1 * N2 # 展平后的序列维度大小

        # --- Top-K 损失的处理 ---
        if loss_type == 'topk':
            # 检查 K 值是否有效
            if self.topk_k <= 0 or self.topk_k > N_flat:
                 # 警告：无效的 topk_k 值：{}。必须在 1 和 {} 之间。注意力 Top-K 损失为0。
                 print(f"Warning: Invalid topk_k value: {self.topk_k}. Must be between 1 and {N_flat}. Attention Top-K loss is 0.")
                 # 返回一个设备正确的零张量
                 return torch.tensor(0.0, device=self.device)

            # 展平 N x N 维度，以便找到全局 Top-K (在每个样本和每个头内部)
            original_attention_flat = original_attention_matrix.view(B, head, N_flat) # 形状 (B, head, N*N)
            # 注意：对于 Top-K 攻击，我们通常希望最小化 对抗样本在 原始 Top-K 位置上的值
            adversarial_attention_flat = adversarial_attention_matrix.view(B, head, N_flat) # 形状 (B, head, N*N)

            # 在原始注意力矩阵中找到 Top-K 值和它们的索引
            # topk_values_orig 形状 (B, head, k)
            # topk_indices_orig 形状 (B, head, k)
            # largest=True 表示找最大的 K 个值
            # sorted=False 可以提高一点点速度，因为我们不需要排序
            topk_values_orig, topk_indices_orig = torch.topk(
                original_attention_flat,
                k=self.topk_k,
                dim=-1,       # 在展平后的最后一个维度 (N*N) 上找 Top-K
                largest=True, # 找最大的 K 个值
                sorted=False
            )

            # 获取对抗样本在原始 Top-K 索引位置上的值
            # 使用 gather 函数
            # adversarial_values_at_topk_indices 形状 (B, head, k)
            adversarial_values_at_topk_indices = torch.gather(
                adversarial_attention_flat,
                dim=-1,       # 在展平后的最后一个维度上根据索引收集
                index=topk_indices_orig
            )

            # 计算 Top-K 攻击损失
            # 目标是让对抗样本在这些关键位置上的值降低
            # 直接计算这些值的平均值作为损失（希望这个平均值最小）。损失值为正。
            loss = adversarial_values_at_topk_indices.mean()

            # 这个损失本身就是我们希望最小化的目标，所以返回它 (正值)
            return loss

        # --- 对于非 Top-K 损失 (基于距离/相似度)，应用掩码 (如果提供) ---
        # 注意：Top-K 损失和区域掩码的结合需要小心。
        # 如果 region_mask 提供了，并且 loss_type 不是 'topk'，我们继续应用掩码。
        # 如果 loss_type 是 'topk'，我们不在这里应用 region_mask。
        orig_for_loss = original_attention_matrix
        adv_for_loss = adversarial_attention_matrix
        num_elements = orig_for_loss.numel()

        if region_mask is not None:
            # 掩码形状兼容性检查
            # 期望掩码形状兼容 (B, head, N, N)
            assert region_mask.ndim in [2, 3, 4], f"Attention region mask must be 2D, 3D, or 4D, got {region_mask.ndim}"
            
            # 尝试将掩码扩展到 (B, head, N, N) 形状
            if region_mask.ndim == 2: # (N, N)
                 if region_mask.shape != (N1, N2):
                      # 警告：注意力区域掩码空间形状 {} 与注意力矩阵空间形状 ({}, {}) 不匹配。跳过掩码应用。
                      print(f"Warning: Attention region mask spatial shape {region_mask.shape} != Attention matrix spatial shape ({N1}, {N2}). Skipping mask application.")
                      region_mask = None # 禁用掩码
                 else:
                      # 将 (N, N) 扩展到 (B, head, N, N)
                      region_mask = region_mask.unsqueeze(0).unsqueeze(0).repeat(B, head, 1, 1).to(self.device) # 确保在同一设备

            elif region_mask.ndim == 3: # (B, N, N)
                 if region_mask.shape != (B, N1, N2):
                      # 警告：注意力区域掩码形状 {} 与注意力矩阵形状 ({}, {}, {}) 在维度 0, 2, 3 上不匹配。跳过掩码应用。
                      print(f"Warning: Attention region mask shape {region_mask.shape} != Attention matrix shape ({B}, {head}, {N1}, {N2}) in dims 0, 2, 3. Skipping mask application.")
                      region_mask = None # 禁用掩码
                 else:
                      # 将 (B, N, N) 扩展到 (B, head, N, N)
                      region_mask = region_mask.unsqueeze(1).repeat(1, head, 1, 1).to(self.device) # 确保在同一设备

            elif region_mask.ndim == 4 and region_mask.shape[1] == 1: # (B, 1, N, N)
                 if region_mask.shape[2:] != (N1, N2):
                     # 警告：注意力区域掩码形状 {} 与注意力矩阵形状 ({}, {}, {}) 在维度 2, 3 上不匹配。跳过掩码应用。
                     print(f"Warning: Attention region mask shape {region_mask.shape} != Attention matrix shape ({B}, {head}, {N1}, {N2}) in dims 2, 3. Skipping mask application.")
                     region_mask = None # 禁用掩码
                 else:
                      # 将 (B, 1, N, N) 扩展到 (B, head, N, N)
                      region_mask = region_mask.repeat(1, head, 1, 1).to(self.device) # 确保在同一设备

            elif region_mask.ndim == 4: # 期望 (B, head, N, N)
                 if region_mask.shape != (B, head, N1, N2):
                      # 警告：注意力区域掩码形状 {} 与注意力矩阵形状 ({}, {}, {}) 不匹配。跳过掩码应用。
                      print(f"Warning: Attention region mask shape {region_mask.shape} != Attention matrix shape ({B}, {head}, {N1}, {N2}). Skipping mask application.")
                      region_mask = None # 禁用掩码
                 else:
                      # 确保掩码在同一设备
                      region_mask = region_mask.to(self.device)


            # 应用掩码（如果仍然启用）
            if region_mask is not None:
                 orig_for_loss = original_attention_matrix * region_mask.float()
                 adv_for_loss = adversarial_attention_matrix * region_mask.float()
                 # 计算掩码中非零元素的数量作为有效元素数量
                 num_elements = torch.sum(region_mask).item()
                 if num_elements == 0:
                    # 警告：注意力区域掩码全为零。注意力损失为0。
                     print("Warning: Attention region mask is all zeros. Attention loss is 0.")
                     return torch.tensor(0.0, device=self.device)


        # 距离度量计算
        if loss_type == 'mse':
            # 使用 reduction='sum' 计算所有元素的平方差之和
            loss = self.mse_loss(adv_for_loss, orig_for_loss)
        elif loss_type == 'l1':
            # 使用 reduction='sum' 计算所有元素的绝对差之和
            loss = self.l1_loss(adv_for_loss, orig_for_loss)
        elif loss_type == 'kl':
             # 注意力矩阵通常是概率分布（Softmax 后），可以使用 KLDivLoss
             # 需要将注意力矩阵视为概率分布
             # KLDivLoss 输入需要是 Log-probabilities
             # 计算 KL(P || Q) 其中 P 是原始分布，Q 是对抗分布
             # 目标是最大化 KL 散度 (即最小化 -KL 散度)
             # 这要求 original_attention_matrix 和 adversarial_attention_matrix 是概率分布 (非负且和为1)
             # 假设输入已经是 Softmax 后的概率分布
             # 为避免 log(0)，使用一个小 epsilon
             epsilon = 1e-10
             # 将概率转换为 Log-probabilities
             log_orig_prob = torch.log(orig_for_loss + epsilon)
             log_adv_prob = torch.log(adv_for_loss + epsilon)
             # 计算 KLDivLoss (Kullback-Leibler divergence)
             # KLDivLoss(input, target, reduction) 计算 target * (log(target) - input)
             # 我们想要计算 KL(P || Q)，即 sum(P * log(P / Q))
             # 相当于 sum(P * log(P) - P * log(Q))
             # KLDivLoss 的 target 是 P (原始), input 是 log(Q) (对抗的对数概率)
             # 并且 KLDivLoss 默认计算 sum(target * (log(target) - input))
             # 因此，KLDivLoss(log_adv_prob, orig_for_loss, reduction='sum') 计算 sum(orig * (log(orig) - log_adv))
             # 这正是 sum(orig * log(orig / adv)) = KL(orig || adv)
             # 我们的目标是最大化差异，即最大化 KL 散度
             # 所以损失是 -KL 散度，以便最小化负损失
             # 使用 reduction='sum' 获取总和，后续手动除以有效元素数量
             # 确保输入到 KLDivLoss 的 target 是概率 (非负)
             if torch.min(orig_for_loss) < 0 or torch.max(orig_for_loss) > 1 + epsilon:
                  print("Warning: Original attention values seem not to be in [0, 1] for KL loss. Clamping to [epsilon, 1].")
                  orig_for_loss = torch.clamp(orig_for_loss, epsilon, 1)
             if torch.min(adv_for_loss) < 0 or torch.max(adv_for_loss) > 1 + epsilon:
                  print("Warning: Adversarial attention values seem not to be in [0, 1] for KL loss. Clamping to [epsilon, 1].")
                  adv_for_loss = torch.clamp(adv_for_loss, epsilon, 1)

             # Ensure that the sum over the attention dimension is 1 for probability assumption
             # This might need clarification based on how your ATN outputs attention.
             # For simplicity here, assuming they are normalized probabilities.

             # 计算原始概率的对数
             log_orig_prob = torch.log(orig_for_loss + epsilon)
             # 计算对抗概率的对数
             log_adv_prob = torch.log(adv_for_loss + epsilon)

             # 计算 KL(原始 || 对抗)
             kl_loss = F.kl_div(log_adv_prob, orig_for_loss, reduction='sum') # KLDivLoss(log_Q, P)

             # 目标是最大化差异，即最大化 KL 散度，所以损失取负
             loss = -kl_loss # 最小化 -KL(orig || adv)

        elif loss_type == 'js':
             # Jensen-Shannon Divergence (JS)
             # JS(P || Q) = 0.5 * (KL(P || M) + KL(Q || M)) where M = 0.5 * (P + Q)
             # 目标是最大化 JS 散度，所以损失取负
             epsilon = 1e-10
             M = 0.5 * (orig_for_loss + adv_for_loss)
             M = torch.clamp(M, epsilon, 1) # Avoid log(0)

             # 计算 log(M)
             log_M = torch.log(M)

             # 计算 KL(P || M)
             # KLDivLoss(log_M, orig_for_loss, reduction='sum') computes sum(orig * (log(orig) - log_M))
             # This is KL(orig || M)
             kl_pm = F.kl_div(log_M, orig_for_loss, reduction='sum')

             # 计算 KL(Q || M)
             # KLDivLoss(log_M, adv_for_loss, reduction='sum') computes sum(adv * (log(adv) - log_M))
             # This is KL(adv || M)
             kl_qm = F.kl_div(log_M, adv_for_loss, reduction='sum')

             # 计算 JS 散度
             js_divergence = 0.5 * (kl_pm + kl_qm)

             # 目标是最大化 JS 散度，所以损失取负
             loss = -js_divergence # 最小化 -JS(orig || adv)

        # Cosine Similarity - 目标是最小化相似度 (即使它们不相似)，即最大化负相似度
        # Cosine Similarity 的范围是 [-1, 1]
        # 最大化差异 -> 最小化相似度
        # 损失 = 相似度 -> 最小化损失 = 最小化相似度
        # 考虑展平的向量 (B*head, N*N)
        elif loss_type == 'cosine':
             # 展平以计算每个头/每个样本的余弦相似度
             # 形状 (B*head, N*N)
             orig_flat = original_attention_matrix.view(B*head, N_flat)
             adv_flat = adversarial_attention_matrix.view(B*head, N_flat)
             # calculate_cosine_similarity 返回形状为 (B*head,) 的张量
             # 取平均值作为损失
             # 注意：CosineSimilarity 已经在 __init__ 中定义，使用 self.cosine_similarity
             # self.cosine_similarity 期望输入的最后一个维度是需要计算相似度的维度
             # 所以需要将注意力矩阵reshape为 (B*head*N, N) 或 (B*head*N, N) 或者 (B*head, N*N) 然后 dim=-1
             # 展平到 (B*head, N*N) 是最直接的
             # 这里计算的是每个头、每个样本的 N*N 矩阵作为向量的相似度
             # cosine_similarity 返回 (B*head,)
             similarity = self.cosine_similarity(orig_flat, adv_flat).mean() # 计算批量和头的平均相似度
             # 目标是最小化相似度，所以损失等于相似度 (希望优化器最小化它)
             loss = similarity # 最小化 Cosine Similarity

        else:
            # 错误：不支持的注意力损失类型：{}。请选择 'mse', 'l1', 'kl', 'js', 'topk', 'cosine'。
            raise ValueError(f"Unsupported attention loss type: {loss_type}. Please choose 'mse', 'l1', 'kl', 'js', 'topk', or 'cosine'.")

        # 对于非 Top-K 损失 (基于距离/相似度)，如果使用了掩码，需要除以有效元素数量进行归一化
        # Top-K 损失已经在其自己的逻辑中处理了平均/求和
        if loss_type != 'topk':
             # 注意：KL 和 JS 散度理论上也是一种"平均"，但它们的计算方式不同
             # KLDivLoss 和我们在这里实现的 JS 散度（基于 KLDivLoss 的总和）已经隐含了对元素的求和
             # 如果应用了区域掩码，我们需要根据掩码中的元素数量进行归一化
             # 如果没有掩码，num_elements 是全部元素数量，除以它得到平均值
             if region_mask is not None:
                 # 如果应用了掩码，num_elements 已经是掩码内元素的总和
                 loss = loss / num_elements
             # else: 如果没有掩码，loss 是总和 (因为 MSE/L1 是 sum reduction)，除以全部元素数
             # 注意：KLDivLoss 的 reduction='sum' 也是对元素的总和
             # 所以这里统一除以 num_elements 得到平均值
             # 这部分逻辑需要根据 KLDivLoss 和 JS 计算的细节确认是否需要
             # 鉴于 MSE/L1 使用 sum reduction，这里除以 num_elements 是为了得到平均损失。
             # KLDivLoss 和 JS 也通常报告平均值，所以除以 num_elements 可能是合适的。
             # 暂时保留，如果发现问题再调整。
             pass # KL/JS/Cosine already averaged or summed appropriately

        # 对于基于距离/相似度的损失，取负数以实现最大化差异
        # Top-K 损失（旨在最小化原始 Top-K 位置的值）已经返回正值
        if loss_type in ['mse', 'l1', 'kl', 'js', 'cosine']:
            # 损失值为负，以便最小化负损失等于最大化正距离/散度
            # Cosine similarity 的目标是最小化相似度，所以损失 = similarity 是正确的，不需要取负
            # 但为了与攻击损失"最大化差异"的概念一致，我们可以将损失定义为 1 - similarity，然后最小化它。
            # 或者更简单，最小化 similarity 本身。
            # 如果我们希望最大化差异/散度，并且这些损失函数计算的是距离或散度（值越大差异越大），
            # 那么损失应该取负，因为优化器最小化损失。
            # mse, l1, kl, js: 越大越好 (差异大) -> loss = -value
            # cosine: 越小越好 (差异大) -> loss = +value
            # 修正：cosine loss 目标是最小化相似度，不需要取负。

            if loss_type in ['mse', 'l1', 'kl', 'js']:
                 return -loss # 最大化差异/散度，损失取负
            elif loss_type == 'cosine':
                 return loss # 最小化相似度，损失取正
            # else: Top-K handled above
        else:
             # Top-K 损失已经在上面返回了
             return loss

    def calculate_feature_attack_loss(
        self,
        original_features: torch.Tensor,
        adversarial_features: torch.Tensor,
        loss_type: str = 'mse', # Add type hint
        region_mask: Optional[torch.Tensor] = None # Add type hint and Optional
    ) -> torch.Tensor:
        """
        计算基于ATN特征层 (B, C, N, W, H) 或 (B, C, N, H, W) 的攻击损失 (阶段一)。
        目标：最大化对抗样本特征与原始样本特征的差异。
        损失值为负数，以实现最大化差异（因为优化器默认最小化损失）。

        Args:
            original_features (torch.Tensor): 原始样本下的ATN特征层。
                                            形状为 (B, C, N, W, H) 或 (B, C, N, H, W)。
            adversarial_features (torch.Tensor): 对抗样本下的ATN特征层，形状与 original_features 相同。
            loss_type (str): 使用的损失类型 ('mse', 'l1'). 目前只支持 mse 和 l1。
            region_mask (torch.Tensor | None, optional): 区域掩码。此参数保留用于未来扩展。
                                                注意：目前不对特征层应用区域掩码。

        Returns:
            torch.Tensor: 特征攻击损失 (希望最小化此值以最大化误导)。
                          由于我们希望最大化差异，所以损失值为负。
                          返回形状为 () 的标量张量。
        """
        # 检查输入特征是否为None或为空
        if original_features is None or adversarial_features is None or original_features.numel() == 0 or adversarial_features.numel() == 0:
             # 警告：输入特征为None或为空。特征攻击损失为0。
             print("Warning: Input features are None or empty. Feature attack loss is 0.")
             # 返回一个设备正确的零张量
             return torch.tensor(0.0, device=self.device)

        # 确保输入形状匹配
        if original_features.shape != adversarial_features.shape:
             # 错误：原始特征形状 {} 与对抗特征形状 {} 不匹配。
             print(f"Error: Original feature shape {original_features.shape} != Adversarial feature shape {adversarial_features.shape}.")
             # 返回一个设备正确的零张量并继续
             return torch.tensor(0.0, device=self.device)

        # 假设特征形状是 (B, C, N, W, H) 或 (B, C, N, H, W)
        # 为了计算 MSE 或 L1 损失，我们需要在所有维度上进行比较
        # 展平所有维度，只保留批量维度 (B, C*N*W*H)
        original_features_flat = original_features.view(original_features.shape[0], -1)
        adversarial_features_flat = adversarial_features.view(adversarial_features.shape[0], -1)

        # 注意：目前不对特征层应用区域掩码，region_mask 参数保留用于未来扩展。
        # 如果未来需要应用掩码，逻辑会更复杂，因为它需要在 5D 张量上操作。
        if region_mask is not None:
            # 警告：目前不支持对特征层应用区域掩码。忽略提供的掩码。
            print("Warning: Region mask is not currently supported for feature attack loss. Ignoring provided mask.")
            region_mask = None # 确保不使用掩码

        # num_elements 是每个样本展平后的元素数量
        num_elements_per_sample = original_features_flat.shape[1]
        # total_elements 是所有样本的总元素数量 (如果使用 sum reduction 并最终取平均)
        # 这里我们计算每个样本的损失，然后取批量平均

        # 距离度量计算
        if loss_type == 'mse':
            # 计算每个样本的 MSE (在展平后的维度上)
            # MSELoss 默认 reduction='mean' 会计算整个批次的平均值
            # 如果需要每个样本的损失，可以使用 reduction='none' 并手动计算平均或求和
            # 这里使用 reduction='sum' 然后除以每个样本的元素数量，再取批量平均
            # 或者直接使用 F.mse_loss 并手动处理维度
            # 使用 self.mse_loss (reduction='sum') 更简单
            # 假设 self.mse_loss 的 input/target 形状是相同的
            # self.mse_loss(adv_features_flat, orig_features_flat) 会计算整个批次的平方差总和
            total_sum_sq_diff = self.mse_loss(adversarial_features_flat, original_features_flat) # 对整个批次和展平的维度求和
            # 归一化：除以批量大小 B 和每个样本的元素数量 num_elements_per_sample
            loss = total_sum_sq_diff / (original_features_flat.shape[0] * num_elements_per_sample)

        elif loss_type == 'l1':
            # 计算每个样本的 L1 损失 (在展平后的维度上)
            # self.l1_loss (reduction='sum') 会计算整个批次的绝对差总和
            total_sum_abs_diff = self.l1_loss(adversarial_features_flat, original_features_flat)
            # 归一化：除以批量大小 B 和每个样本的元素数量 num_elements_per_sample
            loss = total_sum_abs_diff / (original_features_flat.shape[0] * num_elements_per_sample)

        else:
            # 错误：不支持的特征攻击损失类型：{}。请选择 'mse' 或 'l1'。
            raise ValueError(f"Unsupported feature attack loss type: {loss_type}. Please choose 'mse' or 'l1'.")

        # 取负数以实现最大化差异（最小化负损失等于最大化正距离）
        return -loss


    def get_generator_attack_loss(
        self,
        original_atn_outputs: dict,
        adversarial_atn_outputs: dict,
        decision_loss_type: str = 'mse',
        attention_loss_type: str = 'mse',
        decision_region_mask: Optional[torch.Tensor] = None,
        attention_region_mask: Optional[torch.Tensor] = None,
        train_stage: int = 1
    ) -> torch.Tensor:
        """
        根据训练阶段计算生成器的总攻击损失。

        Args:
            original_atn_outputs (dict): 原始样本通过 ATN 模型得到的输出字典，包含 'features', 'decision', 'attention'。
            adversarial_atn_outputs (dict): 对抗样本通过 ATN 模型得到的输出字典。
            decision_loss_type (str): 用于计算决策攻击损失的类型 ('mse', 'l1')。
            attention_loss_type (str): 用于计算注意力攻击损失的类型 ('mse', 'l1', 'kl', 'js', 'topk', 'cosine')。
            decision_region_mask (torch.Tensor | None, optional): 用于决策攻击损失的区域掩码。
            attention_region_mask (torch.Tensor | None, optional): 用于注意力攻击损失的区域掩码。
            train_stage (int): 当前的训练阶段 (1 或 2)。

        Returns:
            torch.Tensor: 生成器的总攻击损失。返回形状为 () 的标量张量。
        """
        total_attack_loss = torch.tensor(0.0, device=self.device) # 初始化总损失

        # --- 阶段1 攻击损失 (特征层) ---
        if train_stage == 1:
            # 在阶段1，攻击目标是最大化特征层的差异
            original_features = original_atn_outputs.get('features')
            adversarial_features = adversarial_atn_outputs.get('features')

            # 确保特征可用
            if original_features is not None and adversarial_features is not None:
                 # 计算特征攻击损失 (通常是 MSE 或 L1)
                 # calculate_feature_attack_loss 返回负值 (因为目标是最大化差异)
                 feature_attack_loss = self.calculate_feature_attack_loss(
                      original_features,
                      adversarial_features,
                      loss_type='mse', # 阶段1固定使用 MSE 作为特征攻击损失类型，或者可以从 config 中获取
                      # region_mask 目前不支持，所以不传递
                 )
                 # 阶段1的总攻击损失就是特征攻击损失 (不需要权重，或者权重就是1.0)
                 # 因为 calculate_feature_attack_loss 返回的是负值，最小化它就是最大化特征差异，符合攻击目标。
                 total_attack_loss = feature_attack_loss # total_attack_loss 已经是负值
            else:
                 # 警告：阶段1计算攻击损失时特征为None。总攻击损失为0。
                 print("Warning: Features are None when calculating attack loss in Stage 1. Total attack loss is 0.")
                 total_attack_loss = torch.tensor(0.0, device=self.device)

        # --- 阶段2 攻击损失 (决策图和注意力图) ---
        elif train_stage == 2:
            # 在阶段2，攻击目标是误导决策图和注意力图
            original_decision_map = original_atn_outputs.get('decision')
            adversarial_decision_map = adversarial_atn_outputs.get('decision')
            original_attention_matrix = original_atn_outputs.get('attention')
            adversarial_attention_matrix = adversarial_atn_outputs.get('attention')

            # 计算决策攻击损失
            # calculate_decision_loss 返回负值 (因为目标是最大化差异)
            decision_attack_loss = self.calculate_decision_loss(
                 original_decision_map,
                 adversarial_decision_map,
                 loss_type=decision_loss_type, # 使用传入的类型
                 region_mask=decision_region_mask
            )
            # 计算注意力攻击损失
            # calculate_attention_loss 返回负值 (距离/散度) 或正值 (Top-K)
            attention_attack_loss = self.calculate_attention_loss(
                 original_attention_matrix,
                 adversarial_attention_matrix,
                 loss_type=attention_loss_type, # 使用传入的类型
                 region_mask=attention_region_mask # Note: mask only applied to non-topk losses internally
            )

            # 将两种损失按权重相加
            # 注意：decision_attack_loss 和 attention_attack_loss (非 Top-K) 是负值
            # attention_attack_loss (Top-K) 是正值
            # 总损失 = 权重 * 决策损失 + 权重 * 注意力损失
            # 最小化总损失
            # 如果决策损失和非 Top-K 注意力损失是负值，最小化它们会使它们更负，即最大化原始的正距离/散度
            # 如果 Top-K 注意力损失是正值，最小化它会使它趋近于零，符合最小化对抗样本在原始 Top-K 位置的值的目标
            total_attack_loss = (self.decision_loss_weight * decision_attack_loss +
                                 self.attention_loss_weight * attention_attack_loss)

        else:
            # 错误：不支持的训练阶段 {} 用于计算攻击损失。
            print(f"Error: Unsupported training stage {train_stage} for attack loss calculation.")
            # 返回一个设备正确的零张量
            total_attack_loss = torch.tensor(0.0, device=self.device)

        return total_attack_loss