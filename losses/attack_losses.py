import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence
from torch.distributions import Categorical

class AttackLosses:
    def __init__(self, device, attention_loss_weight=1.0, decision_loss_weight=1.0, topk_k=10):
        self.device = device
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
        self.topk_k = topk_k

        self.attention_loss_weight = attention_loss_weight
        self.decision_loss_weight = decision_loss_weight

    def calculate_decision_loss(self, original_out, adversarial_out, loss_type='mse', region_mask=None):
        """
        计算基于ATN最终决策输出 (B, W, H) 的攻击损失。
        目标：最大化对抗样本输出与原始样本输出的差异，以误导AI决策。

        Args:
            original_out (torch.Tensor): 原始样本下的ATN最终输出 (B, W, H)。
            adversarial_out (torch.Tensor): 对抗样本下的ATN最终输出 (B, W, H)。
            loss_type (str): 使用的损失类型 ('mse', 'l1'). (KL/JS 对连续输出不直接适用)
            region_mask (torch.Tensor, optional): 区域掩码 (B, W, H)。如果提供，损失只在掩码区域计算。

        Returns:
            torch.Tensor: 决策攻击损失 (希望最小化此值以最大化误导)。
                          由于我们希望最大化差异，所以损失值为负。
        """
        assert original_out.shape == adversarial_out.shape
        B, W, H = original_out.shape

        # 如果提供了掩码，则只考虑掩码区域
        if region_mask is not None:
            assert region_mask.shape == (B, W, H) or region_mask.shape == (1, W, H) # 支持批次或单样本掩码
            if region_mask.shape == (1, W, H):
                 region_mask = region_mask.repeat(B, 1, 1) # 扩展到批次维度
            # 应用掩码
            original_out_masked = original_out * region_mask
            adversarial_out_masked = adversarial_out * region_mask
            # 计算有效像素数量
            num_masked_elements = torch.sum(region_mask).item()
            if num_masked_elements == 0:
                return torch.tensor(0.0, device=self.device)

            orig_for_loss = original_out_masked
            adv_for_loss = adversarial_out_masked

        else:
            orig_for_loss = original_out
            adv_for_loss = adversarial_out
            num_masked_elements = orig_for_loss.numel()


        # 距离度量
        if loss_type == 'mse':
            loss = self.mse_loss(adv_for_loss, orig_for_loss) # reduction='mean'
        elif loss_type == 'l1':
            loss = self.l1_loss(adv_for_loss, orig_for_loss) # reduction='mean'
        # TODO: 如果需要更复杂的决策损失，可以在这里添加

        else:
            raise ValueError(f"Unsupported decision loss type: {loss_type}")


        # 取负数以实现最大化差异
        return -loss


    def calculate_attention_loss(self,
                                original_attention_matrix: torch.Tensor,
                                adversarial_attention_matrix: torch.Tensor,
                                loss_type: str = 'mse', # 添加损失类型参数
                                region_mask: torch.Tensor = None): # 添加区域掩码参数
        """
        计算基于ATN序列注意力矩阵 (B, head, N, N) 的攻击损失。
        目标：最大化对抗样本注意力与原始样本注意力的差异，特别是针对 Top-K 区域。

        Args:
            original_attention_matrix (torch.Tensor): 原始样本下的ATN注意力矩阵 (B, head, N, N)。
            adversarial_attention_matrix (torch.Tensor): 对抗样本下的ATN注意力矩阵 (B, head, N, N)。
            loss_type (str): 使用的损失类型 ('mse', 'l1', 'l2_diff', 'linf_diff', 'cosine_sim', 'topk').
                           'topk' 类型专门针对 Top-K 位置进行攻击。
            region_mask (torch.Tensor, optional): 区域掩码 (B, head, N, N)。如果提供，损失只在掩码区域计算。
                                                注意：Top-K 损失与区域掩码的结合需要仔细设计。
                                                这里先实现单独的 Top-K 损失，不与 region_mask 直接结合。

        Returns:
            torch.Tensor: 注意力攻击损失 (希望最小化此值以最大化误导)。
                          基于距离或相似度的损失通常取负，Top-K 损失通常取正 (目标是降低 Top-K 位置的值)。
        """
        if original_attention_matrix is None or adversarial_attention_matrix is None:
            return torch.tensor(0.0, device=self.device)

            assert original_attention_matrix.shape == adversarial_attention_matrix.shape
            B, head, N1, N2 = original_attention_matrix.shape
            assert N1 == N2, "Attention matrix is expected to be square (N x N)"
            N_flat = N1 * N2 # 展平后的维度大小

        # --- Top-K 损失的处理 ---
        if loss_type == 'topk':
            # 检查 K 值是否有效
            if self.topk_k <= 0 or self.topk_k > N_flat:
                 print(f"Warning: Invalid topk_k value: {self.topk_k}. Must be between 1 and {N_flat}. Attention Top-K loss is 0.")
                 return torch.tensor(0.0, device=self.device)

            # 展平 N x N 维度，以便找到全局 Top-K (在每个样本和每个头内部)
            original_attention_flat = original_attention_matrix.view(B, head, N_flat) # Shape (B, head, N*N)
            adversarial_attention_flat = adversarial_attention_matrix.view(B, head, N_flat) # Shape (B, head, N*N)

            # 在原始注意力矩阵中找到 Top-K 值和它们的索引
            # topk_values_orig shape (B, head, k)
            # topk_indices_orig shape (B, head, k)
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
            # adversarial_values_at_topk_indices shape (B, head, k)
            adversarial_values_at_topk_indices = torch.gather(
                adversarial_attention_flat,
                dim=-1,       # 在展平后的最后一个维度上根据索引收集
                index=topk_indices_orig # 使用原始 Top-K 的索引
            )

            # 计算 Top-K 攻击损失
            # 目标是让对抗样本在这些关键位置上的值降低
            # 直接计算这些值的平均值作为损失（希望这个平均值最小）
            loss = adversarial_values_at_topk_indices.mean()

            # 这个损失本身就是我们希望最小化的目标，所以不需要取负
            return loss # 返回一个正值，生成器会最小化它

        # --- 对于非 Top-K 损失，应用掩码 (如果提供) ---
        # 注意：Top-K 损失和区域掩码的结合需要小心。
        # 如果 region_mask 提供了，并且 loss_type 不是 'topk'，我们继续应用掩码。
        # 如果 loss_type 是 'topk'，我们不在这里应用 region_mask。
        if region_mask is not None:
            # 掩码形状兼容性检查 (与之前类似)
            assert region_mask.ndim == 4 or region_mask.ndim == 3 or region_mask.ndim == 2
            if region_mask.ndim == 2: # (N, N) 假设所有样本和所有头共用一个掩码
                 region_mask = region_mask.unsqueeze(0).unsqueeze(0).repeat(B, head, 1, 1)
            elif region_mask.ndim == 3: # (B, N, N) 假设所有头共用掩码
                 region_mask = region_mask.unsqueeze(1).repeat(1, head, 1, 1)
            elif region_mask.ndim == 4 and region_mask.shape[1] == 1: # (B, 1, N, N) 扩展到所有头
                 region_mask = region_mask.repeat(1, head, 1, 1)

            assert region_mask.shape == original_attention_matrix.shape, "Attention region mask shape mismatch after expansion"

            # 应用掩码
            original_attention_matrix_masked = original_attention_matrix * region_mask
            adversarial_attention_matrix_masked = adversarial_attention_matrix * region_mask

            num_masked_elements = torch.sum(region_mask).item()
            if num_masked_elements == 0:
             return torch.tensor(0.0, device=self.device)

            orig_for_loss = original_attention_matrix_masked
            adv_for_loss = adversarial_attention_matrix_masked

        else: # 没有提供掩码
            orig_for_loss = original_attention_matrix
            adv_for_loss = adversarial_attention_matrix
            # num_masked_elements = orig_for_loss.numel() # 这个变量对于这里的损失计算不直接使用，保留只是为了逻辑清晰


        # --- 距离或相似度度量 (非 Top-K 损失) ---
        if loss_type == 'mse':
            loss = self.mse_loss(adv_for_loss, orig_for_loss) # reduction='mean'
            # 希望最大化差异 (距离越大越好)，所以损失取负
            return -loss
        elif loss_type == 'l1':
            loss = self.l1_loss(adv_for_loss, orig_for_loss) # reduction='mean'
             # 希望最大化差异，损失取负
            return -loss
        elif loss_type == 'l2_diff':
            diff = adv_for_loss - orig_for_loss
            # 展平除 batch 和 head 之外的维度，计算 L2 范数，然后对 head 和 batch 求平均
            loss = torch.norm(diff.view(B, head, -1), p=2, dim=-1).mean()
            # 希望最大化差异，损失取负
            return -loss
        elif loss_type == 'linf_diff':
            diff = adv_for_loss - orig_for_loss
            # 展平除 batch 和 head 之外的维度，计算 L-inf 范数，然后对 head 和 batch 求平均
            loss = torch.max(torch.abs(diff).view(B, head, -1), dim=-1).values.mean()
             # 希望最大化差异，损失取负
            return -loss
        elif loss_type == 'cosine_sim':
            # 余弦相似度，需要将矩阵展平为向量 (每个样本每个头独立展平)
            orig_flat = orig_for_loss.view(B * head, -1)
            adv_flat = adv_for_loss.view(B * head, -1)
            # 计算相似度，结果形状 (B * head)
            similarity = self.cosine_similarity(adv_flat, orig_flat)

            # 希望最小化相似度 (最大化差异)，使用 (1 - similarity) / 2 作为差异度量，希望最小化这个差异
            # 对所有样本和头求平均
            loss = (1.0 - similarity).mean() / 2.0
            return loss # 返回正值，希望最小化它

        elif loss_type == 'kl': # 重新添加 KL 损失框架
             # 注意力权重通常是经过 softmax 的，可以视为概率分布。
             # 对于 (B, head, N, N) 形状，对最后一维 (N2) 进行 softmax。
             # 然后计算 KL(original || adversarial) 或 KL(adversarial || original)
             epsilon = 1e-10
             # 对最后一维进行 softmax 获取概率分布
             original_probs = F.softmax(orig_for_loss, dim=-1) + epsilon
             adversarial_probs = F.softmax(adv_for_loss, dim=-1) + epsilon

             # 计算 KL(original || adversarial) 以最大化差异 (让 adversarial 远离 original)
             # 对所有 B, head, N1 维度的 KL 散度求平均
             # F.kl_div 期望输入是 log_prob 和 prob，并且是逐元素的。
             # 更合适的方式是使用 torch.distributions.kl_divergence，但它期望 Distribution 对象。
             # 我们可以手动计算 -sum(p * log(q/p))
             # 或使用 F.kl_div(input, target, reduction='batchmean') 其中 input 是 log_probs, target 是 probs
             # 如果 input 是 log_probs, target 是 probs, 则计算的是 KL(target || exp(input))
             # 我们想要 KL(original || adversarial)，所以 original 是 target (probs), adversarial 是 input (log_probs)
             # log_adversarial_probs = torch.log(adversarial_probs)
             # kl_loss = F.kl_div(log_adversarial_probs, original_probs, reduction='batchmean')

             # 另一种更直观的方式：手动计算或使用torch.distributions (如果输入是概率)
             # KL(P || Q) = sum(P * log(P / Q))
             kl_loss_manual = (original_probs * torch.log(original_probs / adversarial_probs)).sum(dim=-1).mean() # 对 (B, head, N1) 求平均

             # 作为损失项，我们最小化 -KL(...)，所以返回 -kl_loss_manual
             return -kl_loss_manual

        elif loss_type == 'js': # 重新添加 JS 损失框架
             # JS(P || Q) = 0.5 * (KL(P || M) + KL(Q || M)), M = 0.5 * (P + Q)
             epsilon = 1e-10
             original_probs = F.softmax(orig_for_loss, dim=-1) + epsilon
             adversarial_probs = F.softmax(adv_for_loss, dim=-1) + epsilon
             m = 0.5 * (original_probs + adversarial_probs)

             # 计算 KL(original || m) 和 KL(adversarial || m)
             kl_orig_m = (original_probs * torch.log(original_probs / m)).sum(dim=-1)
             kl_adv_m = (adversarial_probs * torch.log(adversarial_probs / m)).sum(dim=-1)

             js_loss_manual = 0.5 * (kl_orig_m + kl_adv_m).mean() # 对 (B, head, N1) 求平均

             # 作为损失项，我们最小化 -JS(...)，所以返回 -js_loss_manual
             return -js_loss_manual

        elif loss_type == 'rank_corr':
            # TODO: 实现基于排序的相关性损失 (例如 Spearman 或 Kendall)
            raise NotImplementedError("Rank correlation loss not implemented.")
        else:
            raise ValueError(f"Unsupported attention loss type: {loss_type}")


    def get_generator_attack_loss(self,
                                original_atn_outputs: dict,
                                adversarial_atn_outputs: dict,
                                decision_loss_type: str = 'mse',
                                attention_loss_type: str = 'mse',
                                decision_region_mask: torch.Tensor = None,
                                attention_region_mask: torch.Tensor = None):
        """
        整合决策攻击损失和注意力攻击损失。

        Args:
            original_atn_outputs (dict): 原始样本下的 ATN 输出字典 {'decision': ..., 'attention': ...}。
            adversarial_atn_outputs (dict): 对抗样本下的 ATN 输出字典。
            decision_loss_type (str): 决策损失类型。
            attention_loss_type (str): 注意力损失类型 ('mse', 'l1', 'l2_diff', 'linf_diff', 'cosine_sim', 'topk').
            decision_region_mask (torch.Tensor, optional): 决策图区域掩码。
            attention_region_mask (torch.Tensor, optional): 注意力矩阵区域掩码。

        Returns:
            torch.Tensor: 生成器的总攻击损失。
        """
        original_decision_output = original_atn_outputs.get('decision')
        adversarial_decision_output = adversarial_atn_outputs.get('decision')
        original_attention_matrix = original_atn_outputs.get('attention')
        adversarial_attention_matrix = adversarial_atn_outputs.get('attention')

        decision_loss = torch.tensor(0.0, device=self.device)
        if original_decision_output is not None and adversarial_decision_output is not None:
             # calculate_decision_loss 已经更新以支持 loss_type 和 region_mask
             decision_loss = self.calculate_decision_loss(
                 original_decision_output,
                 adversarial_decision_output,
                 loss_type=decision_loss_type,
                 region_mask=decision_region_mask
             )
        else:
             print("Warning: Decision output not available, decision loss is 0.")


        attention_loss = torch.tensor(0.0, device=self.device)
        if original_attention_matrix is not None and adversarial_attention_matrix is not None:
             # Top-K 损失通常不需要区域掩码 (因为它关注全局 Top-K 位置)
             # 但其他注意力损失类型可以结合掩码
             current_attention_mask = attention_region_mask if attention_loss_type != 'topk' else None

             attention_loss = self.calculate_attention_loss(
                 original_attention_matrix,
                 adversarial_attention_matrix,
                 loss_type=attention_loss_type, # 使用传入的损失类型 ('topk' 或其他)
                 region_mask=current_attention_mask # 根据损失类型决定是否使用掩码
             )
        else:
             print("Warning: Attention matrix not available, attention loss is 0.")

        total_attack_loss = self.decision_loss_weight * decision_loss + self.attention_loss_weight * attention_loss

        return total_attack_loss

# --- 如何在 train_generator.py 中使用 ---
# from losses.attack_losses import AttackLosses
# # ... 初始化 AttackLosses，传入 Top-K 的 K 值 ...
# # 在 argparse 中添加 --topk_k 参数
# attack_losses = AttackLosses(
#     device,
#     attention_loss_weight=args.attention_loss_weight,
#     decision_loss_weight=args.decision_loss_weight,
#     topk_k=args.topk_k # 从命令行参数获取 K 值
# )
# # ... 在训练循环中计算损失 ...
# # 假设你有 decision_mask 和 attention_mask 张量 (需要从数据加载部分获取或生成)
# # decision_mask: (B, W, H), attention_mask: (B, head, N, N) 或其他兼容形状
#
# loss_G_attack = attack_losses.get_generator_attack_loss(
#     original_atn_outputs,
#     adversarial_atn_outputs,
#     decision_loss_type=args.decision_loss_type,
#     attention_loss_type=args.attention_loss_type, # 可以是 'topk' 或其他
#     decision_region_mask=decision_mask,
#     attention_region_mask=attention_mask # 如果 attention_loss_type == 'topk'，这里传什么都无所谓了
# )
# # ... 生成器总损失计算 ...