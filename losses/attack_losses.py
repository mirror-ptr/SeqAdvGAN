import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class AttackLosses:
    def __init__(self, device, attention_loss_weight=1.0, decision_loss_weight=1.0, topk_k=10):
        self.device = device
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.l1_loss = nn.L1Loss(reduction='sum')
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
        self.topk_k = topk_k

        self.attention_loss_weight = attention_loss_weight
        self.decision_loss_weight = decision_loss_weight

    def calculate_decision_loss(self, original_out, adversarial_out, loss_type='mse', region_mask=None):
        """
        计算基于ATN最终决策输出 (B, W, H) 的攻击损失。
        目标：最大化对抗样本输出与原始样本输出的差异，以误导AI决策。
        损失值为负数，以实现最大化差异。

        Args:
            original_out (torch.Tensor): 原始样本下的ATN最终输出 (B, W, H)。
            adversarial_out (torch.Tensor): 对抗样本下的ATN最终输出 (B, W, H)。
            loss_type (str): 使用的损失类型 ('mse', 'l1').
            region_mask (torch.Tensor, optional): 区域掩码 (B, W, H)。如果提供，损失只在掩码区域计算。

        Returns:
            torch.Tensor: 决策攻击损失 (希望最小化此值以最大化误导)。
                          由于我们希望最大化差异，所以损失值为负。
        """
        if original_out is None or adversarial_out is None:
             print("Warning: Decision maps are None. Decision loss is 0.")
             return torch.tensor(0.0, device=self.device)

        assert original_out.shape == adversarial_out.shape
        B, W, H = original_out.shape

        orig_for_loss = original_out
        adv_for_loss = adversarial_out
        num_elements = orig_for_loss.numel() # Default to total elements

        # 如果提供了掩码，则只考虑掩码区域
        if region_mask is not None:
            # 掩码形状兼容性检查 (B, W, H) 或 (1, W, H)
            assert region_mask.ndim == 3 or region_mask.ndim == 2, f"Decision region mask must be 2D or 3D, got {region_mask.ndim}"
            if region_mask.ndim == 2: # (W, H) or (H, W) - assuming (W, H) might be a typo and it's (H, W)
                 # Assuming region_mask is (H, W) and needs to be (B, H, W)
                 if region_mask.shape != (H, W):
                     print(f"Warning: Decision region mask shape {region_mask.shape} != Decision map spatial shape ({H}, {W}). Assuming mask is (W, H) and transposing.")
                     # Try transposing as a fallback if shape doesn't match expected (H, W)
                     if region_mask.shape == (W, H):
                          region_mask = region_mask.T # Transpose if it's (W, H)
                     else:
                          raise ValueError(f"Decision region mask shape {region_mask.shape} is incompatible with decision map shape ({B}, {H}, {W}). Expected (H, W) or (B, H, W).")

                 region_mask = region_mask.unsqueeze(0).repeat(B, 1, 1) # Expand to batch dimension
            elif region_mask.ndim == 3: # (B, H, W)
                 assert region_mask.shape == (B, H, W), f"Decision region mask shape {region_mask.shape} != Decision map shape ({B}, {H}, {W})"


            # 应用掩码
            # 使用 float() 确保乘法不会出错
            orig_for_loss = original_out * region_mask.float()
            adv_for_loss = adversarial_out * region_mask.float()
            # 计算有效像素数量
            num_elements = torch.sum(region_mask).item()
            if num_elements == 0:
                print("Warning: Region mask is all zeros. Decision loss is 0.")
                return torch.tensor(0.0, device=self.device)


        # 距离度量
        if loss_type == 'mse':
            # MSELoss with reduction='sum' gives sum of squared differences over elements
            loss = self.mse_loss(adv_for_loss, orig_for_loss)
        elif loss_type == 'l1':
            # L1Loss with reduction='sum' gives sum of absolute differences
            loss = self.l1_loss(adv_for_loss, orig_for_loss)
        # KL/JS 对连续输出不直接适用
        elif loss_type in ['kl', 'js']:
             print(f"Warning: {loss_type} loss is not suitable for continuous decision map output. Using MSE/L1 instead.")
             loss = self.mse_loss(adv_for_loss, orig_for_loss) # Fallback to MSE
        else:
            raise ValueError(f"Unsupported decision loss type: {loss_type}. Choose 'mse' or 'l1'.")

        # Normalize the loss by the number of elements included in the loss calculation
        loss = loss / num_elements


        # 取负数以实现最大化差异
        return -loss


    def calculate_attention_loss(self,
                                original_attention_matrix: torch.Tensor,
                                adversarial_attention_matrix: torch.Tensor,
                                loss_type: str = 'mse',
                                region_mask: torch.Tensor = None):
        """
        计算基于ATN序列注意力矩阵 (B, head, N, N) 的攻击损失。
        目标：最大化对抗样本注意力与原始样本注意力或改变Top-K位置。
        基于距离/相似度的损失取负，Top-K 损失取正。

        Args:
            original_attention_matrix (torch.Tensor): 原始样本下的ATN注意力矩阵 (B, head, N, N)。
            adversarial_attention_matrix (torch.Tensor): 对抗样本下的ATN注意力矩阵 (B, head, N, N)。
            loss_type (str): 使用的损失类型 ('mse', 'l1', 'kl', 'js', 'topk').
            region_mask (torch.Tensor, optional): 区域掩码 (B, head, N, N)。如果提供，损失只在掩码区域计算。
                                                注意：Top-K 损失与区域掩码的结合需要仔细设计。
                                                这里将分开处理：Top-K 损失不直接应用 region_mask。

        Returns:
            torch.Tensor: 注意力攻击损失 (希望最小化此值以最大化误导)。
                          基于距离或相似度的损失通常取负，Top-K 损失通常取正 (目标是降低 Top-K 位置的值)。
        """
        if original_attention_matrix is None or adversarial_attention_matrix is None:
            print("Warning: Attention matrices are None. Attention loss is 0.")
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

            # 这个损失本身就是我们希望最小化的目标，所以返回它 (正值)
            return loss

        # --- 对于非 Top-K 损失，应用掩码 (如果提供) ---
        # 注意：Top-K 损失和区域掩码的结合需要小心。
        # 如果 region_mask 提供了，并且 loss_type 不是 'topk'，我们继续应用掩码。
        # 如果 loss_type 是 'topk'，我们不在这里应用 region_mask。
        orig_for_loss = original_attention_matrix
        adv_for_loss = adversarial_attention_matrix
        num_elements = orig_for_loss.numel()

        if region_mask is not None:
            # 掩码形状兼容性检查 (B, head, N, N) 或其他兼容形状
            # 需要根据你的实际掩码生成/加载方式来调整这里的逻辑
            # 假设 mask 可以是 (B, 1, N, N), (B, head, N, N), (1, 1, N, N), (N, N) 等
            assert region_mask.ndim in [2, 3, 4], f"Attention region mask must be 2D, 3D, or 4D, got {region_mask.ndim}"
            
            # 尝试将掩码扩展到 (B, head, N, N) 形状
            if region_mask.ndim == 2: # (N, N)
                 if region_mask.shape != (N1, N2):
                     print(f"Warning: Attention region mask shape {region_mask.shape} != Attention map spatial shape ({N1}, {N2}). Assuming mask is (N2, N1) and transposing.")
                     if region_mask.shape == (N2, N1):
                          region_mask = region_mask.T
                     else:
                          raise ValueError(f"Attention region mask shape {region_mask.shape} is incompatible with attention map shape ({B}, {head}, {N1}, {N2}). Expected (N, N), (B, N, N), (B, 1, N, N), or (B, head, N, N).")

                 region_mask = region_mask.unsqueeze(0).unsqueeze(0).repeat(B, head, 1, 1)
            elif region_mask.ndim == 3: # (B, N, N)
                 assert region_mask.shape == (B, N1, N2), f"Attention region mask shape {region_mask.shape} != Attention map shape ({B}, {N1}, {N2}) in dims 0, 2, 3"
                 region_mask = region_mask.unsqueeze(1).repeat(1, head, 1, 1)
            elif region_mask.ndim == 4 and region_mask.shape[1] == 1: # (B, 1, N, N)
                 assert region_mask.shape[2:] == (N1, N2), f"Attention region mask shape {region_mask.shape} != Attention map shape ({B}, {head}, {N1}, {N2}) in dims 2, 3"
                 region_mask = region_mask.repeat(1, head, 1, 1)
            elif region_mask.ndim == 4: # Should be (B, head, N, N)
                 assert region_mask.shape == (B, head, N1, N2), f"Attention region mask shape {region_mask.shape} != Attention map shape ({B}, {head}, {N1}, {N2})"


            # 应用掩码
            orig_for_loss = original_attention_matrix * region_mask.float()
            adv_for_loss = adversarial_attention_matrix * region_mask.float()
            num_elements = torch.sum(region_mask).item()
            if num_elements == 0:
                print("Warning: Attention region mask is all zeros. Attention loss is 0.")
                return torch.tensor(0.0, device=self.device)


        # 距离度量
        if loss_type == 'mse':
            # MSELoss with reduction='sum' gives sum of squared differences over elements
            loss = self.mse_loss(adv_for_loss, orig_for_loss)
        elif loss_type == 'l1':
            # L1Loss with reduction='sum' gives sum of absolute differences
            loss = self.l1_loss(adv_for_loss, orig_for_loss)
        # KL/JS 对连续输出不直接适用
        elif loss_type in ['kl', 'js']:
             print(f"Warning: {loss_type} loss is not suitable for continuous decision map output. Using MSE/L1 instead.")
             loss = self.mse_loss(adv_for_loss, orig_for_loss) # Fallback to MSE
        else:
            raise ValueError(f"Unsupported decision loss type: {loss_type}. Choose 'mse' or 'l1'.")

        # Normalize the loss by the number of elements considered
        loss = loss / num_elements


        # 取负数以实现最大化差异
        return -loss


    def calculate_feature_attack_loss(self, original_features, adversarial_features, loss_type='mse', region_mask=None):
        """
        计算基于ATN特征层 (B, C, N, W, H) 的攻击损失 (阶段一)。
        目标：最大化对抗样本特征与原始样本特征的差异。
        损失值为负数，以实现最大化差异。

        Args:
            original_features (torch.Tensor): 原始样本下的ATN特征层 (B, C, N, W, H) 或 (B, C, N, H, W)。
            adversarial_features (torch.Tensor): 对抗样本下的ATN特征层 (B, C, N, W, H) 或 (B, C, N, H, W)。
            loss_type (str): 使用的损失类型 ('mse', 'l1').
            region_mask (torch.Tensor, optional): 区域掩码 (B, C, N, W, H)。如果提供，损失只在掩码区域计算。
                                                注意：目前不对特征层应用区域掩码，此参数保留用于未来扩展。

        Returns:
            torch.Tensor: 特征攻击损失 (希望最小化此值以最大化误导)。
                          由于我们希望最大化差异，所以损失值为负。
        """
        if original_features is None or adversarial_features is None:
            print("Warning: Features are None. Feature attack loss is 0.")
            return torch.tensor(0.0, device=self.device)

        assert original_features.shape == adversarial_features.shape
        # B, C, N, H, W = original_features.shape # Assuming (B, C, N, H, W) or (B, C, N, W, H)
        # num_elements = original_features.numel() # Default to total elements

        # 目前不对特征层应用区域掩码，仅计算所有元素的损失
        # TODO: 如果需要对特征层应用掩码，在此处实现逻辑
        orig_for_loss = original_features
        adv_for_loss = adversarial_features
        num_elements = orig_for_loss.numel()
        if num_elements == 0:
            print("Warning: Feature tensor is empty. Feature attack loss is 0.")
            return torch.tensor(0.0, device=self.device)

        # 距离度量
        if loss_type == 'mse':
            loss = self.mse_loss(adv_for_loss, orig_for_loss)
        elif loss_type == 'l1':
            loss = self.l1_loss(adv_for_loss, orig_for_loss)
        else:
            # 对于特征层攻击，主要使用 MSE 或 L1
            print(f"Warning: Feature attack loss type '{loss_type}' not directly supported. Using MSE instead.")
            loss = self.mse_loss(adv_for_loss, orig_for_loss)

        # Normalize the loss by the number of elements
        loss = loss / num_elements

        # 取负数以实现最大化差异
        return -loss


    def get_generator_attack_loss(self,
                                original_atn_outputs: dict,
                                adversarial_atn_outputs: dict,
                                decision_loss_type: str = 'mse',
                                attention_loss_type: str = 'mse',
                                decision_region_mask: torch.Tensor = None,
                                attention_region_mask: torch.Tensor = None,
                                train_stage: int = 1):
        """
        整合攻击损失。

        Args:
            original_atn_outputs (dict): 原始样本下的 ATN 输出字典 {'decision': ..., 'attention': ...}。
            adversarial_atn_outputs (dict): 对抗样本下的 ATN 输出字典。
            decision_loss_type (str): 决策损失类型。
            attention_loss_type (str): 注意力损失类型 ('mse', 'l1', 'kl', 'js', 'topk').
            decision_region_mask (torch.Tensor, optional): 决策图区域掩码。
            attention_region_mask (torch.Tensor, optional): 注意力矩阵区域掩码。
            train_stage (int): 当前训练阶段 (1 或 2)。

        Returns:
            torch.Tensor: 生成器的总攻击损失。
        """
        # 根据训练阶段选择损失计算逻辑
        if train_stage == 1:
            # 阶段一：仅特征攻击
            original_features = original_atn_outputs.get('features') # 假设 ATN 输出中包含了 features
            adversarial_features = adversarial_atn_outputs.get('features')

            feature_attack_loss = self.calculate_feature_attack_loss(
                original_features,
                adversarial_features,
                loss_type=decision_loss_type # 在阶段一，decision_loss_type 用于控制特征攻击损失类型
                # region_mask 目前不对特征层应用
            )
            # 阶段一的总攻击损失就是特征攻击损失
            total_attack_loss = feature_attack_loss
            print("Calculating Stage 1 Attack Loss (Feature Attack)")
            return total_attack_loss

        elif train_stage == 2:
            # 阶段二：决策图和注意力攻击
            original_decision_output = original_atn_outputs.get('decision')
            adversarial_decision_output = adversarial_atn_outputs.get('decision')
            original_attention_matrix = original_atn_outputs.get('attention')
            adversarial_attention_matrix = adversarial_atn_outputs.get('attention')

            decision_loss = torch.tensor(0.0, device=self.device)
            if original_decision_output is not None and adversarial_decision_output is not None:
                 decision_loss = self.calculate_decision_loss(
                     original_decision_output,
                     adversarial_decision_output,
                     loss_type=decision_loss_type,
                     region_mask=decision_region_mask
                 )
            else:
                 print("Warning: Decision output not available for Stage 2 decision loss, decision loss is 0.")


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
                 print("Warning: Attention matrix not available for Stage 2 attention loss, attention loss is 0.")

            # 整合损失
            total_attack_loss = self.decision_loss_weight * decision_loss + self.attention_loss_weight * attention_loss
            print("Calculating Stage 2 Attack Loss (Decision + Attention Attack)")
            return total_attack_loss

        else:
            raise ValueError(f"Unsupported training stage: {train_stage}. Must be 1 or 2.")

# --- 如何在 train_generator.py 中使用 ---
# from losses.attack_losses import AttackLosses
# # ... 初始化 AttackLosses，传入 Top-K 的 K 值 ...
# attack_losses = AttackLosses(
#     device,
#     attention_loss_weight=args.attention_loss_weight,
#     decision_loss_weight=args.decision_loss_weight,
#     topk_k=args.topk_k
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