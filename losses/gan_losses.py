import torch
import torch.nn as nn

class GANLosses:
    def __init__(
        self,
        device: torch.device, # 计算设备（例如 'cuda' 或 'cpu'）
        gan_loss_type: str = 'bce' # GAN损失类型，支持 'bce' 或 'lsgan'
    ):
        """
        初始化 GAN 损失计算器。

        Args:
            device (torch.device): 计算设备（例如 'cuda' 或 'cpu'）。
            gan_loss_type (str): GAN 的损失类型，目前支持 'bce' (Binary Cross Entropy) 和 'lsgan' (Least Squares GAN)。
        """
        super(GANLosses, self).__init__()
        self.device = device
        self.gan_loss_type = gan_loss_type

        if gan_loss_type == 'bce':
            # BCELoss 适合判别器输出为 sigmoid 激活后的概率值的情况
            # 使用 BCEWithLogitsLoss 可以提高数值稳定性，它会在内部对 logits 进行 Sigmoid
            # reduction='mean' 会计算所有非批量维度的平均损失
            self.adversarial_loss = nn.BCEWithLogitsLoss(reduction='mean').to(self.device)
        elif gan_loss_type == 'lsgan':
            # LSGAN 使用 MSELoss
            # 对于判别器，真实目标是 1，伪造目标是 0
            # 对于生成器，目标是 1
            # reduction='mean' 会计算所有非批量维度的平均损失
            self.adversarial_loss = nn.MSELoss(reduction='mean').to(self.device)
        else:
            raise ValueError(f"Unsupported GAN loss type: {gan_loss_type}. Choose 'bce' or 'lsgan'.")

    def discriminator_loss(
        self,
        D_real_output: torch.Tensor, # 判别器对真实样本的输出
        D_fake_output: torch.Tensor # 判别器对伪造样本的输出
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算判别器的损失。
        目标：最大化判别器区分真实样本和伪造样本的能力。

        Args:
            D_real_output (torch.Tensor): 判别器对真实样本的输出 (原始 logits)。
            D_fake_output (torch.Tensor): 判别器对伪造样本的输出 (原始 logits)。

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 包含总损失、真实样本损失、伪造样本损失的元组，均为标量张量。
        """
        # 确保输入张量在正确的设备上
        D_real_output = D_real_output.to(self.device)
        D_fake_output = D_fake_output.to(self.device)

        if self.gan_loss_type == 'bce':
            # 使用 BCEWithLogitsLoss，目标标签与 logits 直接计算
            # 真实样本的目标标签是 1
            real_labels = torch.ones_like(D_real_output, device=self.device)
            loss_real = self.adversarial_loss(D_real_output, real_labels)

            # 伪造样本的目标标签是 0
            fake_labels = torch.zeros_like(D_fake_output, device=self.device)
            loss_fake = self.adversarial_loss(D_fake_output, fake_labels)

            total_loss_D = loss_real + loss_fake
            return total_loss_D, loss_real, loss_fake

        elif self.gan_loss_type == 'lsgan':
            # LSGAN 损失：mean((D_real - 1)^2) + mean((D_fake - 0)^2)
            # 目标标签是 1 和 0
            loss_real = self.adversarial_loss(D_real_output, torch.ones_like(D_real_output, device=self.device)) # (D_real - 1)^2
            loss_fake = self.adversarial_loss(D_fake_output, torch.zeros_like(D_fake_output, device=self.device)) # (D_fake - 0)^2

            total_loss_D = loss_real + loss_fake
            return total_loss_D, loss_real, loss_fake

        else:
            # 如果 gan_loss_type 不支持，在 __init__ 中已经抛出错误，这里理论上不会到达
            raise ValueError(f"Unsupported GAN loss type: {self.gan_loss_type}. Cannot calculate discriminator loss.")

    def generator_loss(
        self,
        D_fake_output: torch.Tensor # 判别器对伪造样本的输出
    ) -> torch.Tensor:
        """
        计算生成器的对抗性损失。
        目标：误导判别器，使伪造样本被判别器认为是真实样本。

        Args:
            D_fake_output (torch.Tensor): 判别器对伪造样本的输出 (原始 logits)。

        Returns:
            torch.Tensor: 生成器的对抗性损失，形状为 () 的标量张量。
        """
        # 确保输入张量在正确的设备上
        D_fake_output = D_fake_output.to(self.device)

        if self.gan_loss_type == 'bce':
            # 生成器希望伪造样本被判别器认为是真实样本 (目标标签是 1)
            target_labels = torch.ones_like(D_fake_output, device=self.device)
            # 使用 BCEWithLogitsLoss，目标标签与 logits 直接计算
            loss_G_gan = self.adversarial_loss(D_fake_output, target_labels)
            return loss_G_gan

        elif self.gan_loss_type == 'lsgan':
            # LSGAN 损失：mean((D_fake - 1)^2)
            # 目标标签是 1
            loss_G_gan = self.adversarial_loss(D_fake_output, torch.ones_like(D_fake_output, device=self.device)) # (D_fake - 1)^2
            return loss_G_gan

        else:
             # 如果 gan_loss_type 不支持，在 __init__ 中已经抛出错误，这里理论上不会到达
             raise ValueError(f"Unsupported GAN loss type: {self.gan_loss_type}. Cannot calculate generator loss.")

# 示例用法 (在训练脚本中): # 添加注释以说明示例用途
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # 初始化损失计算器，根据配置选择类型
# # 例如从 config 对象获取: gan_loss_type = cfg.losses.gan_type
# gan_loss_calculator = GANLosses(device=device, gan_loss_type='lsgan') # 示例使用 LSGAN
#
# # 假设 D_real 和 D_fake 是判别器对真实和伪造批次的原始 logits 输出
# # 对于 CNN Discriminator: D_real_outputs, D_fake_outputs shape: (batch_size, 1)
# # 对于 PatchGAN Discriminator: D_real_outputs, D_fake_outputs shape: (batch_size, 1, patch_N, patch_H, patch_W)
# # 注意：这里假设判别器输出的是 logits，而不是 sigmoid 后的概率。
# # 如果判别器输出了 sigmoid 后的概率，并且你使用的是 BCELoss (而非 BCEWithLogitsLoss)，需要调整。
# # 示例：假设 discriminator(real_batch) 和 discriminator(fake_batch) 返回 logits
# D_real_outputs = discriminator(real_batch)
# D_fake_outputs = discriminator(fake_batch)
#
# # 计算判别器损失
# total_d_loss, loss_real, loss_fake = gan_loss_calculator.discriminator_loss(D_real_outputs, D_fake_outputs)
#
# # 计算生成器损失 (仅基于判别器对伪造样本的判断)
# # 在训练生成器时，需要重新通过判别器获取对生成样本的判断
# # G_output = generator(input_batch) # 生成对抗样本或扰动
# # Adversarial_input = original_input + G_output # 根据阶段应用扰动
# # D_fake_outputs_for_G_train = discriminator(Adversarial_input) # 将对抗样本输入判别器获取 logits
# # 示例：假设 D_fake_outputs_for_G_train 是判别器对生成样本的 logits 输出
# D_fake_outputs_for_G_train = discriminator(generator(input_batch)) # 简化示例
# g_loss_gan = gan_loss_calculator.generator_loss(D_fake_outputs_for_G_train)

# 其他损失 (攻击损失、正则化损失) 需要单独计算和整合。
# 总生成器损失通常是 g_loss_gan + attack_loss + regularization_losses
# 总判别器损失是 total_d_loss
