import torch
import torch.nn as nn

class GANLosses:
    def __init__(self, device, gan_loss_type='bce'):
        super(GANLosses, self).__init__()
        self.device = device
        self.gan_loss_type = gan_loss_type

        if gan_loss_type == 'bce':
            # BCELoss 适合判别器输出为 sigmoid 激活后的概率值的情况
            # 将 reduction 改为 'mean'，以便对 PatchGAN 的输出进行平均
            self.adversarial_loss = nn.BCELoss(reduction='mean').to(self.device)
        elif gan_loss_type == 'lsgan':
            # LSGAN 使用 MSELoss
            # 对于判别器，真实目标是 1，伪造目标是 0
            # 对于生成器，目标是 1
            # reduction='mean' 会计算所有非批次维度的平均损失
            self.adversarial_loss = nn.MSELoss(reduction='mean').to(self.device)
        else:
            raise ValueError(f"Unsupported GAN loss type: {gan_loss_type}. Choose 'bce' or 'lsgan'.")

    def discriminator_loss(self, D_real_output, D_fake_output):
        """
        计算判别器的损失。
        支持处理 (B, ...) 形状的输出，例如 (B, 1) 或 (B, 1, patch_N, patch_H, patch_W)。
        损失将计算在所有非批次维度的平均值。

        Args:
            D_real_output (torch.Tensor): 判别器对真实样本的输出。
                                        - 'bce': 经过 Sigmoid 激活的概率值 (0-1)。
                                        - 'lsgan': 原始 logits。
            D_fake_output (torch.Tensor): 判别器对伪造样本的输出。
                                        - 'bce': 经过 Sigmoid 激活的概率值 (0-1)。
                                        - 'lsgan': 原始 logits。
        Returns:
            torch.Tensor: 判别器损失 (标量)。
        """
        if self.gan_loss_type == 'bce':
            # 真实样本的目标标签是 1
            real_labels = torch.ones_like(D_real_output, device=self.device)
            loss_real = self.adversarial_loss(D_real_output, real_labels)

            # 伪造样本的目标标签是 0
            fake_labels = torch.zeros_like(D_fake_output, device=self.device)
            loss_fake = self.adversarial_loss(D_fake_output, fake_labels)

            total_loss_D = loss_real + loss_fake
            return total_loss_D

        elif self.gan_loss_type == 'lsgan':
            # LSGAN 损失：mean((D_real - 1)^2) + mean((D_fake - 0)^2)
            # 注意：这里期望 D_real_output 和 D_fake_output 是原始 logits，而不是 sigmoid 后的概率
            # 如果判别器输出了 Sigmoid 后的概率，这个计算方式会不符合 LSGAN 原文。
            # 假设为了兼容性，即使输出是概率，我们也直接使用。
            # 更标准的 LSGAN 应该配合判别器输出 logits。
            loss_real = self.adversarial_loss(D_real_output, torch.ones_like(D_real_output, device=self.device)) # (D_real - 1)^2
            loss_fake = self.adversarial_loss(D_fake_output, torch.zeros_like(D_fake_output, device=self.device)) # (D_fake - 0)^2

            total_loss_D = loss_real + loss_fake
            return total_loss_D

        else:
            raise ValueError(f"Unsupported GAN loss type: {self.gan_loss_type}. Cannot calculate discriminator loss.")

    def generator_loss(self, D_fake_output):
        """
        计算生成器的对抗性损失 (希望判别器将伪造样本判为真实)。
        支持处理 (B, ...) 形状的输出。

        Args:
            D_fake_output (torch.Tensor): 判别器对伪造样本的输出。
                                        - 'bce': 经过 Sigmoid 激活的概率值 (0-1)。
                                        - 'lsgan': 原始 logits。
        Returns:
            torch.Tensor: 生成器的对抗性损失 (标量)。
        """
        if self.gan_loss_type == 'bce':
            # 生成器希望伪造样本被判别器认为是真实样本 (目标标签是 1)
            target_labels = torch.ones_like(D_fake_output, device=self.device)
            loss_G_gan = self.adversarial_loss(D_fake_output, target_labels)
            return loss_G_gan

        elif self.gan_loss_type == 'lsgan':
            # LSGAN 损失：mean((D_fake - 1)^2)
            # 注意：这里期望 D_fake_output 是原始 logits
            loss_G_gan = self.adversarial_loss(D_fake_output, torch.ones_like(D_fake_output, device=self.device)) # (D_fake - 1)^2
            return loss_G_gan

        else:
             raise ValueError(f"Unsupported GAN loss type: {self.gan_loss_type}. Cannot calculate generator loss.")

# 示例用法 (在训练脚本中):
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# gan_loss_calculator_bce = GANLosses(device=device, gan_loss_type='bce')
# gan_loss_calculator_lsgan = GANLosses(device=device, gan_loss_type='lsgan')
#
# # 假设 D_real 和 D_fake 是判别器对真实和伪造批次的输出
# # 如果使用 BCELoss，判别器输出需要 Sigmoid。
# # 如果使用 LSGAN Loss，判别器输出层通常不加 Sigmoid，直接输出 logits。
# # 对于 CNN Discriminator: D_real_outputs, D_fake_outputs shape: (batch_size, 1) (Sigmoid for BCE, Logits for LSGAN)
# # 对于 PatchGAN Discriminator: D_real_outputs, D_fake_outputs shape: (batch_size, 1, patch_N, patch_H, patch_W) (Sigmoid for BCE, Logits for LSGAN)
# D_real_outputs = discriminator(real_batch)
# D_fake_outputs = discriminator(fake_batch)
#
# # 计算判别器损失
# d_loss = gan_loss_calculator.discriminator_loss(D_real_outputs, D_fake_outputs)
#
# # 计算生成器损失 (仅基于判别器对伪造样本的判断)
# # 在训练生成器时，D_fake_outputs_for_G_train 是 discriminator(generator(input_batch))
# g_loss_gan = gan_loss_calculator.generator_loss(D_fake_outputs_for_G_train)
