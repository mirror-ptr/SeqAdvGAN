import sys
import os
# 将项目根目录添加到 Python 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random

# 导入你的模型、损失函数和工具
from models.generator_cnn import Generator
from models.discriminator_cnn import SequenceDiscriminatorCNN
from models.discriminator_patchgan import PatchDiscriminator3d
from losses.gan_losses import GANLosses
from losses.attack_losses import AttackLosses
from losses.regularization_losses import linf_norm, l2_norm, l2_penalty, total_variation_loss

# 导入可视化函数
from utils.vis_utils import visualize_samples_and_outputs, visualize_attention_maps

# 导入 ATN 工具和数据工具
from utils.atn_utils import load_atn_model, get_atn_outputs
from utils.data_utils import create_mock_dataloader

# 导入评估函数
from utils.eval_utils import evaluate_model # 导入评估函数

# 训练主函数
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 加载模型
    generator = Generator(in_channels=args.channels, out_channels=args.channels, epsilon=args.epsilon).to(device) # 确保通道数一致
    # 根据 args.discriminator_type 选择判别器
    if args.discriminator_type == 'cnn':
        discriminator = SequenceDiscriminatorCNN(in_channels=args.channels).to(device)
        print("Using CNN Discriminator")
    elif args.discriminator_type == 'patchgan':
        discriminator = PatchDiscriminator3d(in_channels=args.channels).to(device)
        print("Using PatchGAN Discriminator")
    else:
        raise ValueError(f"Unsupported discriminator type: {args.discriminator_type}")
    atn_model = load_atn_model(args.atn_model_path).to(device) # 加载 ATN 模型

    # 设置 ATN 模型为评估模式，因为我们不对其进行训练
    atn_model.eval()
    for param in atn_model.parameters():
        param.requires_grad = False # 冻结 ATN 模型参数

    # TODO: 如果使用 LSGAN，需要确保判别器的输出层没有 Sigmoid
    # 根据选择的 GAN loss 类型调整判别器
    if args.gan_loss_type == 'lsgan':
        if isinstance(discriminator, SequenceDiscriminatorCNN) and isinstance(discriminator.classifier[-1], nn.Sigmoid):
             # 移除 CNN 判别器最后一层的 Sigmoid
             discriminator.classifier = nn.Sequential(*list(discriminator.classifier.children())[:-1])
             print("Removed Sigmoid from CNN Discriminator for LSGAN.")
        elif isinstance(discriminator, PatchDiscriminator3d) and isinstance(discriminator.sigmoid, nn.Sigmoid):
             # 移除 PatchGAN 判别器的 Sigmoid 模块
             discriminator.sigmoid = nn.Identity() # 使用 Identity 替代 Sigmoid
             print("Replaced Sigmoid with Identity in PatchGAN Discriminator for LSGAN.")

    # 2. 设置优化器
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))

    # 3. 设置损失函数
    gan_losses = GANLosses(device=device, gan_loss_type=args.gan_loss_type)
    attack_losses = AttackLosses(device,
                                 attention_loss_weight=args.attention_loss_weight,
                                 decision_loss_weight=args.decision_loss_weight,
                                 topk_k=args.topk_k)

    # 4. 数据加载（使用模拟数据）
    # TODO: 未来切换到真实数据加载器，并加载区域掩码
    # dataloader = create_real_dataloader(...) # 需要在 utils/data_utils.py 中实现
    dataloader = create_mock_dataloader(batch_size=args.batch_size,
                                        num_samples=args.num_mock_samples, # 使用 args 中的 num_mock_samples
                                        sequence_length=args.sequence_length,
                                        channels=args.channels,
                                        height=args.height,
                                        width=args.width
    )
    # TODO: 创建评估数据加载器 (可以使用与训练相同的数据加载器，或者一个单独的数据集)
    eval_dataloader = create_mock_dataloader(batch_size=args.batch_size, # 可以使用不同的 batch size
                                            num_samples=getattr(args, 'num_eval_samples', args.num_mock_samples), # 使用 args 中的 num_eval_samples 或默认值
                                            sequence_length=args.sequence_length,
                                            channels=args.channels,
                                            height=args.height,
                                            width=args.width
    )

    # 5. TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=args.log_dir)

    # 6. 模型保存路径
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 7. 加载检查点 (如果存在)
    start_epoch = 0
    if args.resume_checkpoint:
        if os.path.exists(args.resume_checkpoint):
            print(f"Loading checkpoint from {args.resume_checkpoint}")
            checkpoint = torch.load(args.resume_checkpoint, map_location=device)
            generator.load_state_dict(checkpoint['generator_state_dict'])
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming training from epoch {start_epoch}")
        else:
            print(f"Checkpoint not found at {args.resume_checkpoint}. Starting from epoch 0.")


    print("Starting training...")
    global_step = 0

    # 训练主循环
    for epoch in range(start_epoch, args.epochs):
        generator.train() # 设置生成器为训练模式
        discriminator.train() # 设置判别器为训练模式

        # TODO: 未来从 dataloader 获取 batch 数据，包括 original_features, decision_mask, attention_mask
        # for i, (original_features_batch, decision_mask_batch, attention_mask_batch) in enumerate(dataloader):
        # original_features = original_features_batch.to(device)
        # decision_mask = decision_mask_batch.to(device) # (B, W, H)
        # attention_mask = attention_mask_batch.to(device) # (B, head, N, N) 或兼容形状

        for i, original_features_batch in enumerate(dataloader): # 重命名以避免与占位符函数冲突
            original_features = original_features_batch.to(device) # (B, 128, N, W, H)

            # TODO: 在使用模拟数据时，可以创建全1的掩码，或者根据需要创建模拟掩码用于测试区域攻击框架
            # 例如：
            # decision_mask = torch.ones(original_features.shape[0], args.height, args.width, device=device)
            # attention_mask = torch.ones(original_features.shape[0], args.num_vis_heads, args.sequence_length, args.sequence_length, device=device) # 假设 head=num_vis_heads
            decision_mask = None # 暂时不使用掩码
            attention_mask = None # 暂时不使用掩码


            #############################
            # 训练判别器
            #############################
            optimizer_D.zero_grad()

            # 对真实样本的判断
            # 如果使用 LSGAN，这里的判别器输出是 logits，如果使用 BCE，输出是 Sigmoid 后的概率
            D_real_output = discriminator(original_features)
            # 在 gan_losses.py 中，discriminator_loss 的第一个参数是 D_real_output, 第二个是 D_fake_output
            # loss_D_real = gan_losses.adversarial_loss(D_real_output, torch.ones_like(D_real_output, device=device)) # LSGAN target 1, BCE target 1

            # 生成对抗扰动
            delta = generator(original_features)
            # 确保扰动在 epsilon 范围内 (通过生成器结构已经保证，这里可以作为额外的检查或监控)
            # delta = torch.clamp(delta, -args.epsilon, args.epsilon) # 如果生成器结构不能保证，需要 clamp

            adversarial_features = original_features + delta
            # 再次 clamp 确保对抗样本也在合理范围内（如果需要）
            # adversarial_features = torch.clamp(adversarial_features, min_val, max_val) # 根据实际数据范围设定

            # 对伪造样本 (对抗样本) 的判断
            # 在训练D时，G的参数需要冻结
            # 如果使用 LSGAN，这里的判别器输出是 logits，如果使用 BCE，输出是 Sigmoid 后的概率
            D_fake_output = discriminator(adversarial_features.detach())
            # loss_D_fake = gan_losses.adversarial_loss(D_fake_output, torch.zeros_like(D_fake_output, device=device)) # LSGAN target 0, BCE target 0

            # 判别器总损失 (使用 gan_losses.py 中的 discriminator_loss)
            # loss_D = loss_D_real + loss_D_fake # 这是手动计算 BCE Loss 的方式
            loss_D = gan_losses.discriminator_loss(D_real_output, D_fake_output) # 使用封装好的函数


            # 反向传播和更新参数
            loss_D.backward()
            optimizer_D.step()

            #############################
            # 训练生成器
            #############################
            optimizer_G.zero_grad()

            # 生成对抗扰动 (不需要重新生成，delta 和 adversarial_features 已经计算过)
            # delta = generator(original_features) # 这会重新计算，如果 G 的参数更新了则需要
            # adversarial_features = original_features + delta # 这也需要重新计算

            # 为了确保梯度流正确，并且在G的训练步骤中使用最新的G输出：
            current_delta = generator(original_features)
            current_adversarial_features = original_features + current_delta


            # 获取对抗样本下的判别器输出 (希望判真)
            # 如果使用 LSGAN，这里的判别器输出是 logits，如果使用 BCE，输出是 Sigmoid 后的概率
            D_fake_output_for_G = discriminator(current_adversarial_features) # 使用 current_adversarial_features
            loss_G_gan = gan_losses.generator_loss(D_fake_output_for_G) # 目标是骗过判别器

            # 获取原始样本下的 ATN 输出
            original_atn_outputs = get_atn_outputs(atn_model, original_features)

            # 获取对抗样本下的 ATN 输出
            adversarial_atn_outputs = get_atn_outputs(atn_model, current_adversarial_features) # 使用 current_adversarial_features

            # 计算攻击损失
            # TODO: 未来在这里传入 decision_loss_type, attention_loss_type, decision_region_mask, attention_region_mask
            # 可以从 args 中获取损失类型，从 dataloader 获取掩码
            loss_G_attack = attack_losses.get_generator_attack_loss(
                original_atn_outputs,
                adversarial_atn_outputs,
                decision_loss_type=args.decision_loss_type, # 从 args 获取
                attention_loss_type=args.attention_loss_type, # 从 args 获取
                decision_region_mask=decision_mask, # 传入掩码
                attention_region_mask=attention_mask # 传入掩码
            )

            # 计算正则化损失 (扰动范数 + TV loss)
            loss_G_reg_l2 = l2_penalty(current_delta) # 使用 L2 范数作为惩罚项
            loss_G_reg_tv = total_variation_loss(current_delta) # 计算 TV loss

            # 生成器总损失
            loss_G = loss_G_gan + args.attack_loss_weight * loss_G_attack + \
                     args.reg_loss_weight * loss_G_reg_l2 + args.tv_loss_weight * loss_G_reg_tv # 添加 TV loss

            # 反向传播和更新参数
            loss_G.backward()
            optimizer_G.step()

            #############################
            # 记录和可视化
            #############################
            if global_step % args.log_interval == 0:
                print(f"Epoch [{epoch}/{args.epochs}] Step [{i}/{len(dataloader)}] \
                      Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f} \
                      Loss_G_GAN: {loss_G_gan.item():.4f} Loss_G_Attack: {loss_G_attack.item():.4f} \
                      Loss_G_Reg_L2: {loss_G_reg_l2.item():.4f} Loss_G_Reg_TV: {loss_G_reg_tv.item():.4f}") # 记录 L2 和 TV loss

                writer.add_scalar('Loss/Discriminator', loss_D.item(), global_step)
                writer.add_scalar('Loss/Generator', loss_G.item(), global_step)
                writer.add_scalar('Loss/Generator_GAN', loss_G_gan.item(), global_step)
                writer.add_scalar('Loss/Generator_Attack', loss_G_attack.item(), global_step)
                writer.add_scalar('Loss/Generator_Regularization_L2', loss_G_reg_l2.item(), global_step) # 记录 L2 loss
                writer.add_scalar('Loss/Generator_Regularization_TV', loss_G_reg_tv.item(), global_step) # 记录 TV loss

                # 记录扰动范数
                with torch.no_grad():
                    # 使用当前步骤的扰动 current_delta
                    linf = linf_norm(current_delta).mean().item()
                    l2 = l2_norm(current_delta).mean().item()
                    writer.add_scalar('Perturbation/Linf_Norm', linf, global_step)
                    writer.add_scalar('Perturbation/L2_Norm', l2, global_step)

                # 可视化部分
                if global_step % args.vis_interval == 0: # 假设有 vis_interval 参数
                    with torch.no_grad():
                         visualize_samples_and_outputs(
                             writer,
                             original_features, # (B, C, N, W, H)
                             current_delta,     # (B, C, N, W, H)
                             current_adversarial_features, # (B, C, N, W, H)
                             original_atn_outputs.get('decision'), # (B, W, H)
                             adversarial_atn_outputs.get('decision'),# (B, W, H)
                             global_step,
                             num_samples=args.num_vis_samples, # 假设有 num_vis_samples 参数
                             sequence_step_to_vis=args.sequence_step_to_vis # 假设有 sequence_step_to_vis 参数
                         )

                         # 可视化注意力矩阵 (如果 ATN 输出了注意力，并且想在训练时可视化)
                         # if original_atn_outputs.get('attention') is not None or adversarial_atn_outputs.get('attention') is not None:
                         #     visualize_attention_maps(
                         #         writer,
                         #         original_atn_outputs.get('attention'), # (B, head, N, N)
                         #         adversarial_atn_outputs.get('attention'),# (B, head, N, N)
                         #         global_step,
                         #         num_samples=args.num_vis_samples, # 可以控制可视化样本数量
                         #         num_heads_to_vis=1 # 可以控制可视化多少个注意力头
                         #     )


            global_step += 1

        # --- 评估阶段 ---
        if (epoch + 1) % args.eval_interval == 0:
            print(f"\nEvaluating model at epoch {epoch + 1}...")
            # 使用评估数据加载器和评估函数
            eval_results = evaluate_model(generator, discriminator, atn_model, eval_dataloader, device, args) # 将 args 传入评估函数
            for key, value in eval_results.items():
                if not np.isnan(value): # 只记录非 NaN 的指标
                    writer.add_scalar(f'Evaluation/{key}', value, epoch + 1)

            generator.train() # Switch back to training mode
            discriminator.train() # Switch back to training mode
            print("Evaluation finished for this epoch.")


        # 保存检查点
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    print("Training finished.")
    writer.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train SeqAdvGAN Generator and Discriminator")
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=4, help='input batch size for training')
    parser.add_argument('--lr_g', type=float, default=0.0002, help='learning rate for Generator')
    parser.add_argument('--lr_d', type=float, default=0.0002, help='learning rate for Discriminator')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--epsilon', type=float, default=0.03, help='L-infinity constraint for perturbation')
    parser.add_argument('--attack_loss_weight', type=float, default=10.0, help='weight for generator attack loss')
    parser.add_argument('--reg_loss_weight', type=float, default=1.0, help='weight for perturbation regularization loss (e.g., L2)')
    parser.add_argument('--tv_loss_weight', type=float, default=0.0, help='weight for total variation loss')
    parser.add_argument('--attention_loss_weight', type=float, default=1.0, help='weight for attention loss within attack loss')
    parser.add_argument('--decision_loss_weight', type=float, default=1.0, help='weight for decision loss within attack loss')

    # 添加攻击损失类型和掩码相关参数
    parser.add_argument('--decision_loss_type', type=str, default='mse', choices=['mse', 'l1', 'kl', 'js'], help='type of loss for decision map attack')
    parser.add_argument('--attention_loss_type', type=str, default='mse', choices=['mse', 'l1', 'kl', 'js', 'topk'], help='type of loss for attention map attack')
    parser.add_argument('--use_region_mask', action='store_true', help='whether to use region masks for attack loss')
    # TODO: 未来添加 mask_path 参数，指定掩码文件路径或生成掩码的策略
    # parser.add_argument('--mask_strategy', type=str, default='none', help='Strategy for generating/loading masks')
    # parser.add_argument('--mask_path', type=str, default=None, help='Path to mask file (if using file strategy)')
    parser.add_argument('--topk_k', type=int, default=1, help='K value for Top-K attention loss (also used for evaluation criterion)')

    # 添加判别器选择参数
    parser.add_argument('--discriminator_type', type=str, default='cnn', choices=['cnn', 'patchgan'], help='type of discriminator to use')

    # 添加 GAN Loss 类型选择参数
    parser.add_argument('--gan_loss_type', type=str, default='bce', choices=['bce', 'lsgan'], help='type of GAN loss to use')

    parser.add_argument('--log_dir', type=str, default='runs/seqadvgan_train', help='tensorboard log directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='directory to save model checkpoints')
    parser.add_argument('--log_interval', type=int, default=10, help='how many batches to wait before logging training status')
    parser.add_argument('--save_interval', type=int, default=10, help='how many epochs to wait before saving checkpoint')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='path to checkpoint to resume training from')
    parser.add_argument('--atn_model_path', type=str, default='path/to/atn_model', help='path to the trained ATN model')

    # 模拟数据参数
    parser.add_argument('--num_mock_samples', type=int, default=100, help='number of mock data samples for training')
    parser.add_argument('--num_eval_samples', type=int, default=50, help='number of mock data samples for evaluation') # 添加评估样本数量参数
    parser.add_argument('--sequence_length', type=int, default=16, help='sequence length of mock data')
    parser.add_argument('--channels', type=int, default=128, help='number of channels of mock data (ATN feature size)')
    parser.add_argument('--height', type=int, default=64, help='height of mock data')
    parser.add_argument('--width', type=int, default=64, help='width of mock data')

    # 可视化参数
    parser.add_argument('--vis_interval', type=int, default=50, help='how many global steps to wait before visualizing samples')
    parser.add_argument('--num_vis_samples', type=int, default=4, help='number of samples to visualize')
    parser.add_argument('--sequence_step_to_vis', type=int, default=0, help='the sequence step to visualize (0-indexed)')

    # 评估参数
    parser.add_argument('--eval_interval', type=int, default=5, help='how many epochs to wait before evaluating') # 添加评估间隔参数
    parser.add_argument('--eval_success_threshold', type=float, default=0.1, help='threshold for attack success in evaluation') # 添加评估成功阈值参数
    parser.add_argument('--eval_success_criterion', type=str, default='mse_diff_threshold', choices=['mse_diff_threshold', 'mean_change_threshold', 'topk_value_drop', 'topk_position_change'], help='criterion for attack success in evaluation') # 添加评估成功标准参数，限制选项


    args = parser.parse_args()

    # 根据 use_region_mask 标志决定是否生成或加载掩码 (目前只使用模拟数据，创建全1或全0掩码用于测试)
    # TODO: 未来根据 args.mask_strategy 或 args.mask_path 加载真实掩码
    decision_mask = None
    attention_mask = None
    if args.use_region_mask:
        print("Warning: use_region_mask is enabled, but using mock full masks. Implement real mask loading/generation in data_utils.py.")
        # 简单的全1掩码示例，模拟对整个区域进行攻击
        # future: 需要根据 dataloader 返回的真实数据形状创建掩码
        # 假设 decision map 形状是 (B, H, W)， attention map 形状是 (B, head, N, N)
        mock_batch_size = args.batch_size # 模拟批次大小
        mock_height = args.height # 模拟高度
        mock_width = args.width # 模拟宽度
        mock_sequence_length = args.sequence_length # 模拟序列长度
        mock_num_heads = 4 # 模拟注意力头数量 (与 MockATN 一致)

        decision_mask = torch.ones(mock_batch_size, mock_height, mock_width, device=device)
        attention_mask = torch.ones(mock_batch_size, mock_num_heads, mock_sequence_length, mock_sequence_length, device=device)
        # TODO: 如果需要测试局部攻击，可以创建局部为1的掩码
        # decision_mask[:, 10:20, 30:40] = 0 # 示例：将部分区域置零


    train(args) 