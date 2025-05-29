import sys
import os
# 将项目根目录添加到 Python 路径，以便导入项目内的模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random # 导入 random 库用于设置随机种子
import argparse # 导入 argparse 用于命令行参数解析
from typing import Optional, Dict, Any # 导入类型提示 Optional, Dict, Any
from collections import deque # 导入 deque 用于实现固定长度的队列 (用于动态平衡损失历史)
from tqdm import tqdm # Import tqdm function from tqdm library

# 导入模型、损失函数和工具模块
from models.generator_cnn import Generator # 导入生成器模型
from models.discriminator_cnn import SequenceDiscriminatorCNN # 导入 CNN 判别器
from models.discriminator_patchgan import PatchDiscriminator3d # 导入 PatchGAN 判别器
from losses.gan_losses import GANLosses # 导入 GAN 损失函数类
from losses.attack_losses import AttackLosses # 导入攻击损失函数类
# 导入正则化损失函数 (L-inf, L2, L2参数惩罚, TV)
from losses.regularization_losses import linf_norm, l2_norm, l2_penalty, total_variation_loss

# 导入可视化工具函数
from utils.vis_utils import visualize_samples_and_outputs, visualize_attention_maps, visualize_stage2_pixel_attack

# 导入 ATN 工具和数据工具
from utils.atn_utils import load_atn_model, get_atn_outputs
from utils.data_utils import GameVideoDataset, create_mock_dataloader, worker_init_fn

# 导入评估工具函数
from utils.eval_utils import evaluate_model # 导入评估函数

# 导入配置工具函数
from utils.config_utils import parse_args_and_config

def set_seed(seed: int) -> None:
    """
    设置所有必要的随机种子以确保实验的可复现性。
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(cfg: Any) -> None:
    """
    SeqAdvGAN 生成器和判别器的训练主函数 (阶段二: 像素级攻击)。

    Stage 2 数据流:
    1. 原始图像 (B, 3, T, H, W) -> Generator -> 像素扰动 (B, 3, T, H, W)
    2. 原始图像 + 像素扰动 = 对抗图像 (B, 3, T, H, W)
    3. 对抗图像 -> IrisXeonNet -> 感知特征 (B, 128, T, H', W')
    4. 感知特征 -> AttentionTransformer + TriggerNet -> 最终输出
    5. 攻击目标: 使对抗图像的TriggerNet输出与原始图像的TriggerNet输出尽可能不同
    """
    # 确保当前训练阶段是 2
    assert cfg.training.train_stage == 2, f"This script is only for training stage 2, but config specifies stage {cfg.training.train_stage}."
    print(f"Starting training for Stage {cfg.training.train_stage} (Pixel-level Attack)...")

    # 设置随机种子以确保实验的可重现性
    set_seed(cfg.training.seed)

    # 获取计算设备
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load ATN Model Components (Frozen) ---
    print("Loading ATN model components...")
    atn_model_dict = load_atn_model(cfg=cfg, device=device)

    if atn_model_dict is None:
        print("Error: Failed to load ATN model components. Exiting training.")
        sys.exit(1)
        
    # 验证ATN模型组件
    required_components = ['xeon_net', 'attention_transformer', 'trigger_net']
    for component in required_components:
        if component not in atn_model_dict or atn_model_dict[component] is None:
            print(f"Error: Missing or None ATN component: {component} in atn_model_dict")
            sys.exit(1)
            
    # 确保所有ATN模型组件都被冻结
    for model_name, model in atn_model_dict.items():
        if model is not None:
            for param in model.parameters():
                param.requires_grad = False
            model.eval()
    print("ATN model components loaded successfully and frozen.")

    # --- 2. Initialize Generator and Discriminator Models ---
    print("Initializing Generator and Discriminator...")

    # Stage 2: Generator接收原始图像并输出像素级扰动
    generator = Generator(
        in_channels=cfg.model.generator.in_channels,  # 通常是3 (RGB)
        out_channels=cfg.model.generator.out_channels,  # 通常是3 (RGB扰动)
        num_bottlenecks=cfg.model.generator.num_bottlenecks,
        base_channels=cfg.model.generator.base_channels,
        epsilon=cfg.model.generator.epsilon  # 像素级扰动上限 (L-inf)
    ).to(device)

    # Stage 2: Discriminator接收对抗图像作为输入
    if cfg.model.discriminator.type == 'cnn':
        discriminator = SequenceDiscriminatorCNN(
            in_channels=cfg.model.discriminator.in_channels,  # 通常是3 (RGB)
            base_channels=cfg.model.discriminator.base_channels
        ).to(device)
        print("Using CNN Discriminator for Stage 2")
    elif cfg.model.discriminator.type == 'patchgan':
        discriminator = PatchDiscriminator3d(
            in_channels=cfg.model.discriminator.in_channels,  # 通常是3 (RGB)
            base_channels=cfg.model.discriminator.base_channels
        ).to(device)
        print("Using PatchGAN Discriminator for Stage 2")
    else:
        print(f"Error: Unsupported discriminator type: {cfg.model.discriminator.type}")
        sys.exit(1)

    print("Generator and Discriminator models initialized.")

    # --- 3. Define Loss Functions ---
    print("Defining loss functions...")
    gan_losses = GANLosses(device=device, gan_loss_type=cfg.losses.gan_type)
    print(f"Using GAN loss type: {cfg.losses.gan_type}")

    # 正则化损失权重
    lambda_l_inf = cfg.regularization.lambda_l_inf
    lambda_l2 = cfg.regularization.lambda_l2
    lambda_tv = cfg.regularization.lambda_tv
    lambda_l2_penalty = cfg.regularization.lambda_l2_penalty
    print(f"Using regularization weights: L_inf={lambda_l_inf}, L2={lambda_l2}, TV={lambda_tv}, L2_Params={lambda_l2_penalty}")

    # --- 4. Define Optimizers ---
    print("Defining optimizers...")
    optimizer_G = optim.Adam(
        generator.parameters(),
        lr=cfg.training.lr_g,
        betas=(cfg.training.b1, cfg.training.b2)
    )
    optimizer_D = optim.Adam(
        discriminator.parameters(),
        lr=cfg.training.lr_d,
        betas=(cfg.training.b1, cfg.training.b2)
    )
    print(f"Optimizers initialized with lr_g={cfg.training.lr_g}, lr_d={cfg.training.lr_d}")

    # --- 5. Data Loading ---
    print("Loading data...")
    if cfg.data.use_mock_data:
        print("Using mock data for training.")
        train_dataloader = create_mock_dataloader(
            batch_size=cfg.training.batch_size,
            num_samples=cfg.data.mock_num_samples,
            sequence_length=cfg.data.sequence_length,
            channels=cfg.data.channels,  # 应该是3 (RGB)
            height=cfg.data.height,
            width=cfg.data.width,
            shuffle=True,
            num_workers=cfg.data.num_workers
        )
        eval_dataloader = create_mock_dataloader(
            batch_size=getattr(cfg.evaluation, 'num_eval_samples', cfg.training.batch_size),
            num_samples=getattr(cfg.evaluation, 'num_eval_samples', 100),
            sequence_length=cfg.data.sequence_length,
            channels=cfg.data.channels,
            height=cfg.data.height,
            width=cfg.data.width,
            shuffle=False,
            num_workers=cfg.data.num_workers
        )
    else:
        # 实际数据加载逻辑
        if not os.path.exists(cfg.data.video_path):
            print(f"Error: Video file not found at {cfg.data.video_path}")
            sys.exit(1)
        if not os.path.exists(cfg.data.level_json_path):
            print(f"Error: Level JSON file not found at {cfg.data.level_json_path}")
            sys.exit(1)

        train_dataset = GameVideoDataset(
            video_path=cfg.data.video_path,
            level_json_path=cfg.data.level_json_path,
            sequence_length=cfg.data.sequence_length,
            transform=None,
            target_height=cfg.data.height,
            target_width=cfg.data.width,
            device=device
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            worker_init_fn=worker_init_fn,
            pin_memory=True
        )
        
        eval_dataset = GameVideoDataset(
            video_path=cfg.data.video_path,
            level_json_path=cfg.data.level_json_path,
            sequence_length=cfg.data.sequence_length,
            transform=None,
            target_height=cfg.data.height,
            target_width=cfg.data.width,
            device=device
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=getattr(cfg.evaluation, 'num_eval_samples', cfg.training.batch_size),
            shuffle=False,
            num_workers=cfg.data.num_workers,
            worker_init_fn=worker_init_fn,
            pin_memory=True
        )

    # 检查数据加载器
    if len(train_dataloader) == 0:
        print("Error: Training dataloader is empty.")
        sys.exit(1)
    if len(eval_dataloader) == 0:
        print("Warning: Evaluation dataloader is empty. Evaluation will be skipped.")

    print(f"Training data loaded. Number of batches per epoch: {len(train_dataloader)}")
    print(f"Evaluation data loaded. Number of batches: {len(eval_dataloader)}")

    # --- TensorBoard Logging Setup ---
    log_dir = cfg.logging.log_dir
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    # --- Checkpoint Loading (Optional) ---
    start_epoch = 0
    global_step = 0
    best_eval_metric = -float('inf')  # Stage 2目标是最大化攻击成功率

    if cfg.training.resume_checkpoint:
        print(f"Attempting to resume from checkpoint: {cfg.training.resume_checkpoint}")
        try:
            checkpoint = torch.load(cfg.training.resume_checkpoint, map_location=device)
            generator.load_state_dict(checkpoint['generator_state_dict'])
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            global_step = checkpoint.get('global_step', start_epoch * len(train_dataloader))
            best_eval_metric = checkpoint.get('best_eval_metric', -float('inf'))
            print(f"Checkpoint loaded successfully. Resuming from epoch {start_epoch}, global_step {global_step}.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting training from scratch.")

    # Dynamic GAN Balance Variables
    dynamic_balance_cfg = cfg.training.dynamic_gan_balance
    dynamic_balance_enabled = dynamic_balance_cfg.enabled
    current_D_freq = dynamic_balance_cfg.initial_D_freq
    current_G_freq = dynamic_balance_cfg.initial_G_freq

    if dynamic_balance_enabled:
        d_loss_history = deque(maxlen=dynamic_balance_cfg.loss_history_window)
        g_gan_loss_history = deque(maxlen=dynamic_balance_cfg.loss_history_window)
        print(f"Dynamic GAN balance enabled. Initial D freq: {current_D_freq}, Initial G freq: {current_G_freq}")
    else:
        print("Dynamic GAN balance disabled. Using fixed frequencies.")

    # 损失函数
    mse_loss_fn = nn.MSELoss()
    cosine_similarity_fn = nn.CosineSimilarity(dim=-1)

    print("Starting training...")

    # --- 6. Training Loop (Stage 2: Pixel-level Attack) ---
    for epoch in tqdm(range(start_epoch, cfg.training.num_epochs), desc="Training Epochs"):
        generator.train()
        discriminator.train()

        for i, batch_data in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch}/{cfg.training.num_epochs}"):
            # batch_data shape is likely (B, T, H, C, W) from DataLoader based on the error message
            real_x = batch_data.to(device)

            # --- Train Discriminator ---
            if global_step % current_D_freq == 0:
                optimizer_D.zero_grad()

                # 1. Generator生成像素级扰动delta
                with torch.no_grad():  # D训练时不需要G的梯度
                    delta = generator(real_x)  # delta shape: (B, C, T, H, W)
                    # 约束delta到epsilon球内
                    delta = torch.clamp(delta, -cfg.model.generator.epsilon, cfg.model.generator.epsilon)
                    # 生成对抗图像
                    adversarial_x = real_x + delta
                    # 确保对抗图像在有效范围内
                    adversarial_x = torch.clamp(adversarial_x, 0, 1)

                # 2. Discriminator判别
                D_real_output = discriminator(real_x)  # 真实图像
                D_fake_output = discriminator(adversarial_x)  # 对抗图像（detached）

                # 3. 计算Discriminator损失
                d_total_loss, d_real_loss, d_fake_loss = gan_losses.discriminator_loss(
                    D_real_output, D_fake_output
                )

                # 4. 更新Discriminator
                if torch.isfinite(d_total_loss):
                    d_total_loss.backward()
                    optimizer_D.step()

            # --- Train Generator ---
            if global_step % current_G_freq == 0:
                optimizer_G.zero_grad()

                # 1. Generator生成像素级扰动
                delta_for_G = generator(real_x)  # delta shape: (B, C, T, H, W)
                delta_for_G = torch.clamp(delta_for_G, -cfg.model.generator.epsilon, cfg.model.generator.epsilon)
                adversarial_x_for_G = real_x + delta_for_G
                adversarial_x_for_G = torch.clamp(adversarial_x_for_G, 0, 1)

                # 2. 获取ATN输出 - 完整的数据流
                # 2.1 获取原始图像的ATN输出（无梯度）
                with torch.no_grad():
                    # 感知特征 -> AttentionTransformer + TriggerNet
                    original_atn_outputs = get_atn_outputs(
                        atn_model_dict,
                        real_x, # Pass the raw image
                        cfg=cfg,
                        device=device # Pass the device
                        # perception_features=original_perception_features # Removed redundant parameter
                    )

                # 2.2 获取对抗图像的ATN输出（需要梯度）
                # 感知特征 -> AttentionTransformer + TriggerNet
                adversarial_atn_outputs = get_atn_outputs(
                    atn_model_dict,
                    adversarial_x_for_G, # Pass the adversarial image
                    cfg=cfg,
                    device=device # Pass the device
                    # perception_features=adversarial_perception_features # Removed redundant parameter
                )

                # 3. 计算Generator GAN损失
                D_fake_output_for_G = discriminator(adversarial_x_for_G)
                gen_gan_loss = gan_losses.generator_loss(D_fake_output_for_G)

                # 4. 计算Generator攻击损失 (Stage 2: 攻击TriggerNet输出)
                gen_attack_loss = torch.tensor(0.0, device=device)
                
                # 提取TriggerNet输出
                original_trigger_output = original_atn_outputs.get('trigger_output')
                adversarial_trigger_output = adversarial_atn_outputs.get('trigger_output')

                if original_trigger_output is not None and adversarial_trigger_output is not None:
                    if original_trigger_output.shape == adversarial_trigger_output.shape and original_trigger_output.numel() > 0:
                        # 攻击目标：最大化原始输出与对抗输出之间的差异
                        # 使用MSE损失衡量差异，然后取负值以最大化差异
                        trigger_mse_diff = mse_loss_fn(
                            original_trigger_output.float(), 
                            adversarial_trigger_output.float()
                        )
                        
                        # 也可以使用余弦相似度作为辅助攻击损失
                        if original_trigger_output.dim() > 1:
                            # 将输出展平以计算余弦相似度
                            orig_flat = original_trigger_output.view(original_trigger_output.size(0), -1)
                            adv_flat = adversarial_trigger_output.view(adversarial_trigger_output.size(0), -1)
                            cosine_sim = cosine_similarity_fn(orig_flat, adv_flat).mean()
                            # 攻击目标：最小化余弦相似度（即最大化-cosine_sim）
                            cosine_attack_loss = -cosine_sim
                        else:
                            cosine_attack_loss = torch.tensor(0.0, device=device)

                        # 组合攻击损失
                        gen_attack_loss = (
                            -cfg.losses.decision_loss_weight * trigger_mse_diff + 
                            cfg.losses.get('cosine_loss_weight', 0.1) * cosine_attack_loss
                        )
                    else:
                        print("Warning: TriggerNet outputs shape mismatch or empty.")
                else:
                    print("Warning: TriggerNet outputs are None.")

                # 5. 计算正则化损失
                reg_loss_l_inf = torch.tensor(0.0, device=device)
                if lambda_l_inf > 0 and delta_for_G is not None and delta_for_G.numel() > 0:
                    reg_loss_l_inf = lambda_l_inf * linf_norm(delta_for_G).mean()

                reg_loss_l2 = torch.tensor(0.0, device=device)
                if lambda_l2 > 0 and delta_for_G is not None and delta_for_G.numel() > 0:
                    reg_loss_l2 = lambda_l2 * l2_penalty(delta_for_G).mean()

                reg_loss_tv = torch.tensor(0.0, device=device)
                if lambda_tv > 0 and delta_for_G is not None and delta_for_G.numel() > 0:
                    reg_loss_tv = lambda_tv * total_variation_loss(delta_for_G).mean()

                reg_loss_l2_params = torch.tensor(0.0, device=device)
                if lambda_l2_penalty > 0:
                    gen_l2_reg = torch.tensor(0.0, device=device)
                    for param in generator.parameters():
                        if param.requires_grad:
                            gen_l2_reg += param.square().sum()
                    reg_loss_l2_params = lambda_l2_penalty * gen_l2_reg

                # 6. 组合Generator总损失
                gen_total_loss = (
                    cfg.losses.gan_loss_weight * gen_gan_loss +
                    gen_attack_loss +
                    reg_loss_l_inf +
                    reg_loss_l2 +
                    reg_loss_tv +
                    reg_loss_l2_params
                )

                # 7. 更新Generator
                if torch.isfinite(gen_total_loss):
                    gen_total_loss.backward()
                    optimizer_G.step()
                else:
                    print(f"Warning: Generator loss is not finite at step {global_step}")

            # --- Dynamic GAN Balance Adjustment ---
            if dynamic_balance_enabled and dynamic_balance_cfg.strategy == "loss_ratio_freq_adjust":
                if torch.isfinite(d_total_loss) and torch.isfinite(gen_gan_loss):
                    d_loss_history.append(d_total_loss.item())
                    g_gan_loss_history.append(gen_gan_loss.item())

                    if (global_step + 1) % dynamic_balance_cfg.freq_adjust_interval == 0:
                        avg_d_loss = np.mean(list(d_loss_history)) if d_loss_history else 0.0
                        avg_g_gan_loss = np.mean(list(g_gan_loss_history)) if g_gan_loss_history else 0.0

                        balance_ratio = float('inf')
                        if abs(avg_g_gan_loss) > 1e-8:
                            balance_ratio = avg_d_loss / avg_g_gan_loss
                        elif abs(avg_d_loss) < 1e-8:
                            balance_ratio = 1.0

                        print(f"Step {global_step+1}: Balance Ratio (D/G_GAN): {balance_ratio:.2f}")

                        # 调整训练频率
                        D_dominant_threshold = dynamic_balance_cfg.D_dominant_threshold
                        G_dominant_threshold = dynamic_balance_cfg.G_dominant_threshold
                        
                        if balance_ratio > D_dominant_threshold:
                            current_D_freq = min(current_D_freq + 1, 10)
                            current_G_freq = max(1, current_G_freq - 1)
                        elif balance_ratio < G_dominant_threshold:
                            current_G_freq = min(current_G_freq + 1, 10)
                            current_D_freq = max(1, current_D_freq - 1)

            # --- Logging and Visualization ---
            if global_step % cfg.logging.log_interval == 0:
                # 记录损失
                if torch.isfinite(gen_total_loss):
                    writer.add_scalar('Loss/Generator_Total', gen_total_loss.item(), global_step)
                if torch.isfinite(gen_gan_loss):
                    writer.add_scalar('Loss/Generator_GAN', gen_gan_loss.item(), global_step)
                if torch.isfinite(gen_attack_loss):
                    writer.add_scalar('Loss/Generator_Attack_TriggerNet', gen_attack_loss.item(), global_step)
                if torch.isfinite(d_total_loss):
                    writer.add_scalar('Loss/Discriminator_Total', d_total_loss.item(), global_step)

                # 记录正则化损失
                writer.add_scalar('Loss/Regularization_L_inf', reg_loss_l_inf.item(), global_step)
                writer.add_scalar('Loss/Regularization_L2', reg_loss_l2.item(), global_step)
                writer.add_scalar('Loss/Regularization_TV', reg_loss_tv.item(), global_step)
                writer.add_scalar('Loss/Regularization_L2_Params', reg_loss_l2_params.item(), global_step)

                # 记录扰动统计
                if delta_for_G is not None and delta_for_G.numel() > 0:
                    with torch.no_grad():
                        writer.add_scalar('Perturbation/Pixel_Linf_Norm', linf_norm(delta_for_G).mean().item(), global_step)
                        writer.add_scalar('Perturbation/Pixel_L2_Norm', l2_norm(delta_for_G).mean().item(), global_step)

                print(f"Epoch [{epoch}/{cfg.training.num_epochs}], Step [{global_step}]")
                print(f"  D Loss: {d_total_loss.item():.4f}")
                print(f"  G Loss: {gen_total_loss.item():.4f} (GAN: {gen_gan_loss.item():.4f}, Attack: {gen_attack_loss.item():.4f})")

            # 可视化
            if (global_step + 1) % cfg.logging.vis_interval == 0:
                num_vis_samples = min(cfg.logging.num_vis_samples, real_x.shape[0])
                sequence_step_to_vis = cfg.logging.sequence_step_to_vis

                # 使用专门的Stage 2可视化函数
                with torch.no_grad():
                    visualize_stage2_pixel_attack(
                        writer=writer,
                        original_images=real_x[:num_vis_samples].detach().cpu(),
                        adversarial_images=adversarial_x_for_G[:num_vis_samples].detach().cpu(),
                        pixel_deltas=delta_for_G[:num_vis_samples].detach().cpu(),
                        original_trigger_output=original_atn_outputs.get('trigger_output')[:num_vis_samples].detach().cpu() if original_atn_outputs and original_atn_outputs.get('trigger_output') is not None else None,
                        adversarial_trigger_output=adversarial_atn_outputs.get('trigger_output')[:num_vis_samples].detach().cpu() if adversarial_atn_outputs and adversarial_atn_outputs.get('trigger_output') is not None else None,
                        step=global_step,
                        num_samples=num_vis_samples,
                        sequence_step_to_vis=sequence_step_to_vis
                    )

            global_step += 1

        # --- Evaluation ---
        if (epoch + 1) % cfg.training.eval_interval == 0 or (epoch + 1) == cfg.training.num_epochs:
            if len(eval_dataloader) > 0:
                print(f"Running evaluation at epoch {epoch+1}...")
                eval_metrics = evaluate_model(
                    generator=generator,
                    discriminator=discriminator,
                    atn_model=atn_model_dict,
                    dataloader=eval_dataloader,
                    device=device,
                    cfg=cfg,
                    current_train_stage=cfg.training.train_stage
                )

                if eval_metrics:
                    print(f"Evaluation Metrics at epoch {epoch+1}: {eval_metrics}")
                    for metric_name, metric_value in eval_metrics.items():
                        if np.isfinite(metric_value):
                            writer.add_scalar(f'Evaluation/{metric_name}', float(metric_value), global_step)

                    # 保存最佳模型
                    primary_eval_metric_name = getattr(cfg.evaluation, 'primary_metric', 'Attack_Success_Rate_TriggerNet')
                    is_lower_metric_better = getattr(cfg.evaluation, 'primary_metric_lower_is_better', False)
                    primary_eval_metric = eval_metrics.get(primary_eval_metric_name)

                    if primary_eval_metric is not None and np.isfinite(primary_eval_metric):
                        is_better = (is_lower_metric_better and primary_eval_metric < best_eval_metric) or \
                                   (not is_lower_metric_better and primary_eval_metric > best_eval_metric)

                        if is_better:
                            best_eval_metric = primary_eval_metric
                            checkpoint_dir = os.path.join(cfg.logging.log_dir, 'checkpoints')
                            os.makedirs(checkpoint_dir, exist_ok=True)
                            best_checkpoint_path = os.path.join(checkpoint_dir, f'best_stage_{cfg.training.train_stage}.pth')
                            print(f"Saving best model: {primary_eval_metric_name}={best_eval_metric:.4f}")

                            torch.save({
                                'epoch': epoch,
                                'global_step': global_step,
                                'generator_state_dict': generator.state_dict(),
                                'discriminator_state_dict': discriminator.state_dict(),
                                'optimizer_G_state_dict': optimizer_G.state_dict(),
                                'optimizer_D_state_dict': optimizer_D.state_dict(),
                                'best_eval_metric': best_eval_metric,
                                'cfg': cfg # Save config for reproducibility
                            }, best_checkpoint_path)
                            print(f"Best model saved to {best_checkpoint_path}")

                            # Always save the latest checkpoint
                            checkpoint_dir = os.path.join(cfg.logging.log_dir, 'checkpoints')
                            os.makedirs(checkpoint_dir, exist_ok=True)
                            latest_checkpoint_path = os.path.join(checkpoint_dir, f'latest_stage_{cfg.training.train_stage}.pth')
                            torch.save({
                                'epoch': epoch,
                                'global_step': global_step,
                                'generator_state_dict': generator.state_dict(),
                                'discriminator_state_dict': discriminator.state_dict(),
                                'optimizer_G_state_dict': optimizer_G.state_dict(),
                                'optimizer_D_state_dict': optimizer_D.state_dict(),
                                'best_eval_metric': best_eval_metric,
                                'cfg': cfg
                            }, latest_checkpoint_path)
                            print(f"Latest model saved to {latest_checkpoint_path}")

if __name__ == "__main__":
    # 解析命令行参数和配置文件
    cfg = parse_args_and_config(default_config_path="configs/stage2_config.yaml")
    # 运行训练函数
    train(cfg)
