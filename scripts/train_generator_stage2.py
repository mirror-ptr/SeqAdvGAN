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
# visualize_samples_and_outputs 用于可视化样本、特征、扰动、决策图等
# visualize_attention_maps 用于可视化注意力图
from utils.vis_utils import visualize_samples_and_outputs, visualize_attention_maps

# 导入 ATN 工具和数据工具
# load_atn_model 用于加载预训练的 ATN 模型
# get_atn_outputs 用于获取 ATN 模型的输出 (特征, 决策, 注意力)
# GameVideoDataset 用于加载视频数据
# create_mock_dataloader 用于创建模拟数据加载器
# worker_init_fn 用于初始化数据加载器的工作进程的随机种子
from utils.atn_utils import load_atn_model, get_atn_outputs
from utils.data_utils import GameVideoDataset, create_mock_dataloader, worker_init_fn

# 导入评估工具函数
from utils.eval_utils import evaluate_model # 导入评估函数

# 导入配置工具函数
# parse_args_and_config 用于解析命令行参数并加载/合并配置
from utils.config_utils import parse_args_and_config

# 设置随机种子
def set_seed(seed: int) -> None:
    """
    设置所有必要的随机种子以确保实验的可复现性。
    包括 PyTorch 的 CPU 和 GPU 种子、NumPy 种子和 Python 标准库的 random 种子。
    
    Args:
        seed (int): 用于设置随机生成器的整数种子。
    """
    # 设置 PyTorch 的 CPU 种子
    torch.manual_seed(seed)
    # 如果 CUDA 可用，设置所有 GPU 的种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 设置 NumPy 的种子
    np.random.seed(seed)
    # 设置 Python 标准库 random 的种子
    random.seed(seed)
    # 配置 PyTorch 的 cuDNN 后端，设置为确定性模式有助于提高可复现性，但可能会牺牲一些性能
    torch.backends.cudnn.deterministic = True
    # 关闭 cuDNN 的基准测试模式，设置为 False 以避免随机选择算法
    torch.backends.cudnn.benchmark = False

# 训练主函数
def train(cfg: Any) -> None: # 使用 Any 类型提示 cfg，因为它通常是 EasyDict 对象
    """
    SeqAdvGAN 生成器和判别器的训练主函数 (阶段二: 像素级攻击)。

    该函数根据配置加载模型、数据、损失函数和优化器，
    执行训练循环、计算损失、更新模型参数，
    并进行定期的日志记录、可视化和模型保存。
    支持检查点恢复。
    实现了基于损失比率的 GAN 动态平衡策略。

    Args:
        cfg (Any): 包含所有训练配置参数的配置对象（例如 EasyDict）。
                   预计包含 training, model, losses, regularization, data, logging, evaluation 等子结构。
    """
    # 确保当前训练阶段是 2
    assert cfg.training.train_stage == 2, f"This script is only for training stage 2, but config specifies stage {cfg.training.train_stage}."
    print(f"Starting training for Stage {cfg.training.train_stage} (Pixel-level Attack)...")

    # 设置随机种子以确保实验的可重现性
    set_seed(cfg.training.seed)

    # 获取计算设备 (根据配置和 CUDA 可用性选择 GPU 或 CPU)
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load ATN Model (Frozen) ---
    # 加载预训练的 ATN 模型，用于在训练期间获取决策/注意力输出。
    # ATN 模型在整个 GAN 训练过程中保持冻结，不计算梯度。
    print("Loading ATN model...")
    if not hasattr(cfg.model, 'atn') or not hasattr(cfg.model.atn, 'model_path'):
         print("Error: ATN model configuration (cfg.model.atn) or model_path is missing in the config.")
         sys.exit(1)

    # 在 Stage 2，我们需要完整的 ATN 模型来获取决策图和注意力图
    atn_model = load_atn_model(
        cfg=cfg,
        device=device
    )

    # 检查 ATN 模型是否加载成功
    if atn_model is None:
        print(f"Error: Failed to load ATN model from {cfg.model.atn.model_path}. Exiting training.")
        sys.exit(1)
    print("ATN model loaded successfully and frozen.") # 明确指出模型已冻结

    # --- 2. Initialize Generator and Discriminator Models ---
    print("Initializing Generator and Discriminator...")

    # Stage 2: Generator takes image input and outputs image perturbation
    generator = Generator(
        in_channels=cfg.model.generator.in_channels, # Generator 输入通道数 (应与原始图像通道数匹配，通常是 3)
        out_channels=cfg.model.generator.out_channels, # Generator 输出通道数 (应与原始图像通道数匹配，通常是 3)
        num_bottlenecks=cfg.model.generator.num_bottlenecks,
        base_channels=cfg.model.generator.base_channels,
        epsilon=cfg.model.generator.epsilon # 像素级扰动上限 (L-inf)
    ).to(device)

    # Stage 2: Discriminator takes adversarial image as input
    if cfg.model.discriminator.type == 'cnn':
        discriminator = SequenceDiscriminatorCNN(
            in_channels=cfg.model.discriminator.in_channels, # Discriminator 输入通道数 (应与原始图像通道数匹配，通常是 3)
            base_channels=cfg.model.discriminator.base_channels
        ).to(device)
        print("Using CNN Discriminator for Stage 2")
    elif cfg.model.discriminator.type == 'patchgan':
         discriminator = PatchDiscriminator3d(
              in_channels=cfg.model.discriminator.in_channels, # Discriminator 输入通道数 (应与原始图像通道数匹配，通常是 3)
              base_channels=cfg.model.discriminator.base_channels
         ).to(device)
         print("Using PatchGAN Discriminator for Stage 2")
    else:
        print(f"Error: Unsupported discriminator type for Stage 2: {cfg.model.discriminator.type}. Supported types are 'cnn' and 'patchgan'.")
        sys.exit(1)

    print("Generator and Discriminator models initialized.")

    # --- 3. Define Loss Functions ---
    print("Defining loss functions...")
    gan_losses = GANLosses(device=device, gan_loss_type=cfg.losses.gan_type)
    print(f"Using GAN loss type: {cfg.losses.gan_type}")

    # 初始化攻击损失计算器 (决策, 注意力)
    # Stage 2 重点关注决策图和注意力图的攻击损失。
    attack_losses = AttackLosses(
        device=device,
        attention_loss_weight=cfg.losses.attention_loss_weight, # 注意力损失权重
        decision_loss_weight=cfg.losses.decision_loss_weight, # 决策损失权重
        topk_k=cfg.losses.topk_k
    )
    print(f"Using attack loss weights: Decision={cfg.losses.decision_loss_weight}, Attention={cfg.losses.attention_loss_weight}")
    print(f"Using regularization weights: L_inf={cfg.regularization.lambda_l_inf}, L2={cfg.regularization.lambda_l2}, TV={cfg.regularization.lambda_tv}, L2_Params={cfg.regularization.lambda_l2_penalty}")


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
    print(f"Optimizers initialized with lr_g={cfg.training.lr_g}, lr_d={cfg.training.lr_d}, betas=({cfg.training.b1}, {cfg.training.b2})")

    # --- 5. Data Loading ---
    print("Loading data...")
    if cfg.data.use_mock_data:
        print("Using mock data for training.")
        train_dataloader = create_mock_dataloader(
            batch_size=cfg.training.batch_size,
            num_samples=cfg.data.mock_num_samples,
            sequence_length=cfg.data.sequence_length,
            channels=cfg.data.channels,
            height=cfg.data.height,
            width=cfg.data.width,
            shuffle=True,
            num_workers=cfg.data.num_workers
        )
        eval_dataloader = create_mock_dataloader(
            batch_size=cfg.evaluation.num_eval_samples,
            num_samples=cfg.evaluation.num_eval_samples,
            sequence_length=cfg.data.sequence_length,
            channels=cfg.data.channels,
            height=cfg.data.height,
            width=cfg.data.width,
            shuffle=False,
            num_workers=cfg.data.num_workers
        )
    else:
        print(f"Using real data from video: {cfg.data.video_path}")
        if not os.path.exists(cfg.data.video_path):
             print(f"Error: Video file not found at {cfg.data.video_path}. Please check the path in config.")
             sys.exit(1)
        if not os.path.exists(cfg.data.level_json_path):
             print(f"Error: Level JSON file not found at {cfg.data.level_json_path}. Please check the path in config.")
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
            pin_memory=True,
            multiprocessing_context='spawn'
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
             batch_size=cfg.evaluation.num_eval_samples,
             shuffle=False,
             num_workers=cfg.data.num_workers,
             worker_init_fn=worker_init_fn,
             pin_memory=True,
             multiprocessing_context='spawn'
        )

    print(f"Training data loaded. Number of batches per epoch: {len(train_dataloader)}")
    print(f"Evaluation data loaded. Number of batches: {len(eval_dataloader)}")

    # --- TensorBoard Logging Setup ---
    log_dir = cfg.logging.log_dir
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    # --- Checkpoint Loading (Optional) ---
    start_epoch = 0
    global_step = 0
    best_eval_metric = -float('inf') # For Stage 2, may maximize attack success or minimize something else

    if cfg.training.resume_checkpoint:
         print(f"Attempting to resume from checkpoint: {cfg.training.resume_checkpoint}")
         try:
              checkpoint = torch.load(cfg.training.resume_checkpoint, map_location=device)
              generator.load_state_dict(checkpoint['generator_state_dict'])
              discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
              optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
              optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
              start_epoch = checkpoint['epoch'] + 1
              global_step = checkpoint.get('global_step', start_epoch * len(train_dataloader)) # Compatible with older checkpoints
              best_eval_metric = checkpoint.get('best_eval_metric', -float('inf')) # Compatible with older checkpoints
              print(f"Checkpoint loaded successfully. Resuming from epoch {start_epoch}, global_step {global_step}.")
              # TODO: Resume dynamic balance state if needed

         except FileNotFoundError:
              print(f"Error: Checkpoint file not found at {cfg.training.resume_checkpoint}. Starting training from scratch (epoch 0).")
         except KeyError as e:
              print(f"Error: Key '{e}' not found in checkpoint. Checkpoint format mismatch? Starting training from scratch (epoch 0).")
         except Exception as e:
              print(f"Error loading checkpoint: {e}. Starting training from scratch (epoch 0).")

    print("Starting training...")

    if not cfg.training.resume_checkpoint or global_step == 0:
        global_step = 0

    # Dynamic GAN Balance Variables
    dynamic_balance_cfg = cfg.training.dynamic_gan_balance
    dynamic_balance_enabled = dynamic_balance_cfg.enabled
    current_D_freq = dynamic_balance_cfg.initial_D_freq
    current_G_freq = dynamic_balance_cfg.initial_G_freq

    if dynamic_balance_enabled:
         d_loss_history = deque(maxlen=dynamic_balance_cfg.loss_history_window)
         g_gan_loss_history = deque(maxlen=dynamic_balance_cfg.loss_history_window)
         # TODO: If resuming, load loss history
         print(f"Dynamic GAN balance enabled. Initial D freq: {current_D_freq}, Initial G freq: {current_G_freq}")
    else:
         print("Dynamic GAN balance disabled. Using fixed frequencies.")


    # --- 6. Training Loop (Stage 2: Pixel-level Attack) ---
    for epoch in tqdm(range(start_epoch, cfg.training.num_epochs), desc="Training Epochs"):
        generator.train()
        discriminator.train()

        for i, batch_data in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch}/{cfg.training.num_epochs}"):
            # batch_data shape: (B, C, T, H, W) - original image data
            real_x = batch_data.to(device) # Real image data

            # --- Train Discriminator ---
            optimizer_D.zero_grad()

            # 1. Generator generates pixel-level delta
            # Input to generator is original image (real_x)
            delta = generator(real_x) # delta shape: (B, C, T, H, W)

            # 2. Create adversarial image: original image + delta
            # Clamp delta to epsilon ball for L-inf constraint
            delta = torch.clamp(delta, -cfg.model.generator.epsilon, cfg.model.generator.epsilon)
            # Add delta to original image to get adversarial image
            adversarial_x = real_x + delta
            # Clamp adversarial_x to image valid range (e.g., 0-1 or 0-255) if needed, though usually handled by dataset/preprocessing
            # If original images are 0-255, clamp adversarial_x to 0-255:
            # adversarial_x = torch.clamp(adversarial_x, 0, 255) 
            # If original images are 0-1, clamp adversarial_x to 0-1:
            # adversarial_x = torch.clamp(adversarial_x, 0, 1)

            # Use original images (real_x) as real samples for Discriminator
            D_real_output = discriminator(real_x) # Discriminator sees real images

            # Use adversarial images (adversarial_x) as fake samples for Discriminator
            # Detach adversarial_x to prevent gradients flowing back to Generator during D training
            D_fake_output_detached = discriminator(adversarial_x.detach()) # Discriminator sees fake images

            # Calculate Discriminator GAN loss
            d_total_loss, d_real_loss, d_fake_loss = gan_losses.discriminator_loss(D_real_output, D_fake_output_detached)

            # Update Discriminator parameters (conditional on D_freq)
            if dynamic_balance_enabled and (global_step % current_D_freq == 0):
                 d_total_loss.backward()
                 optimizer_D.step()

            # --- Train Generator ---
            optimizer_G.zero_grad()

            # 1. Generate pixel-level delta again (ensure gradients for G training)
            # This delta will be added to real_x to get adversarial_x for G training
            delta_for_G = generator(real_x) # delta_for_G shape: (B, C, T, H, W)
            delta_for_G = torch.clamp(delta_for_G, -cfg.model.generator.epsilon, cfg.model.generator.epsilon)
            adversarial_x_for_G_train = real_x + delta_for_G
            # Clamp adversarial_x_for_G_train if needed

            # 2. Get ATN outputs for original and adversarial images
            # Ensure no gradients back to ATN
            with torch.no_grad():
                original_atn_outputs = get_atn_outputs(atn_model, real_x, cfg=cfg)

            # Get ATN outputs for adversarial images (gradients flow through adversarial_x_for_G_train)
            # Ensure ATN outputs are on the correct device if not already handled by get_atn_outputs
            # This call should now correctly return decision and attention for Stage 2
            adversarial_atn_outputs = get_atn_outputs(atn_model, adversarial_x_for_G_train, cfg=cfg)

            # 3. Calculate Generator GAN loss
            D_fake_output_for_G_train = discriminator(adversarial_x_for_G_train)
            gen_gan_loss = gan_losses.generator_loss(D_fake_output_for_G_train)

            # 4. Calculate Generator Attack loss (Stage 2: Decision Map and Attention)
            # Goal: Maximize difference in decision/attention between original and adversarial images
            gen_attack_loss_stage2 = attack_losses.get_generator_attack_loss(
                 original_atn_outputs=original_atn_outputs,
                 adversarial_atn_outputs=adversarial_atn_outputs,
                 train_stage=cfg.training.train_stage, # Pass current stage (2)
                 decision_loss_type=cfg.losses.decision_loss_type,
                 attention_loss_type=cfg.losses.attention_loss_type,
                 decision_region_mask=None, # Use masks if configured
                 attention_region_mask=None # Use masks if configured
            ) # This loss calculates attack objective (negative if maximization)

            # 5. Calculate Generator Regularization losses on delta (pixel perturbation)
            # L-inf norm is implicitly handled by the clamp, but can be added as a penalty
            reg_loss_l_inf = torch.tensor(0.0, device=device)
            if cfg.regularization.lambda_l_inf > 0:
                 # Check L-inf norm on delta_for_G
                 reg_loss_l_inf = cfg.regularization.lambda_l_inf * linf_norm(delta_for_G)

            reg_loss_l2 = torch.tensor(0.0, device=device)
            if cfg.regularization.lambda_l2 > 0:
                 reg_loss_l2 = cfg.regularization.lambda_l2 * l2_norm(delta_for_G) # L2 norm of the pixel delta

            reg_loss_tv = torch.tensor(0.0, device=device)
            if cfg.regularization.lambda_tv > 0:
                 reg_loss_tv = cfg.regularization.lambda_tv * total_variation_loss(delta_for_G) # TV loss on the pixel delta

            reg_loss_l2_params = torch.tensor(0.0, device=device)
            if cfg.regularization.lambda_l2_penalty > 0:
                gen_l2_reg = torch.tensor(0.0, device=device)
                for param in generator.parameters():
                    if param.requires_grad:
                        gen_l2_reg += torch.norm(param, 2)**2
                reg_loss_l2_params = cfg.regularization.lambda_l2_penalty * gen_l2_reg

            # Combine all Generator losses
            # Stage 2 total loss includes GAN loss, Attack loss (Decision/Attention), and Regularization
            gen_total_loss = cfg.losses.gan_loss_weight * gen_gan_loss + gen_attack_loss_stage2 + reg_loss_l_inf + reg_loss_l2 + reg_loss_tv + reg_loss_l2_params

            # Backpropagate Generator loss and update parameters (conditional on G_freq)
            if dynamic_balance_enabled and (global_step % current_G_freq == 0):
                 gen_total_loss.backward()
                 optimizer_G.step()

            # --- Dynamic GAN Balance Adjustment ---
            if dynamic_balance_enabled and dynamic_balance_cfg.strategy == "loss_ratio_freq_adjust":
                 d_loss_history.append(d_total_loss.item())
                 g_gan_loss_history.append(gen_gan_loss.item()) # Use G_GAN loss for balance

                 if (global_step + 1) % dynamic_balance_cfg.freq_adjust_interval == 0:
                      avg_d_loss = np.mean(list(d_loss_history)) if d_loss_history else 0.0
                      avg_g_gan_loss = np.mean(list(g_gan_loss_history)) if g_gan_loss_history else 0.0

                      balance_ratio = float('inf')
                      if abs(avg_g_gan_loss) > 1e-8:
                           balance_ratio = avg_d_loss / avg_g_gan_loss

                      print(f"Step {global_step+1} (Epoch {epoch}): D Loss Avg: {avg_d_loss:.4f}, G GAN Loss Avg: {avg_g_gan_loss:.4f}, Balance Ratio (D/G_GAN): {balance_ratio:.2f}")

                      D_dominant_threshold = dynamic_balance_cfg.D_dominant_threshold
                      G_dominant_threshold = dynamic_balance_cfg.G_dominant_threshold
                      min_freq = 1
                      max_freq = 10

                      if balance_ratio > D_dominant_threshold:
                          old_D_freq = current_D_freq
                          old_G_freq = current_G_freq
                          current_D_freq = min(current_D_freq + 1, max_freq)
                          current_G_freq = max(min_freq, current_G_freq - 1)
                          if current_D_freq != old_D_freq or current_G_freq != old_G_freq:
                              print(f"  [Balance Adjust] D dominant (Ratio: {balance_ratio:.2f}) -> Freqs adjusted: D {old_D_freq}->{current_D_freq}, G {old_G_freq}->{current_G_freq}")
                      elif balance_ratio < G_dominant_threshold:
                          old_D_freq = current_D_freq
                          old_G_freq = current_G_freq
                          current_G_freq = min(current_G_freq + 1, max_freq)
                          current_D_freq = max(min_freq, current_D_freq - 1)
                          if current_D_freq != old_D_freq or current_G_freq != old_G_freq:
                               print(f"  [Balance Adjust] G dominant (Ratio: {balance_ratio:.2f}) -> Freqs adjusted: D {old_D_freq}->{current_D_freq}, G {old_G_freq}->{current_G_freq}")
                      else:
                          if current_D_freq != dynamic_balance_cfg.initial_D_freq or current_G_freq != dynamic_balance_cfg.initial_G_freq:
                              old_D_freq = current_D_freq
                              old_G_freq = current_G_freq
                              current_D_freq = dynamic_balance_cfg.initial_D_freq
                              current_G_freq = dynamic_balance_cfg.initial_G_freq
                              print(f"  [Balance Adjust] Balanced -> Freqs reset to initial: D {old_D_freq}->{current_D_freq}, G {old_G_freq}->{current_G_freq}")


            # --- Logging and Visualization ---
            if global_step % cfg.logging.log_interval == 0:
                writer.add_scalar('Loss/Generator_Total', gen_total_loss.item(), global_step)
                writer.add_scalar('Loss/Generator_GAN', gen_gan_loss.item(), global_step)
                writer.add_scalar('Loss/Generator_Attack (Decision/Attention)', gen_attack_loss_stage2.item(), global_step) # Log Stage 2 attack loss

                writer.add_scalar('Loss/Regularization_L_inf', reg_loss_l_inf.item(), global_step)
                writer.add_scalar('Loss/Regularization_L2', reg_loss_l2.item(), global_step)
                writer.add_scalar('Loss/Regularization_TV', reg_loss_tv.item(), global_step)
                writer.add_scalar('Loss/Regularization_L2_Params', reg_loss_l2_params.item(), global_step)

                writer.add_scalar('Loss/Discriminator_Total', d_total_loss.item(), global_step)
                writer.add_scalar('Loss/Discriminator_Real', d_real_loss.item(), global_step)
                writer.add_scalar('Loss/Discriminator_Fake', d_fake_loss.item(), global_step)

                # Log perturbation norms (calculated on pixel delta)
                if delta_for_G is not None and delta_for_G.numel() > 0:
                    with torch.no_grad():
                         # linf_norm and l2_norm expect (B, C, N, H, W)
                         writer.add_scalar('Perturbation/Pixel_Linf_Norm', linf_norm(delta_for_G.detach()).mean().item(), global_step)
                         writer.add_scalar('Perturbation/Pixel_L2_Norm', l2_norm(delta_for_G.detach()).mean().item(), global_step)

                # Log Discriminator scores
                with torch.no_grad():
                    real_score = D_real_output.mean() if D_real_output.ndim == 0 else D_real_output.view(D_real_output.size(0), -1).mean(dim=-1).mean().item()
                    fake_score = D_fake_output_detached.mean() if D_fake_output_detached.ndim == 0 else D_fake_output_detached.view(D_fake_output_detached.size(0), -1).mean(dim=-1).mean().item()
                    writer.add_scalar('Discriminator_Scores/Real', real_score, global_step)
                    writer.add_scalar('Discriminator_Scores/Fake', fake_score, global_step)

                print(f"Epoch [{epoch}/{cfg.training.num_epochs}], Step [{global_step}], ")
                print(f"  D Loss: {d_total_loss.item():.4f} (Real: {d_real_loss.item():.4f}, Fake: {d_fake_loss.item():.4f})")
                print(f"  G Loss: {gen_total_loss.item():.4f} (GAN: {gen_gan_loss.item():.4f}, Attack: {gen_attack_loss_stage2.item():.4f}, Reg_Linf: {reg_loss_l_inf.item():.4f}, Reg_L2: {reg_loss_l2.item():.4f}, Reg_TV: {reg_loss_tv.item():.4f}, Reg_L2_Params: {reg_loss_l2_params.item():.4f})")

            # Visualize samples and outputs periodically
            if (global_step + 1) % cfg.logging.vis_interval == 0:
                 num_vis_samples = min(cfg.logging.num_vis_samples, real_x.shape[0])
                 sequence_step_to_vis = cfg.logging.sequence_step_to_vis

                 # Get ATN outputs for visualization (ensure no gradients)
                 with torch.no_grad():
                      original_atn_outputs_vis = get_atn_outputs(atn_model, real_x[:num_vis_samples].detach(), cfg=cfg)
                      adversarial_atn_outputs_vis = get_atn_outputs(atn_model, adversarial_x[:num_vis_samples].detach(), cfg=cfg)

                 # Extract decision maps and attention maps for visualization
                 original_decision_map_vis = original_atn_outputs_vis.get('decision')
                 adversarial_decision_map_vis = adversarial_atn_outputs_vis.get('decision')
                 original_attention_map_vis = original_atn_outputs_vis.get('attention')
                 adversarial_attention_map_vis = adversarial_atn_outputs_vis.get('attention')

                 # Visualize samples, perturbation, and decision maps
                 visualize_samples_and_outputs(
                     writer,
                     original_image=real_x[:num_vis_samples].detach().cpu(),
                     original_features=None, # Stage 2 does not visualize features
                     feature_delta=None,     # Stage 2 delta is on pixels
                     adversarial_features=None, # Stage 2 adversarial is on pixels
                     original_decision_map=original_decision_map_vis.detach().cpu() if original_decision_map_vis is not None else None,
                     adversarial_decision_map=adversarial_decision_map_vis.detach().cpu() if adversarial_decision_map_vis is not None else None,
                     step=global_step,
                     num_samples=num_vis_samples,
                     sequence_step_to_vis=sequence_step_to_vis,
                     visualize_decision_diff=True # Visualize decision map difference in Stage 2
                 )

                 # 可视化注意力图
                 visualize_attention_maps(
                     writer, 
                     attention_matrix_orig=original_atn_outputs.get('attention'), # 使用正确的参数名
                     attention_matrix_adv=adversarial_atn_outputs.get('attention'),   # 使用正确的参数名
                     step=global_step,
                     num_samples=num_vis_samples,
                     num_heads_to_vis=cfg.logging.num_vis_heads # Use num_vis_heads from config
                 )

            # Increment global step
            global_step += 1

        # --- Evaluation ---
        if (epoch + 1) % cfg.training.eval_interval == 0 or (epoch + 1) == cfg.training.num_epochs:
            print(f"Running evaluation at epoch {epoch+1}...")

            # Evaluate the model (Generator and Discriminator) on the evaluation dataset
            eval_metrics = evaluate_model(
                generator=generator, # Evaluate Generator performance
                discriminator=discriminator, # Evaluate Discriminator performance
                atn_model=atn_model, # Evaluation needs ATN model to get outputs
                dataloader=eval_dataloader, # Use evaluation data loader
                device=device,
                cfg=cfg, # Pass config for evaluation parameters (e.g., evaluation criteria)
                current_train_stage=cfg.training.train_stage # Pass current training stage (2)
            )

            # Log evaluation metrics to TensorBoard
            if eval_metrics:
               print(f"Evaluation Metrics at epoch {epoch+1}: {eval_metrics}")
               for metric_name, metric_value in eval_metrics.items():
                   if np.isfinite(metric_value):
                        writer.add_scalar(f'Evaluation/{metric_name}', float(metric_value), global_step)
                   else:
                        print(f"Warning: Evaluation metric '{metric_name}' is not finite ({metric_value}). Skipping logging.")

               # --- Save best model based on evaluation metric (Stage 2: Maximize Attack Success Rate) ---
               # In Stage 2, we usually want to maximize the attack success rate (e.g., decision or attention attack success).
               # Choose a primary evaluation metric to track for saving the best model.
               # Example: Maximize Decision Attack Success Rate.
               primary_eval_metric_name = 'Attack_Success_Rate_Decision' # Stage 2 focuses on decision/attention attack success
               is_lower_metric_better = False # For success rate, higher is better

               primary_eval_metric = eval_metrics.get(primary_eval_metric_name)

               if primary_eval_metric is not None and np.isfinite(primary_eval_metric):
                   if (is_lower_metric_better and primary_eval_metric < best_eval_metric) or \
                      (not is_lower_metric_better and primary_eval_metric > best_eval_metric):
                        best_eval_metric = primary_eval_metric

                        checkpoint_dir = os.path.join(cfg.logging.log_dir, 'checkpoints')
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        best_checkpoint_path = os.path.join(checkpoint_dir, f'best_stage_{cfg.training.train_stage}.pth')
                        print(f"Saving best model to {best_checkpoint_path} based on {primary_eval_metric_name}: {best_eval_metric:.4f}")

                        torch.save({
                            'epoch': epoch,
                            'global_step': global_step,
                            'generator_state_dict': generator.state_dict(),
                            'discriminator_state_dict': discriminator.state_dict(),
                            'optimizer_G_state_dict': optimizer_G.state_dict(),
                            'optimizer_D_state_dict': optimizer_D.state_dict(),
                            'best_eval_metric': best_eval_metric,
                            'current_train_stage': cfg.training.train_stage,
                            # TODO: Save random states
                        }, best_checkpoint_path)
                        print("Best model checkpoint saved.")
               else:
                   print(f"Warning: Primary evaluation metric '{primary_eval_metric_name}' is not available or not finite. Skipping best model saving.")
            else:
               print("Warning: Evaluation metrics dictionary is empty. Skipping logging and best model saving.")


        # --- Save Checkpoint ---
        if (epoch + 1) % cfg.training.save_interval == 0 or (epoch + 1) == cfg.training.num_epochs:
            checkpoint_dir = os.path.join(cfg.logging.log_dir, 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}_stage_{cfg.training.train_stage}.pth')
            print(f"Saving checkpoint to {checkpoint_path}")

            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'best_eval_metric': best_eval_metric,
                'current_train_stage': cfg.training.train_stage,
                # TODO: Save random states
            }, checkpoint_path)
            print("Checkpoint saved.")

    writer.close()
    print("Training finished.")

if __name__ == '__main__':
    cfg = parse_args_and_config(default_config_path='configs/default_config.yaml', task_config_arg='config')

    if cfg is not None:
        print("Configuration loaded:")
        import json
        from easydict import EasyDict
        print(json.dumps(dict(cfg), indent=4))
        print("--------------------------")

        os.makedirs(cfg.logging.log_dir, exist_ok=True)
        os.makedirs(os.path.join(cfg.logging.log_dir, 'checkpoints'), exist_ok=True)

        train(cfg)
    else:
        print("Failed to load configuration. Exiting.")
        sys.exit(1) 