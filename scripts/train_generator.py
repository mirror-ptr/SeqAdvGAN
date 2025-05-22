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
    SeqAdvGAN 生成器和判别器的训练主函数。

    该函数根据配置加载模型、数据、损失函数和优化器，
    执行训练循环、计算损失、更新模型参数，
    并进行定期的日志记录、可视化和模型保存。
    支持训练阶段的选择和检查点恢复。
    实现了基于损失比率的 GAN 动态平衡策略。

    Args:
        cfg (Any): 包含所有训练配置参数的配置对象（例如 EasyDict）。
                   预计包含 training, model, losses, regularization, data, logging, evaluation 等子结构。
    """
    # 确保当前训练阶段在支持的范围内（目前支持阶段 1 和 2）
    assert cfg.training.train_stage in [1, 2], f"Unsupported training stage: {cfg.training.train_stage}. Supported stages are 1 and 2."
    print(f"Starting training for Stage {cfg.training.train_stage}...")

    # 设置随机种子以确保实验的可复现性
    set_seed(cfg.training.seed)

    # 获取计算设备 (根据配置和 CUDA 可用性选择 GPU 或 CPU)
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load ATN Model (Frozen) ---
    # 加载预训练的 ATN 模型，用于在训练期间提取特征或获取决策/注意力输出。
    # ATN 模型在整个 GAN 训练过程中保持冻结，不计算梯度。
    print("Loading ATN model...")
    # 根据配置中的 ATN 模型路径加载模型权重和结构。
    # atn_utils.load_atn_model 函数负责实例化模型和加载权重。
    # 确保配置对象 cfg.model.atn 存在且包含模型路径 cfg.model.atn.model_path。
    # 还需要传递 ATN 模型初始化所需的输入尺寸参数 (in_channels, sequence_length, height, width, num_heads)，这些参数应从配置中读取。
    # TODO: 确保 cfg 包含 ATN 初始化所需的所有参数 (已在 config 文件中添加相应字段并在此读取)。
    if not hasattr(cfg.model, 'atn') or not hasattr(cfg.model.atn, 'model_path'):
         print("Error: ATN model configuration (cfg.model.atn) or model_path is missing in the config.")
         sys.exit(1)

    atn_model = load_atn_model(
        model_path=cfg.model.atn.model_path, # ATN 模型权重文件的路径
        device=device, # 将模型加载到指定的计算设备
        in_channels=cfg.model.atn.atn_in_channels, # ATN 模型期望的输入通道数 (通常是原始图像通道，例如 3)
        sequence_length=cfg.model.atn.atn_sequence_length, # ATN 模型期望的输入序列长度
        height=cfg.model.atn.atn_height, # ATN 模型期望的输入高度
        width=cfg.model.atn.atn_width, # ATN 模型期望的输入宽度
        num_heads=cfg.model.atn.atn_num_heads # ATN 模型注意力头部数量
        # TODO: 考虑是否需要 load_feature_head_only 参数，以及如何根据 Stage 1/2 加载不同部分。
        #      例如， Stage 1 可能只需要特征提取部分，Stage 2 需要整个模型。
        #      目前 load_atn_model 会尝试加载整个模型并返回 nn.Module。
    )

    # 检查 ATN 模型是否加载成功
    if atn_model is None:
        print(f"Error: Failed to load ATN model from {cfg.model.atn.model_path}. Exiting training.")
        sys.exit(1) # 如果加载失败，退出脚本
    # 将加载的 ATN 模型移动到指定设备，并设置为评估模式 (不计算梯度)
    atn_model.eval() # ATN 模型在 GAN 训练期间保持评估模式
    atn_model.to(device) # 确保 ATN 模型在正确的设备上
    print("ATN model loaded successfully and frozen.") # 明确指出模型已冻结

    # --- 2. Initialize Generator and Discriminator Models ---
    print("Initializing Generator and Discriminator...")
    # 根据当前的训练阶段初始化生成器和判别器模型。

    # 阶段 1 的模型配置：
    # Generator 输入是 ATN 特征 (通道数由 config 定义，例如 128C)，输出是特征扰动 (相同通道数)。
    # Discriminator 输入是 ATN 特征 (与 Generator 输入通道数相同)。
    if cfg.training.train_stage == 1:
        generator = Generator(
            in_channels=cfg.model.generator.in_channels, # Generator 输入通道数 (应与 ATN 特征通道数匹配)
            out_channels=cfg.model.generator.out_channels, # Generator 输出通道数 (应与 ATN 特征通道数匹配)
            num_bottlenecks=cfg.model.generator.num_bottlenecks, # Generator 的 Bottleneck 块数量
            base_channels=cfg.model.generator.base_channels, # Generator 的基础通道数
            epsilon=cfg.model.generator.epsilon # Generator 输出扰动的 L-inf 范数上限
        ).to(device)

        # 根据配置选择并初始化判别器类型 (SequenceDiscriminatorCNN 或 PatchDiscriminator3d)
        if cfg.model.discriminator.type == 'cnn':
            discriminator = SequenceDiscriminatorCNN(
                in_channels=cfg.model.generator.out_channels, # Discriminator 输入通道数 (应与 ATN 特征通道数匹配)
                base_channels=cfg.model.discriminator.base_channels
            ).to(device)
            print("Using CNN Discriminator")
        elif cfg.model.discriminator.type == 'patchgan':
             discriminator = PatchDiscriminator3d(
                  in_channels=cfg.model.generator.out_channels, # Discriminator 输入通道数 (应与 ATN 特征通道数匹配)
                  base_channels=cfg.model.discriminator.base_channels
             ).to(device)
             print("Using PatchGAN Discriminator")
        else:
            # 如果配置的判别器类型不支持，打印错误并退出
            print(f"Error: Unsupported discriminator type: {cfg.model.discriminator.type}. Supported types are 'cnn' and 'patchgan'.")
            sys.exit(1)

    # 阶段 2 的模型配置：
    # TODO: 阶段 2 模型初始化。Generator 输入可能是原始图像，输出图像扰动。Discriminator 输入也可能是图像或处理后的图像。
    elif cfg.training.train_stage == 2:
        # TODO: 实现阶段 2 的生成器和判别器模型初始化逻辑。
        # Stage 2 Generator might need different architecture (e.g., takes image input)
        # Stage 2 Discriminator might need different architecture (e.g., takes image input)
        print("Stage 2 model initialization not yet fully implemented.")
        sys.exit(1)

    print("Generator and Discriminator models initialized.")

    # --- 3. Define Loss Functions ---
    print("Defining loss functions...")
    # 初始化 GAN 损失计算器 (支持 BCE 或 LSGAN)
    # 将计算设备 device 传递给 GANLosses 类，确保损失函数在正确设备上。
    gan_losses = GANLosses(device=device, gan_loss_type=cfg.losses.gan_type)
    print(f"Using GAN loss type: {cfg.losses.gan_type}")

    # 初始化攻击损失计算器 (特征、决策、注意力)
    # 将计算设备 device 以及注意力/决策损失权重、Top-K 参数传递给 AttackLosses 类。
    attack_losses = AttackLosses(
        device=device, # 传递计算设备
        attention_loss_weight=cfg.losses.attention_loss_weight, # 注意力损失权重
        decision_loss_weight=cfg.losses.decision_loss_weight, # 决策损失权重 (在 Stage 1 用于特征攻击)
        topk_k=cfg.losses.topk_k # Top-K 参数，用于 Top-K 注意力损失和评估
    )
    print(f"Using attack loss weights: Decision (Feature)={cfg.losses.decision_loss_weight}, Attention={cfg.losses.attention_loss_weight}")
    # 打印使用的正则化权重
    print(f"Using regularization weights: L_inf={cfg.regularization.lambda_l_inf}, L2={cfg.regularization.lambda_l2}, TV={cfg.regularization.lambda_tv}, L2_Params={cfg.regularization.lambda_l2_penalty}")


    # --- 4. Define Optimizers ---
    print("Defining optimizers...")
    # 为生成器和判别器定义 Adam 优化器。
    # 从配置中读取学习率 (lr_g, lr_d) 和 Adam 的 beta 参数 (b1, b2)。
    optimizer_G = optim.Adam(
        generator.parameters(), # 优化生成器的所有参数
        lr=cfg.training.lr_g, # 生成器学习率
        betas=(cfg.training.b1, cfg.training.b2) # Adam 的 beta 参数
    )
    optimizer_D = optim.Adam(
        discriminator.parameters(), # 优化判别器的所有参数
        lr=cfg.training.lr_d, # 判别器学习率
        betas=(cfg.training.b1, cfg.training.b2) # Adam 的 beta 参数
    )
    print(f"Optimizers initialized with lr_g={cfg.training.lr_g}, lr_d={cfg.training.lr_d}, betas=({cfg.training.b1}, {cfg.training.b2})")

    # --- 5. Data Loading ---
    print("Loading data...")
    # 根据配置创建数据集 (Dataset) 和数据加载器 (DataLoader)。
    # 支持使用模拟数据或真实视频数据。

    if cfg.data.use_mock_data:
        # 如果配置使用模拟数据，则创建模拟数据加载器。
        print("Using mock data for training.")
        dataloader = create_mock_dataloader(
            batch_size=cfg.training.batch_size, # 批次大小
            num_samples=cfg.data.mock_num_samples, # 模拟样本数量
            sequence_length=cfg.data.sequence_length, # 序列长度
            channels=cfg.data.channels, # 图像通道数
            height=cfg.data.height, # 图像高度
            width=cfg.data.width, # 图像宽度
            shuffle=True, # 在每个 epoch 开始时打乱数据
            num_workers=cfg.data.num_workers # 数据加载器工作进程数
        )
    else:
        # 如果配置使用真实视频数据，则创建 GameVideoDataset。
        print(f"Using real data from video: {cfg.data.video_path}")
        # 检查视频文件和关卡 JSON 文件是否存在
        if not os.path.exists(cfg.data.video_path):
             print(f"Error: Video file not found at {cfg.data.video_path}. Please check the path in config.")
             sys.exit(1)
        if not os.path.exists(cfg.data.level_json_path):
             print(f"Error: Level JSON file not found at {cfg.data.level_json_path}. Please check the path in config.")
             sys.exit(1)

        # 初始化 GameVideoDataset
        train_dataset = GameVideoDataset(
            video_path=cfg.data.video_path, # 视频文件路径
            level_json_path=cfg.data.level_json_path, # 关卡 JSON 文件路径
            sequence_length=cfg.data.sequence_length, # 每个样本的序列长度 (帧数)
            transform=None, # TODO: 在这里添加数据增强或预处理变换
            target_height=cfg.data.height, # 视频帧的目标高度
            target_width=cfg.data.width # 视频帧的目标宽度
        )
        # 使用 DataLoader 包装数据集，并配置相关参数。
        # worker_init_fn 用于确保多进程数据加载时的随机性。
        # pin_memory=True 可以将数据加载到 CUDA 锁页内存，加速 GPU 数据传输。
        dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.training.batch_size, # 批次大小
            shuffle=True, # 在每个 epoch 开始时打乱数据
            num_workers=cfg.data.num_workers, # 数据加载器工作进程数
            worker_init_fn=worker_init_fn, # 工作进程初始化函数
            pin_memory=True # 启用锁页内存
        )

    print(f"Data loaded. Number of batches per epoch: {len(dataloader)}")

    # --- TensorBoard Logging Setup ---
    # 创建 SummaryWriter 对象，用于将训练过程中的标量、图像等写入 TensorBoard 日志文件。
    # 日志目录根据配置 cfg.logging.log_dir 和当前训练阶段 train_stage 确定。
    log_dir = os.path.join(cfg.logging.log_dir, f'stage{cfg.training.train_stage}_train') # 日志保存目录，区分训练阶段
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    # --- Checkpoint Loading (Optional) ---
    # 检查是否需要从检查点恢复训练。
    start_epoch = 0 # 训练开始的 epoch 编号
    # TODO: 完善检查点加载逻辑，确保可以加载所有必要的状态 (模型、优化器、epoch、global_step 等)。
    if cfg.training.resume_checkpoint:
         print(f"Attempting to resume from checkpoint: {cfg.training.resume_checkpoint}")
         try:
              # 加载检查点文件到指定的设备
              checkpoint = torch.load(cfg.training.resume_checkpoint, map_location=device)
              # 加载生成器和判别器的模型状态字典
              generator.load_state_dict(checkpoint['generator_state_dict'])
              discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
              # 加载生成器和判别器的优化器状态字典
              optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
              optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
              # 更新开始的 epoch 编号
              start_epoch = checkpoint['epoch'] + 1
              # 如果检查点中保存了 global_step，也进行恢复
              if 'global_step' in checkpoint:
                   global_step = checkpoint['global_step']
              else:
                   # 如果检查点没有保存 global_step，根据 epoch 和 dataloader 长度估算 (可能不准确)
                   global_step = start_epoch * len(dataloader) # 简单的估算
                   print(f"Warning: global_step not found in checkpoint. Estimating global_step as {global_step}.")
              # 如果检查点中保存了 best_eval_metric，也进行恢复
              # if 'best_eval_metric' in checkpoint:
              #      best_eval_metric = checkpoint['best_eval_metric']
              print(f"Checkpoint loaded successfully. Resuming from epoch {start_epoch}, global_step {global_step}.")
         except FileNotFoundError:
              print(f"Error: Checkpoint file not found at {cfg.training.resume_checkpoint}. Starting training from scratch (epoch 0).")
              global_step = 0 # 从头开始，global_step 为 0
         except KeyError as e:
              print(f"Error: Key '{e}' not found in checkpoint. Checkpoint format mismatch? Starting training from scratch (epoch 0).")
              global_step = 0
         except Exception as e:
              print(f"Error loading checkpoint: {e}. Starting training from scratch (epoch 0).")
              global_step = 0

    print("Starting training...")
    # global_step 变量用于跟踪总的训练步数，在整个训练过程中累积
    # 如果从检查点恢复，global_step 会在检查点加载逻辑中被设定
    # 否则，它将从 0 开始
    # Initialize global_step based on start_epoch and dataloader length if resuming
    # global_step = start_epoch * len(dataloader) # This is a simple approximation

    # For accurate global_step when resuming, you might need to save it in the checkpoint.
    # If checkpoint has global_step:
    if cfg.training.resume_checkpoint and 'global_step' in checkpoint:
         global_step = checkpoint['global_step']
         print(f"Resuming global_step from checkpoint: {global_step}")
    else:
         global_step = 0 # Start from 0 if not resuming or global_step not in checkpoint

    # 用于保存最优模型的评估指标值。通常在评估阶段计算，并用于决定是否保存模型检查点。
    # 初始化为一个很小的负值，以便第一次评估的任何有效指标都能被认为是最好的。
    best_eval_metric = -float('inf') # For saving the best model based on evaluation metric

    # --- Initialize Dynamic GAN Balance Variables ---
    # 从配置中读取 GAN 动态平衡相关的参数
    dynamic_balance_cfg = cfg.training.dynamic_gan_balance
    dynamic_balance_enabled = dynamic_balance_cfg.enabled # 是否启用动态平衡
    current_D_freq = dynamic_balance_cfg.initial_D_freq # 判别器初始更新频率
    current_G_freq = dynamic_balance_cfg.initial_G_freq # 生成器初始更新频率

    # 使用 deque (双端队列) 存储最近的损失历史，用于计算移动平均损失
    # maxlen 参数限制队列的长度，超出长度的旧元素会自动移除
    if dynamic_balance_enabled:
         # 判别器总损失的历史队列
         d_loss_history = deque(maxlen=dynamic_balance_cfg.loss_history_window)
         # 生成器 GAN 损失的历史队列
         g_gan_loss_history = deque(maxlen=dynamic_balance_cfg.loss_history_window)
         print(f"Dynamic GAN balance enabled. Initial D freq: {current_D_freq}, Initial G freq: {current_G_freq}")
    else:
         print("Dynamic GAN balance disabled. Using fixed frequencies.")


    # --- 6. Training Loop ---
    # 主训练循环，迭代指定的 epoch 数量。
    # 使用 tqdm 库在控制台显示 epoch 的进度条。
    for epoch in range(start_epoch, cfg.training.num_epochs):
        # 在每个 epoch 开始时，将模型设置为训练模式。
        # 这会启用 Dropout 和 BatchNorm 等层的训练行为。
        generator.train()
        discriminator.train()

        # 使用 tqdm 在控制台显示当前 epoch 的批次处理进度条。
        # enumerate(dataloader) 提供批次的索引 i 和数据 batch_data。
        # total=len(dataloader) 指定总的批次数量。
        # desc 设置进度条的描述信息。
        for i, batch_data in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}/{cfg.training.num_epochs}"):
            # batch_data 的形状通常是 (B, C, T, H, W)，表示原始的图像序列数据。
            # 这是 ATN 模型期望的输入格式。
            # 将批次数据移动到指定的计算设备 (GPU 或 CPU)。
            real_x = batch_data.to(device) # Real image data
            # print(f"Debug: Epoch {epoch}, Batch {i} - Loaded real_x, shape: {real_x.shape}") # Debug print after loading batch

            # --- Stage 1 Training Logic (Feature Attack) ---
            # 第一阶段训练的目标：训练生成器产生针对 ATN 特征层的扰动，
            # 使对抗样本在特征空间与原始样本差异最大化，并欺骗判别器。
            # Goal: Generator generates perturbation on features to max feature difference and fool Discriminator

            # Debug print before calling get_atn_outputs
            # print(f"Debug: Epoch {epoch}, Batch {i} - Before get_atn_outputs, real_x shape: {real_x.shape}")

            # 1. Get original features from ATN (ATN takes original image data as input)
            # 从预训练的 ATN 模型获取原始输入的特征。
            # 由于 ATN 模型是冻结的，使用 torch.no_grad() 上下文管理器，确保不为 ATN 的参数计算梯度。
            # get_atn_outputs 函数根据参数返回 ATN 的不同输出。
            # 在 Stage 1 训练中，我们只需要 ATN 的 features 输出 (return_features=True)。
            # decision 和 attention 输出通常在 Stage 2 或评估时才需要，这里设为 False 以提高效率。
            with torch.no_grad(): # Ensure no gradients flow back to ATN model
                 original_atn_outputs = get_atn_outputs(
                      atn_model,
                      real_x, # 传递原始图像数据作为 ATN 的输入 (形状: B, C_img, T, H_img, W_img)
                      return_features=True, # 获取特征层输出
                      # Stage 1 doesn't primarily attack decision/attention, so no need to return them
                      # unless needed for separate visualization/logging
                      return_decision=False, # Stage 1 训练中不主要关注决策图，设为 False
                      return_attention=False # Stage 1 训练中不主要关注注意力图，设为 False
                  )
            # original_features 的形状大约是 (B, C_feat, N_feat, H_feat, W_feat)，取决于 ATN 特征提取器的输出。
            original_features = original_atn_outputs.get('features')

            # Debug print after calling get_atn_outputs
            # print(f"Debug: Epoch {epoch}, Batch {i} - After get_atn_outputs, original_features shape: {original_features.shape if original_features is not None else None}")

            # Check if original_features is None or empty. If so, skip this batch.
            # 有时 ATN 可能因为各种原因未能产生有效的输出，此检查可以避免后续错误。
            if original_features is None or original_features.numel() == 0:
                 # 如果特征为空，打印警告并跳过当前批次
                 print(f"Warning: Epoch {epoch}, Batch {i} - ATN failed to return features. Skipping batch.")
                 continue # Skip this batch if features are not available

            # Ensure original_features requires grad for Generator training
            # 为了计算生成器的梯度，我们需要将梯度从生成器的输出 (对抗特征) 反向传播到生成器的输入。
            # 虽然 original_features 来自冻结的 ATN，但它将作为生成器的输入，因此需要启用梯度跟踪。
            # 我们克隆并分离 (detach) 原始特征，以切断与 ATN 的计算图，然后对其启用 requires_grad=True。
            original_features_for_gen = original_features.clone().detach().requires_grad_(True)

            # --- Train Discriminator ---
            # 训练判别器的目标：使判别器能够区分真实的特征 (来自原始 ATN 输出) 和伪造的特征 (来自生成器输出)。
            # Goal: Discriminator distinguishes real features (from original ATN) and fake features (from Generator)

            # 清除判别器优化器的梯度，为当前训练步骤做准备。
            optimizer_D.zero_grad() # Clear gradients for Discriminator

            # 获取判别器对真实特征的输出。
            # 真实特征是原始 ATN 的输出特征。
            # 使用 detach() 确保在判别器训练期间不计算生成器的梯度。判别器训练时，生成器被视为固定的。
            # Discriminator processes the feature tensor output from ATN/Generator
            D_real_output = discriminator(original_features.detach()) # 判别器处理真实的特征张量

            # 生成伪造特征：原始特征 + 生成器产生的扰动。
            # 这需要通过生成器进行一次前向传播。
            # 注意：在训练判别器时，我们需要分离 (detach) 生成器的输出扰动 delta，
            # 以便判别器训练时将生成器的输出视为一个固定的输入，而不是一个需要计算梯度的变量。

            # 生成器输入是 original_features_for_gen (已requires_grad=True，但在这里 for D training 我们通过外层 no_grad() 屏蔽)
            # 方式1: 使用 with torch.no_grad(): 生成 delta
            with torch.no_grad(): # 确保在判别器训练时，梯度不会通过生成器反向传播
                 # 生成扰动 delta，输入是原始特征的副本 (已分离)
                 delta_for_D = generator(original_features_for_gen.detach()) # 生成 delta
                 # 计算用于判别器训练的对抗特征
                 # 对抗特征 = 原始特征 (已分离) + 扰动 delta (已分离)
                 # 注意：这里的 original_features.detach() 也可以直接用 original_features，因为外层有 no_grad()
                 # 但使用 detach() 更清晰地表明这是独立于 Generator 计算图的固定输入
                 adversarial_features_for_D = original_features.detach() + delta_for_D

            # 获取判别器对伪造特征的输出。
            # 这是判别器处理对抗特征后的输出。
            D_fake_output_detached = discriminator(adversarial_features_for_D)

            # 计算判别器的 GAN 损失。
            # gan_losses.discriminator_loss 函数根据 GAN 类型 (BCE 或 LSGAN) 计算损失。
            # 在 LSGAN 中，判别器希望 D_real_output 接近 1，D_fake_output_detached 接近 0。
            # 函数返回总损失、真实样本损失和伪造样本损失。
            d_total_loss, d_real_loss, d_fake_loss = gan_losses.discriminator_loss(D_real_output, D_fake_output_detached)

            # 根据动态平衡策略或固定频率，判断是否更新判别器的参数。
            # global_step % current_D_freq == 0 表示每 current_D_freq 步更新一次判别器。
            if global_step % current_D_freq == 0:
                 # 计算判别器损失的梯度。
                 d_total_loss.backward()
                 # 根据梯度更新判别器的参数。
                 optimizer_D.step() # 更新判别器参数
                 # print(f"Debug: Step {global_step} - Updated Discriminator") # Debug print

            # --- Train Generator ---
            # 训练生成器的目标：
            # 1. 使生成器产生的对抗特征能够欺骗判别器 (生成器希望判别器认为伪造样本是真实的)。
            # 2. 使生成器产生的对抗特征与原始特征之间的差异最大化 (攻击目标)。
            # Goal: Make Generator output (adversarial features) fool the Discriminator AND maximize feature difference

            # 清除生成器优化器的梯度。
            optimizer_G.zero_grad() # 清除生成器参数的梯度

            # 生成扰动，这一次需要确保梯度能够从生成器的输出反向传播到生成器的参数。
            # 使用 original_features_for_gen 作为输入，因为其 requires_grad=True。
            # Debug print before calling generator for G train
            # print(f"Debug: Epoch {epoch}, Batch {i} - Before generator forward for G train, original_features_for_gen shape: {original_features_for_gen.shape}")
            delta = generator(original_features_for_gen) # 生成扰动，梯度会流经 generator
            # print(f"Debug: Epoch {epoch}, Batch {i} - After generator forward for G train, delta shape: {delta.shape}")

            # 计算用于生成器训练的对抗特征。
            # 对抗特征 = 原始特征 + 生成器产生的扰动 delta。
            # 由于 original_features_for_gen 和 delta 都跟踪梯度，这里的 adversarial_features 也会跟踪梯度。
            adversarial_features = original_features_for_gen + delta
            # print(f"Debug: Epoch {epoch}, Batch {i} - Calculated adversarial_features for G train, shape: {adversarial_features.shape}")

            # 获取判别器对这些对抗特征的输出，用于计算生成器的 GAN 损失。
            # 注意：判别器此时处于 train() 模式 (在 epoch 开始时设置)，但其参数只在判别器优化步骤中更新。
            # 在计算生成器损失时，我们只关心判别器的输出值，梯度会从 GAN 损失通过判别器的前向传播流回生成器。
            D_fake_output_for_G_train = discriminator(adversarial_features)

            # 计算生成器的 GAN 损失。
            # gan_losses.generator_loss 函数根据 GAN 类型计算损失。
            # 在 LSGAN 中，生成器希望 D_fake_output_for_G_train 接近 1。
            gen_gan_loss = gan_losses.generator_loss(D_fake_output_for_G_train)

            # 计算生成器的攻击损失 (第一阶段：特征攻击)。
            # 目标是最大化原始特征和对抗特征之间的差异。
            # attack_losses.get_generator_attack_loss 函数根据配置计算攻击损失。
            # 在 Stage 1 中，它使用 decision_loss_type (通常是 MSE) 计算 original_features 和 adversarial_features 之间的差异。
            # 函数设计为返回一个**负值**（如果损失类型是最大化差异），因此最小化这个负值就等价于最大化正向的差异。
            # 传递原始特征 (来自 ATN，不需要梯度) 和对抗特征 (来自生成器，需要梯度)。
            gen_attack_loss_feature = attack_losses.get_generator_attack_loss(
                 original_atn_outputs={'features': original_features}, # 传递原始特征字典
                 adversarial_atn_outputs={'features': adversarial_features}, # 传递对抗特征字典
                 train_stage=cfg.training.train_stage, # 传递当前训练阶段
                 decision_loss_type=cfg.losses.decision_loss_type, # 使用配置中的决策损失类型作为特征攻击损失类型
                 attention_loss_type=cfg.losses.attention_loss_type # 传递注意力损失类型 (Stage 1 通常为 'none')
                 # region_mask 参数默认 None，表示不使用区域掩码
            ) # 此损失计算的是攻击目标，如果是最大化目标，返回的是负值

            # 计算生成器的正则化损失。
            # 正则化通常应用于生成的扰动 delta，以控制其大小和光滑度。
            # 这些损失需要 delta 具有梯度，因为它们参与损失计算。
            # delta 已经需要梯度，因为它来自 generator(original_features_for_gen)。

            # L-infinity 范数正则化 (通常通过生成器末端的 clamp 硬性限制 epsilon)。
            # 如果配置中 lambda_l_inf > 0 且 losses.regularization_losses 中实现了 linf_penalty，则计算此损失。
            reg_loss_l_inf = torch.tensor(0.0, device=device) # 默认为 0
            if cfg.regularization.lambda_l_inf > 0:
                 # TODO: 如果需要，在 losses.regularization_losses 中实现 linf_penalty 函数，并在此调用。
                 # if hasattr(regularization_losses, 'linf_penalty'):
                 #      reg_loss_l_inf = cfg.regularization.lambda_l_inf * regularization_losses.linf_penalty(delta)
                 # Note: Current regularization_losses only has linf_norm, not penalty loss
                 pass # Placeholder if L-inf penalty loss is added later

            # L2 范数正则化，惩罚扰动的整体 L2 范数。
            # 如果配置中 lambda_l2 > 0，则计算此损失。
            reg_loss_l2 = torch.tensor(0.0, device=device) # 默认为 0
            if cfg.regularization.lambda_l2 > 0:
                 reg_loss_l2 = cfg.regularization.lambda_l2 * l2_norm(delta) # Apply L2 norm as penalty (minimize overall delta size)

            # TV 正则化，惩罚扰动的总变分。
            # 如果配置中 lambda_tv > 0，则计算此损失。
            reg_loss_tv = torch.tensor(0.0, device=device) # 默认为 0
            if cfg.regularization.lambda_tv > 0:
                 reg_loss_tv = cfg.regularization.lambda_tv * total_variation_loss(delta) # Apply TV loss (minimize noise/encourage smoothness)

            # L2 参数正则化，惩罚生成器权重的 L2 范数。
            # 如果配置中 lambda_l2_penalty > 0，则计算此损失。
            reg_loss_l2_params = torch.tensor(0.0, device=device) # 默认为 0
            if cfg.regularization.lambda_l2_penalty > 0:
                # Calculate L2 penalty on generator weights
                gen_l2_reg = torch.tensor(0.0, device=device)
                for param in generator.parameters():
                    if param.requires_grad:
                        gen_l2_reg += torch.norm(param, 2)**2 # Sum of squared L2 norms of parameters
                reg_loss_l2_params = cfg.regularization.lambda_l2_penalty * gen_l2_reg # Apply L2 penalty weight


            # Combine all Generator losses
            # 合并所有生成器的损失项，形成生成器的总损失。
            # 总损失 = GAN 损失 (乘以权重) + 攻击损失 + 各项正则化损失。
            # 注意：Stage 1 的攻击损失 (gen_attack_loss_feature) 设计为负值，因此加到总损失中相当于最大化它。
            # Regularization losses are positive penalties for constraints
            gen_total_loss = cfg.losses.gan_loss_weight * gen_gan_loss + gen_attack_loss_feature + reg_loss_l2 + reg_loss_tv + reg_loss_l2_params
            # Note: L-inf penalty is commented out if not implemented yet

            # Backpropagate Generator loss and update parameters, conditioned on G_freq
            if global_step % current_G_freq == 0:
                 gen_total_loss.backward()
                 optimizer_G.step() # Update Generator parameters
                 # print(f"Debug: Step {global_step} - Updated Generator") # Debug print

            # --- Dynamic GAN Balance Adjustment (Frequency Adjustment Strategy) ---
            if dynamic_balance_enabled and dynamic_balance_cfg.strategy == "loss_ratio_freq_adjust":
                 # Append current losses to history
                 # 将当前的判别器总损失和生成器 GAN 损失添加到对应的历史队列中。
                 # 使用 .item() 方法从 PyTorch 张量中提取标量值。
                 # Use item() to get scalar values from tensors
                 d_loss_history.append(d_total_loss.item())
                 # 对于生成器，我们通常关注其对抗性能力，所以使用 gen_gan_loss 来进行平衡计算
                 g_gan_loss_history.append(gen_gan_loss.item()) # Use the GAN loss part of G loss for balance

                 # Adjust frequencies every freq_adjust_interval steps
                 if (global_step + 1) % dynamic_balance_cfg.freq_adjust_interval == 0:
                      # Calculate average losses from history
                      # 计算损失历史队列中的平均值。
                      # 使用 np.mean() 并将 deque 转换为 list。
                      # 如果队列为空，则平均值为 0.0。
                      avg_d_loss = np.mean(list(d_loss_history)) if d_loss_history else 0.0
                      avg_g_gan_loss = np.mean(list(g_gan_loss_history)) if g_gan_loss_history else 0.0

                      balance_ratio = float('inf') # Initialize ratio to infinity
                      # Avoid division by zero or very small values for G_loss_gan
                      if abs(avg_g_gan_loss) > 1e-8: # Use a small epsilon to check if G loss is effectively zero
                           balance_ratio = avg_d_loss / avg_g_gan_loss
                      # else: if avg_g_gan_loss is near zero, balance_ratio remains inf, indicating G is very strong

                      print(f"Step {global_step+1} (Epoch {epoch}): D Loss Avg: {avg_d_loss:.4f}, G GAN Loss Avg: {avg_g_gan_loss:.4f}, Balance Ratio (D/G_GAN): {balance_ratio:.2f}")

                      # Adjust frequencies based on balance ratio
                      D_dominant_threshold = dynamic_balance_cfg.D_dominant_threshold
                      G_dominant_threshold = dynamic_balance_cfg.G_dominant_threshold
                      min_freq = 1 # Minimum update frequency (update every step)
                      max_freq = 10 # Maximum update frequency (update every 10 steps)

                      if balance_ratio > D_dominant_threshold:
                          # Discriminator is too strong (D loss >> G GAN loss)
                          # Decrease D frequency (update less often), Increase G frequency (update more often)
                          old_D_freq = current_D_freq
                          old_G_freq = current_G_freq
                          current_D_freq = min(current_D_freq + 1, max_freq)
                          current_G_freq = max(min_freq, current_G_freq - 1)
                          if current_D_freq != old_D_freq or current_G_freq != old_G_freq:
                              print(f"  [Balance Adjust] D dominant (Ratio: {balance_ratio:.2f}) -> Freqs adjusted: D {old_D_freq}->{current_D_freq}, G {old_G_freq}->{current_G_freq}")
                      elif balance_ratio < G_dominant_threshold:
                          # Generator is too strong (G GAN loss >> D loss, or D loss is near zero)
                          # Decrease G frequency (update less often), Increase D frequency (update more often)
                          old_D_freq = current_D_freq
                          old_G_freq = current_G_freq
                          current_G_freq = min(current_G_freq + 1, max_freq)
                          current_D_freq = max(min_freq, current_D_freq - 1)
                          if current_D_freq != old_D_freq or current_G_freq != old_G_freq:
                              print(f"  [Balance Adjust] G dominant (Ratio: {balance_ratio:.2f}) -> Freqs adjusted: D {old_D_freq}->{current_D_freq}, G {old_G_freq}->{current_G_freq}")
                      else:
                          # Fairly balanced, reset to initial frequencies if they have changed
                          if current_D_freq != dynamic_balance_cfg.initial_D_freq or current_G_freq != dynamic_balance_cfg.initial_G_freq:
                              old_D_freq = current_D_freq
                              old_G_freq = current_G_freq
                              current_D_freq = dynamic_balance_cfg.initial_D_freq
                              current_G_freq = dynamic_balance_cfg.initial_G_freq
                              print(f"  [Balance Adjust] Balanced -> Freqs reset to initial: D {old_D_freq}->{current_D_freq}, G {old_G_freq}->{current_G_freq}")

            # --- Log Dynamic Balance Metrics ---
            # 记录动态平衡相关的指标到 TensorBoard
            if dynamic_balance_enabled and global_step % cfg.logging.log_interval == 0:
                 writer.add_scalar('GAN_Balance/D_Update_Frequency', current_D_freq, global_step) # 判别器更新频率
                 writer.add_scalar('GAN_Balance/G_Update_Frequency', current_G_freq, global_step) # 生成器更新频率
                 # Log learning rates (will be useful for adaptive_lr strategy later)
                 writer.add_scalar('GAN_Balance/Generator_Learning_Rate', optimizer_G.param_groups[0]['lr'], global_step) # 生成器学习率
                 writer.add_scalar('GAN_Balance/Discriminator_Learning_Rate', optimizer_D.param_groups[0]['lr'], global_step) # 判别器学习率


            # --- Logging and Visualization (Existing) ---
            # The existing logging and visualization code is after the optimizer steps.
            # It uses the losses and delta values from the last step.
            # No major changes needed here, but ensure it uses the updated global_step.
            if global_step % cfg.logging.log_interval == 0:
                # Existing logging code (losses, norms, scores)
                # 记录生成器和判别器的损失到 TensorBoard
                writer.add_scalar('Loss/Generator_Total', gen_total_loss.item(), global_step) # 生成器总损失
                writer.add_scalar('Loss/Generator_GAN', gen_gan_loss.item(), global_step) # 生成器 GAN 损失
                # In Stage 1, Attack (Feature) loss is what gen_attack_loss_feature represents
                # Note: gen_attack_loss_feature is negative, log the absolute value or track it separately
                # Let's log the raw negative value to see it decrease (become more negative)
                writer.add_scalar('Loss/Generator_Attack (Feature)', gen_attack_loss_feature.item(), global_step) # Log attack loss (negative)

                # Log regularization losses (these are positive)
                writer.add_scalar('Loss/Regularization_L2', reg_loss_l2.item(), global_step)
                writer.add_scalar('Loss/Regularization_TV', reg_loss_tv.item(), global_step)
                writer.add_scalar('Loss/Regularization_L2_Params', reg_loss_l2_params.item(), global_step)

                # Log Discriminator losses (these are positive)
                writer.add_scalar('Loss/Discriminator_Total', d_total_loss.item(), global_step) # 判别器总损失
                writer.add_scalar('Loss/Discriminator_Real', d_real_loss.item(), global_step) # 判别器真实样本损失
                writer.add_scalar('Loss/Discriminator_Fake', d_fake_loss.item(), global_step) # 判别器伪造样本损失

                # Log perturbation norms (uses delta from the last G step)
                # 记录生成的扰动范数 (L-inf 和 L2) 到 TensorBoard
                # 计算范数时对 delta 使用 .detach()，确保不计算梯度
                # Calculate norms on the delta generated from original_features_for_gen
                # Ensure delta is detached for logging if needed, though linf_norm/l2_norm should not require gradients
                writer.add_scalar('Perturbation/Feature_Linf_Norm', linf_norm(delta.detach()).mean().item(), global_step) # 记录平均 L-inf 范数
                writer.add_scalar('Perturbation/Feature_L2_Norm', l2_norm(delta.detach()).mean().item(), global_step)   # 记录平均 L2 范数

                # Log Discriminator scores
                # D_real_output and D_fake_output_detached are already detached
                # Calculate mean score over the batch and any patchgan dimensions
                writer.add_scalar('Discriminator_Scores/Real', D_real_output.mean().item(), global_step) # 记录平均真实样本得分
                writer.add_scalar('Discriminator_Scores/Fake', D_fake_output_detached.mean().item(), global_step) # 记录平均伪造样本得分

                # Visualize samples and outputs occasionally
                if (global_step + 1) % cfg.logging.vis_interval == 0: # 每隔 vis_interval 步进行可视化，+1 是为了包含 global_step = 0 的情况
                    # Use a small subset of samples for visualization
                    num_vis_samples = min(cfg.logging.num_vis_samples, real_x.shape[0])
                    # Select which sequence step to visualize for the 2D representation
                    sequence_step_to_vis = cfg.logging.sequence_step_to_vis

                    # Original image for visualization (first few samples)
                    original_image_vis = real_x[:num_vis_samples].detach() # Shape (num_vis, C, T, H, W)

                    # Features for visualization (first few samples)
                    # Use original_features and adversarial_features from the last G step before detaching for D
                    # Ensure they are detached now for visualization
                    original_features_vis = original_features[:num_vis_samples].detach()
                    adversarial_features_vis = adversarial_features[:num_vis_samples].detach() # Use adversarial_features from G train
                    feature_delta_vis = delta[:num_vis_samples].detach() # Use delta from G train

                    # For Stage 1, we are primarily visualizing Feature Perturbation and Feature Difference
                    # Decision/Attention maps are less relevant in Stage 1 visualization unless you specifically want to see
                    # how feature attack implicitly affects them (not the primary goal/loss)
                    visualize_samples_and_outputs(
                        writer,
                        original_image=original_image_vis, # 原始图像切片
                        original_features=original_features_vis, # 原始特征切片
                        feature_delta=feature_delta_vis, # 特征扰动切片
                        adversarial_features=adversarial_features_vis, # 对抗特征切片
                        original_decision_map=None, # Stage 1 不可视化决策图
                        adversarial_decision_map=None, # Stage 1 不可视化决策图
                        step=global_step,
                        num_samples=num_vis_samples, # 可视化的样本数量
                        sequence_step_to_vis=sequence_step_to_vis, # 可视化哪个序列步骤
                        visualize_decision_diff=False # Stage 1 不可视化决策图差异
                    )

                    # If ATN also outputs attention in Stage 1 and you want to visualize its implicit change
                    # (optional, as Stage 1 doesn't attack attention directly)
                    # You would need to get original_attention and adversarial_attention from ATN
                    # adversarial_atn_outputs would need to include attention if needed
                    # For now, skip attention visualization in Stage 1 training loop unless explicitly configured.
                    # if hasattr(cfg.logging, 'visualize_attention_in_stage1') and cfg.logging.visualize_attention_in_stage1:
                    #    # Need adversarial_atn_outputs including attention from the last G step
                    #    # This requires calling get_atn_outputs on adversarial_features, which might be slow
                    #    # Consider doing this only during evaluation or less frequently
                    #    pass # Placeholder for attention visualization in stage 1 training


            # Increment global step
            # 每个批次处理完成后，全局步数加 1
            global_step += 1

        # --- Evaluation ---
        # 在每个 epoch 结束时（或按照 eval_interval 指定的周期）进行模型评估。
        # 这用于在独立的评估数据集上衡量模型的实际攻击效果和性能。
        # Evaluate the model every few epochs
        # 检查当前 epoch 是否达到评估周期或是否是最后一个 epoch
        if (epoch + 1) % cfg.training.eval_interval == 0 or (epoch + 1) == cfg.training.num_epochs:
            print(f"Running evaluation at epoch {epoch+1}")
            # 在评估数据集上评估模型性能。
            # evaluate_model 函数需要生成器、判别器、ATN 模型、评估数据加载器、设备、配置和当前训练阶段。
            # 注意：这里的代码是占位符，需要一个专门的评估数据加载器 (eval_dataloader)。
            # 目前 dataloader 是训练数据加载器，直接用于评估不准确。
            # Note: Need to ensure evaluation dataset and dataloader are setup
            # For now, let's assume evaluation uses the same dataloader or a separate eval_dataloader
            # Passed dataloader is the training dataloader, need a separate eval_dataloader
            # TODO: Setup a separate evaluation dataloader if needed

            # Placeholder evaluation call - need a dedicated evaluation dataloader
            # Assuming for now we use the training dataloader for a quick check (not ideal)
            # eval_metrics = evaluate_model(generator, discriminator, atn_model, dataloader, device, cfg, cfg.training.train_stage)
            # print(f"Evaluation Metrics at epoch {epoch+1}: {eval_metrics}")

            # TODO: Implement proper evaluation with a dedicated eval_dataloader
            # 目前评估逻辑被跳过，打印提示信息。
            print("Evaluation skipped in training loop. Implement with a dedicated eval dataloader.")

            # Example of how to use evaluation metrics for logging:
            # 如果实现了评估，可以将评估指标记录到 TensorBoard
            # if eval_metrics:
            #    for metric_name, metric_value in eval_metrics.items():
            #        writer.add_scalar(f'Evaluation/{metric_name}', metric_value, global_step) # global_step 记录评估发生时的总步数
            #
            #    # Save best model based on a chosen evaluation metric (e.g., Attack_Success_Rate_Feature_Stage1)
            #    # 如果评估结果是当前最优，则保存模型检查点。
            #    # TODO: 选择用于判断最优模型的评估指标 (例如 Stage 1 的特征攻击成功率)。
            #    # primary_eval_metric = eval_metrics.get(f'Attack_Success_Rate_Feature_Stage{cfg.training.train_stage}', -float('inf'))
            #    # if primary_eval_metric > best_eval_metric:
            #    #    best_eval_metric = primary_eval_metric
            #    #    # 构建最佳检查点保存路径
            #    #    best_checkpoint_path = os.path.join(checkpoint_dir, 'best_stage_{}.pth'.format(cfg.training.train_stage))
            #    #    print(f"Saving best model to {best_checkpoint_path} based on metric: {best_eval_metric:.4f}")
            #    #    # 保存模型状态、优化器状态、epoch、global_step 等信息
            #    #    torch.save({
            #    #        'epoch': epoch,
            #    #        'global_step': global_step,
            #    #        'generator_state_dict': generator.state_dict(),
            #    #        'discriminator_state_dict': discriminator.state_dict(),
            #    #        'optimizer_G_state_dict': optimizer_G.state_dict(),
            #    #        'optimizer_D_state_dict': optimizer_D.state_dict(),
            #    #        'best_eval_metric': best_eval_metric,
            #    #        'current_train_stage': cfg.training.train_stage
            #    #    }, best_checkpoint_path)
            #    #    print("Best model checkpoint saved.")


        # --- Save Checkpoint ---
        # 定期或在每个 epoch 结束时保存模型的检查点。
        # 这使得训练可以在中断后从上次保存的状态恢复。
        # Save checkpoint every few epochs or at the end of training
        # 检查当前 epoch 是否达到保存周期或是否是最后一个 epoch
        if (epoch + 1) % cfg.training.save_interval == 0 or (epoch + 1) == cfg.training.num_epochs:
            # Ensure checkpoint directory exists (already done outside loop, but safe to re-check)
            # 确保检查点保存目录存在，如果不存在则创建
            checkpoint_dir = os.path.join(cfg.logging.log_dir, 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Define checkpoint path, maybe include stage number
            # 构建检查点文件的完整路径，包含当前 epoch 和训练阶段
            checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}_stage_{cfg.training.train_stage}.pth')
            print(f"Saving checkpoint to {checkpoint_path}")

            # Save the state dictionaries and other relevant info
            # 使用 torch.save 将模型的 state_dict、优化器的 state_dict、当前的 epoch、global_step 等信息保存到文件。
            torch.save({
                'epoch': epoch,
                'global_step': global_step, # 保存当前的 global_step
                'generator_state_dict': generator.state_dict(), # 保存生成器模型的状态
                'discriminator_state_dict': discriminator.state_dict(), # 保存判别器模型的状态
                'optimizer_G_state_dict': optimizer_G.state_dict(), # 保存生成器优化器的状态
                'optimizer_D_state_dict': optimizer_D.state_dict(), # 保存判别器优化器的状态
                # Add best_eval_metric if you implement evaluation and saving best model
                # 'best_eval_metric': best_eval_metric, # 如果保存了最优模型逻辑，也保存最优指标值
                'current_train_stage': cfg.training.train_stage, # 保存当前的训练阶段
                # TODO: Save random states for full reproducibility
                # 为了实现完全的可复现性，通常还需要保存随机数生成器的状态。
                # 在从检查点恢复时加载这些状态。
                # 'random_states': {
                #    'torch_rng_state': torch.get_rng_state(),
                #    'torch_cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                #    'numpy_rng_state': np.random.get_state(),
                #    'random_rng_state': random.getstate(), # 需要导入 random
                # }

            }, checkpoint_path)
            print("Checkpoint saved.")

    # 训练循环结束
    # 在所有 epoch 训练完成后执行的代码。
    # Close the TensorBoard writer
    writer.close() # 关闭 SummaryWriter，确保所有日志写入文件
    print("Training finished.")


# If the script is run directly, parse arguments and start training
# Python 脚本的标准入口点。
# 当脚本直接运行时 (而不是作为模块导入)，执行此块中的代码。
if __name__ == '__main__':
    # 使用 config_utils.parse_args_and_config 函数解析命令行参数，并加载和合并配置文件。
    # 该函数会加载默认配置 (configs/default_config.yaml)，然后加载通过 --config 参数指定的任务配置文件，
    # 最后使用其他命令行参数覆盖已有配置。
    # Pass the default config path and the name of the command line arg for the task config file
    cfg = parse_args_and_config(default_config_path='configs/default_config.yaml', task_config_arg='config')

    # Check if config loading was successful
    # 检查配置文件是否成功加载 (parse_args_and_config 在失败时返回 None)。
    if cfg is not None:
        print("Configuration loaded:")
        # 打印加载的配置内容，使用 json 库进行格式化输出，使其更易读。
        # 需要导入 json 和 EasyDict (因为它通常是 EasyDict 对象)。
        # Using json.dumps provides nice indentation
        import json # Import json here as it is only used in this block
        from easydict import EasyDict # Import EasyDict here
        print(json.dumps(dict(cfg), indent=4)) # 将 EasyDict 转换为标准字典再用 json 格式化打印
        print("--------------------------")

        # Ensure the log directory exists before starting training/logging
        # 确保日志和检查点目录存在。如果不存在，则创建它们。
        # 使用 os.makedirs() 并设置 exist_ok=True，即使目录已存在也不会报错。
        # Use os.path.join to correctly handle paths across different OS
        os.makedirs(cfg.logging.log_dir, exist_ok=True) # 创建日志目录
        # 也确保检查点目录存在
        os.makedirs(os.path.join(cfg.logging.log_dir, 'checkpoints'), exist_ok=True) # 创建检查点目录

        # Call the main training function with the loaded configuration
        # 调用主训练函数 train()，将加载的配置对象 cfg 传递进去，开始训练过程。
        train(cfg)
    else:
        # If config loading failed, print an error message and exit
        # 如果配置文件加载失败 (cfg is None)，打印错误信息并退出脚本。
        print("Failed to load configuration. Exiting.")
        sys.exit(1) # 使用非零退出码表示脚本执行失败 