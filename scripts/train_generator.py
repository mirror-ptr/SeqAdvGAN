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
        cfg=cfg,
        device=device
    )

    # 检查 ATN 模型是否加载成功
    if atn_model is None:
        print(f"Error: Failed to load ATN model from {cfg.model.atn.model_path}. Exiting training.")
        sys.exit(1) # 如果加载失败，退出脚本
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
        print("Initializing Stage 2 models...")
        # Stage 2 Generator takes image input and outputs image perturbation
        generator = Generator(
            in_channels=cfg.model.generator.in_channels, # Generator 输入通道数 (应与原始图像通道数匹配)
            out_channels=cfg.model.generator.out_channels, # Generator 输出通道数 (应与原始图像通道数匹配)
            num_bottlenecks=cfg.model.generator.num_bottlenecks,
            base_channels=cfg.model.generator.base_channels,
            epsilon=cfg.model.generator.epsilon # 像素级扰动上限
        ).to(device)

        # Stage 2 Discriminator takes adversarial image as input
        if cfg.model.discriminator.type == 'cnn':
            discriminator = SequenceDiscriminatorCNN(
                in_channels=cfg.model.discriminator.in_channels, # Discriminator 输入通道数 (应与原始图像通道数匹配)
                base_channels=cfg.model.discriminator.base_channels
            ).to(device)
            print("Using CNN Discriminator for Stage 2")
        elif cfg.model.discriminator.type == 'patchgan':
             discriminator = PatchDiscriminator3d(
                  in_channels=cfg.model.discriminator.in_channels, # Discriminator 输入通道数 (应与原始图像通道数匹配)
                  base_channels=cfg.model.discriminator.base_channels
             ).to(device)
             print("Using PatchGAN Discriminator for Stage 2")
        else:
            print(f"Error: Unsupported discriminator type for Stage 2: {cfg.model.discriminator.type}. Supported types are 'cnn' and 'patchgan'.")
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
        # 训练数据加载器
        train_dataloader = create_mock_dataloader(
            batch_size=cfg.training.batch_size, # 批次大小
            num_samples=cfg.data.mock_num_samples, # 模拟样本数量
            sequence_length=cfg.data.sequence_length, # 序列长度
            channels=cfg.data.channels, # 图像通道数
            height=cfg.data.height, # 图像高度
            width=cfg.data.width, # 图像宽度
            shuffle=True, # 在每个 epoch 开始时打乱数据
            num_workers=cfg.data.num_workers # 数据加载器工作进程数
        )
        # 评估数据加载器 (模拟数据，使用部分样本，不打乱)
        eval_dataloader = create_mock_dataloader(
            batch_size=cfg.evaluation.num_eval_samples, # 评估批次大小，使用评估样本数量
            num_samples=cfg.evaluation.num_eval_samples, # 模拟评估样本数量
            sequence_length=cfg.data.sequence_length, # 序列长度
            channels=cfg.data.channels, # 图像通道数
            height=cfg.data.height, # 图像高度
            width=cfg.data.width, # 图像宽度
            shuffle=False, # 评估时不打乱数据
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
            target_width=cfg.data.width, # 视频帧的目标宽度
            device=device # Pass the determined device
        )
        # 使用 DataLoader 包装数据集，并配置相关参数。
        # worker_init_fn 用于确保多进程数据加载时的随机性。
        # pin_memory=True 可以将数据加载到 CUDA 锁页内存，加速 GPU 数据传输。
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.training.batch_size, # 批次大小
            shuffle=True, # 在每个 epoch 开始时打乱数据
            num_workers=cfg.data.num_workers, # 数据加载器工作进程数
            worker_init_fn=worker_init_fn, # 工作进程初始化函数
            pin_memory=True, # 启用锁页内存
            multiprocessing_context='spawn' # 使用 'spawn' 启动方法以兼容 CUDA 和多进程
        )
        # 评估数据加载器 (使用相同数据集，但批量大小为评估样本数量，不打乱)
        # 注意：在大型项目中，通常会使用独立的评估数据集
        eval_dataset = GameVideoDataset( # 可以重用训练数据集，或者加载一个独立的评估数据集
             video_path=cfg.data.video_path, # 视频文件路径
             level_json_path=cfg.data.level_json_path, # 关卡 JSON 文件路径
             sequence_length=cfg.data.sequence_length, # 每个样本的序列长度 (帧数)
             transform=None,
             target_height=cfg.data.height,
             target_width=cfg.data.width,
             device=device
        )
        eval_dataloader = DataLoader(
             eval_dataset,
             batch_size=cfg.evaluation.num_eval_samples, # 评估批次大小，使用评估样本数量
             shuffle=False, # 评估时不打乱数据
             num_workers=cfg.data.num_workers,
             worker_init_fn=worker_init_fn,
             pin_memory=True,
             multiprocessing_context='spawn'
        )


    print(f"Training data loaded. Number of batches per epoch: {len(train_dataloader)}")
    print(f"Evaluation data loaded. Number of batches: {len(eval_dataloader)}")


    # --- TensorBoard Logging Setup ---
    # 创建 SummaryWriter 对象，用于将训练过程中的标量、图像等写入 TensorBoard 日志文件。
    # 日志目录根据配置 cfg.logging.log_dir 和当前训练阶段 train_stage 确定。
    log_dir = cfg.logging.log_dir # 直接使用配置文件中的日志目录
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    # --- Checkpoint Loading (Optional) ---
    # 检查是否需要从检查点恢复训练。
    start_epoch = 0 # 训练开始的 epoch 编号
    global_step = 0 # 训练开始的全局步数
    best_eval_metric = -float('inf') # 用于保存最优模型的评估指标值，初始化为一个很小的值

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
              # 更新开始的 epoch 编号和全局步数
              start_epoch = checkpoint['epoch'] + 1
              global_step = checkpoint.get('global_step', start_epoch * len(train_dataloader)) # 兼容旧检查点
              # 如果检查点中保存了 best_eval_metric，也进行恢复
              best_eval_metric = checkpoint.get('best_eval_metric', -float('inf')) # 兼容旧检查点
              print(f"Checkpoint loaded successfully. Resuming from epoch {start_epoch}, global_step {global_step}.")
              # TODO: Resume dynamic balance state if needed (loss history, freqs)

         except FileNotFoundError:
              print(f"Error: Checkpoint file not found at {cfg.training.resume_checkpoint}. Starting training from scratch (epoch 0).")
              # global_step 保持 0，start_epoch 保持 0
         except KeyError as e:
              print(f"Error: Key '{e}' not found in checkpoint. Checkpoint format mismatch? Starting training from scratch (epoch 0).")
              # global_step 保持 0，start_epoch 保持 0
         except Exception as e:
              print(f"Error loading checkpoint: {e}. Starting training from scratch (epoch 0).")
              # global_step 保持 0，start_epoch 保持 0

    print("Starting training...")

    # Initialize global_step if not resuming or resuming failed
    if not cfg.training.resume_checkpoint or global_step == 0:
        global_step = 0

    # TODO: Initialize Dynamic GAN Balance Variables (Need to handle resuming)
    # 从配置中读取 GAN 动态平衡相关的参数
    dynamic_balance_cfg = cfg.training.dynamic_gan_balance
    dynamic_balance_enabled = dynamic_balance_cfg.enabled # 是否启用动态平衡
    current_D_freq = dynamic_balance_cfg.initial_D_freq # 判别器初始更新频率
    current_G_freq = dynamic_balance_cfg.initial_G_freq # 生成器初始更新频率

    # 使用 deque (双端队列) 存储最近的损失历史，用于计算移动平均损失
    if dynamic_balance_enabled:
         # 判别器总损失的历史队列
         d_loss_history = deque(maxlen=dynamic_balance_cfg.loss_history_window)
         # 生成器 GAN 损失的历史队列
         g_gan_loss_history = deque(maxlen=dynamic_balance_cfg.loss_history_window)
         # TODO: If resuming, load loss history from checkpoint
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
        for i, batch_data in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch}/{cfg.training.num_epochs}"):
            # batch_data 的形状通常是 (B, C, T, H, W)，表示原始的图像序列数据。
            # 这是 ATN 模型期望的输入格式。
            # 将批次数据移动到指定的计算设备 (GPU 或 CPU)。
            real_x = batch_data.to(device) # Real image data

            # --- Stage 1 Training Logic (Feature Attack) ---
            # 第一阶段训练的目标：训练生成器产生针对 ATN 特征层的扰动，
            # 使对抗样本在特征空间与原始样本差异最大化，并欺骗判别器。
            # Goal: Generator generates perturbation on features to max feature difference and fool Discriminator

            # 1. Get original features from ATN (ATN takes original image data as input)
            # 从预训练的 ATN 模型获取原始输入的特征。
            # 由于 ATN 模型是冻结的，使用 torch.no_grad() 上下文管理器，确保不为 ATN 的参数计算梯度。
            with torch.no_grad(): # Ensure no gradients flow back to ATN model
                 original_atn_outputs = get_atn_outputs(
                      atn_model,
                      real_x, # 传递原始图像数据作为 ATN 的输入 (形状: B, C_img, T, H_img, W_img)
                      cfg=cfg # 传递配置对象
                  )
            # original_features 的形状大约是 (B, C_feat, N_feat, H_feat, W_feat)，取决于 ATN 特征提取器的输出。
            original_features = original_atn_outputs.get('features')

            # Check if original_features is None or empty. If so, skip this batch.
            if original_features is None or original_features.numel() == 0:
                 print(f"Warning: Epoch {epoch}, Batch {i} - ATN failed to return features. Skipping batch.")
                 continue # Skip this batch if features are not available

            # Ensure original_features requires grad for Generator training
            original_features_for_gen = original_features.clone().detach().requires_grad_(True)

            # --- Train Discriminator ---
            # 训练判别器的目标：使判别器能够区分真实的特征 (来自原始 ATN 输出) 和伪造的特征 (来自生成器输出)。
            # Goal: Discriminator distinguishes real features (from original ATN) and fake features (from Generator)

            # 清除判别器优化器的梯度，为当前训练步骤做准备。
            optimizer_D.zero_grad() # Clear gradients for Discriminator

            # 获取判别器对真实特征的输出。
            D_real_output = discriminator(original_features.detach()) # 判别器处理真实的特征张量

            # 生成伪造特征：原始特征 + 生成器产生的扰动。
            with torch.no_grad(): # 确保在判别器训练时，梯度不会通过生成器反向传播
                 # 生成扰动 delta，输入是原始特征的副本 (已分离)
                 delta_for_D = generator(original_features_for_gen.detach()) # 生成 delta
                 # 计算用于判别器训练的对抗特征
                 adversarial_features_for_D = original_features.detach() + delta_for_D

            # 获取判别器对伪造特征的输出。
            D_fake_output_detached = discriminator(adversarial_features_for_D)

            # 计算判别器的 GAN 损失。
            d_total_loss, d_real_loss, d_fake_loss = gan_losses.discriminator_loss(D_real_output, D_fake_output_detached)

            # 根据动态平衡策略或固定频率，判断是否更新判别器的参数。
            if global_step % current_D_freq == 0:
                 # 计算判别器损失的梯度。
                 d_total_loss.backward()
                 # 根据梯度更新判别器的参数。
                 optimizer_D.step() # 更新判别器参数

            # --- Train Generator ---
            # 训练生成器的目标：
            # 1. 使生成器产生的对抗特征能够欺骗判别器 (生成器希望判别器认为伪造样本是真实的)。
            # 2. 使生成器产生的对抗特征与原始特征之间的差异最大化 (攻击目标)。
            # Goal: Make Generator output (adversarial features) fool the Discriminator AND maximize feature difference

            # 清除生成器优化器的梯度。
            optimizer_G.zero_grad() # 清除生成器参数的梯度

            # 生成扰动，这一次需要确保梯度能够从生成器的输出反向传播到生成器的参数。
            delta = generator(original_features_for_gen) # 生成扰动，梯度会流经 generator

            # 计算用于生成器训练的对抗特征。
            adversarial_features = original_features_for_gen + delta

            # 获取判别器对这些对抗特征的输出，用于计算生成器的 GAN 损失。
            D_fake_output_for_G_train = discriminator(adversarial_features)

            # 计算生成器的 GAN 损失。
            gen_gan_loss = gan_losses.generator_loss(D_fake_output_for_G_train)

            # 计算生成器的攻击损失 (第一阶段：特征攻击)。
            # 目标是最大化原始特征和对抗特征之间的差异。
            # attack_losses.get_generator_attack_loss 函数根据配置计算攻击损失。
            # 在 Stage 1 中，它使用 decision_loss_type (通常是 MSE) 计算 original_features 和 adversarial_features 之间的差异。
            gen_attack_loss_feature = attack_losses.get_generator_attack_loss(
                 original_atn_outputs={'features': original_features}, # 传递原始特征字典
                 adversarial_atn_outputs={'features': adversarial_features}, # 传递对抗特征字典
                 train_stage=cfg.training.train_stage, # 传递当前训练阶段
                 decision_loss_type=cfg.losses.decision_loss_type, # 使用配置中的决策损失类型作为特征攻击损失类型
                 attention_loss_type=cfg.losses.attention_loss_type # 传递注意力损失类型 (Stage 1 通常为 'none')
            ) # 此损失计算的是攻击目标，如果是最大化目标，返回的是负值

            # 计算生成器的正则化损失。
            # 正则化通常应用于生成的扰动 delta，以控制其大小和光滑度。

            reg_loss_l_inf = torch.tensor(0.0, device=device) # 默认为 0

            reg_loss_l2 = torch.tensor(0.0, device=device) # 默认为 0
            if cfg.regularization.lambda_l2 > 0:
                 reg_loss_l2 = cfg.regularization.lambda_l2 * l2_norm(delta) # Apply L2 norm as penalty (minimize overall delta size)

            reg_loss_tv = torch.tensor(0.0, device=device) # 默认为 0
            if cfg.regularization.lambda_tv > 0:
                 reg_loss_tv = cfg.regularization.lambda_tv * total_variation_loss(delta) # Apply TV loss (minimize noise/encourage smoothness)

            reg_loss_l2_params = torch.tensor(0.0, device=device) # 默认为 0
            if cfg.regularization.lambda_l2_penalty > 0:
                gen_l2_reg = torch.tensor(0.0, device=device)
                for param in generator.parameters():
                    if param.requires_grad:
                        gen_l2_reg += torch.norm(param, 2)**2 # Sum of squared L2 norms of parameters
                reg_loss_l2_params = cfg.regularization.lambda_l2_penalty * gen_l2_reg # Apply L2 penalty weight


            # Combine all Generator losses
            gen_total_loss = cfg.losses.gan_loss_weight * gen_gan_loss + gen_attack_loss_feature + reg_loss_l2 + reg_loss_tv + reg_loss_l2_params

            # Backpropagate Generator loss and update parameters, conditioned on G_freq
            if global_step % current_G_freq == 0:
                 gen_total_loss.backward()
                 optimizer_G.step() # Update Generator parameters

            # --- Dynamic GAN Balance Adjustment (Frequency Adjustment Strategy) ---
            if dynamic_balance_enabled and dynamic_balance_cfg.strategy == "loss_ratio_freq_adjust":
                 # Append current losses to history
                 d_loss_history.append(d_total_loss.item())
                 g_gan_loss_history.append(gen_gan_loss.item()) # Use the GAN loss part of G loss for balance

                 # Adjust frequencies every freq_adjust_interval steps
                 if (global_step + 1) % dynamic_balance_cfg.freq_adjust_interval == 0:
                      # Calculate average losses from history
                      avg_d_loss = np.mean(list(d_loss_history)) if d_loss_history else 0.0
                      avg_g_gan_loss = np.mean(list(g_gan_loss_history)) if g_gan_loss_history else 0.0

                      balance_ratio = float('inf') # Initialize ratio to infinity
                      # Avoid division by zero or very small values for G_loss_gan
                      if abs(avg_g_gan_loss) > 1e-8: # Use a small epsilon to check if G loss is effectively zero
                           balance_ratio = avg_d_loss / avg_g_gan_loss

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
            if dynamic_balance_enabled and global_step % cfg.logging.log_interval == 0:
                 writer.add_scalar('GAN_Balance/D_Update_Frequency', current_D_freq, global_step) # 判别器更新频率
                 writer.add_scalar('GAN_Balance/G_Update_Frequency', current_G_freq, global_step) # 生成器更新频率
                 writer.add_scalar('GAN_Balance/Generator_Learning_Rate', optimizer_G.param_groups[0]['lr'], global_step) # 生成器学习率
                 writer.add_scalar('GAN_Balance/Discriminator_Learning_Rate', optimizer_D.param_groups[0]['lr'], global_step) # 判别器学习率


            # --- Logging and Visualization (Existing) ---
            if global_step % cfg.logging.log_interval == 0:
                # Existing logging code (losses, norms, scores)
                writer.add_scalar('Loss/Generator_Total', gen_total_loss.item(), global_step) # 生成器总损失
                writer.add_scalar('Loss/Generator_GAN', gen_gan_loss.item(), global_step) # 生成器 GAN 损失
                writer.add_scalar('Loss/Generator_Attack (Feature)', gen_attack_loss_feature.item(), global_step) # Log attack loss (negative)

                writer.add_scalar('Loss/Regularization_L2', reg_loss_l2.item(), global_step)
                writer.add_scalar('Loss/Regularization_TV', reg_loss_tv.item(), global_step)
                writer.add_scalar('Loss/Regularization_L2_Params', reg_loss_l2_params.item(), global_step)

                writer.add_scalar('Loss/Discriminator_Total', d_total_loss.item(), global_step) # 判别器总损失
                writer.add_scalar('Loss/Discriminator_Real', d_real_loss.item(), global_step) # 判别器真实样本损失
                writer.add_scalar('Loss/Discriminator_Fake', d_fake_loss.item(), global_step) # 判别器伪造样本损失

                writer.add_scalar('Perturbation/Feature_Linf_Norm', linf_norm(delta.detach()).mean().item(), global_step) # 记录平均 L-inf 范数
                writer.add_scalar('Perturbation/Feature_L2_Norm', l2_norm(delta.detach()).mean().item(), global_step)   # 记录平均 L2 范数

                writer.add_scalar('Discriminator_Scores/Real', D_real_output.mean().item(), global_step) # 记录平均真实样本得分
                writer.add_scalar('Discriminator_Scores/Fake', D_fake_output_detached.mean().item(), global_step) # 记录平均伪造样本得分

                # Visualize samples and outputs occasionally
                if (global_step + 1) % cfg.logging.vis_interval == 0: # 每隔 vis_interval 步进行可视化，+1 是为了包含 global_step = 0 的情况
                    num_vis_samples = min(cfg.logging.num_vis_samples, real_x.shape[0])
                    sequence_step_to_vis = cfg.logging.sequence_step_to_vis

                    original_image_vis = real_x[:num_vis_samples].detach()
                    original_features_vis = original_features[:num_vis_samples].detach()
                    feature_delta_vis = delta[:num_vis_samples].detach()
                    adversarial_features_vis = adversarial_features[:num_vis_samples].detach()

                    visualize_samples_and_outputs(
                        writer,
                        original_image=original_image_vis,
                        original_features=original_features_vis,
                        feature_delta=feature_delta_vis,
                        adversarial_features=adversarial_features_vis,
                        original_decision_map=None,
                        adversarial_decision_map=None,
                        step=global_step,
                        num_samples=num_vis_samples,
                        sequence_step_to_vis=sequence_step_to_vis,
                        visualize_decision_diff=False
                    )


            # Increment global step
            global_step += 1

        # --- Evaluation ---
        # 在每个 epoch 结束时（或按照 eval_interval 指定的周期）进行模型评估。
        # 检查当前 epoch 是否达到评估周期或是否是最后一个 epoch
        if (epoch + 1) % cfg.training.eval_interval == 0 or (epoch + 1) == cfg.training.num_epochs:
            print(f"Running evaluation at epoch {epoch+1}")

            # 调用完善后的 evaluate_model 函数
            eval_metrics = evaluate_model(
                generator=generator if cfg.training.train_stage == 1 else None, # Stage 1 评估需要 Generator
                discriminator=discriminator, # 评估 Discriminator 性能
                atn_model=atn_model, # 评估需要 ATN 模型
                dataloader=eval_dataloader, # 使用评估数据加载器
                device=device,
                cfg=cfg,
                current_train_stage=cfg.training.train_stage # 传递当前训练阶段
            )

            # 将评估指标记录到 TensorBoard
            if eval_metrics:
               print(f"Evaluation Metrics at epoch {epoch+1}: {eval_metrics}")
               for metric_name, metric_value in eval_metrics.items():
                   # 确保指标值不是 NaN 或 Inf，并转换为 float
                   if np.isfinite(metric_value):
                        writer.add_scalar(f'Evaluation/{metric_name}', float(metric_value), global_step) # global_step 记录评估发生时的总步数
                   else:
                        print(f"Warning: Evaluation metric '{metric_name}' is not finite ({metric_value}). Skipping logging.")

               # --- Save best model based on evaluation metric (Stage 1: Maximize Feature L2 Diff) ---
               # 选择用于判断最优模型的评估指标。在 Stage 1，通常是最大化特征差异。
               # 例如，使用 'Feature_L2_Diff_Avg' 作为指标，目标是最大化它。
               primary_eval_metric_name = 'Feature_L2_Diff_Avg' # Stage 1 关注特征 L2 差异
               is_lower_metric_better = False # 对于特征差异指标，越高越好 (攻击效果越好)

               primary_eval_metric = eval_metrics.get(primary_eval_metric_name)

               # 如果成功获取到主要评估指标且是有限值
               if primary_eval_metric is not None and np.isfinite(primary_eval_metric):
                   # 根据目标 (最小化或最大化) 判断是否是最好的结果
                   # 注意这里修改了判断条件，现在是 primary_eval_metric > best_eval_metric 时保存
                   if (is_lower_metric_better and primary_eval_metric < best_eval_metric) or \
                      (not is_lower_metric_better and primary_eval_metric > best_eval_metric):
                        best_eval_metric = primary_eval_metric # 更新最优指标值

                        # 构建最佳检查点保存路径
                        checkpoint_dir = os.path.join(cfg.logging.log_dir, 'checkpoints') # Ensure path is correct
                        os.makedirs(checkpoint_dir, exist_ok=True) # Ensure checkpoint dir exists
                        best_checkpoint_path = os.path.join(checkpoint_dir, f'best_stage_{cfg.training.train_stage}.pth')
                        print(f"Saving best model to {best_checkpoint_path} based on {primary_eval_metric_name}: {best_eval_metric:.4f}")

                        # 保存模型状态、优化器状态、epoch、global_step 等信息
                        torch.save({
                            'epoch': epoch,
                            'global_step': global_step, # 保存当前的 global_step
                            'generator_state_dict': generator.state_dict(), # 保存生成器模型的状态
                            'discriminator_state_dict': discriminator.state_dict(), # 保存判别器模型的状态
                            'optimizer_G_state_dict': optimizer_G.state_dict(), # 保存生成器优化器的状态
                            'optimizer_D_state_dict': optimizer_D.state_dict(), # 保存判别器优化器的状态
                            'best_eval_metric': best_eval_metric, # 保存最优指标值
                            'current_train_stage': cfg.training.train_stage, # 保存当前的训练阶段
                            # TODO: Save random states for full reproducibility
                        }, best_checkpoint_path)
                        print("Best model checkpoint saved.")
               else:
                   print(f"Warning: Primary evaluation metric '{primary_eval_metric_name}' is not available or not finite. Skipping best model saving.")
            else:
               print("Warning: Evaluation metrics dictionary is empty. Skipping logging and best model saving.")


        # --- Save Checkpoint ---
        # 定期或在每个 epoch 结束时保存模型的检查点。
        if (epoch + 1) % cfg.training.save_interval == 0 or (epoch + 1) == cfg.training.num_epochs:
            # Ensure checkpoint directory exists
            checkpoint_dir = os.path.join(cfg.logging.log_dir, 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Define checkpoint path
            checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}_stage_{cfg.training.train_stage}.pth')
            print(f"Saving checkpoint to {checkpoint_path}")

            # Save the state dictionaries and other relevant info
            torch.save({
                'epoch': epoch,
                'global_step': global_step, # 保存当前的 global_step
                'generator_state_dict': generator.state_dict(), # 保存生成器模型的状态
                'discriminator_state_dict': discriminator.state_dict(), # 保存判别器模型的状态
                'optimizer_G_state_dict': optimizer_G.state_dict(), # 保存生成器优化器的状态
                'optimizer_D_state_dict': optimizer_D.state_dict(), # 保存判别器优化器的状态
                'best_eval_metric': best_eval_metric, # 保存最优指标值
                'current_train_stage': cfg.training.train_stage, # 保存当前的训练阶段
                # TODO: Save random states for full reproducibility
            }, checkpoint_path)
            print("Checkpoint saved.")

    # 训练循环结束
    # Close the TensorBoard writer
    writer.close()
    print("Training finished.")


# If the script is run directly, parse arguments and start training
if __name__ == '__main__':
    # 使用 config_utils.parse_args_and_config 函数解析命令行参数，并加载和合并配置文件。
    cfg = parse_args_and_config(default_config_path='configs/default_config.yaml', task_config_arg='config')

    # Check if config loading was successful
    if cfg is not None:
        print("Configuration loaded:")
        import json # Import json here as it is only used in this block
        from easydict import EasyDict # Import EasyDict here
        print(json.dumps(dict(cfg), indent=4)) # 将 EasyDict 转换为标准字典再用 json 格式化打印
        print("--------------------------")

        # Ensure the log directory exists before starting training/logging
        os.makedirs(cfg.logging.log_dir, exist_ok=True)
        os.makedirs(os.path.join(cfg.logging.log_dir, 'checkpoints'), exist_ok=True)

        # Call the main training function with the loaded configuration
        train(cfg)
    else:
        print("Failed to load configuration. Exiting.")
        sys.exit(1) 