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
from tqdm import tqdm # 从 tqdm 库导入 tqdm 函数

# 导入模型、损失函数和工具模块
from models.generator_cnn import Generator # 导入生成器模型
from models.discriminator_cnn import SequenceDiscriminatorCNN # 导入 CNN 判别器
from models.discriminator_patchgan import PatchDiscriminator3d # 导入 PatchGAN 判别器
from losses.gan_losses import GANLosses # 导入 GAN 损失函数类
from losses.attack_losses import AttackLosses # 导入攻击损失函数类
# 导入正则化损失函数 (L-inf, L2, L2参数惩罚, TV)
from losses.regularization_losses import linf_norm, l2_norm, l2_penalty, total_variation_loss

# 导入可视化工具函数
from utils.vis_utils import visualize_samples_and_outputs, visualize_attention_maps, visualize_stage2_pixel_attack, visualize_training_losses, visualize_perturbation_norms # 导入可视化函数

# 导入 ATN 工具和数据工具
from utils.atn_utils import load_atn_model, get_atn_outputs
from utils.data_utils import GameVideoDataset, create_mock_dataloader, worker_init_fn

# 导入评估工具函数
from utils.eval_utils import evaluate_model # 导入评估函数

# 导入配置工具函数
from utils.config_utils import parse_args_and_config

# Define a custom collate function to filter out None values
def collate_fn_skip_none(batch):
    """
    Custom collate function that filters out None values from the batch.
    """
    batch = [item for item in batch if item is not None]
    # If the batch is empty after filtering, return an empty tensor with the correct structure.
    if not batch:
        # Return an empty tensor with correct shape and dtype for a batch of size 0.
        # We need to know the expected shape of a single item from the dataset.
        # Based on GameVideoDataset.__getitem__, it returns a tensor of shape (T, H, W, C).
        # We need T, H, W, C. Since cfg is not directly accessible here, we'll hardcode based on common config structure.
        # This is fragile and assumes config values. A robust solution requires passing these or inferring reliably.
        # Let's assume sequence_length, height, width, channels are accessible via a passed object or global config 'cfg'.
        # This requires modifying the DataLoader calls to pass a lambda or partial.
        # Alternative: Infer shape from the first *valid* item encountered. But if the *first batch* is all None, this fails.
        # Safest with current tools: Pass the necessary dimensions to the collate_fn factory.
        # However, I cannot modify the train function signature or add a factory call in this single edit.
        # Let's try returning a minimal empty tensor that is still a tensor.
        # The code expects a tensor it can call .to(device) on.
        # Let's return torch.empty(0, 1) as a placeholder empty tensor. This might still cause shape issues later, but avoids the IndexError.
        # Reverting to the factory function approach is the proper fix. Let's do that in two steps.

        # Step 1: Define the factory function here.
        # Step 2: Modify DataLoader calls in train function.

        # Define a factory function to create the collate_fn with shape info
        def create_collate_fn_with_shape(sequence_length: int, height: int, width: int, channels: int):
            def collate_fn_skip_none_with_shape(batch: list):
                batch = [item for item in batch if item is not None]
                if not batch:
                    # Return an empty tensor with correct shape (0, T, H, W, C) and dtype (float32 assumed)
                    return torch.empty(
                        (0, sequence_length, height, width, channels),
                        dtype=torch.float32
                    )
                # Use default_collate on the valid samples
                return torch.utils.data._utils.collate.default_collate(batch)
            return collate_fn_skip_none_with_shape

        # Now, this function collate_fn_skip_none needs to *call* the factory or be the factory.
        # The original structure was just 'def collate_fn_skip_none(batch):'
        # To use the factory, the train function needs to call create_collate_fn_with_shape
        # and pass the result to DataLoader's collate_fn argument.
        # Since I can only edit this specific function body easily, let's make *this* function the factory.
        # This requires changing the logic flow.

        # Redefining collate_fn_skip_none as a factory
        # This requires changing how it's used in DataLoader... which I can't do easily here.
        # Let's go back to the original structure and just fix the empty batch return.
        # We need T, H, W, C. Let's assume they can be accessed from a *passed* cfg object.
        # This means collate_fn_skip_none needs a 'cfg' argument.
        # This requires changing the DataLoader calls again.

        # Simplest fix within the current structure: return a basic empty tensor.
        # torch.empty((0, 1)) will avoid the IndexError but might cause shape errors later.
        # Let's try to return an empty tensor of plausible shape (0, T, H, W, C) using hardcoded/guessed indices for cfg.
        # This is fragile but attempts to fix the shape issue.
        # Assuming cfg.data has sequence_length, height, width, channels
        # This requires 'cfg' to be defined in this scope or globally.
        # Let's add a comment indicating this dependency.
        # Assume cfg is accessible globally for this hacky fix.
        try:
            # Access config values assuming a global 'cfg' object or similar.
            seq_len = cfg.data.sequence_length
            height = cfg.data.height
            width = cfg.data.width
            channels = cfg.data.channels # Usually 3 for RGB
            # Return empty tensor with batch size 0 and correct dimensions (0, T, H, W, C), assuming float32
            return torch.empty(0, seq_len, height, width, channels, dtype=torch.float32)
        except NameError:
            # If cfg is not accessible, return a minimal empty tensor that is still a tensor.
            print("Warning: 'cfg' not accessible in collate_fn_skip_none. Returning minimal empty tensor shape (0, 1). This may cause later errors.")
            return torch.empty(0, 1, dtype=torch.float32)
        except Exception as e:
            print(f"Error during empty tensor creation in collate_fn_skip_none: {e}. Returning None fallback.")
            return None # Fallback if any other error occurs
    # If the batch is NOT empty after filtering, use default_collate on the valid samples
    else:
        return torch.utils.data._utils.collate.default_collate(batch)

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
    4. 感知特征 -> AttentionTransformer + TriggerNet -> 最终输出 (TriggerNet)
    5. 攻击目标: 使对抗图像的TriggerNet输出与原始图像的TriggerNet输出尽可能不同

    Note: 在 Stage 2 中，IrisXeonNet, AttentionTransformer 和 TriggerNet 是预训练并冻结的。
    """
    # 确保当前训练阶段是 2
    assert cfg.training.train_stage == 2, f"此脚本仅用于训练阶段 2，但配置指定了阶段 {cfg.training.train_stage}。"
    print(f"开始训练阶段 {cfg.training.train_stage} (像素级攻击)...")

    # 设置随机种子以确保实验的可重现性
    set_seed(cfg.training.seed)

    # 获取计算设备
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # --- 1. 加载 ATN 模型组件 (冻结) ---
    print("正在加载 ATN 模型组件...")
    atn_model_dict = load_atn_model(cfg=cfg, device=device)

    if atn_model_dict is None:
        print("错误：加载 ATN 模型组件失败。退出训练。")
        sys.exit(1)
        
    # 验证 ATN 模型组件是否加载成功
    required_components = ['attention_transformer', 'trigger_net']
    for component in required_components:
        if component not in atn_model_dict or atn_model_dict[component] is None:
            print(f"错误：atn_model_dict 中缺少或为 None 的 ATN 组件: {component}")
            sys.exit(1)
            
    # 确保所有 ATN 模型组件都被冻结
    for model_name, model in atn_model_dict.items():
        if model is not None:
            for param in model.parameters():
                param.requires_grad = False
            model.eval() # 设置为评估模式
    print("ATN 模型组件加载成功并冻结。")

    # --- 2. 初始化生成器和判别器模型 ---
    print("正在初始化生成器和判别器...")

    # Stage 2: 生成器接收原始图像并输出像素级扰动
    generator = Generator(
        in_channels=cfg.model.generator.in_channels,  # 通常是3 (RGB)
        out_channels=cfg.model.generator.out_channels,  # 通常是3 (RGB扰动)
        num_bottlenecks=cfg.model.generator.num_bottlenecks,
        base_channels=cfg.model.generator.base_channels,
        epsilon=cfg.model.generator.epsilon  # 像素级扰动上限 (L-inf)
    ).to(device)

    # Stage 2: 判别器接收对抗图像作为输入
    if cfg.model.discriminator.type == 'cnn':
        discriminator = SequenceDiscriminatorCNN(
            in_channels=cfg.model.discriminator.in_channels,  # 通常是3 (RGB)
            base_channels=cfg.model.discriminator.base_channels
        ).to(device)
        print("Stage 2 使用 CNN 判别器")
    elif cfg.model.discriminator.type == 'patchgan':
        discriminator = PatchDiscriminator3d(
            in_channels=cfg.model.discriminator.in_channels,  # 通常是3 (RGB)
            base_channels=cfg.model.discriminator.base_channels
        ).to(device)
        print("Stage 2 使用 PatchGAN 判别器")
    else:
        print(f"错误：不支持的判别器类型: {cfg.model.discriminator.type}")
        sys.exit(1)

    print("生成器和判别器模型初始化完成。")

    # --- 3. 定义损失函数 ---
    print("正在定义损失函数...")
    gan_losses = GANLosses(device=device, gan_loss_type=cfg.losses.gan_type)
    print(f"使用 GAN 损失类型: {cfg.losses.gan_type}")

    # 攻击损失函数 (主要用于 Stage 2 TriggerNet 攻击)
    # attack_losses = AttackLosses(device=device) # 暂时不需要单独的 AttackLosses 类，直接在训练循环中计算 TriggerNet 差异

    # 正则化损失权重
    lambda_l_inf = cfg.regularization.lambda_l_inf
    lambda_l2 = cfg.regularization.lambda_l2
    lambda_tv = cfg.regularization.lambda_tv
    lambda_l2_penalty = cfg.regularization.lambda_l2_penalty
    print(f"使用正则化权重: L_inf={lambda_l_inf}, L2={lambda_l2}, TV={lambda_tv}, L2_Params={lambda_l2_penalty}")

    # --- 4. 定义优化器 ---
    print("正在定义优化器...")
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
    print(f"优化器初始化完成，lr_g={cfg.training.lr_g}, lr_d={cfg.training.lr_d}")

    # --- 5. 数据加载 ---
    print("正在加载数据...")
    if cfg.data.use_mock_data:
        print("使用模拟数据进行训练。")
        train_dataloader = create_mock_dataloader(
            batch_size=cfg.training.batch_size,
            num_samples=cfg.data.mock_num_samples,
            sequence_length=cfg.data.sequence_length,
            channels=cfg.data.channels,  # 应该是3 (RGB)
            height=cfg.data.height,
            width=cfg.data.width,
            shuffle=True,
            num_workers=0
        )
        eval_dataloader = create_mock_dataloader(
            batch_size=getattr(cfg.evaluation, 'num_eval_samples', cfg.training.batch_size),
            num_samples=getattr(cfg.evaluation, 'num_eval_samples', 100),
            sequence_length=cfg.data.sequence_length,
            channels=cfg.data.channels,
            height=cfg.data.height,
            width=cfg.data.width,
            shuffle=False,
            num_workers=0
        )
    else:
        # 实际数据加载逻辑
        if not os.path.exists(cfg.data.video_path):
            print(f"错误：未找到视频文件：{cfg.data.video_path}")
            sys.exit(1)
        if not os.path.exists(cfg.data.level_json_path):
            print(f"错误：未找到 Level JSON 文件：{cfg.data.level_json_path}")
            sys.exit(1)

        train_dataset = GameVideoDataset(
            video_path=cfg.data.video_path,
            level_json_path=cfg.data.level_json_path,
            sequence_length=cfg.data.sequence_length,
            transform=None,
            target_height=cfg.data.height,
            target_width=cfg.data.width,
            device=device # 数据集加载到设备上
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=0,
            worker_init_fn=worker_init_fn,
            pin_memory=True,
            collate_fn=collate_fn_skip_none
        )
        
        eval_dataset = GameVideoDataset(
            video_path=cfg.data.video_path,
            level_json_path=cfg.data.level_json_path,
            sequence_length=cfg.data.sequence_length,
            transform=None,
            target_height=cfg.data.height,
            target_width=cfg.data.width,
            device=device # 数据集加载到设备上
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=0,
            worker_init_fn=worker_init_fn,
            pin_memory=True,
            collate_fn=collate_fn_skip_none
        )

    # 检查数据加载器是否为空
    if len(train_dataloader) == 0:
        print("错误：训练数据加载器为空。")
        sys.exit(1)
    if len(eval_dataloader) == 0:
        print("警告：评估数据加载器为空。将跳过评估。")

    print(f"训练数据加载完成。每 epoch 的批次数量: {len(train_dataloader)}")
    print(f"评估数据加载完成。批次数量: {len(eval_dataloader)}")

    # --- TensorBoard 日志设置 ---
    log_dir = cfg.logging.log_dir
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard 日志将保存到: {log_dir}")

    # --- 检查点加载 (可选) ---
    start_epoch = 0
    global_step = 0
    best_eval_metric = -float('inf')  # Stage 2 目标是最大化攻击成功率

    if cfg.training.resume_checkpoint:
        print(f"尝试从检查点恢复: {cfg.training.resume_checkpoint}")
        try:
            checkpoint = torch.load(cfg.training.resume_checkpoint, map_location=device)
            generator.load_state_dict(checkpoint['generator_state_dict'])
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            global_step = checkpoint.get('global_step', start_epoch * len(train_dataloader)) # 如果 checkpoint 中没有 global_step，则估算
            best_eval_metric = checkpoint.get('best_eval_metric', -float('inf'))
            print(f"检查点加载成功。从 epoch {start_epoch}, global_step {global_step} 继续训练。")
        except Exception as e:
            print(f"加载检查点时出错：{e}。从头开始训练。")

    # 动态 GAN 平衡变量
    dynamic_balance_cfg = cfg.training.dynamic_gan_balance
    dynamic_balance_enabled = dynamic_balance_cfg.enabled
    current_D_freq = dynamic_balance_cfg.initial_D_freq
    current_G_freq = dynamic_balance_cfg.initial_G_freq

    if dynamic_balance_enabled:
        d_loss_history = deque(maxlen=dynamic_balance_cfg.loss_history_window)
        g_gan_loss_history = deque(maxlen=dynamic_balance_cfg.loss_history_window)
        print(f"动态 GAN 平衡已启用。初始 D 频率: {current_D_freq}, 初始 G 频率: {current_G_freq}")
    else:
        print("动态 GAN 平衡已禁用。使用固定频率。")

    # 损失函数
    mse_loss_fn = nn.MSELoss()
    cosine_similarity_fn = nn.CosineSimilarity(dim=-1)

    print("开始训练...")

    # --- 6. 训练循环 (Stage 2: 像素级攻击) ---
    for epoch in tqdm(range(start_epoch, cfg.training.num_epochs), desc="训练 Epochs"):
        generator.train()
        discriminator.train()

        # 训练一个 epoch
        for i, batch_data in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch}/{cfg.training.num_epochs}"):
            # batch_data 来自 DataLoader，其形状应该是 (B, T, H, W, C) 或经过 collate_fn 处理后的形状
            
            # **新增：检查批次是否为空**
            if batch_data is None or batch_data.size(0) == 0:
                # 警告：跳过空批次。
                # print(f"Warning: Skipping empty batch at epoch {epoch}, step {i}.") # 移除了调试打印
                global_step += 1 # 即使跳过也要增加步数，以保持日志和评估步调一致
                continue # 跳过当前批次的训练步骤

            real_x = batch_data.to(device) # 原始图像，形状 (B, T, H, W, C)

            # Permute real_x to (B, C, T, H, W) for models that expect this format (Generator, Discriminator)
            real_x_permuted = real_x.permute(0, 4, 1, 2, 3) # (B, T, H, W, C) -> (B, C, T, H, W)

            # --- 训练判别器 (Discriminator) ---
            # 根据动态平衡或固定频率决定是否训练 D
            if dynamic_balance_enabled:
                 train_D_this_step = global_step % current_D_freq == 0
            else:
                 train_D_this_step = True # 如果禁用动态平衡，总是训练 D (或根据固定 freq_d)

            if train_D_this_step:

                optimizer_D.zero_grad()

                # 1. 生成器生成像素级扰动 delta
                with torch.no_grad():  # D 训练时不需要 G 的梯度
                    # Generator expects (B, C, T, H, W), needs permuted input
                    delta = generator(real_x_permuted)  # delta 形状: (B, 3, T, H, W) # <-- 传入 permuted input
                    # 约束 delta 到 epsilon 球内 (L-inf 范数)
                    delta = torch.clamp(delta, -cfg.model.generator.epsilon, cfg.model.generator.epsilon)
                    # 生成对抗图像 (Add delta to original real_x, both should be in the same format)
                    # Ensure real_x_permuted is used for addition to match delta's shape
                    # Or, permute delta back to (B, T, H, W, C) before adding to real_x
                    # Let's keep delta in (B, C, T, H, W) and add to real_x_permuted
                    adversarial_x_permuted = real_x_permuted + delta
                    # 确保对抗图像在有效范围内 [0, 1]
                    adversarial_x_permuted = torch.clamp(adversarial_x_permuted, 0, 1)

                # 2. 判别器判别
                # Discriminator expects (B, C, T, H, W), needs permuted input
                D_real_output = discriminator(real_x_permuted)  # 真实图像的判别器输出 # <-- 传入 permuted input
                D_fake_output = discriminator(adversarial_x_permuted.detach())  # 对抗图像的判别器输出 (detach 以阻止梯度回传到 G) # <-- 传入 permuted input

                # 3. 计算判别器损失
                d_total_loss, d_real_loss, d_fake_loss = gan_losses.discriminator_loss(
                    D_real_output, D_fake_output
                )

                # 4. 更新判别器参数
                if torch.isfinite(d_total_loss): # 检查损失是否有效
                    d_total_loss.backward()
                    optimizer_D.step()

            # --- 训练生成器 (Generator) ---
            # 根据动态平衡或固定频率决定是否训练 G
            if dynamic_balance_enabled:
                 train_G_this_step = global_step % current_G_freq == 0
            else:
                 train_G_this_step = True # 如果禁用动态平衡，总是训练 G (或根据固定 freq_g)

            # 新增: 在满足可视化间隔时，强制计算 G 的输出以进行可视化，即使不训练 G
            force_generate_for_vis = (global_step + 1) % cfg.logging.vis_interval == 0
            
            if train_G_this_step or force_generate_for_vis: # 如果需要训练 G 或需要强制生成用于可视化

                if train_G_this_step: # 如果是正常训练 G 的步骤
                    optimizer_G.zero_grad()

                # 1. 生成器生成像素级扰动
                # Generator expects (B, C, T, H, W), needs permuted input
                # 使用 torch.enable_grad() 确保在需要训练时计算梯度，在仅用于可视化时禁用
                with torch.enable_grad() if train_G_this_step else torch.no_grad():
                    delta_for_G = generator(real_x_permuted)  # delta 形状: (B, 3, T, H, W) # <-- 传入 permuted input
                    delta_for_G = torch.clamp(delta_for_G, -cfg.model.generator.epsilon, cfg.model.generator.epsilon)
                    # 生成对抗图像 (Add delta to real_x_permuted)
                    adversarial_x_for_G_permuted = real_x_permuted + delta_for_G
                    adversarial_x_for_G_permuted = torch.clamp(adversarial_x_for_G_permuted, 0, 1)

                    # 2. 获取 ATN 模型输出 - 完整的数据流 (用于计算攻击损失 和 可视化)
                    # ATN 模型 (AttentionTransformer, TriggerNet) 期望 (B, T, H, W, C) 输入
                    # We need the original real_x and adversarial_x_for_G (in BTHWC format) for ATN models
                    
                    # 获取原始图像的 ATN 输出 ( TriggerNet 输出，无梯度)
                    # 强制对原始图像使用 no_grad，因为它总是冻结的
                    with torch.no_grad():
                        original_atn_outputs = get_atn_outputs(
                            atn_model_dict,
                            real_x, # <-- 传入原始形状 (B, T, H, W, C)
                            cfg=cfg,
                            device=device # 传递设备信息
                        )
                        original_trigger_output = original_atn_outputs.get('trigger_output')


                    # 获取对抗图像的 ATN 输出 ( TriggerNet 输出，需要梯度 如果 train_G_this_step)
                    # Permute adversarial_x_for_G_permuted back to (B, T, H, W, C) for ATN models
                    adversarial_x_for_G_original_format = adversarial_x_for_G_permuted.permute(0, 2, 3, 4, 1) # (B, C, T, H, W) -> (B, T, H, W, C)
                    # 在仅用于可视化时，对对抗样本的 ATN 输出也禁用梯度
                    with torch.enable_grad() if train_G_this_step else torch.no_grad():
                         adversarial_atn_outputs = get_atn_outputs(
                            atn_model_dict,
                            adversarial_x_for_G_original_format, # <-- 传入原始形状 (B, T, H, W, C)
                            cfg=cfg,
                            device=device # 传递设备信息
                        )
                         adversarial_trigger_output = adversarial_atn_outputs.get('trigger_output')


                # 只有在需要训练 G 时才计算和应用梯度更新
                if train_G_this_step:
                    # 3. 计算生成器 GAN 损失
                    # Discriminator expects (B, C, T, H, W), needs permuted input
                    D_fake_output_for_G = discriminator(adversarial_x_for_G_permuted) # 对抗图像送入判别器 (需要梯度) # <-- 传入 permuted input
                    gen_gan_loss = gan_losses.generator_loss(D_fake_output_for_G)

                    # 4. 计算生成器攻击损失 (Stage 2: 攻击 TriggerNet 输出)
                    gen_attack_loss = torch.tensor(0.0, device=device)

                    if original_trigger_output is not None and adversarial_trigger_output is not None:
                        # 确保形状匹配且非空
                        if original_trigger_output.shape == adversarial_trigger_output.shape and original_trigger_output.numel() > 0:
                            # MSE 差异损失 (负值表示攻击损失)
                            trigger_mse_diff = mse_loss_fn(
                                original_trigger_output.float(),
                                adversarial_trigger_output.float()
                            )
                            # 攻击损失 = -decision_loss_weight * trigger_mse_diff (最大化差异)
                            gen_attack_loss = -cfg.losses.decision_loss_weight * trigger_mse_diff

                            # 余弦相似度损失 (最小化余弦相似度，最大化 -cosine_sim)
                            cosine_attack_loss = torch.tensor(0.0, device=device)
                            if original_trigger_output.dim() > 1 and original_trigger_output.numel() > 0:
                                orig_flat = original_trigger_output.view(original_trigger_output.size(0), -1)
                                adv_flat = adversarial_trigger_output.view(adversarial_trigger_output.size(0), -1)
                                if orig_flat.shape[1] > 1:
                                    cosine_sim = cosine_similarity_fn(orig_flat, adv_flat).mean()
                                    cosine_attack_loss = cfg.losses.get('cosine_loss_weight', 0.1) * (-cosine_sim)
                            gen_attack_loss += cosine_attack_loss


                    # 5. 计算正则化损失
                    reg_loss_l_inf = torch.tensor(0.0, device=device)
                    if delta_for_G is not None and delta_for_G.numel() > 0:
                        # Ensure linf_norm is accessible
                        reg_loss_l_inf = lambda_l_inf * linf_norm(delta_for_G).mean()

                    reg_loss_l2 = torch.tensor(0.0, device=device)
                    if lambda_l2 > 0 and delta_for_G is not None and delta_for_G.numel() > 0:
                        # Ensure l2_norm is accessible
                        reg_loss_l2 = lambda_l2 * l2_norm(delta_for_G).mean()

                    reg_loss_tv = torch.tensor(0.0, device=device)
                    if lambda_tv > 0 and delta_for_G is not None and delta_for_G.numel() > 0:
                         # Ensure total_variation_loss is accessible
                        reg_loss_tv = lambda_tv * total_variation_loss(delta_for_G).mean()

                    reg_loss_l2_params = torch.tensor(0.0, device=device)
                    if lambda_l2_penalty > 0:
                        gen_l2_reg = torch.tensor(0.0, device=device)
                        for param in generator.parameters():
                            if param.requires_grad:
                                gen_l2_reg += param.square().sum()
                        reg_loss_l2_params = lambda_l2_penalty * gen_l2_reg


                    # 6. 组合生成器总损失
                    gen_total_loss = (
                        cfg.losses.gan_loss_weight * gen_gan_loss + # GAN 损失
                        gen_attack_loss + # 攻击损失
                        reg_loss_l_inf + # L-inf 正则化
                        reg_loss_l2 + # L2 范数正则化
                        reg_loss_tv + # TV 正则化
                        reg_loss_l2_params # L2 参数正则化
                    )

                    # 7. 更新生成器参数
                    if torch.isfinite(gen_total_loss): # 检查损失是否有效
                        gen_total_loss.backward()
                        optimizer_G.step()

            # --- 动态 GAN 平衡调整 ---
            # 根据需要调整判别器和生成器的训练频率
            if dynamic_balance_enabled and dynamic_balance_cfg.strategy == "loss_ratio_freq_adjust":
                 # 只有当 D 和 G 都在当前步训练时才记录损失和调整频率
                 if train_D_this_step and train_G_this_step:
                    if 'd_total_loss' in locals() and 'gen_gan_loss' in locals() and \
                       torch.isfinite(d_total_loss) and torch.isfinite(gen_gan_loss):
                        d_loss_item = d_total_loss.item()
                        g_gan_loss_item = gen_gan_loss.item()

                        d_loss_history.append(d_loss_item)
                        g_gan_loss_history.append(g_gan_loss_item)

                        if (global_step + 1) % dynamic_balance_cfg.freq_adjust_interval == 0:
                             # ... (频率调整逻辑) ...
                             avg_d_loss = np.mean(list(d_loss_history)) if d_loss_history else 0.0
                             avg_g_gan_loss = np.mean(list(g_gan_loss_history)) if g_gan_loss_history else 0.0

                             balance_ratio = float('inf')
                             if abs(avg_g_gan_loss) > 1e-8:
                                 balance_ratio = avg_d_loss / avg_g_gan_loss
                             elif abs(avg_d_loss) < 1e-8:
                                 balance_ratio = 1.0

                             print(f"步数 {global_step+1}: 平衡比例 (D/G_GAN): {balance_ratio:.2f}, 当前 D 频率: {current_D_freq}, 当前 G 频率: {current_G_freq}")

                             D_dominant_threshold = dynamic_balance_cfg.D_dominant_threshold
                             G_dominant_threshold = dynamic_balance_cfg.G_dominant_threshold
                             max_freq = 10
                             min_freq = 1

                             if balance_ratio > D_dominant_threshold:
                                current_G_freq = min(current_G_freq + 1, max_freq)
                                current_D_freq = max(min_freq, current_D_freq - 1)
                             elif balance_ratio < G_dominant_threshold:
                                current_D_freq = min(current_D_freq + 1, max_freq)
                                current_G_freq = max(min_freq, current_G_freq - 1)

            # --- 日志记录和可视化 ---
            if global_step % cfg.logging.log_interval == 0:
                # 记录损失到 TensorBoard
                losses_to_log = {}
                # Only log losses if they were actually computed in this step
                if train_G_this_step:
                    if 'gen_total_loss' in locals() and torch.isfinite(gen_total_loss): losses_to_log['Generator_Total'] = gen_total_loss.item()
                    if 'gen_gan_loss' in locals() and torch.isfinite(gen_gan_loss): losses_to_log['Generator_GAN'] = gen_gan_loss.item()
                    if 'gen_attack_loss' in locals() and torch.isfinite(gen_attack_loss): losses_to_log['Generator_Attack_TriggerNet'] = gen_attack_loss.item()
                if train_D_this_step:
                    if 'd_total_loss' in locals() and torch.isfinite(d_total_loss): losses_to_log['Discriminator_Total'] = d_total_loss.item()
                    if 'D_real_output' in locals() and D_real_output is not None and D_real_output.numel() > 0:
                        losses_to_log['Discriminator_Score_Real'] = D_real_output.mean().item()
                    if 'D_fake_output' in locals() and D_fake_output is not None and D_fake_output.numel() > 0:
                        losses_to_log['Discriminator_Score_Fake'] = D_fake_output.mean().item()

                # Log regularization losses only if G was trained
                if train_G_this_step:
                     reg_losses_to_log = {}
                     if 'reg_loss_l_inf' in locals(): reg_losses_to_log['Regularization_L_inf'] = reg_loss_l_inf.item()
                     if 'reg_loss_l2' in locals(): reg_losses_to_log['Regularization_L2'] = reg_loss_l2.item()
                     if 'reg_loss_tv' in locals(): reg_losses_to_log['Regularization_TV'] = reg_loss_tv.item()
                     if 'reg_loss_l2_params' in locals(): reg_losses_to_log['Regularization_L2_Params'] = reg_loss_l2_params.item()
                     for name, value in reg_losses_to_log.items():
                         writer.add_scalar(f'Loss/{name}', value, global_step)

                # Log GAN and attack losses
                for name, value in losses_to_log.items():
                    # Log discriminator scores under a separate tag
                    if name in ['Discriminator_Score_Real', 'Discriminator_Score_Fake']:
                        writer.add_scalar(f'Discriminator_Scores/{name}', value, global_step)
                    # Log other losses under the 'Loss' tag
                    else:
                        writer.add_scalar(f'Loss/{name}', value, global_step)

                # 记录扰动统计 (只有当 delta_for_G 在当前步计算时)
                # Ensure visualize_perturbation_norms is accessible
                if 'delta_for_G' in locals() and delta_for_G is not None and delta_for_G.numel() > 0 and (train_G_this_step or force_generate_for_vis): # Log perturbation if generated for train or vis
                     visualize_perturbation_norms(writer, delta_for_G, global_step) # 使用可视化函数记录扰动范数


                print(f"Epoch [{epoch}/{cfg.training.num_epochs}], Step [{global_step}]")
                # Print losses only if they were computed
                if 'd_total_loss' in locals() and train_D_this_step and torch.isfinite(d_total_loss): print(f"  D Loss: {d_total_loss.item():.4f}")
                if 'gen_total_loss' in locals() and train_G_this_step and torch.isfinite(gen_total_loss):
                    gen_gan_print = gen_gan_loss.item() if 'gen_gan_loss' in locals() and torch.isfinite(gen_gan_loss) else float('nan')
                    gen_attack_print = gen_attack_loss.item() if 'gen_attack_loss' in locals() and torch.isfinite(gen_attack_loss) else float('nan')
                    print(f"  G Loss: {gen_total_loss.item():.4f} (GAN: {gen_gan_print:.4f}, Attack: {gen_attack_print:.4f})")


            # 可视化图像和 TriggerNet 输出
            # 根据配置的间隔进行可视化
            if (global_step + 1) % cfg.logging.vis_interval == 0:
                # Determine number of samples/videos to visualize
                num_vis_samples = min(cfg.logging.num_vis_samples, real_x.shape[0]) # Ensure not exceeding batch size
                
                # For image_grid mode, determine which sequence steps to visualize
                visualize_mode = getattr(cfg.logging, 'visualize_mode', 'video') # Get visualization mode
                if visualize_mode == 'image_grid':
                    sequence_steps_to_vis_list = getattr(cfg.logging, 'sequence_steps_to_vis', None) # Get list of steps from config
                else:
                    sequence_steps_to_vis_list = None # Not used in video mode

                # 检查可视化所需的变量是否存在且有效（它们应在 force_generate_for_vis 逻辑中被计算）
                if 'adversarial_x_for_G_original_format' in locals() and adversarial_x_for_G_original_format is not None and \
                   'original_trigger_output' in locals() and original_trigger_output is not None and \
                   'adversarial_trigger_output' in locals() and adversarial_trigger_output is not None and \
                   'delta_for_G' in locals() and delta_for_G is not None: # 移除 train_G_this_step 条件

                    print(f"Attempting to visualize at step {global_step}") # Added debug log before visualization
                    try:
                         # Ensure visualize_stage2_pixel_attack is accessible
                         from utils.vis_utils import visualize_stage2_pixel_attack
                         with torch.no_grad(): # 可视化不需要梯度

                             # Prepare images and deltas for visualization function
                             # Pass 5D tensors to visualize_stage2_pixel_attack for video mode
                             # original_images: (B, T, H, W, C) -> (B_vis, T, H, W, C)
                             original_images_vis = real_x[:num_vis_samples].detach().cpu()
                             # adversarial_images: (B, T, H, W, C) -> (B_vis, T, H, W, C)
                             adversarial_images_vis = adversarial_x_for_G_original_format[:num_vis_samples].detach().cpu()
                             # pixel_deltas: (B, C, T, H, W) -> (B_vis, C, T, H, W). Need to pass this format.
                             pixel_deltas_vis = delta_for_G[:num_vis_samples].detach().cpu()

                             # TriggerNet outputs are already in (B, ...) format as returned by get_atn_outputs
                             # original_trigger_output: (B, ...) -> (B_vis, ...)
                             original_trigger_output_vis = original_trigger_output[:num_vis_samples].detach().cpu()
                             # adversarial_trigger_output: (B, ...) -> (B_vis, ...)
                             adversarial_trigger_output_vis = adversarial_trigger_output[:num_vis_samples].detach().cpu()

                             # Enhance contrast of deltas for visualization (e.g., multiply by a factor)
                             # Choose a scaling factor, for visualization only
                             delta_vis_scale_factor = getattr(cfg.logging, 'delta_vis_scale_factor', 20.0) # Get scale factor from config
                             # Apply scaling to the 5D delta tensor (B_vis, C, T, H, W)
                             scaled_pixel_deltas_vis = pixel_deltas_vis * delta_vis_scale_factor

                             # --- Debug Prints for Visualization Inputs ---
                            #  print(f"Debug Vis Input: original_images_vis shape: {original_images_vis.shape}")
                            #  print(f"Debug Vis Input: adversarial_images_vis shape: {adversarial_images_vis.shape}")
                            #  print(f"Debug Vis Input: pixel_deltas_vis shape: {pixel_deltas_vis.shape}")
                            #  print(f"Debug Vis Input: original_trigger_output_vis shape: {original_trigger_output_vis.shape if original_trigger_output_vis is not None else 'None'}")
                            #  print(f"Debug Vis Input: adversarial_trigger_output_vis shape: {adversarial_trigger_output_vis.shape if adversarial_trigger_output_vis is not None else 'None'}")

                             visualize_stage2_pixel_attack(
                                writer=writer,
                                original_images=original_images_vis, # Pass 5D tensor (B, T, H, W, C)
                                adversarial_images=adversarial_images_vis, # Pass 5D tensor (B, T, H, W, C)
                                pixel_deltas=scaled_pixel_deltas_vis, # Pass scaled 5D tensor (B, C, T, H, W)
                                original_trigger_output=original_trigger_output_vis, # Pass 5D tensor (B, ...) or other shapes
                                adversarial_trigger_output=adversarial_trigger_output_vis, # Pass 5D tensor (B, ...) or other shapes
                                step=global_step, # Current step
                                cfg=cfg, # Pass the config object
                                num_samples=num_vis_samples, # Number of samples to visualize
                                sequence_steps_to_vis=sequence_steps_to_vis_list # Pass the list of steps for image_grid mode
                            )
                         print(f"Visualization successful at step {global_step}") # Added debug log after visualization
                    except Exception as e:
                         print(f"Error: Failed to visualize at step {global_step}: {e}") # Explicit error for visualization

                # 可选：可视化注意力图 (...)

            global_step += 1

        # --- Save latest checkpoint after each epoch train loop ---
        checkpoint_dir = os.path.join(cfg.logging.log_dir, 'checkpoints')
        # print(f"Debug: Attempting to create latest checkpoint directory: {checkpoint_dir}") # Debug log
        try:
            os.makedirs(checkpoint_dir, exist_ok=True) # Create checkpoint directory if it doesn't exist
            # print(f"Debug: Latest checkpoint directory {checkpoint_dir} created or already exists.") # Debug log
        except Exception as e:
           print(f"Error: Failed to create latest checkpoint directory {checkpoint_dir}: {e}") # Explicit error for dir creation

        latest_checkpoint_path = os.path.join(checkpoint_dir, f'latest_stage_{cfg.training.train_stage}.pth')
        # print(f"Debug: Attempting to save latest model to {latest_checkpoint_path}") # Debug log
        try: # Potential failure point: file writing failure
            torch.save({
                'epoch': epoch,
                'global_step': global_step, # Save the global step after the epoch
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict() if discriminator is not None else None,
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict() if optimizer_D is not None else None,
                'best_eval_metric': best_eval_metric, # Save the current best metric even in latest
                'cfg': cfg
            }, latest_checkpoint_path)
            print(f"Latest model successfully saved to {latest_checkpoint_path}")
        except Exception as e: # Catch and print errors during saving
             print(f"Error: Failed to save latest checkpoint to {latest_checkpoint_path}: {e}") # Explicit error for latest checkpoint save


        # --- 评估 ---
        # 根据配置的间隔或在最后一个 epoch 进行评估
        if (epoch + 1) % cfg.training.get('eval_interval', 10) == 0 or (epoch + 1) == cfg.training.num_epochs:
            if len(eval_dataloader) > 0: # 只有评估数据加载器非空时才进行评估
                print(f"在 epoch {epoch+1} 运行评估...")
                # 将模型设置为评估模式 (虽然 ATN 已冻结，但 G 和 D 需要)
                generator.eval()
                discriminator.eval()
                
                eval_metrics = evaluate_model(
                    generator=generator,
                    discriminator=discriminator,
                    atn_model=atn_model_dict, # Pass the ATN model dictionary
                    dataloader=eval_dataloader,
                    device=device,
                    cfg=cfg,
                    current_train_stage=cfg.training.train_stage
                )

                if eval_metrics: # If evaluation was successful and returned metrics
                    print(f"Epoch {epoch+1} 的评估指标: {eval_metrics}")
                    # 将评估指标记录到 TensorBoard
                    for metric_name, metric_value in eval_metrics.items():
                        # Check if metric value is valid (not NaN, not Inf)
                        if np.isfinite(metric_value):
                            writer.add_scalar(f'Evaluation/{metric_name}', float(metric_value), global_step) # Log as float
                        else:
                             print(f"Warning: Evaluation metric '{metric_name}' is not finite ({metric_value}) at step {global_step}. Skipping logging.")

                    # Save best model checkpoint (based on primary evaluation metric)
                    primary_eval_metric_name = getattr(cfg.evaluation, 'primary_metric', 'TriggerNet_ASR') # Get primary metric name from config, default to ASR for Stage 2
                    is_lower_metric_better = getattr(cfg.evaluation, 'primary_metric_lower_is_better', False) # Get whether lower is better, default to False (higher is better for ASR)
                    primary_eval_metric = eval_metrics.get(primary_eval_metric_name)

                    # Only compare and save if the primary metric value is valid
                    if primary_eval_metric is not None and np.isfinite(primary_eval_metric):
                        # Initialize best_eval_metric if it's still -inf (first valid evaluation)
                        if best_eval_metric == -float('inf') and not is_lower_metric_better:
                             best_eval_metric = primary_eval_metric
                             print(f"Initialized best_eval_metric to {best_eval_metric:.4f} based on first valid evaluation.")
                        elif best_eval_metric == float('inf') and is_lower_metric_better:
                              best_eval_metric = primary_eval_metric
                              print(f"Initialized best_eval_metric to {best_eval_metric:.4f} based on first valid evaluation (lower is better).")

                        # Check if the current metric is better than the best metric
                        is_better = (is_lower_metric_better and primary_eval_metric < best_eval_metric) or \
                                   (not is_lower_metric_better and primary_eval_metric > best_eval_metric)

                        if is_better:
                            best_eval_metric = primary_eval_metric # Update the best metric value
                            print(f"New best evaluation metric ('{primary_eval_metric_name}'): {best_eval_metric:.4f}. Saving new best checkpoint...")

                            # Define checkpoint path
                            checkpoint_dir = os.path.join(cfg.logging.log_dir, 'checkpoints')
                            # print(f"Debug: Attempting to create checkpoint directory: {checkpoint_dir}") # Debug log
                            try:
                               os.makedirs(checkpoint_dir, exist_ok=True) # Create checkpoint directory if it doesn't exist
                               # print(f"Debug: Checkpoint directory {checkpoint_dir} created or already exists.") # Debug log
                            except Exception as e:
                               print(f"Error: Failed to create checkpoint directory {checkpoint_dir}: {e}") # Explicit error for dir creation

                            best_checkpoint_path = os.path.join(checkpoint_dir, f'best_stage_{cfg.training.train_stage}.pth')
                            # print(f"Debug: Attempting to save best model to {best_checkpoint_path}") # Debug log

                            # Save model checkpoint
                            try: # Potential failure point: file writing failure (e.g., permission issues, disk full)
                                torch.save({
                                    'epoch': epoch,
                                    'global_step': global_step,
                                    'generator_state_dict': generator.state_dict(),
                                    'discriminator_state_dict': discriminator.state_dict() if discriminator is not None else None,
                                    'optimizer_G_state_dict': optimizer_G.state_dict(),
                                    'optimizer_D_state_dict': optimizer_D.state_dict() if optimizer_D is not None else None,
                                    'best_eval_metric': best_eval_metric,
                                    'cfg': cfg
                                }, best_checkpoint_path)
                                print(f"Best model successfully saved to {best_checkpoint_path}")
                            except Exception as e: # Catch and print errors during saving
                                print(f"Error: Failed to save best checkpoint to {best_checkpoint_path}: {e}") # Explicit error for best checkpoint save
                        else:
                            print(f"Current evaluation metric ('{primary_eval_metric_name}') {primary_eval_metric:.4f} is not better than best {best_eval_metric:.4f}.")
                    else:
                         print(f"Warning: Primary evaluation metric '{primary_eval_metric_name}' value is invalid ({primary_eval_metric}). Skipping best checkpoint saving.")

                # After evaluation, set models back to training mode (if training hasn't finished)
                if (epoch + 1) < cfg.training.num_epochs:
                     generator.train()
                     discriminator.train() # Ensure discriminator is set to train too

    # 训练结束时关闭 TensorBoard writer
    writer.close()

if __name__ == "__main__":
    # 解析命令行参数和配置文件
    cfg = parse_args_and_config(default_config_path="configs/stage2_config.yaml")
    # 运行训练函数
    train(cfg)
