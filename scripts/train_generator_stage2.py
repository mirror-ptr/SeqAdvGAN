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
    required_components = ['xeon_net', 'attention_transformer', 'trigger_net']
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
            device=device # 数据集加载到设备上
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=getattr(cfg.evaluation, 'num_eval_samples', cfg.training.batch_size),
            shuffle=False,
            num_workers=cfg.data.num_workers,
            worker_init_fn=worker_init_fn,
            pin_memory=True
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
            # batch_data 来自 DataLoader，其形状应该是 (B, C, T, H, W)
            real_x = batch_data.to(device) # 原始图像

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
                    delta = generator(real_x)  # delta 形状: (B, 3, T, H, W)
                    # 约束 delta 到 epsilon 球内 (L-inf 范数)
                    delta = torch.clamp(delta, -cfg.model.generator.epsilon, cfg.model.generator.epsilon)
                    # 生成对抗图像
                    adversarial_x = real_x + delta
                    # 确保对抗图像在有效范围内 [0, 1]
                    adversarial_x = torch.clamp(adversarial_x, 0, 1)

                # 2. 判别器判别
                D_real_output = discriminator(real_x)  # 真实图像的判别器输出
                D_fake_output = discriminator(adversarial_x.detach())  # 对抗图像的判别器输出 (detach 以阻止梯度回传到 G)

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

            if train_G_this_step:
                optimizer_G.zero_grad()

                # 1. 生成器生成像素级扰动
                delta_for_G = generator(real_x)  # delta 形状: (B, 3, T, H, W)
                delta_for_G = torch.clamp(delta_for_G, -cfg.model.generator.epsilon, cfg.model.generator.epsilon)
                # 生成对抗图像
                adversarial_x_for_G = real_x + delta_for_G
                adversarial_x_for_G = torch.clamp(adversarial_x_for_G, 0, 1)

                # 2. 获取 ATN 模型输出 - 完整的数据流 (用于计算攻击损失)
                # ATN 模型是冻结的，但对于对抗样本，我们需要计算梯度。
                # 对于原始图像，可以禁用梯度计算以节省内存和计算。
                
                # 获取原始图像的 ATN 输出 ( TriggerNet 输出，无梯度)
                with torch.no_grad():
                    original_atn_outputs = get_atn_outputs(
                        atn_model_dict,
                        real_x, # 输入原始图像
                        cfg=cfg,
                        device=device # 传递设备信息
                    )

                # 获取对抗图像的 ATN 输出 ( TriggerNet 输出，需要梯度)
                adversarial_atn_outputs = get_atn_outputs(
                    atn_model_dict,
                    adversarial_x_for_G, # 输入对抗图像
                    cfg=cfg,
                    device=device # 传递设备信息
                )

                # 3. 计算生成器 GAN 损失
                D_fake_output_for_G = discriminator(adversarial_x_for_G) # 对抗图像送入判别器 (需要梯度)
                gen_gan_loss = gan_losses.generator_loss(D_fake_output_for_G)

                # 4. 计算生成器攻击损失 (Stage 2: 攻击 TriggerNet 输出)
                gen_attack_loss = torch.tensor(0.0, device=device)
                
                # 提取 TriggerNet 输出
                original_trigger_output = original_atn_outputs.get('trigger_output')
                adversarial_trigger_output = adversarial_atn_outputs.get('trigger_output')

                if original_trigger_output is not None and adversarial_trigger_output is not None:
                    # 确保形状匹配且非空
                    if original_trigger_output.shape == adversarial_trigger_output.shape and original_trigger_output.numel() > 0:
                        # 攻击目标：最大化原始输出与对抗输出之间的差异
                        # 使用 MSE 损失衡量差异，然后取负值以最大化差异
                        # 或者使用余弦相似度，最小化余弦相似度（最大化 -cosine_sim）

                        # MSE 差异损失 (负值表示攻击损失)
                        trigger_mse_diff = mse_loss_fn(
                            original_trigger_output.float(), 
                            adversarial_trigger_output.float()
                        )
                        # 攻击损失 = -decision_loss_weight * trigger_mse_diff (最大化差异)
                        attack_mse_loss = -cfg.losses.decision_loss_weight * trigger_mse_diff

                        # 余弦相似度损失 (最小化余弦相似度，最大化 -cosine_sim)
                        cosine_attack_loss = torch.tensor(0.0, device=device)
                        # 只有当输出维度 > 1 时计算余弦相似度
                        if original_trigger_output.dim() > 1 and original_trigger_output.numel() > 0:
                            # 将输出展平以计算余弦相似度
                            orig_flat = original_trigger_output.view(original_trigger_output.size(0), -1)
                            adv_flat = adversarial_trigger_output.view(adversarial_trigger_output.size(0), -1)
                            # 检查展平后的维度是否大于 1 以计算余弦相似度
                            if orig_flat.shape[1] > 1:
                                cosine_sim = cosine_similarity_fn(orig_flat, adv_flat).mean()
                                # 攻击目标：最小化余弦相似度（即最大化-cosine_sim）
                                # 余弦攻击损失 = cosine_loss_weight * (-cosine_sim)
                                cosine_attack_loss = cfg.losses.get('cosine_loss_weight', 0.1) * (-cosine_sim)
                            # else: print("Warning: TriggerNet output flattened dim is 1, skipping cosine similarity.") # 移除了调试打印
                        # else: print("Warning: TriggerNet output dim <= 1, skipping cosine similarity.") # 移除了调试打印

                        # 组合攻击损失
                        gen_attack_loss = attack_mse_loss + cosine_attack_loss

                    # else: print("Warning: TriggerNet outputs shape mismatch or empty.") # 移除了调试打印
                # else: print("Warning: TriggerNet outputs are None.") # 移除了调试打印

                # 5. 计算正则化损失
                reg_loss_l_inf = torch.tensor(0.0, device=device)
                # 只有 delta_for_G 有效时才计算
                if delta_for_G is not None and delta_for_G.numel() > 0:
                    reg_loss_l_inf = lambda_l_inf * linf_norm(delta_for_G).mean()

                reg_loss_l2 = torch.tensor(0.0, device=device)
                # L2 范数正则化 (对扰动本身)
                # 只有 delta_for_G 有效时才计算
                if lambda_l2 > 0 and delta_for_G is not None and delta_for_G.numel() > 0:
                    reg_loss_l2 = lambda_l2 * l2_norm(delta_for_G).mean() # 使用 l2_norm 函数计算 L2 范数

                reg_loss_tv = torch.tensor(0.0, device=device)
                # 只有 delta_for_G 有效时才计算
                if lambda_tv > 0 and delta_for_G is not None and delta_for_G.numel() > 0:
                    reg_loss_tv = lambda_tv * total_variation_loss(delta_for_G).mean()

                reg_loss_l2_params = torch.tensor(0.0, device=device)
                # L2 参数正则化 (对生成器参数)
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
                # else: print(f"Warning: 生成器损失在步数 {global_step} 无效") # 移除了调试打印

            # --- 动态 GAN 平衡调整 ---
            # 根据需要调整判别器和生成器的训练频率
            if dynamic_balance_enabled and dynamic_balance_cfg.strategy == "loss_ratio_freq_adjust":
                # 只有当 D 总损失和 G GAN 损失都有效时才进行调整
                if 'd_total_loss' in locals() and 'gen_gan_loss' in locals() and \
                   torch.isfinite(d_total_loss) and torch.isfinite(gen_gan_loss):
                    d_loss_item = d_total_loss.item()
                    g_gan_loss_item = gen_gan_loss.item()
                    
                    # 仅当当前步训练了相应的模型时才记录损失
                    if train_D_this_step:
                        d_loss_history.append(d_loss_item)
                    if train_G_this_step:
                         g_gan_loss_history.append(g_gan_loss_item)

                    if (global_step + 1) % dynamic_balance_cfg.freq_adjust_interval == 0:
                        # 计算损失历史的平均值
                        avg_d_loss = np.mean(list(d_loss_history)) if d_loss_history else 0.0
                        avg_g_gan_loss = np.mean(list(g_gan_loss_history)) if g_gan_loss_history else 0.0

                        balance_ratio = float('inf')
                        # 避免除以零或接近零的值
                        if abs(avg_g_gan_loss) > 1e-8:
                            balance_ratio = avg_d_loss / avg_g_gan_loss
                        elif abs(avg_d_loss) < 1e-8: # 如果 D 和 G 都接近零
                            balance_ratio = 1.0

                        print(f"步数 {global_step+1}: 平衡比例 (D/G_GAN): {balance_ratio:.2f}, 当前 D 频率: {current_D_freq}, 当前 G 频率: {current_G_freq}")

                        # 调整训练频率 (确保频率至少为 1)
                        D_dominant_threshold = dynamic_balance_cfg.D_dominant_threshold
                        G_dominant_threshold = dynamic_balance_cfg.G_dominant_threshold
                        max_freq = 10 # 频率上限，防止某个模型训练过少
                        min_freq = 1 # 频率下限
                        
                        if balance_ratio > D_dominant_threshold: # D 损失相对于 G GAN 损失较高，说明 D 太强或 G 太弱，增加 G 训练频率，减少 D 训练频率
                            current_G_freq = min(current_G_freq + 1, max_freq)
                            current_D_freq = max(min_freq, current_D_freq - 1)
                        elif balance_ratio < G_dominant_threshold: # D 损失相对于 G GAN 损失较低，说明 D 太弱或 G 太强，增加 D 训练频率，减少 G 训练频率
                            current_D_freq = min(current_D_freq + 1, max_freq)
                            current_G_freq = max(min_freq, current_G_freq - 1)
                        # 如果比例在阈值之间，保持当前频率
                        
                        # 可选：清空损失历史，使用新的频率开始累积
                        # d_loss_history.clear()
                        # g_gan_loss_history.clear()

            # --- 日志记录和可视化 ---
            if global_step % cfg.logging.log_interval == 0:
                # 记录损失到 TensorBoard
                losses_to_log = {}
                if 'gen_total_loss' in locals() and torch.isfinite(gen_total_loss): losses_to_log['Generator_Total'] = gen_total_loss.item()
                if 'gen_gan_loss' in locals() and torch.isfinite(gen_gan_loss): losses_to_log['Generator_GAN'] = gen_gan_loss.item()
                if 'gen_attack_loss' in locals() and torch.isfinite(gen_attack_loss): losses_to_log['Generator_Attack_TriggerNet'] = gen_attack_loss.item()
                if 'd_total_loss' in locals() and torch.isfinite(d_total_loss): losses_to_log['Discriminator_Total'] = d_total_loss.item()
                
                visualize_training_losses(writer, losses_to_log, global_step) # 使用可视化函数记录损失

                # 记录正则化损失
                reg_losses_to_log = {}
                if 'reg_loss_l_inf' in locals(): reg_losses_to_log['Regularization_L_inf'] = reg_loss_l_inf.item()
                if 'reg_loss_l2' in locals(): reg_losses_to_log['Regularization_L2'] = reg_loss_l2.item()
                if 'reg_loss_tv' in locals(): reg_losses_to_log['Regularization_TV'] = reg_loss_tv.item()
                if 'reg_loss_l2_params' in locals(): reg_losses_to_log['Regularization_L2_Params'] = reg_loss_l2_params.item()
                
                for name, value in reg_losses_to_log.items():
                    writer.add_scalar(f'Loss/{name}', value, global_step)

                # 记录扰动统计
                # 只有当 delta_for_G 在当前步计算时才记录 (即 train_G_this_step)
                if 'delta_for_G' in locals() and delta_for_G is not None and delta_for_G.numel() > 0 and train_G_this_step:
                     visualize_perturbation_norms(writer, delta_for_G, global_step) # 使用可视化函数记录扰动范数

                print(f"Epoch [{epoch}/{cfg.training.num_epochs}], Step [{global_step}]")
                # 使用 locals() 检查变量是否存在并打印
                if 'd_total_loss' in locals() and torch.isfinite(d_total_loss): print(f"  D Loss: {d_total_loss.item():.4f}")
                if 'gen_total_loss' in locals() and torch.isfinite(gen_total_loss): 
                    gen_gan_print = gen_gan_loss.item() if 'gen_gan_loss' in locals() and torch.isfinite(gen_gan_loss) else float('nan')
                    gen_attack_print = gen_attack_loss.item() if 'gen_attack_loss' in locals() and torch.isfinite(gen_attack_loss) else float('nan')
                    print(f"  G Loss: {gen_total_loss.item():.4f} (GAN: {gen_gan_print:.4f}, Attack: {gen_attack_print:.4f})")

            # 可视化图像和 TriggerNet 输出
            # 根据配置的间隔进行可视化
            if (global_step + 1) % cfg.logging.vis_interval == 0:
                num_vis_samples = min(cfg.logging.num_vis_samples, real_x.shape[0]) # 确保不超出批次大小
                sequence_step_to_vis = cfg.logging.sequence_step_to_vis

                # 只有当对抗样本和 TriggerNet 输出在当前步计算时才进行可视化 (即 train_G_this_step)
                if 'adversarial_x_for_G' in locals() and \
                   'original_trigger_output' in locals() and original_trigger_output is not None and \
                   'adversarial_trigger_output' in locals() and adversarial_trigger_output is not None and \
                   'delta_for_G' in locals() and delta_for_G is not None and train_G_this_step:

                    with torch.no_grad(): # 可视化不需要梯度
                        visualize_stage2_pixel_attack(
                            writer=writer,
                            original_images=real_x[:num_vis_samples].detach().cpu(), # 原始图像
                            adversarial_images=adversarial_x_for_G[:num_vis_samples].detach().cpu(), # 对抗图像
                            pixel_deltas=delta_for_G[:num_vis_samples].detach().cpu(), # 像素扰动
                            original_trigger_output=original_trigger_output[:num_vis_samples].detach().cpu(), # 原始 TriggerNet 输出
                            adversarial_trigger_output=adversarial_trigger_output[:num_vis_samples].detach().cpu(), # 对抗 TriggerNet 输出
                            step=global_step, # 当前步数
                            num_samples=num_vis_samples, # 可视化样本数
                            sequence_step_to_vis=sequence_step_to_vis # 要可视化的序列步骤
                        )
                    
                # 可选：可视化注意力图 (如果 attention_features 可用且需要可视化)
                # 这取决于 IrisAttentionTransformers 的输出结构和 visualize_attention_maps 函数的适用性
                # if 'original_atn_outputs' in locals() and original_atn_outputs and original_atn_outputs.get('attention') is not None and \
                #    'adversarial_atn_outputs' in locals() and adversarial_atn_outputs and adversarial_atn_outputs.get('attention') is not None and train_G_this_step:
                #    with torch.no_grad():
                #        visualize_attention_maps(
                #            writer=writer,
                #            attention_matrix_orig=original_atn_outputs.get('attention')[:num_vis_samples].detach().cpu(),
                #            attention_matrix_adv=adversarial_atn_outputs.get('attention')[:num_vis_samples].detach().cpu(),
                #            step=global_step,
                #            num_samples=min(cfg.logging.num_vis_samples, real_x.shape[0]),
                #            num_heads_to_vis=getattr(cfg.logging, 'num_attn_heads_to_vis', 1) # 从配置获取可视化注意力头的数量
                #        )

            global_step += 1

        # --- 评估 ---
        # 根据配置的间隔或在最后一个 epoch 进行评估
        if (epoch + 1) % cfg.training.eval_interval == 0 or (epoch + 1) == cfg.training.num_epochs:
            if len(eval_dataloader) > 0: # 只有评估数据加载器非空时才进行评估
                print(f"在 epoch {epoch+1} 运行评估...")
                # 将模型设置为评估模式 (虽然 ATN 已冻结，但 G 和 D 需要)
                generator.eval()
                discriminator.eval()
                
                eval_metrics = evaluate_model(
                    generator=generator,
                    discriminator=discriminator,
                    atn_model=atn_model_dict, # 传递整个 ATN 模型字典
                    dataloader=eval_dataloader,
                    device=device,
                    cfg=cfg,
                    current_train_stage=cfg.training.train_stage
                )

                if eval_metrics: # 如果评估成功并返回指标
                    print(f"Epoch {epoch+1} 的评估指标: {eval_metrics}")
                    # 将评估指标记录到 TensorBoard
                    for metric_name, metric_value in eval_metrics.items():
                        # 检查指标值是否有效 (非 NaN, 非 Inf)
                        if np.isfinite(metric_value):
                            writer.add_scalar(f'Evaluation/{metric_name}', float(metric_value), global_step)

                    # 保存最佳模型 (基于主要评估指标)
                    primary_eval_metric_name = getattr(cfg.evaluation, 'primary_metric', 'Attack_Success_Rate_TriggerNet') # 从配置获取主要评估指标名称，默认为攻击成功率
                    is_lower_metric_better = getattr(cfg.evaluation, 'primary_metric_lower_is_better', False) # 从配置获取指标是否越低越好，默认为越高越好
                    primary_eval_metric = eval_metrics.get(primary_eval_metric_name) # 获取主要指标值

                    # 只有主要指标值有效时才进行比较和保存
                    if primary_eval_metric is not None and np.isfinite(primary_eval_metric):
                        # 检查当前指标是否比最佳指标更好
                        is_better = (is_lower_metric_better and primary_eval_metric < best_eval_metric) or \
                                   (not is_lower_metric_better and primary_eval_metric > best_eval_metric)

                        if is_better:
                            best_eval_metric = primary_eval_metric # 更新最佳指标值
                            checkpoint_dir = os.path.join(cfg.logging.log_dir, 'checkpoints')
                            os.makedirs(checkpoint_dir, exist_ok=True) # 创建检查点目录
                            best_checkpoint_path = os.path.join(checkpoint_dir, f'best_stage_{cfg.training.train_stage}.pth')
                            print(f"保存最佳模型：{primary_eval_metric_name}={best_eval_metric:.4f}")

                            # 保存模型检查点
                            torch.save({
                                'epoch': epoch,
                                'global_step': global_step,
                                'generator_state_dict': generator.state_dict(),
                                'discriminator_state_dict': discriminator.state_dict(),
                                'optimizer_G_state_dict': optimizer_G.state_dict(),
                                'optimizer_D_state_dict': optimizer_D.state_dict(),
                                'best_eval_metric': best_eval_metric,
                                'cfg': cfg # 保存配置以供复现
                            }, best_checkpoint_path)
                            print(f"最佳模型已保存到 {best_checkpoint_path}")

                    # 总是保存最新的检查点
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
                    print(f"最新模型已保存到 {latest_checkpoint_path}")

                # 评估结束后，将模型设置回训练模式 (如果训练还没有结束)
                if (epoch + 1) < cfg.training.num_epochs:
                     generator.train()
                     discriminator.train()

    # 训练结束时关闭 TensorBoard writer
    writer.close()

if __name__ == "__main__":
    # 解析命令行参数和配置文件
    cfg = parse_args_and_config(default_config_path="configs/stage2_config.yaml")
    # 运行训练函数
    train(cfg)
