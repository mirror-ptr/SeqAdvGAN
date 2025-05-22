import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import argparse
import os
import sys
from torch.utils.tensorboard import SummaryWriter
from typing import Any # Import Any for cfg type hint
import json # Import json for pretty printing config

# 将项目根目录添加到 Python 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入模型和可视化工具
from models.generator_cnn import Generator
from utils.vis_utils import visualize_samples_and_outputs, visualize_attention_maps, feature_to_visualizable

# 导入 ATN 工具和数据工具
from utils.atn_utils import load_atn_model, get_atn_outputs
from utils.data_utils import GameVideoDataset, create_mock_dataloader, worker_init_fn # Import worker_init_fn if needed for real data

# Import config utility
from utils.config_utils import parse_args_and_config

def visualize_results(cfg: Any):
    """
    根据配置加载训练好的生成器和 ATN 模型，生成对抗样本，并使用 TensorBoard 进行可视化。

    Args:
        cfg (Any): 包含所有可视化和模型配置参数的配置对象（例如 EasyDict）。
                   预计包含 training, model, data, logging, visualization 等子结构。
    """
    # 确保设备是可用的 CUDA 或 CPU，并将其设置为计算设备
    device = torch.device(cfg.training.device if torch.cuda.is_available() and cfg.training.device == 'cuda' else "cpu")
    print(f"Using device: {device}")

    # --- 1. 加载生成器模型 ---
    # 根据配置初始化 Generator 模型
    # Generator 的输入输出通道数应该与 ATN 特征通道数一致 (通常为 128)
    # 注意：这里的 Generator 定义可能与 Stage 1 训练时的 Generator 定义不完全一致，
    # 如果 Stage 1 的 Generator 输入是特征，而这里期望的是图像，需要调整。
    # 根据 train_generator.py，阶段 1 的 Generator 输入和输出都是特征 (128通道)。
    # 因此，这里的 in_channels 和 out_channels 应该匹配特征通道数。
    # ATN 的特征通道数默认为 128。
    feature_channels = 128 # 假设 ATN 输出特征是 128 通道

    generator = Generator(
        in_channels=feature_channels, # Generator 输入是特征
        out_channels=feature_channels, # Generator 输出是特征扰动
        epsilon=cfg.model.generator.epsilon, # 扰动上限
        num_bottlenecks=cfg.model.generator.num_bottlenecks, # Generator 的 Bottleneck 数量
        base_channels=cfg.model.generator.base_channels # Generator 的基础通道数
    ).to(device)

    # 加载训练好的 Generator 权重 (如果指定了路径且文件存在)
    if cfg.visualization.generator_weights_path and os.path.exists(cfg.visualization.generator_weights_path):
        print(f"Loading generator weights from {cfg.visualization.generator_weights_path}")
        try:
             # 加载权重文件
             checkpoint = torch.load(cfg.visualization.generator_weights_path, map_location=device)
             # 加载模型的 state_dict
             generator.load_state_dict(checkpoint['generator_state_dict'])
             print("Generator weights loaded successfully.")
        except Exception as e:
             # 如果加载权重失败，打印警告
             print(f"Warning: Failed to load generator weights: {e}. Using randomly initialized generator instead.")
    else:
        # 如果权重文件路径未指定或文件不存在，打印警告
        print(f"Warning: Generator weights not found at {cfg.visualization.generator_weights_path}. Using randomly initialized generator.")

    # 将生成器设置为评估模式，关闭 Dropout 和 BatchNorm 的训练行为
    generator.eval()

    # --- 2. 加载 ATN 模型 ---
    # ATN 模型用于获取原始样本的特征、决策图和注意力图，以及对抗样本的对应输出。
    print(f"Loading ATN model/feature head from: {cfg.model.atn.model_path}")
    # load_atn_model 函数负责初始化 IrisXeonNet 模型并加载其权重。
    # ATN 模型的输入参数应匹配原始图像的维度 (通道、序列长度、高、宽) 和注意力头部数量。
    atn_model = load_atn_model(
        cfg.model.atn.model_path, # ATN 模型权重文件路径
        device, # 计算设备
        in_channels=cfg.data.channels, # ATN 输入是原始图像通道数 (通常为 3)
        sequence_length=cfg.data.sequence_length, # ATN 输入序列长度
        height=cfg.data.height, # ATN 输入高度
        width=cfg.data.width, # ATN 输入宽度
        num_heads=cfg.model.atn.atn_num_heads # ATN 注意力头部数量
    )

    # 检查 ATN 模型是否成功加载
    if atn_model is None:
        print("Failed to load ATN model. Exiting visualization.")
        return

    # 将 ATN 模型设置为评估模式，并冻结所有参数（在可视化过程中不需要训练 ATN）
    atn_model.eval()
    for param in atn_model.parameters():
        param.requires_grad = False # Freeze ATN model parameters
    print("ATN model loaded and frozen.")

    # --- 3. 数据加载 (用于可视化) ---
    print(f"Loading visualization data from video: {cfg.data.video_path} / {cfg.data.level_json_path}...")

    try:
        # 使用 GameVideoDataset 加载真实的视频和关卡数据
        # Sequence length, target height/width 来自配置
        vis_dataset = GameVideoDataset(
            video_path=cfg.data.video_path,
            level_json_path=cfg.data.level_json_path,
            sequence_length=cfg.data.sequence_length,
            target_height=cfg.data.height,
            target_width=cfg.data.width
        )
        
        # 根据配置确定要可视化的样本数量，并创建数据集子集
        num_vis_samples = cfg.visualization.num_vis_samples
        # 确保要可视化的样本数量不超过实际数据集大小
        if num_vis_samples > len(vis_dataset):
            print(f"Warning: num_vis_samples ({num_vis_samples}) is greater than dataset size ({len(vis_dataset)}). Visualizing all samples.")
            num_vis_samples = len(vis_dataset)

        # 创建一个包含前 num_vis_samples 个样本索引的列表，并构建 Subset 数据集
        subset_indices = list(range(num_vis_samples))
        vis_dataset = Subset(vis_dataset, subset_indices)

        # 创建 DataLoader 用于批量加载可视化样本
        dataloader = DataLoader(
            vis_dataset,
            batch_size=cfg.visualization.batch_size, # 使用可视化专用的批量大小
            shuffle=False, # 可视化时不打乱样本顺序
            num_workers=cfg.data.num_workers, # 使用配置的 worker 数量加载数据
            pin_memory=True # 启用 pin_memory 加快数据传输到 GPU
            # 如果使用真实视频数据且需要 worker_init_fn，这里也应添加 worker_init_fn=worker_init_fn
        )
        print(f"Visualization data loaded. Number of samples: {len(vis_dataset)}, Number of batches: {len(dataloader)}")

    except FileNotFoundError as e:
        # 如果加载真实数据失败，使用模拟数据加载器作为备选
        print(f"Error loading visualization dataset: {e}. Using mock data instead.")
        dataloader = create_mock_dataloader(
            batch_size=cfg.visualization.batch_size,
            num_samples=cfg.visualization.num_vis_samples, # 模拟数据生成 num_vis_samples 个样本
            sequence_length=cfg.data.sequence_length,
            channels=cfg.data.channels,
            height=cfg.data.height,
            width=cfg.data.width,
            shuffle=False, # 模拟数据不打乱
            num_workers=cfg.data.num_workers # 使用配置的 worker 数量
        )
        print(f"Using mock visualization data. Number of samples to generate: {cfg.visualization.num_vis_samples}, Number of batches: {len(dataloader)}")

    # --- 4. 执行可视化并生成 TensorBoard 日志 ---
    print("Generating adversarial samples and visualizing...")

    # 确定 TensorBoard 日志保存目录
    # 日志目录结构：log_dir / visualization / Generator 权重文件名 (不含扩展名)
    # 如果没有加载权重，使用 'random_gen' 作为目录名
    gen_name = os.path.basename(cfg.visualization.generator_weights_path or 'random_gen').replace('.pth', '')
    vis_log_dir = os.path.join(cfg.logging.log_dir, 'visualization', gen_name)
    # 创建 SummaryWriter 实例，指定日志保存目录
    writer = SummaryWriter(log_dir=vis_log_dir)
    print(f"TensorBoard logs will be saved to: {vis_log_dir}")

    # 在 torch.no_grad() 上下文中执行可视化，不计算梯度
    with torch.no_grad():
        # 遍历数据加载器中的每个批次
        for i, batch_data in enumerate(dataloader):
            # batch_data shape: (B, C_img, T, H_img, W_img) - 原始图像数据
            real_x = batch_data.to(device) # 将数据移动到计算设备
            
            # 将原始图像数据调整为模型期望的格式 (Generator/ATN 输入格式可能不同)
            # 根据 Generator 的 forward 函数定义 (B, C, N, W, H)，这里需要调整
            # 假设原始图像数据是 (B, C_img, T, H_img, W_img)，需要匹配 Generator 输入 (B, C_feat, N_feat, W_feat, H_feat)
            # 但是 Generator 阶段 1 输入是特征，而这里 real_x 是原始图像。
            # visualize_results 函数似乎假设 Generator 输入是原始图像并输出图像扰动。
            # 如果 Generator 是阶段 1 的特征扰动生成器，则其输入应为 ATN 特征，输出为特征扰动。
            # 需要根据当前脚本的实际功能来调整这里的逻辑。
            # 假设此脚本用于 Stage 2 可视化，Generator 输入是图像。
            # 如果此脚本用于 Stage 1 可视化，则 Generator 输入是 ATN 特征。
            # 根据代码，Generator(real_x_model_format) 假设 Generator 输入是图像数据。
            # 这里的 real_x_model_format 的 permute 逻辑 (0, 1, 2, 4, 3) 看起来是从 (B, C, T, H, W) -> (B, C, T, W, H)
            # 这似乎是将 H 和 W 交换了。
            # 需要确认 Generator 和 ATN 期望的空间维度顺序是 W, H 还是 H, W。
            # 假设它们期望 (B, C, N, H, W) 或 (B, C, T, H, W)
            # 根据 Generator 定义的 Conv3d kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
            # 这意味着空间维度是 H, W，序列维度是 N/T。
            # Generator 的 forward 也期望 (B, C, N, W, H)，这与 Conv3d 定义有点矛盾。
            # 让我们暂时假设 Generator 期望 (B, C, N, H, W)，那么 real_x 已经是这个形状。
            # 如果 Generator 期望 (B, C, N, W, H)，那么 real_x.permute(0, 1, 2, 4, 3) 是正确的。
            # 检查 Generator CNN 的 forward 函数，它期望 (B, 128, N, W, H)。
            # 但是 ATN 的输出特征形状是 (B, 128, N, H, W)。
            # 这意味着 Generator 接收 ATN 特征时需要 H/W 转换，或者 Generator 设计时考虑了 ATN 输出形状。
            # 鉴于 Generator 在 train_generator 中输入是 features，且 features 形状是 (B, 128, N, H, W)
            # 那么 visualize_results 中的 Generator 输入也应该是特征。
            # 这里的代码 `delta = generator(real_x_model_format)` 假设 Generator 输入是图像 (real_x)。
            # 这与 Stage 1 训练 Generator 输入特征的逻辑不符。
            # 此脚本更像是用于 Stage 2 的可视化，Generator 输入是图像。
            # 让我们假设 visualize_results 用于 Stage 2，Generator 输入原始图像 (B, C_img, T, H_img, W_img)，输出图像扰动 (B, C_img, T, H_img, W_img)。
            # 那么 Generator 初始化时的 in_channels/out_channels 应该是图像通道数 (例如 3)。
            # ATN 模型加载也应该使用原始图像维度。

            # 根据 Stage 2 逻辑修正 Generator 初始化通道数
            generator_vis = Generator(
                in_channels=cfg.data.channels, # Generator 输入是原始图像通道数
                out_channels=cfg.data.channels, # Generator 输出是图像扰动通道数
                epsilon=cfg.model.generator.epsilon,
                num_bottlenecks=cfg.model.generator.num_bottlenecks,
                base_channels=cfg.model.generator.base_channels
            ).to(device)

            # 重新加载 Generator 权重到修正后的模型结构
            if cfg.visualization.generator_weights_path and os.path.exists(cfg.visualization.generator_weights_path):
                print(f"Loading generator weights into visualization model from {cfg.visualization.generator_weights_path}")
                try:
                     checkpoint = torch.load(cfg.visualization.generator_weights_path, map_location=device)
                     # 加载 state_dict，注意如果模型结构不完全匹配可能会报错 (例如通道数)
                     # strict=False 可以在一定程度上忽略不匹配的键
                     generator_vis.load_state_dict(checkpoint['generator_state_dict'], strict=True) # 严格匹配以确保结构正确
                     print("Generator visualization weights loaded successfully.")
                except Exception as e:
                     print(f"Error loading generator weights into visualization model: {e}. Using randomly initialized generator_vis.")
                     # 如果加载失败，可能需要重新初始化 generator_vis 或者退出
                     # 为了继续，我们使用随机初始化的模型，并打印警告
                     generator_vis = Generator(
                        in_channels=cfg.data.channels,
                        out_channels=cfg.data.channels,
                        epsilon=cfg.model.generator.epsilon,
                        num_bottlenecks=cfg.model.generator.num_bottlenecks,
                        base_channels=cfg.model.generator.base_channels
                     ).to(device)

            generator_vis.eval()

            # 原始图像数据已经是 (B, C, T, H, W)
            original_image = real_x # Use real_x directly as original_image

            # 生成图像扰动 delta
            # Generator 输入原始图像
            image_delta = generator_vis(original_image) # Generator outputs image delta
            
            # 计算对抗图像：原始图像 + 图像扰动，并进行裁剪到有效范围 [0, 1]
            # 注意：这里的裁剪范围应与训练时 Generator 输出层保持一致 (例如 Tanh 输出 [-1, 1] 再缩放)
            # 如果 Generator 输出是 [-epsilon, epsilon] 的扰动，裁剪到 [0, 1] 是将 (image + delta) 裁剪
            adversarial_image = torch.clamp(original_image + image_delta, 0, 1)

            # 将原始图像和对抗图像通过 ATN 模型，获取特征、决策图、注意力图
            # ATN 模型期望输入是图像数据 (B, C_img, T, H_img, W_img)
            original_atn_outputs = get_atn_outputs(
                 atn_model, 
                 original_image,
                 return_features=True, # 在 Stage 2 可视化时也获取特征，可能有用
                 return_decision=True, # Stage 2 攻击决策图
                 return_attention=True # Stage 2 攻击注意力图
            )
            adversarial_atn_outputs = get_atn_outputs(
                 atn_model,
                 adversarial_image,
                 return_features=True,
                 return_decision=True,
                 return_attention=True
            )

            # 提取原始和对抗的决策图和注意力图用于可视化
            original_decision_map = original_atn_outputs.get('decision')
            adversarial_decision_map = adversarial_atn_outputs.get('decision')
            original_attention_map = original_atn_outputs.get('attention')
            adversarial_attention_map = adversarial_atn_outputs.get('attention')

            # 调用 visualize_samples_and_outputs 函数进行图像、特征、决策图等的可视化
            # sequence_step_to_vis 选择序列中的哪个步骤进行 2D 可视化
            sequence_step_to_vis = cfg.visualization.sequence_step_to_vis
            
            # 如果可视化函数期望的输入形状与我们这里的图像形状不同，需要进行调整
            # visualize_samples_and_outputs 期望 original_image, feature_delta, adversarial_features, original_decision_map, adversarial_decision_map
            # 这里的 original_image 是图像 (B, C, T, H, W)，feature_delta 是图像扰动 (B, C, T, H, W)，adversarial_features 是对抗特征 (B, C_feat, N_feat, H_feat, W_feat)
            # 参数传递需要匹配 visualize_samples_and_outputs 的定义。
            # visualize_samples_and_outputs 接收 original_image, original_features, feature_delta, adversarial_features, original_decision_map, adversarial_decision_map
            # 其中 original_image, feature_delta, adversarial_features 在 visualize_results 中是图像数据 (B, C, T, H, W)
            # original_features, adversarial_features 在 visualize_samples_and_outputs 中是 ATN 特征 (B, C_feat, N_feat, W_feat, H_feat)
            # 这里需要区分是可视化图像还是特征。
            # 根据 Stage 2 的攻击目标，Generator 输出的是图像扰动，ATN 输出的是决策图/注意力。
            # visualize_samples_and_outputs 应该可视化原始图像、图像扰动、对抗图像，以及原始和对抗的决策图。
            # visualize_attention_maps 应该可视化原始和对抗的注意力图。

            # 调整 visualize_samples_and_outputs 的调用参数以匹配 Stage 2 可视化内容
            visualize_samples_and_outputs(
                writer, # SummaryWriter 对象
                original_image=original_image, # 原始输入图像
                original_features=original_atn_outputs.get('features'), # 原始 ATN 特征 (可选)
                feature_delta=None, # 在 Stage 2，这不是特征扰动，而是图像扰动；传递图像扰动给 visualize_samples_and_outputs 的 feature_delta 参数可能命名不符，但如果函数内部处理了图像扰动，可以这样。
                # 更好的做法是修改 visualize_samples_and_outputs 以接收 image_delta 和 adversarial_image
                # 暂时将 image_delta 传递给 feature_delta 参数，并在 visualize_samples_and_outputs 内部处理。
                # visualize_samples_and_outputs 内部会将这个参数视为扰动进行可视化。
                # 同时也需要将对抗图像传递进去。
                # 修改 visualize_samples_and_outputs 定义或根据其现有实现调整参数。
                # 检查 visualize_samples_and_outputs 的定义，它有 original_image, feature_delta, adversarial_features 参数。
                # feature_delta 被用于可视化扰动范数图。
                # adversarial_features 被用于可视化对抗特征或对抗图像。
                # original_image 用于上下文。
                # 它没有直接的 adversarial_image 参数。
                # 可以将 adversarial_image 传递给 adversarial_features 参数，并在 visualize_samples_and_outputs 内部判断其类型。
                feature_delta=image_delta, # 将图像扰动传递给 feature_delta 参数
                adversarial_features=adversarial_image, # 将对抗图像传递给 adversarial_features 参数
                original_decision_map=original_decision_map, # 原始决策图
                adversarial_decision_map=adversarial_decision_map, # 对抗决策图
                step=i, # 使用批次索引作为步数（或者全局步数 if available）
                num_samples=cfg.visualization.batch_size, # 当前批次的样本数量
                sequence_step_to_vis=sequence_step_to_vis, # 要可视化的序列步骤
                visualize_decision_diff=True # 在 Stage 2 可视化决策图差异
            )

            # 调用 visualize_attention_maps 函数可视化注意力图
            # 检查原始或对抗注意力图是否可用
            if original_attention_map is not None or adversarial_attention_map is not None:
                visualize_attention_maps(
                    writer, # SummaryWriter 对象
                    original_attention_map, # 原始注意力矩阵
                    adversarial_attention_map, # 对抗注意力矩阵
                    step=i, # 使用批次索引作为步数
                    num_samples=cfg.visualization.batch_size, # 当前批次的样本数量
                    num_heads_to_vis=cfg.visualization.num_vis_heads # 要可视化的注意力头部数量
                )

    print("Visualization finished.")
    # 关闭 SummaryWriter，确保日志写入文件
    writer.close()

# 如果脚本作为主程序运行，则执行以下代码块
if __name__ == "__main__":
    # 创建 ArgumentParser 用于解析命令行参数
    parser = argparse.ArgumentParser(description="Visualize SeqAdvGAN Generator Results")
    
    # 添加命令行参数用于指定训练好的 Generator 权重文件路径 (必需参数)
    parser.add_argument(
        '--generator_weights_path', 
        type=str, 
        required=True, 
        help='path to the trained generator weights (.pth file)'
    )
    
    # 添加命令行参数用于指定配置文件路径 (可选参数)
    # parse_args_and_config 函数会处理这个参数
    parser.add_argument(
        '--config', 
        type=str, 
        help='Path to the task-specific YAML configuration file.'
    )

    # 解析命令行参数
    # 使用 parse_known_args() 以便将未知参数传递给 parse_args_and_config
    cmd_args, unknown_args = parser.parse_known_args()

    # 使用 config_utils 中的函数加载和合并配置
    # default_config_path 应该指向默认配置文件
    # task_config_arg 应该与上面 parser.add_argument 中定义的 config 参数名一致
    cfg = parse_args_and_config(default_config_path='configs/default_config.yaml', task_config_arg='config')

    # 检查配置是否成功加载
    if cfg is not None:
        # 将命令行中指定的 generator_weights_path 添加到配置中，覆盖文件中的值
        # 使用 hasattr 检查 cfg.visualization 是否存在，如果不存在则创建
        if not hasattr(cfg, 'visualization'):
             cfg.visualization = edict() # Create visualization section if it doesn't exist
        if cmd_args.generator_weights_path:
            cfg.visualization.generator_weights_path = cmd_args.generator_weights_path

        # 确定用于可视化日志的子目录名称，基于 Generator 权重文件名
        # 如果没有加载权重，使用 'random_gen' 作为目录名
        gen_name = os.path.basename(cfg.visualization.get('generator_weights_path', 'random_gen')).replace('.pth', '')
        # 构建完整的可视化日志目录路径：log_dir / visualization / gen_name
        # 使用 get() 方法安全访问 logging.log_dir，提供默认值 '.' 如果不存在
        vis_log_dir = os.path.join(cfg.logging.get('log_dir', '.'), 'visualization', gen_name)
        # 确保可视化日志目录存在
        os.makedirs(vis_log_dir, exist_ok=True)
        # 将最终确定的可视化日志目录路径存回配置对象，供 visualize_results 函数使用
        cfg.logging.vis_log_dir = vis_log_dir

        # 打印最终加载和合并后的配置，方便用户检查
        print("--- Final Configuration ---")
        # 将 EasyDict 转换为标准字典再用 json 打印，格式更清晰
        print(json.dumps(dict(cfg), indent=4)) # 使用 json.dumps 格式化输出
        print("--------------------------")

        # 调用可视化主函数，开始生成和记录可视化结果
        visualize_results(cfg)
    else:
        # 如果配置加载失败，打印错误消息
        print("Failed to load configuration. Exiting.")
        sys.exit(1) # 退出脚本并返回非零状态码 