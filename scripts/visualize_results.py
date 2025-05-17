import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import sys
from torch.utils.tensorboard import SummaryWriter

# 将项目根目录添加到 Python 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入模型和可视化工具
from models.generator_cnn import Generator
from utils.vis_utils import visualize_samples_and_outputs, visualize_attention_maps, feature_to_visualizable 

# 导入 ATN 工具和数据工具
from utils.atn_utils import load_atn_model, get_atn_outputs
from utils.data_utils import create_mock_dataloader as create_eval_dataloader

def visualize_results(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 加载生成器模型
    generator = Generator(in_channels=args.channels, out_channels=args.channels, epsilon=args.epsilon).to(device)
    if args.generator_weights_path and os.path.exists(args.generator_weights_path):
        print(f"Loading generator weights from {args.generator_weights_path}")
        checkpoint = torch.load(args.generator_weights_path, map_location=device)
        generator.load_state_dict(checkpoint['generator_state_dict'])
    else:
        print(f"Warning: Generator weights not found at {args.generator_weights_path}. Using randomly initialized generator.")

    generator.eval() # 设置生成器为评估模式

    # 2. 加载 ATN 模型
    atn_model = load_atn_model(args.atn_model_path).to(device)
    atn_model.eval() # 设置 ATN 模型为评估模式
    for param in atn_model.parameters():
        param.requires_grad = False # 冻结 ATN 模型参数

    # 3. 数据加载 (使用评估数据加载器)
    dataloader = create_eval_dataloader(
        batch_size=args.batch_size,
        num_samples=args.num_vis_samples, # 只加载需要可视化的样本数量
        sequence_length=args.sequence_length,
        channels=args.channels,
        height=args.height,
        width=args.width
    )

    # 4. 执行可视化
    print("Generating adversarial samples and visualizing...")

    # Use a SummaryWriter to log visualization results
    # You can use a different log_dir to distinguish between training and visualization logs
    vis_log_dir = os.path.join('runs', 'visualization', os.path.basename(args.generator_weights_path or 'random_gen').replace('.pth', ''))
    writer = SummaryWriter(log_dir=vis_log_dir) # type: ignore
    print(f"TensorBoard logs will be saved to: {vis_log_dir}")

    with torch.no_grad(): # 在可视化时不需要计算梯度
        for i, original_features_batch in enumerate(dataloader):
            original_features = original_features_batch.to(device)

            # 生成扰动和对抗样本
            delta = generator(original_features)
            adversarial_features = original_features + delta

            # 获取 ATN 输出
            original_atn_outputs = get_atn_outputs(atn_model, original_features)
            adversarial_atn_outputs = get_atn_outputs(atn_model, adversarial_features)

            # 调用可视化函数
            visualize_samples_and_outputs(
                writer,
                original_features,
                delta,
                adversarial_features,
                original_atn_outputs.get('decision'),
                adversarial_atn_outputs.get('decision'),
                i, # 使用批次索引作为 step
                num_samples=args.batch_size, # 可视化批次中的所有样本
                sequence_step_to_vis=args.sequence_step_to_vis
            )

            #可视化注意力矩阵 (如果 ATN 输出了注意力)
            if original_atn_outputs.get('attention') is not None or adversarial_atn_outputs.get('attention') is not None:
                 visualize_attention_maps(
                     writer,
                     original_atn_outputs.get('attention'), # (B, head, N, N)
                     adversarial_atn_outputs.get('attention'),# (B, head, N, N)
                     i, # 使用批次索引作为 step
                     num_samples=args.batch_size, # 可视化批次中的所有样本
                     num_heads_to_vis=args.num_vis_heads # 假设添加一个参数来控制可视化多少个头
                 )

    print("Visualization finished.")
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize SeqAdvGAN Generator Results")
    parser.add_argument('--generator_weights_path', type=str, required=True, help='path to the trained generator weights (.pth file)')
    parser.add_argument('--atn_model_path', type=str, default='path/to/atn_model', help='path to the trained ATN model')
    parser.add_argument('--num_vis_samples', type=int, default=10, help='number of samples to visualize')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size for visualization dataloader') # Use batching for efficiency
    parser.add_argument('--sequence_step_to_vis', type=int, default=0, help='the sequence step to visualize (0-indexed)')
    parser.add_argument('--num_vis_heads', type=int, default=1, help='number of attention heads to visualize') # For attention visualization

    # 需要与训练时使用的模拟数据参数一致，或者根据实际评估数据调整
    parser.add_argument('--sequence_length', type=int, default=16, help='sequence length of data')
    parser.add_argument('--channels', type=int, default=128, help='number of channels of data (ATN feature size)')
    parser.add_argument('--height', type=int, default=64, help='height of data')
    parser.add_argument('--width', type=int, default=64, help='width of data')
    parser.add_argument('--epsilon', type=float, default=0.03, help='L-infinity constraint used during training (for generator init)')


    args = parser.parse_args()

    visualize_results(args) 