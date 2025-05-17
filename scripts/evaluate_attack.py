#!/usr/bin/env python3

import sys
import os
import argparse
import torch
import torch.nn as nn
import numpy as np

# 将项目根目录添加到 Python 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入模型和工具
from models.generator_cnn import Generator # 假设评估特定生成器
# TODO: 根据需要导入判别器模型，如果评估指标需要判别器输出
from models.discriminator_cnn import SequenceDiscriminatorCNN
from models.discriminator_patchgan import PatchDiscriminator3d
from utils.atn_utils import load_atn_model, get_atn_outputs
from utils.data_utils import create_real_dataloader, create_mock_dataloader # 导入真实和模拟数据加载器
from utils.eval_utils import evaluate_model # 导入评估函数

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 加载生成器模型 (如果需要评估特定生成器的对抗样本)
    generator = None
    if args.generator_weights_path:
        print(f"Loading generator weights from {args.generator_weights_path}")
        generator = Generator(in_channels=args.channels, out_channels=args.channels, epsilon=args.epsilon).to(device) # 确保参数与训练时一致
        if os.path.exists(args.generator_weights_path):
            checkpoint = torch.load(args.generator_weights_path, map_location=device)
            generator.load_state_dict(checkpoint['generator_state_dict'])
            generator.eval() # 设置为评估模式
            print("Generator loaded successfully.")
        else:
            print(f"Warning: Generator weights not found at {args.generator_weights_path}. Proceeding without generator (evaluating original data or pre-saved adversarial data)." )
            generator = None # 如果权重不存在，不使用生成器


    # 2. 加载判别器模型 (如果评估指标需要判别器输出，例如 GAN 评估指标)
    discriminator = None # TODO: 根据需要加载判别器
    if args.discriminator_type == 'cnn':
        discriminator = SequenceDiscriminatorCNN(in_channels=args.channels).to(device)
        print("Using CNN Discriminator for evaluation.")
         # 如果使用 LSGAN，移除 Sigmoid以便evaluate_model能够正确处理
        if args.gan_loss_type == 'lsgan':
             if isinstance(discriminator.classifier[-1], nn.Sigmoid):
                 discriminator.classifier = nn.Sequential(*list(discriminator.classifier.children())[:-1])
                 print("Removed Sigmoid from CNN Discriminator for LSGAN evaluation.")

    elif args.discriminator_type == 'patchgan':
        discriminator = PatchDiscriminator3d(in_channels=args.channels).to(device)
        print("Using PatchGAN Discriminator for evaluation.")
         # 如果使用 LSGAN，移除 Sigmoid
        if args.gan_loss_type == 'lsgan':
             if isinstance(discriminator.sigmoid, nn.Sigmoid):
                 discriminator.sigmoid = nn.Identity()
                 print("Replaced Sigmoid with Identity in PatchGAN Discriminator for LSGAN evaluation.")
    else:
        print(f"Warning: Discriminator type {args.discriminator_type} not recognized or not needed for evaluation. Skipping discriminator loading.")
        discriminator = None # 如果类型不支持或不需要，不加载判别器

    if discriminator:
        # TODO: 如果有判别器的权重，在这里加载
        # if args.discriminator_weights_path and os.path.exists(args.discriminator_weights_path):
        #     print(f"Loading discriminator weights from {args.discriminator_weights_path}")
        #     checkpoint = torch.load(args.discriminator_weights_path, map_location=device)
        #     discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        #     print("Discriminator weights loaded successfully.")
        # else:
        #     print(f"Warning: Discriminator weights not found at {args.discriminator_weights_path}. Using randomly initialized discriminator for evaluation.")
        discriminator.eval() # 设置为评估模式
        for param in discriminator.parameters():
             param.requires_grad = False # 冻结判别器参数


    # 3. 加载 ATN 模型
    atn_model = load_atn_model(args.atn_model_path).to(device)
    atn_model.eval()
    for param in atn_model.parameters():
        param.requires_grad = False # 冻结 ATN 模型参数
    print("ATN model loaded successfully.")

    # 4. 数据加载
    # TODO: 根据 args 中的数据路径和类型选择加载真实数据还是模拟数据
    # TODO: 需要从 args 中获取数据加载相关的参数 (data_root, use_masks, mask_strategy, mask_path, subset_indices, etc.)
    print(f"Loading evaluation data from {args.data_root}...")

    # Placeholder for real data loading
    # eval_dataloader = create_real_dataloader(
    #     data_root=args.data_root,
    #     batch_size=args.batch_size,
    #     sequence_length=args.sequence_length,
    #     channels=args.channels,
    #     height=args.height,
    #     width=args.width,
    #     use_masks=args.use_region_mask, # 根据参数决定是否使用掩码
    #     mask_strategy=args.mask_strategy, # 传递掩码加载策略
    #     mask_path=args.mask_path, # 传递掩码路径
    #     subset_indices=None, # 如果需要评估特定子集，可以在 args 中添加参数
    #     shuffle=False, # 评估时通常不打乱
    #     num_workers=args.num_workers
    # )

    # 使用模拟数据加载器作为占位符
    eval_dataloader = create_mock_dataloader(
        batch_size=args.batch_size,
        num_samples=args.num_eval_samples, # 使用评估样本数量
        sequence_length=args.sequence_length,
        channels=args.channels,
        height=args.height,
        width=args.width,
        shuffle=False # 评估时不打乱
    )

    print(f"Evaluation data loaded. Number of batches: {len(eval_dataloader)}")

    # 5. 执行评估
    print("Starting evaluation...")
    # evaluate_model 函数将处理所有评估指标的计算和汇总
    eval_results = evaluate_model(generator, discriminator, atn_model, eval_dataloader, device, args) # 将 args 传递给 evaluate_model

    # 6. 打印和保存评估结果
    print("\n--- Evaluation Summary ---")
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}") # 格式化输出

    # TODO: 将评估结果保存到文件 (例如 JSON, CSV)
    if args.output_path:
        try:
            # 将结果转换为适合保存的格式 (例如，numpy 转换为 float)
            results_to_save = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in eval_results.items()}
            with open(args.output_path, 'w') as f:
                import json
                json.dump(results_to_save, f, indent=4)
            print(f"Evaluation results saved to {args.output_path}")
        except Exception as e:
            print(f"Error saving evaluation results to {args.output_path}: {e}")

    print("Evaluation finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SeqAdvGAN Attack")
    parser.add_argument('--generator_weights_path', type=str, default=None, help='path to the trained generator weights (.pth file)')
    parser.add_argument('--atn_model_path', type=str, required=True, help='path to the trained ATN model')
    # 添加判别器相关参数，用于需要判别器输出的评估指标
    parser.add_argument('--discriminator_type', type=str, default=None, choices=['cnn', 'patchgan'], help='type of discriminator used in GAN (for evaluation)')
    # TODO: 添加 discriminator_weights_path 参数
    # parser.add_argument('--discriminator_weights_path', type=str, default=None, help='path to the trained discriminator weights (.pth file)')
    parser.add_argument('--gan_loss_type', type=str, default='bce', choices=['bce', 'lsgan'], help='type of GAN loss used during training (for discriminator interpretation)')

    # 数据加载参数
    # TODO: 将这些参数与 data_utils.py 中的 ATNFeatureDataset.__init__ 参数对应
    parser.add_argument('--data_root', type=str, required=True, help='root directory of the evaluation dataset')
    parser.add_argument('--batch_size', type=int, default=4, help='input batch size for evaluation')
    parser.add_argument('--num_eval_samples', type=int, default=100, help='number of samples to evaluate (if using mock data or subset)') # 添加评估样本数量参数
    # 数据形状参数 (需要与实际数据一致)
    parser.add_argument('--sequence_length', type=int, default=16, help='sequence length of data')
    parser.add_argument('--channels', type=int, default=128, help='number of channels of data (ATN feature size)')
    parser.add_argument('--height', type=int, default=64, help='height of data')
    parser.add_argument('--width', type=int, default=64, help='width of data')
    parser.add_argument('--epsilon', type=float, default=0.03, help='L-infinity constraint used during training (for generator init)') # 用于生成器初始化

    # 评估指标参数 (与 eval_utils.py::evaluate_model 参数对应)
    parser.add_argument('--eval_success_threshold', type=float, default=0.1, help='threshold for attack success in evaluation')
    parser.add_argument('--eval_success_criterion', type=str, default='mse_diff_threshold', choices=['mse_diff_threshold', 'mean_change_threshold', 'topk_value_drop', 'topk_position_change'], help='criterion for attack success in evaluation')
    parser.add_argument('--topk_k', type=int, default=10, help='K value for Top-K attack success criterion in evaluation')
    # TODO: 添加掩码相关的评估参数，如果评估也需要掩码
    parser.add_argument('--use_region_mask', action='store_true', help='whether evaluation should use region masks')
    parser.add_argument('--mask_strategy', type=str, default='none', help='Mask loading/generation strategy for evaluation')
    parser.add_argument('--mask_path', type=str, default=None, help='Path to mask files for evaluation (if mask_strategy is file)')

    # 输出参数
    parser.add_argument('--output_path', type=str, default='eval_results.json', help='path to save evaluation results')

    # 其他参数
    parser.add_argument('--num_workers', type=int, default=0, help='number of data loading workers')

    args = parser.parse_args()

    main(args) 