#!/usr/bin/env python3

import sys
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import json # Import json for saving results
from typing import Optional, Dict, Any # Import type hints

# 将项目根目录添加到 Python 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入模型和工具
from models.generator_cnn import Generator # 假设评估特定生成器
# TODO: 根据需要导入判别器模型，如果评估指标需要判别器输出
from models.discriminator_cnn import SequenceDiscriminatorCNN
from models.discriminator_patchgan import PatchDiscriminator3d
from utils.atn_utils import load_atn_model, get_atn_outputs
from utils.data_utils import GameVideoDataset, create_mock_dataloader # Import real and mock data loaders
from utils.eval_utils import evaluate_model # Import evaluation function

# Import config utility
from utils.config_utils import parse_args_and_config


def main(cfg: Any) -> None: # Use Any for cfg for now, can refine later
    """
    SeqAdvGAN 攻击效果评估的主函数。

    该函数负责加载配置、模型（生成器、判别器、ATN）、
    准备评估数据，并调用评估工具计算攻击效果的各项指标，
    最后打印和保存评估结果。

    Args:
        cfg (Any): 包含所有评估配置参数的配置对象（例如 EasyDict）。
                   预计包含 training, model, losses, regularization, data, logging, evaluation 等子结构。
                   注意：评估脚本主要使用 evaluation 和 data 相关的配置。
    """
    # 确保设备是可用的 CUDA 或 CPU
    device = torch.device(cfg.training.device if torch.cuda.is_available() and cfg.training.device == 'cuda' else "cpu")
    print(f"Using device: {device}")

    # --- 1. 加载生成器模型 (如果评估的是由特定生成器产生的对抗样本) ---
    generator = None
    # 检查配置中是否指定了生成器权重路径
    if getattr(cfg.evaluation, 'generator_weights_path', None) is not None: # Get path from config
        generator_weights_path = cfg.evaluation.generator_weights_path
        print(f"Loading generator weights from {generator_weights_path}")
        
        # Generator init parameters should match training config
        # Generator input/output channels should match the FEATURE channels (128)
        feature_channels = 128 # Assuming ATN outputs 128 channels
        try:
            generator = Generator(
                in_channels=feature_channels, # Generator input is features (128)
                out_channels=feature_channels, # Generator output is feature perturbation (128)
                epsilon=cfg.model.generator.epsilon,
                num_bottlenecks=cfg.model.generator.num_bottlenecks,
                base_channels=cfg.model.generator.base_channels
            ).to(device)
            
            if os.path.exists(generator_weights_path):
                # 加载权重，使用 weights_only=False 防止 ModuleNotFoundError
                checkpoint = torch.load(generator_weights_path, map_location=device, weights_only=False)
                # 假设 checkpoint 中保存了 'generator_state_dict'
                generator.load_state_dict(checkpoint['generator_state_dict'])
                generator.eval() # 设置为评估模式
                # 冻结生成器参数
                for param in generator.parameters():
                     param.requires_grad = False
                print("Generator loaded and frozen successfully.")
            else:
                print(f"Warning: Generator weights not found at {generator_weights_path}. Proceeding without generator (evaluating original data or pre-saved adversarial data)." )
                generator = None # 如果权重文件不存在，则不使用生成器
        except Exception as e:
             print(f"Error loading generator: {e}. Proceeding without generator.")
             generator = None


    # --- 2. 加载判别器模型 (如果评估指标需要判别器输出，例如 GAN 评估指标) ---
    discriminator = None 
    # 检查配置中是否指定了判别器类型，并且需要加载判别器
    if getattr(cfg.model, 'discriminator', None) is not None and getattr(cfg.model.discriminator, 'type', None) is not None:
        discriminator_type = cfg.model.discriminator.type
        print(f"Attempting to load {discriminator_type.upper()} Discriminator for evaluation.")
        
        try:
            # Discriminator input channels should match feature channels (128)
            feature_channels = 128 # Assuming ATN outputs 128 channels
            if discriminator_type == 'cnn':
                discriminator = SequenceDiscriminatorCNN(in_channels=feature_channels, base_channels=cfg.model.discriminator.base_channels).to(device)
            elif discriminator_type == 'patchgan':
                discriminator = PatchDiscriminator3d(in_channels=feature_channels, base_channels=cfg.model.discriminator.base_channels).to(device)
            else:
                print(f"Warning: Discriminator type '{discriminator_type}' not recognized. Skipping discriminator loading.")
                discriminator = None

            if discriminator is not None:
                # TODO: Load discriminator weights if available and needed from config
                # if getattr(cfg.evaluation, 'discriminator_weights_path', None) is not None:
                #     discriminator_weights_path = cfg.evaluation.discriminator_weights_path
                #     if os.path.exists(discriminator_weights_path):
                #         print(f"Loading discriminator weights from {discriminator_weights_path}")
                #         checkpoint = torch.load(discriminator_weights_path, map_location=device, weights_only=False)
                #         # Assuming the checkpoint saves 'discriminator_state_dict'
                #         discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
                #         print("Discriminator weights loaded successfully.")
                #     else:
                #         print(f"Warning: Discriminator weights not found at {discriminator_weights_path}. Using randomly initialized discriminator for evaluation.")
                # else:
                #      print("No discriminator weights path specified in config. Using randomly initialized discriminator.")

                discriminator.eval() # 设置为评估模式
                # 冻结判别器参数
                for param in discriminator.parameters():
                     param.requires_grad = False 

                # 根据 GAN 类型调整判别器输出层 (例如 LSGAN 需要移除 Sigmoid)
                if getattr(cfg.losses, 'gan_type', None) == 'lsgan':
                    if isinstance(discriminator, SequenceDiscriminatorCNN) and isinstance(discriminator.classifier[-1], nn.Sigmoid):
                     discriminator.classifier = nn.Sequential(*list(discriminator.classifier.children())[:-1])
                     print("Removed Sigmoid from CNN Discriminator for LSGAN evaluation.")
                    elif isinstance(discriminator, PatchDiscriminator3d) and isinstance(discriminator.sigmoid, nn.Sigmoid):
                     discriminator.sigmoid = nn.Identity()
                     print("Replaced Sigmoid with Identity in PatchGAN Discriminator for LSGAN evaluation.")
                
        except Exception as e:
             print(f"Error loading or configuring discriminator: {e}. Skipping discriminator loading.")
             discriminator = None

    else:
        print("No discriminator configuration found or discriminator not needed for this evaluation. Skipping discriminator loading.")
        discriminator = None


    # --- 3. 加载 ATN 模型 ---
    # ATN model path from config
    # 确保 ATN 模型路径在配置中存在
    if getattr(cfg.model, 'atn', None) is None or getattr(cfg.model.atn, 'model_path', None) is None:
         print("Error: ATN model path not specified in config. Cannot proceed with evaluation.")
         return # 无法加载 ATN 模型，退出

    atn_model_path = cfg.model.atn.model_path
    print(f"Loading ATN model from: {atn_model_path}")
    
    # load_atn_model 函数现在负责实例化和加载权重
    # ATN takes original image dimensions as input parameters
    try:
        atn_model = load_atn_model(
            atn_model_path,
            device,
            in_channels=cfg.data.channels, # ATN input is original image channels (3)
            sequence_length=cfg.data.sequence_length, # ATN input sequence length
            height=cfg.data.height, # ATN input height
            width=cfg.data.width, # ATN input width
            num_heads=cfg.model.atn.atn_num_heads # Pass num_heads
        )
        
        if atn_model is None:
             print("Error: Failed to load ATN model using load_atn_model. Cannot proceed with evaluation.")
             return # 加载失败，退出
             
        atn_model.eval() # 设置 ATN 模型为评估模式
        # 冻结 ATN 模型参数
        for param in atn_model.parameters():
            param.requires_grad = False 
        print("ATN model loaded and frozen successfully.")
        
    except Exception as e:
         print(f"Error loading ATN model: {e}. Cannot proceed with evaluation.")
         return # 加载失败，退出


    # --- 4. 数据加载 ---
    print(f"Loading evaluation data from {cfg.data.video_path} (video) / {cfg.data.level_json_path} (level JSON)...")

    try:
        # 使用 GameVideoDataset 加载真实数据
        eval_dataset = GameVideoDataset(
            video_path=cfg.data.video_path,
            level_json_path=cfg.data.level_json_path,
            sequence_length=cfg.data.sequence_length,
            target_height=cfg.data.height, # Pass target size
            target_width=cfg.data.width # Pass target size
            # transform=... # Optional evaluation transforms
        )
        # 如果配置中指定了 num_eval_samples 并且小于数据集大小，使用一个子集
        num_eval_samples = getattr(cfg.evaluation, 'num_eval_samples', None)
        if num_eval_samples is not None and num_eval_samples < len(eval_dataset):
             print(f"Using a subset of {num_eval_samples} samples for evaluation.")
             subset_indices = list(range(num_eval_samples))
             from torch.utils.data import Subset
             eval_dataset = Subset(eval_dataset, subset_indices)


        # 创建评估数据加载器
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=cfg.training.batch_size, # 使用训练时的批量大小或定义一个独立的评估批量大小
            shuffle=False, # 评估数据不打乱
            num_workers=cfg.data.num_workers,
            pin_memory=True # 将数据加载到 CUDA 锁页内存以加速传输
        )
        print(f"Evaluation data loaded. Number of samples: {len(eval_dataset)}, Number of batches: {len(eval_dataloader)}")

    except FileNotFoundError as e:
        print(f"Error loading evaluation dataset from specified path: {e}. Attempting to use mock data instead.")
        # 如果真实数据加载失败，回退到模拟数据
        try:
            eval_dataloader = create_mock_dataloader(
                batch_size=cfg.training.batch_size, # 使用训练时的批量大小或定义评估批量大小
                num_samples=getattr(cfg.evaluation, 'num_eval_samples', 100), # 使用 config 中的 num_eval_samples，默认为 100
                sequence_length=cfg.data.sequence_length,
                channels=cfg.data.channels, # 匹配数据通道数
                height=cfg.data.height,
                width=cfg.data.width,
                shuffle=False, # 模拟评估数据不打乱
                num_workers=cfg.data.num_workers
            )
            # 检查模拟数据加载器是否为空
            if len(eval_dataloader) == 0:
                 print("Error: Mock dataloader is empty. Cannot perform evaluation.")
                 return # 如果模拟数据也加载失败，退出

            print(f"Using mock evaluation data. Number of samples: {len(eval_dataloader) * cfg.training.batch_size}, Number of batches: {len(eval_dataloader)}")
        except Exception as e_mock:
             print(f"Error creating mock dataloader: {e_mock}. Cannot perform evaluation.")
             return # 如果模拟数据创建失败，退出

    # --- 5. 掩码加载 (如果使用) ---
    # 在评估脚本中，掩码需要加载或生成，并传递给 evaluate_model 函数
    # evaluate_model 函数将负责在批次级别应用掩码
    # 根据 cfg.mask.use_region_mask 决定是否启用掩码逻辑
    # 实际掩码加载/生成逻辑（例如从文件读取，或根据关卡 JSON 生成）需要在这里实现
    # 这里的 decision_mask 和 attention_mask 变量将传递给 evaluate_model
    decision_mask = None
    attention_mask = None
    if getattr(cfg.mask, 'use_region_mask', False):
        print("Warning: use_region_mask is enabled for evaluation. Implement real mask loading/generation based on config (e.g., mask_strategy, mask_path)." )
        # TODO: Implement mask loading/generation logic here based on cfg
        # Example placeholder (creating dummy masks - replace with real logic):
        # Assuming decision map spatial size matches ATN_H, ATN_W and attention matrix size matches ATN_N
        # dummy_decision_mask = torch.ones(cfg.model.atn.atn_height // 4, cfg.model.atn.atn_width // 4, device=device) # Example mask size
        # dummy_attention_mask = torch.ones(cfg.model.atn.atn_num_heads, cfg.model.atn.atn_sequence_length, cfg.model.atn.atn_sequence_length, device=device) # Example mask size
        # decision_mask = dummy_decision_mask
        # attention_mask = dummy_attention_mask
        pass # Placeholder for real mask loading/generation


    # --- 6. 执行评估 ---
    print("Starting evaluation...")
    # 调用 evaluate_model 函数计算并汇总所有评估指标
    # 传递加载的模型、数据加载器、设备、配置对象以及掩码
    # evaluate_model 函数会根据 cfg 中的参数（如 success_criterion, success_threshold, current_train_stage）进行评估
    eval_results = evaluate_model(
        generator=generator, # 可以是 None 如果只评估原始数据
        discriminator=discriminator, # 可以是 None 如果不使用 GAN 指标
        atn_model=atn_model,
        dataloader=eval_dataloader,
        device=device,
        cfg=cfg, # 传递完整的 config 对象
        current_train_stage=getattr(cfg.training, 'train_stage', 1) # 从 config 获取当前训练阶段，默认为 1
    )

    # --- 7. 打印和保存评估结果 ---
    print("\n--- Evaluation Summary ---")
    # 遍历评估结果字典，打印每个指标
    for key, value in eval_results.items():
        print(f"  {key}: {value:.6f}") # 提高打印精度

    # 将评估结果保存到文件 (如果配置中指定了输出路径)
    if getattr(cfg.evaluation, 'output_path', None) is not None: # Get output path from config
        output_path = cfg.evaluation.output_path
        try:
            # 将结果转换为适合保存的格式 (例如，numpy float 转换为 Python float)
            results_to_save = {k: float(v) if isinstance(v, (np.float32, np.float64, torch.Tensor)) else v for k, v in eval_results.items()}
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                 os.makedirs(output_dir)

            # 使用 json 格式保存结果
            with open(output_path, 'w') as f:
                json.dump(results_to_save, f, indent=4)
            print(f"Evaluation results saved to {output_path}")
        except Exception as e:
            print(f"Error saving evaluation results to {output_path}: {e}")

    print("Evaluation finished.")

# 如果脚本作为主程序运行，则解析命令行参数并启动评估
if __name__ == "__main__":
    # Define default config path
    DEFAULT_CONFIG_PATH = 'configs/default_config.yaml' # 默认配置文件路径

    # 使用 config_utils 中的函数解析命令行参数和加载配置
    # 默认加载 default_config.yaml，然后加载 --config 参数指定的 yaml 文件，并用命令行参数覆盖
    # parse_args_and_config 函数会自动处理 EasyDict 和参数覆盖
    cfg = parse_args_and_config(default_config_path=DEFAULT_CONFIG_PATH, task_config_arg='config')

    # 检查配置是否成功加载
    if cfg is not None:
        print("Configuration loaded:")
        # 打印最终配置以供确认
        import json
        print("--- Final Configuration ---")
        # 将 EasyDict 转换为 dict 以便使用 json.dumps
        try:
            print(json.dumps(dict(cfg), indent=4))
        except TypeError:
            # 如果 config 对象无法直接转换为 dict (例如嵌套结构)，尝试更安全的方式
            print("Could not serialize config to JSON, printing raw EasyDict:")
            print(cfg)
        print("--------------------------")

        # 启动评估主函数
        main(cfg) 

    else:
        print("Failed to load configuration. Exiting.") 