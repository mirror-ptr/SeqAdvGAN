#!/usr/bin/env python3

import sys
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import json # Import json for saving results

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


def main(cfg): # Function now accepts config object
    # Ensure device is available
    device = torch.device(cfg.training.device if torch.cuda.is_available() and cfg.training.device == 'cuda' else "cpu")
    print(f"Using device: {device}")

    # 1. 加载生成器模型 (如果需要评估特定生成器的对抗样本)
    generator = None
    if cfg.evaluation.generator_weights_path: # Get path from config
        print(f"Loading generator weights from {cfg.evaluation.generator_weights_path}")
        # Generator init parameters should match training config
        # Generator input/output channels should match the data channels
        generator = Generator(
            in_channels=cfg.data.channels,
            out_channels=cfg.data.channels,
            epsilon=cfg.model.generator.epsilon,
            num_bottlenecks=cfg.model.generator.num_bottlenecks,
            base_channels=cfg.model.generator.base_channels
        ).to(device)
        
        if os.path.exists(cfg.evaluation.generator_weights_path):
            checkpoint = torch.load(cfg.evaluation.generator_weights_path, map_location=device)
            # Assuming the checkpoint saves 'generator_state_dict'
            generator.load_state_dict(checkpoint['generator_state_dict'])
            generator.eval() # Set to evaluation mode
            print("Generator loaded successfully.")
        else:
            print(f"Warning: Generator weights not found at {cfg.evaluation.generator_weights_path}. Proceeding without generator (evaluating original data or pre-saved adversarial data)." )
            generator = None # If weights not found, do not use generator


    # 2. 加载判别器模型 (如果评估指标需要判别器输出，例如 GAN 评估指标)
    discriminator = None # TODO: Load discriminator if needed for eval metrics
    if cfg.model.discriminator.type in ['cnn', 'patchgan']:
        print(f"Using {cfg.model.discriminator.type.upper()} Discriminator for evaluation.")
        if cfg.model.discriminator.type == 'cnn':
            discriminator = SequenceDiscriminatorCNN(in_channels=cfg.data.channels, base_channels=cfg.model.discriminator.base_channels).to(device)
        elif cfg.model.discriminator.type == 'patchgan':
            discriminator = PatchDiscriminator3d(in_channels=cfg.data.channels, base_channels=cfg.model.discriminator.base_channels).to(device)

        if discriminator:
            # TODO: Load discriminator weights if available and needed
            # if cfg.evaluation.discriminator_weights_path and os.path.exists(cfg.evaluation.discriminator_weights_path):
            #     print(f"Loading discriminator weights from {cfg.evaluation.discriminator_weights_path}")
            #     checkpoint = torch.load(cfg.evaluation.discriminator_weights_path, map_location=device)
            #     # Assuming the checkpoint saves 'discriminator_state_dict'
            #     discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            #     print("Discriminator weights loaded successfully.")
            # else:
            #     print(f"Warning: Discriminator weights not found at {cfg.evaluation.discriminator_weights_path}. Using randomly initialized discriminator for evaluation.")

            discriminator.eval() # Set to evaluation mode
            for param in discriminator.parameters():
                 param.requires_grad = False # Freeze discriminator parameters

            # Adjust Discriminator Sigmoid for LSGAN if needed (match training setup)
            if cfg.losses.gan_type == 'lsgan':
                if isinstance(discriminator, SequenceDiscriminatorCNN) and isinstance(discriminator.classifier[-1], nn.Sigmoid):
                 discriminator.classifier = nn.Sequential(*list(discriminator.classifier.children())[:-1])
                 print("Removed Sigmoid from CNN Discriminator for LSGAN evaluation.")
                elif isinstance(discriminator, PatchDiscriminator3d) and isinstance(discriminator.sigmoid, nn.Sigmoid):
                 discriminator.sigmoid = nn.Identity()
                 print("Replaced Sigmoid with Identity in PatchGAN Discriminator for LSGAN evaluation.")

    else:
        print(f"Warning: Discriminator type '{cfg.model.discriminator.type}' not recognized or not needed for evaluation. Skipping discriminator loading.")
        discriminator = None


    # 3. 加载 ATN 模型
    # ATN model path from config
    atn_model = load_atn_model(
        cfg.model.atn.model_path,
        device,
        in_channels=cfg.model.atn.atn_in_channels, # ATN expects its specific input channels
        sequence_length=cfg.model.atn.atn_sequence_length,
        height=cfg.model.atn.atn_height,
        width=cfg.model.atn.atn_width,
        num_heads=cfg.model.atn.atn_num_heads # Pass num_heads
    )
    atn_model.eval()
    for param in atn_model.parameters():
        param.requires_grad = False # Freeze ATN model parameters
    print("ATN model loaded successfully.")

    # 4. 数据加载
    print(f"Loading evaluation data from {cfg.data.video_path} (video) / {cfg.data.level_json_path} (level JSON)...")

    try:
        # Use GameVideoDataset for real data
        eval_dataset = GameVideoDataset(
            video_path=cfg.data.video_path,
            level_json_path=cfg.data.level_json_path,
            sequence_length=cfg.data.sequence_length,
            target_height=cfg.data.height, # Pass target size
            target_width=cfg.data.width # Pass target size
            # transform=... # Optional evaluation transforms
        )
        # If num_eval_samples is specified and less than dataset size, use a subset
        if getattr(cfg.evaluation, 'num_eval_samples', None) is not None and cfg.evaluation.num_eval_samples < len(eval_dataset):
             print(f"Using a subset of {cfg.evaluation.num_eval_samples} samples for evaluation.")
             subset_indices = list(range(cfg.evaluation.num_eval_samples))
             from torch.utils.data import Subset
             eval_dataset = Subset(eval_dataset, subset_indices)


        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=cfg.training.batch_size, # Use training batch size or define a separate eval batch size
            shuffle=False, # Do not shuffle evaluation data
            num_workers=cfg.data.num_workers,
            pin_memory=True
        )
        print(f"Evaluation data loaded. Number of samples: {len(eval_dataset)}, Number of batches: {len(eval_dataloader)}")

    except FileNotFoundError as e:
        print(f"Error loading evaluation dataset: {e}. Using mock data instead.")
        # Fallback to mock data if real data loading fails
    eval_dataloader = create_mock_dataloader(
            batch_size=cfg.training.batch_size, # Use training batch size or define eval batch size
            num_samples=getattr(cfg.evaluation, 'num_eval_samples', 100), # Use num_eval_samples from cfg
            sequence_length=cfg.data.sequence_length,
            channels=cfg.data.channels, # Match data channels
            height=cfg.data.height,
            width=cfg.data.width,
            shuffle=False, # Do not shuffle mock eval data
            num_workers=cfg.data.num_workers
        )
    print(f"Using mock evaluation data. Number of samples: {len(eval_dataloader) * cfg.training.batch_size}, Number of batches: {len(eval_dataloader)}")


    # 5. 掩码加载 (如果使用)
    decision_mask = None
    attention_mask = None
    if cfg.mask.use_region_mask:
        print("Warning: use_region_mask is enabled for evaluation. You need to implement real mask loading/generation.")
        # TODO: Load/generate evaluation masks based on config (mask_strategy, mask_path)
        # These masks should match the size of the decision_map (H_dec, W_dec) and attention_map (head, N, N)
        # assuming decision map spatial size matches ATN_H, ATN_W for mock
        mock_decision_h = cfg.model.atn.atn_height
        mock_decision_w = cfg.model.atn.atn_width
        mock_attention_n = cfg.model.atn.atn_sequence_length
        mock_num_heads = cfg.model.atn.atn_num_heads

        # Create mock full masks on device
        # Note: In evaluate_model, masks are expected per batch (B, ...)
        # Here we load/create them once. Need to handle this in evaluate_model or pass masks per batch.
        # For simplicity now, let's create per-batch masks within evaluate_model if needed.
        # So we don't need to load/create masks here unless the evaluation function expects them batch-wise.
        # Let's modify evaluate_model to handle masks batch-wise if cfg.mask.use_region_mask is True.
        pass # Mask loading/generation will be handled inside evaluate_model for now


    # 6. 执行评估
    print("Starting evaluation...")
    # evaluate_model function calculates and summarizes all evaluation metrics
    # Pass all necessary parameters, including masks config
    eval_results = evaluate_model(
        generator, # Can be None if evaluating original data
        discriminator, # Can be None if not using GAN metrics
        atn_model,
        eval_dataloader,
        device,
        cfg # Pass the entire config object to evaluate_model
    )

    # 7. 打印和保存评估结果
    print("\n--- Evaluation Summary ---")
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}") # Format output

    # Save evaluation results to file
    if cfg.evaluation.output_path: # Get output path from config
        try:
            # Convert results to a format suitable for saving (e.g., numpy to float)
            results_to_save = {k: float(v) if isinstance(v, (np.float32, np.float64, torch.Tensor)) else v for k, v in eval_results.items()}
            # Ensure the output directory exists
            output_dir = os.path.dirname(cfg.evaluation.output_path)
            if output_dir and not os.path.exists(output_dir):
                 os.makedirs(output_dir)

            with open(cfg.evaluation.output_path, 'w') as f:
                json.dump(results_to_save, f, indent=4)
            print(f"Evaluation results saved to {cfg.evaluation.output_path}")
        except Exception as e:
            print(f"Error saving evaluation results to {cfg.evaluation.output_path}: {e}")

    print("Evaluation finished.")

if __name__ == "__main__":
    # Define default config path
    DEFAULT_CONFIG_PATH = 'configs/default_config.yaml'

    # Parse command line arguments and load/merge config
    # Use 'config' as the argument name for task-specific config file
    # Add specific command line argument for generator weights path for visualization
    parser = argparse.ArgumentParser(description="Evaluate SeqAdvGAN Attack")
    parser.add_argument('--generator_weights_path', type=str, required=True, help='path to the trained generator weights (.pth file)')
    # Add other command line arguments you might want to override from config for vis script
    parser.add_argument('--config', type=str, help='Path to the task-specific YAML configuration file.')

    cmd_args = parser.parse_args()

    # Load configuration using the config utility
    cfg = parse_args_and_config(DEFAULT_CONFIG_PATH, task_config_arg='config')

    if cfg is not None:
        # Add the generator_weights_path from command line to the config for the visualization function
        # This assumes generator_weights_path is not already in the YAML config under visualization
        # If it is, the command line argument will overwrite it via parse_args_and_config
        # Let's add evaluation.generator_weights_path to configs and handle it in parse_args_and_config
        # For now, manually add it if it's not already there (less clean)
        if not hasattr(cfg.evaluation, 'generator_weights_path') or cmd_args.generator_weights_path is not None:
             cfg.evaluation.generator_weights_path = cmd_args.generator_weights_path


        # Ensure log dir exists (using the visualization specific subdir)
        gen_name = os.path.basename(cfg.evaluation.generator_weights_path or 'random_gen').replace('.pth', '')
        vis_log_dir = os.path.join(cfg.logging.log_dir, 'visualization', gen_name)
        os.makedirs(vis_log_dir, exist_ok=True)


        # Print final configuration for confirmation
        import json
        print("--- Final Configuration ---")
        print(json.dumps(dict(cfg), indent=4))
        print("--------------------------")


        main(cfg)
    else:
        print("Failed to load configuration. Exiting.") 