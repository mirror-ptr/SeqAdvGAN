import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
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
from utils.data_utils import GameVideoDataset, create_mock_dataloader

# Import config utility
from utils.config_utils import parse_args_and_config

def visualize_results(cfg):
    # Ensure device is available
    device = torch.device(cfg.training.device if torch.cuda.is_available() and cfg.training.device == 'cuda' else "cpu")
    print(f"Using device: {device}")

    # 1. 加载生成器模型
    generator = Generator(
        in_channels=cfg.data.channels,
        out_channels=cfg.data.channels,
        epsilon=cfg.model.generator.epsilon,
        num_bottlenecks=cfg.model.generator.num_bottlenecks,
        base_channels=cfg.model.generator.base_channels
    ).to(device)

    if cfg.visualization.generator_weights_path and os.path.exists(cfg.visualization.generator_weights_path):
        print(f"Loading generator weights from {cfg.visualization.generator_weights_path}")
        checkpoint = torch.load(cfg.visualization.generator_weights_path, map_location=device)
        generator.load_state_dict(checkpoint['generator_state_dict'])
    else:
        print(f"Warning: Generator weights not found at {cfg.visualization.generator_weights_path}. Using randomly initialized generator.")

    generator.eval()

    # 2. 加载 ATN 模型
    atn_model = load_atn_model(
        cfg.model.atn.model_path,
        device,
        in_channels=cfg.model.atn.atn_in_channels,
        sequence_length=cfg.model.atn.atn_sequence_length,
        height=cfg.model.atn.atn_height,
        width=cfg.model.atn.atn_width,
        num_heads=cfg.model.atn.atn_num_heads
    )
    atn_model.eval()
    for param in atn_model.parameters():
        param.requires_grad = False

    # 3. 数据加载 (使用真实数据加载器或模拟加载器)
    print(f"Loading visualization data from video: {cfg.data.video_path} / {cfg.data.level_json_path}...")

    try:
        vis_dataset = GameVideoDataset(
            video_path=cfg.data.video_path,
            level_json_path=cfg.data.level_json_path,
            sequence_length=cfg.data.sequence_length,
            target_height=cfg.data.height,
            target_width=cfg.data.width
        )
        num_vis_samples = cfg.visualization.num_vis_samples
        if num_vis_samples > len(vis_dataset):
            print(f"Warning: num_vis_samples ({num_vis_samples}) is greater than dataset size ({len(vis_dataset)}). Visualizing all samples.")
            num_vis_samples = len(vis_dataset)

        subset_indices = list(range(num_vis_samples))
        vis_dataset = Subset(vis_dataset, subset_indices)

        dataloader = DataLoader(
            vis_dataset,
            batch_size=cfg.visualization.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=True
        )
        print(f"Visualization data loaded. Number of samples: {len(vis_dataset)}, Number of batches: {len(dataloader)}")

    except FileNotFoundError as e:
        print(f"Error loading visualization dataset: {e}. Using mock data instead.")
        dataloader = create_mock_dataloader(
            batch_size=cfg.visualization.batch_size,
            num_samples=cfg.visualization.num_vis_samples,
            sequence_length=cfg.data.sequence_length,
            channels=cfg.data.channels,
            height=cfg.data.height,
            width=cfg.data.width,
            shuffle=False,
            num_workers=cfg.data.num_workers
        )
        print(f"Using mock visualization data. Number of samples: {len(dataloader) * cfg.visualization.batch_size}, Number of batches: {len(dataloader)}")

    # 4. 执行可视化
    print("Generating adversarial samples and visualizing...")

    vis_log_dir = os.path.join(cfg.logging.log_dir, 'visualization', os.path.basename(cfg.visualization.generator_weights_path or 'random_gen').replace('.pth', ''))
    writer = SummaryWriter(log_dir=vis_log_dir)
    print(f"TensorBoard logs will be saved to: {vis_log_dir}")

    with torch.no_grad():
        for i, batch_data in enumerate(dataloader):
            real_x = batch_data.to(device)
            real_x_model_format = real_x.permute(0, 1, 2, 4, 3)

            delta = generator(real_x_model_format)
            adversarial_x_model_format = real_x_model_format + delta
            adversarial_x_model_format = torch.clamp(adversarial_x_model_format, 0, 1)

            original_atn_outputs = get_atn_outputs(atn_model, real_x_model_format.permute(0, 1, 2, 4, 3))
            adversarial_atn_outputs = get_atn_outputs(atn_model, adversarial_x_model_format.permute(0, 1, 2, 4, 3))

            visualize_samples_and_outputs(
                writer,
                real_x_model_format,
                delta,
                adversarial_x_model_format,
                original_atn_outputs.get('decision'),
                adversarial_atn_outputs.get('decision'),
                i,
                num_samples=cfg.visualization.batch_size,
                sequence_step_to_vis=cfg.visualization.sequence_step_to_vis
            )

            if original_atn_outputs.get('attention') is not None or adversarial_atn_outputs.get('attention') is not None:
                visualize_attention_maps(
                    writer,
                    original_atn_outputs.get('attention'),
                    adversarial_atn_outputs.get('attention'),
                    i,
                    num_samples=cfg.visualization.batch_size,
                    num_heads_to_vis=cfg.visualization.num_vis_heads
                )

    print("Visualization finished.")
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize SeqAdvGAN Generator Results")
    parser.add_argument('--generator_weights_path', type=str, required=True, help='path to the trained generator weights (.pth file)')
    parser.add_argument('--config', type=str, help='Path to the task-specific YAML configuration file.')

    cmd_args = parser.parse_args()

    cfg = parse_args_and_config(cmd_args.config, task_config_arg='config')

    if cfg is not None:
        if cmd_args.generator_weights_path:
            cfg.visualization.generator_weights_path = cmd_args.generator_weights_path

        gen_name = os.path.basename(cfg.visualization.generator_weights_path or 'random_gen').replace('.pth', '')
        vis_log_dir = os.path.join(cfg.logging.log_dir, 'visualization', gen_name)
        os.makedirs(vis_log_dir, exist_ok=True)

        print("--- Final Configuration ---")
        print(json.dumps(dict(cfg), indent=4))
        print("--------------------------")

        visualize_results(cfg)
    else:
        print("Failed to load configuration. Exiting.") 