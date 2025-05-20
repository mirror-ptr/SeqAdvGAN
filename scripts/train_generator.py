import sys
import os
# 将项目根目录添加到 Python 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import argparse # Need argparse to define the config path argument

# 导入模型、损失函数和工具
from models.generator_cnn import Generator
from models.discriminator_cnn import SequenceDiscriminatorCNN
from models.discriminator_patchgan import PatchDiscriminator3d
from losses.gan_losses import GANLosses
from losses.attack_losses import AttackLosses
from losses.regularization_losses import linf_norm, l2_norm, l2_penalty, total_variation_loss

# 导入可视化函数
from utils.vis_utils import visualize_samples_and_outputs, visualize_attention_maps

# 导入 ATN 工具和数据工具
from utils.atn_utils import load_atn_model, get_atn_outputs
from utils.data_utils import GameVideoDataset, create_mock_dataloader # Import GameVideoDataset

# 导入评估函数
from utils.eval_utils import evaluate_model # 导入评估函数

# 导入配置工具
from utils.config_utils import parse_args_and_config

# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 训练主函数
def train(cfg): # 现在函数接收一个配置对象
    set_seed(cfg.training.seed)
    # 确保设备是可用的 CUDA 或 CPU
    device = torch.device(cfg.training.device if torch.cuda.is_available() and cfg.training.device == 'cuda' else "cpu")
    print(f"Using device: {device}")

    # 1. 加载模型
    # Generator input/output channels should match the data channels (original image channels)
    generator = Generator(
        in_channels=cfg.data.channels,
        out_channels=cfg.data.channels,
        epsilon=cfg.model.generator.epsilon,
        num_bottlenecks=cfg.model.generator.num_bottlenecks,
        base_channels=cfg.model.generator.base_channels
    ).to(device)

    # Discriminator input channels should match the data channels
    if cfg.model.discriminator.type == 'cnn':
        discriminator = SequenceDiscriminatorCNN(
            in_channels=cfg.data.channels,
            base_channels=cfg.model.discriminator.base_channels
        ).to(device)
        print("Using CNN Discriminator")
    elif cfg.model.discriminator.type == 'patchgan':
        discriminator = PatchDiscriminator3d(
            in_channels=cfg.data.channels,
            base_channels=cfg.model.discriminator.base_channels
        ).to(device)
        print("Using PatchGAN Discriminator")
    else:
        raise ValueError(f"Unknown discriminator type: {cfg.model.discriminator.type}")

    optimizer_G = optim.Adam(generator.parameters(), lr=cfg.training.lr_g, betas=(cfg.training.b1, cfg.training.b2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=cfg.training.lr_d, betas=(cfg.training.b1, cfg.training.b2))

    gan_losses = GANLosses(device=device, gan_loss_type=cfg.losses.gan_type)
    attack_losses = AttackLosses(
        device=device,
        attention_loss_weight=cfg.losses.attention_loss_weight,
        decision_loss_weight=cfg.losses.decision_loss_weight,
        topk_k=cfg.losses.topk_k
    )

    writer = SummaryWriter(log_dir=cfg.logging.log_dir)

    # 2. **数据加载 (使用真实的 GameVideoDataset)**
    print(f"Loading real data from video: {cfg.data.video_path}")
    print(f"Using level JSON: {cfg.data.level_json_path}")

    try:
        dataset = GameVideoDataset(
            video_path=cfg.data.video_path,
            level_json_path=cfg.data.level_json_path,
            sequence_length=cfg.data.sequence_length,
            target_height=cfg.data.height, # Pass target size
            target_width=cfg.data.width # Pass target size
        )
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            pin_memory=True
        )
        print(f"Loaded {len(dataset)} sequences from video.")
    except FileNotFoundError as e:
        print(f"Error loading dataset: {e}. Using mock data instead.")
        # Fallback to mock data if real data loading fails
        dataloader = create_mock_dataloader(
            batch_size=cfg.training.batch_size,
            num_samples=cfg.training.num_mock_samples, # Use num_mock_samples if defined
            sequence_length=cfg.data.sequence_length,
            channels=cfg.data.channels,
            height=cfg.data.height,
            width=cfg.data.width,
            num_workers=cfg.data.num_workers
        )
        print(f"Using mock data. Number of batches: {len(dataloader)}")


    # 3. **加载 ATN 模型 (使用真实的特征头)**
    print(f"Loading ATN model/feature head from: {cfg.model.atn.model_path}")
    # atn_utils.py 中的 load_atn_model 函数现在需要更多参数来实例化 ATN 模型
    # 这些参数应该来自 cfg.model.atn
    atn_model = load_atn_model(
        cfg.model.atn.model_path,
        device,
        in_channels=cfg.model.atn.atn_in_channels, # ATN expects its specific input channels
        sequence_length=cfg.model.atn.atn_sequence_length,
        height=cfg.model.atn.atn_height,
        width=cfg.model.atn.atn_width,
        num_heads=cfg.model.atn.atn_num_heads, # Pass num_heads
        load_feature_head_only=(cfg.training.train_stage == 1) # Add this line to control loading
    )
    atn_model.eval()
    for param in atn_model.parameters():
        param.requires_grad = False # Freeze ATN model parameters


    # 4. 掩码加载 (如果使用)
    decision_mask = None
    attention_mask = None
    if cfg.mask.use_region_mask:
        print("Warning: use_region_mask is enabled. You need to implement real mask loading/generation.")
        # TODO: 根据你的关卡和ATN的输出区域，加载或生成真实的决策图和注意力图掩码。
        # 这些掩码的形状应该与 decision_map (B, H, W) 和 attention_map (B, head, N, N) 匹配。
        # 暂时使用模拟的全1掩码，形状与配置中的 ATN 预期输出尺寸匹配
        mock_batch_size = cfg.training.batch_size
        # Decision map spatial size assumed to be smaller than input, e.g., ATN_H/4, ATN_W/4
        # This needs to match the actual output of your ATN model's decision layer
        mock_decision_h = cfg.model.atn.atn_height # Assuming for mock decision_map size matches input for now
        mock_decision_w = cfg.model.atn.atn_width # Assuming for mock

        mock_attention_n = cfg.model.atn.atn_sequence_length # Attention sequence length
        mock_num_heads = cfg.model.atn.atn_num_heads # Attention heads

        decision_mask = torch.ones(mock_batch_size, mock_decision_h, mock_decision_w, device=device)
        attention_mask = torch.ones(mock_batch_size, mock_num_heads, mock_attention_n, mock_attention_n, device=device)
        # TODO: 如果需要测试局部攻击，可以修改模拟掩码
        # decision_mask[:, 10:20, 30:40] = 0 # Example: set part of decision mask to zero


    # 5. 加载检查点 (如果存在)
    start_epoch = 0
    if cfg.training.resume_checkpoint:
        checkpoint_path = cfg.training.resume_checkpoint
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            generator.load_state_dict(checkpoint['generator_state_dict'])
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming training from epoch {start_epoch}")
        else:
            print(f"Checkpoint not found at {checkpoint_path}. Starting from epoch 0.")


    print("Starting training...")
    global_step = 0

    # 训练主循环
    for epoch in range(start_epoch, cfg.training.num_epochs):
        generator.train() # 设置生成器为训练模式
        discriminator.train() # 设置判别器为训练模式

        for i, batch_data in enumerate(dataloader):
            # dataloader returns (C, T, H, W) for each sample, DataLoader stacks them into (B, C, T, H, W)
            # If GameVideoDataset returned masks, batch_data would be a tuple
            # Assuming batch_data is just the features for now
            real_x = batch_data.to(device) # real_x shape: (B, C, T, H, W)

            # Permute input to match Generator/Discriminator expected shape (B, C, N, W, H)
            # Note: This might be a discrepancy between how data is loaded and models are defined.
            # Generator CNN forward: (B, 128, N, W, H)
            # Discriminator CNN forward: (B, in_channels, N, H, W)
            # Discriminator PatchGAN forward: (B, C, N, H, W)
            # Let's assume the models expect (B, C, N, W, H) where N is sequence_length
            # And the dataloader output (B, C, T, H, W) has T = N
            # So, the data should be permuted from (B, C, T, H, W) to (B, C, N, H, W) which is the same shape. No permute needed if T==N.
            # If dataloader output is (B, C, T, H, W), and models expect (B, C, N, W, H),
            # and T == N, then we need to permute H and W. Let's double check model definitions.
            # Generator forward: x shape (B, 128, N, W, H) - conv3d kernel (1, 4, 4) stride (1, 2, 2) padding (0, 1, 1)
            # This means Generator expects (B, C, N, W, H) -> C: channels, N: sequence_length, W: width, H: height
            # Data loader output is (B, C, T, H, W). Assuming T=N, we need to swap H and W.
            # Let's fix the dataset or permute here. Permuting here is safer for now.
            real_x = real_x.permute(0, 1, 2, 4, 3) # (B, C, T, H, W) -> (B, C, T, W, H) to match Generator/Discriminator expectation

            # TODO: If using masks, ensure masks are on the correct device and have the correct shape (B, ...)
            # if decision_mask is not None: decision_mask = decision_mask.to(device)
            # if attention_mask is not None: attention_mask = attention_mask.to(device)


            #############################
            # Train Discriminator
            #############################
            optimizer_D.zero_grad()

            # Get ATN outputs for real data (need features for Discriminator)
            original_atn_outputs = get_atn_outputs(
                 atn_model,
                 real_x,
                 return_features=True, # Always need features for D
                 return_decision=(cfg.training.train_stage == 2), # Only need decision/attention in stage 2
                 return_attention=(cfg.training.train_stage == 2)
             )
            original_features = original_atn_outputs.get('features')

            if original_features is None:
                print(f"Warning: Original features are None in batch {i}. Skipping discriminator training for this batch.")
                continue # Skip if features are not available (e.g., mock ATN issue or stage 1 loading)

            # Train with real data (using original features as real samples for D)
            # Discriminator expects features, assuming shape (B, C_feat, N, W_feat, H_feat) matching Generator output
            # Need to ensure original_features has the correct shape for the discriminator
            # Current mock ATN returns (B, 128, T, H_feat, W_feat) - check discriminator input expectation again
            # CNN Discriminator expects (B, in_channels, N, H, W) - let's assume in_channels for D is 128 and N, H, W match feature shape
            # PatchGAN Discriminator expects (B, C, N, H, W) - C=128, N, H, W match feature shape
            # So original_features from ATN output should be (B, 128, T, H_feat, W_feat). We need to match model's spatial W, H.
            # The Generator output is (B, 128, N, W, H). Let's assume D expects (B, 128, N, W, H) as well.
            # The features output from mock ATN has shape (B, 128, T_feat, H_feat, W_feat).
            # T_feat should be the same as N (sequence length). H_feat, W_feat are reduced spatial dims.
            # Let's assume Discriminator takes features (B, 128, N, H_feat, W_feat)

            # Resize original_features to match the spatial dimensions expected by the Discriminator if necessary
            # This step might be complex and depends on actual ATN feature dimensions and Discriminator input size
            # For now, assuming original_features spatial dims (H_feat, W_feat) match Discriminator expectation
            # If mismatch happens, interpolation might be needed here:
            # target_h_feat = ... # Discriminator expected height
            # target_w_feat = ... # Discriminator expected width
            # original_features_resized = F.interpolate(original_features, size=(original_features.shape[2], target_h_feat, target_w_feat), mode='trilinear', align_corners=False)
            # Let's proceed assuming shapes are compatible based on mock ATN and Generator output shape

            real_D_output = discriminator(original_features.detach()) # Discriminator input is original features
            # The Discriminator loss calculation depends on the output shape (scalar or PatchGAN map)
            # gan_losses.discriminator_loss is designed to handle this.
            loss_D_real = gan_losses.discriminator_loss_real(real_D_output)

            # Generate fake data (adversarial features)
            # Generator takes real_x (B, C, N, W, H) and outputs delta (B, C, N, W, H)
            # The Generator output delta is in the original input space (e.g., image space)
            # adversarial_x = real_x + delta
            # Clamp adversarial_x
            # adversarial_x = torch.clamp(adversarial_x, 0, 1) # Moved clamping before getting ATN outputs for adv_x

            # Get ATN outputs for adversarial data (need features for Discriminator)
            adversarial_atn_outputs = get_atn_outputs(
                 atn_model,
                 real_x, # ATN takes real_x in original input space
                 return_features=True,
                 return_decision=(cfg.training.train_stage == 2),
                 return_attention=(cfg.training.train_stage == 2)
             )
            adversarial_features = adversarial_atn_outputs.get('features')

            if adversarial_features is None:
                 print(f"Warning: Adversarial features are None in batch {i}. Skipping discriminator training for this batch.")
                 continue

            # Train with fake data (using adversarial features as fake samples for D)
            # Discriminator expects features, shape (B, 128, N, H_feat, W_feat)
            fake_D_output = discriminator(adversarial_features.detach()) # Discriminator input is adversarial features
            loss_D_fake = gan_losses.discriminator_loss_fake(fake_D_output)

            # Total discriminator loss
            loss_D = loss_D_real + loss_D_fake # Assuming gan_losses.discriminator_loss_real/fake return scalars

            loss_D.backward()
            optimizer_D.step()

            #############################
            # Train Generator
            #############################
            # Generator aims to fool Discriminator AND perturb ATN outputs
            optimizer_G.zero_grad()

            # Generator should treat Discriminator's perception of adversarial features as real
            # Get Discriminator output for adversarial features again (but this time allow gradients to flow to G)
            # Discriminator expects features, shape (B, 128, N, H_feat, W_feat)
            fake_D_output_for_G_train = discriminator(adversarial_features) # Input is adversarial features, created by G

            # GAN loss for Generator (Generator wants D to output 1 for fake data)
            loss_G_gan = gan_losses.generator_loss(fake_D_output_for_G_train)

            # Attack loss for Generator (Generator wants to perturb ATN outputs)
            # Calculate attack loss based on the current stage and requested outputs
            # original_atn_outputs and adversarial_atn_outputs already contain the outputs based on stage
            loss_G_attack = attack_losses.get_generator_attack_loss(
                original_atn_outputs, # Pass entire output dicts
                adversarial_atn_outputs,
                decision_loss_type=cfg.losses.decision_loss_type,
                attention_loss_type=cfg.losses.attention_loss_type,
                decision_region_mask=decision_mask, # Pass masks
                attention_region_mask=attention_mask,
                train_stage=cfg.training.train_stage # Pass current stage
            )

            # Regularization losses
            loss_G_reg_l2 = cfg.regularization.lambda_l2 * l2_penalty(generator.parameters()) # L2 penalty on Generator weights
            # TV loss is on the delta output, not generator parameters
            # delta shape: (B, C, N, W, H)
            loss_G_reg_tv = cfg.regularization.lambda_tv * total_variation_loss(delta)

            # Total Generator loss
            # The attack loss is already negative if it's a distance-based loss we want to maximize.
            # So we add it directly.
            loss_G = loss_G_gan + loss_G_attack + loss_G_reg_l2 + loss_G_reg_tv

            loss_G.backward()
            optimizer_G.step()

            #############################
            # Logging and Visualization
            #############################
            if global_step % cfg.logging.log_interval == 0:
                writer.add_scalar('Loss/Generator_GAN', loss_G_gan.item(), global_step)
                writer.add_scalar('Loss/Generator_Attack', loss_G_attack.item(), global_step)
                writer.add_scalar('Loss/Generator_Regularization_L2', loss_G_reg_l2.item(), global_step)
                writer.add_scalar('Loss/Generator_Regularization_TV', loss_G_reg_tv.item(), global_step)
                writer.add_scalar('Loss/Generator_Total', loss_G.item(), global_step)
                writer.add_scalar('Loss/Discriminator_Real', loss_D_real.item(), global_step)
                writer.add_scalar('Loss/Discriminator_Fake', loss_D_fake.item(), global_step)
                writer.add_scalar('Loss/Discriminator_Total', loss_D.item(), global_step)

                # Log perturbation norms (calculate on delta)
                if delta is not None and delta.numel() > 0:
                    with torch.no_grad():
                         # linf_norm and l2_norm expect (B, C, N, W, H)
                         writer.add_scalar('Perturbation/Linf_Norm', linf_norm(delta).mean().item(), global_step)
                         writer.add_scalar('Perturbation/L2_Norm', l2_norm(delta).mean().item(), global_step)

                # Log Discriminator scores (real and fake)
                with torch.no_grad():
                    # Assumes D_real_output and fake_D_output_for_G_train are available and represent scores
                    # If PatchGAN, need to average across patch dimensions
                    real_score = real_D_output.mean() if real_D_output.ndim == 0 else real_D_output.view(real_D_output.size(0), -1).mean(dim=-1).mean().item()
                    fake_score = fake_D_output_for_G_train.mean() if fake_D_output_for_G_train.ndim == 0 else fake_D_output_for_G_train.view(fake_D_output_for_G_train.size(0), -1).mean(dim=-1).mean().item()
                    writer.add_scalar('Discriminator_Scores/Real', real_score, global_step)
                    writer.add_scalar('Discriminator_Scores/Fake', fake_score, global_step)


                print(f"Epoch [{epoch}/{cfg.training.num_epochs}], Step [{global_step}], ")
                print(f"  D Loss: {loss_D.item():.4f} (Real: {loss_D_real.item():.4f}, Fake: {loss_D_fake.item():.4f})")
                print(f"  G Loss: {loss_G.item():.4f} (GAN: {loss_G_gan.item():.4f}, Attack: {loss_G_attack.item():.4f}, Reg_L2: {loss_G_reg_l2.item():.4f}, Reg_TV: {loss_G_reg_tv.item():.4f})")



            if global_step % cfg.logging.vis_interval == 0:
                 # Visualize samples and outputs (pass outputs based on stage)
                         visualize_samples_and_outputs(
                             writer,
                     original_atn_outputs.get('features'), # Always pass features for vis if available
                     delta, # Pass delta for visualization
                     adversarial_atn_outputs.get('features'), # Always pass features for vis if available
                     original_atn_outputs.get('decision'), # Pass decision map if available (stage 2)
                     adversarial_atn_outputs.get('decision'), # Pass decision map if available (stage 2)
                     global_step,
                     num_samples=cfg.logging.num_vis_samples,
                     sequence_step_to_vis=cfg.logging.sequence_step_to_vis
                 )

                 # Visualize attention maps if available (stage 2)
            if cfg.training.train_stage == 2:
                        visualize_attention_maps(
                            writer,
                          original_atn_outputs.get('attention'), # Pass attention matrix if available
                          adversarial_atn_outputs.get('attention'), # Pass attention matrix if available
                          global_step,
                          num_samples=cfg.logging.num_vis_samples, # Or a smaller number for attention maps
                          num_heads_to_vis=cfg.logging.num_vis_heads # Visualize fewer heads if many exist
                      )


            global_step += 1

        # End of Epoch Evaluation and Checkpoint Saving
        if (epoch + 1) % cfg.training.eval_interval == 0:
            print(f"\nEvaluating at end of epoch {epoch+1}...")
            eval_results = evaluate_model(
                generator,
                discriminator, # Pass discriminator for GAN metrics if needed in eval
                atn_model,
                dataloader, # Use training dataloader for simplicity in eval_utils example, use a separate eval_dataloader in practice
                device,
                cfg, # Pass config
                current_train_stage=cfg.training.train_stage # Pass current stage to evaluation
            )
            print("--- Evaluation Results ---")
            for metric_name, value in eval_results.items():
                 writer.add_scalar(f'Evaluation/{metric_name}', value, global_step)
                 print(f"  {metric_name}: {value:.4f}")
            print("------------------------")


        # Save checkpoint
        if (epoch + 1) % cfg.training.checkpoint_interval == 0 or (epoch + 1) == cfg.training.num_epochs:
            checkpoint_dir = cfg.logging.checkpoint_dir
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}_stage_{cfg.training.train_stage}.pth')
            print(f"Saving checkpoint to {checkpoint_path}")
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'cfg': dict(cfg) # Save config as dictionary
            }, checkpoint_path)

    writer.close()
    print("Training finished.")

if __name__ == "__main__":
    # Define default config path
    DEFAULT_CONFIG_PATH = 'configs/default_config.yaml' # Ensure this is the path to your default config

    # Parse command line arguments and load/merge config
    # Use 'config' as the argument name for task-specific config file
    # Add specific command line arguments you want to override from the config
    parser = argparse.ArgumentParser(description="SeqAdvGAN Training Script")
    parser.add_argument('--config', type=str, help='Path to the task-specific YAML configuration file.')
    parser.add_argument('--train_stage', type=int, choices=[1, 2], help='Training stage (1 or 2).')
    parser.add_argument('--num_epochs', type=int, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, help='Batch size.')
    parser.add_argument('--lr_g', type=float, help='Generator learning rate.')
    parser.add_argument('--lr_d', type=float, help='Discriminator learning rate.')
    parser.add_argument('--model.atn.model_path', type=str, help='Path to the ATN model or feature head weights.')
    parser.add_argument('--logging.log_dir', type=str, help='TensorBoard log directory.')
    parser.add_argument('--logging.checkpoint_dir', type=str, help='Model checkpoint directory.')
    parser.add_argument('--training.resume_checkpoint', type=str, help='Path to a checkpoint to resume training from.')

    cmd_args, unknown = parser.parse_known_args()

    # Load configuration using the config utility
    cfg = parse_args_and_config(DEFAULT_CONFIG_PATH, task_config_arg='config')

    if cfg is not None:
        # Print final configuration for confirmation
        import json
        print("--- Final Configuration ---")
        print(json.dumps(dict(cfg), indent=4))
        print("--------------------------")

        # Start training
        train(cfg)

    else:
        print("Failed to load configuration. Exiting.") 