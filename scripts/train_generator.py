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
from utils.data_utils import GameVideoDataset, create_mock_dataloader, worker_init_fn # Import GameVideoDataset and worker_init_fn

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
def train(cfg):
    set_seed(cfg.training.seed)
    # 确保设备是可用的 CUDA 或 CPU
    device = torch.device(cfg.training.device if torch.cuda.is_available() and cfg.training.device == 'cuda' else "cpu")
    print(f"Using device: {device}")

    # 1. 加载模型
    # Generator/Discriminator input/output channels are for FEATURES, not original images
    # According to IrisXeonNet.py, features are 128 channels
    feature_channels = 128 # Assuming IrisXeonNet outputs 128 channels

    # Generator input/output dimensions should match the *feature* dimensions
    # The spatial dimensions and sequence length depend on the ATN output shape
    # We should determine these from the ATN model or configuration.
    # For now, let's use placeholder dimensions that will need adjustment based on actual ATN output.
    # Assuming ATN outputs features of shape (B, 128, N_feat, H_feat, W_feat)
    # The Generator should operate on this shape.
    # The Generator CNN definition (kernel sizes, strides) implies downsampling spatial dims.
    # Let's assume the Generator is designed to take features directly.
    # Its __init__ uses `in_channels`, `out_channels`, `num_bottlenecks`, `base_channels`.
    # The spatial/sequence dimensions are handled implicitly by the Conv3d layers.
    # Its `forward` expects (B, C, N, W, H).

    generator = Generator(
        in_channels=feature_channels, # Generator input is features (128)
        out_channels=feature_channels, # Generator output is feature perturbation (128)
        epsilon=cfg.model.generator.epsilon,
        num_bottlenecks=cfg.model.generator.num_bottlenecks,
        base_channels=cfg.model.generator.base_channels
    ).to(device)

    # Discriminator input channels should match the feature channels
    if cfg.model.discriminator.type == 'cnn':
        discriminator = SequenceDiscriminatorCNN(
            in_channels=feature_channels, # Discriminator input is features (128)
            base_channels=cfg.model.discriminator.base_channels
        ).to(device)
        print("Using CNN Discriminator")
    elif cfg.model.discriminator.type == 'patchgan':
        discriminator = PatchDiscriminator3d(
            in_channels=feature_channels, # Discriminator input is features (128)
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

    # 2. 数据加载
    print(f"Loading real data from video: {cfg.data.video_path}")
    print(f"Using level JSON: {cfg.data.level_json_path}")

    try:
        dataset = GameVideoDataset(
            video_path=cfg.data.video_path,
            level_json_path=cfg.data.level_json_path,
            sequence_length=cfg.data.sequence_length,
            target_height=cfg.data.height,
            target_width=cfg.data.width
        )
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn # Add worker_init_fn
        )
        print(f"Loaded {len(dataset)} sequences from video.")
    except FileNotFoundError as e:
        print(f"Error loading dataset: {e}. Using mock data instead.")
        dataloader = create_mock_dataloader(
            batch_size=cfg.training.batch_size,
            num_samples=cfg.training.num_mock_samples,
            sequence_length=cfg.data.sequence_length,
            channels=cfg.data.channels,
            height=cfg.data.height,
            width=cfg.data.width,
            num_workers=cfg.data.num_workers
        )
        print(f"Using mock data. Number of batches: {len(dataloader)}")


    # 3. 加载 ATN 模型 (IrisXeonNet)
    print(f"Loading ATN model/feature head from: {cfg.model.atn.model_path}")
    # load_atn_model now handles instantiation of IrisXeonNet and loading weights
    # It assumes the weight file contains IrisXeonNet weights
    # ATN takes original image dimensions as input parameters
    atn_model = load_atn_model(
        cfg.model.atn.model_path,
        device,
        in_channels=cfg.data.channels, # ATN input is original image channels (3)
        sequence_length=cfg.data.sequence_length, # ATN input sequence length
        height=cfg.data.height, # ATN input height
        width=cfg.data.width, # ATN input width
        num_heads=cfg.model.atn.atn_num_heads, # Pass num_heads
        load_feature_head_only=(cfg.training.train_stage == 1) # This parameter might not be relevant for IrisXeonNet loading
    )

    if atn_model is None:
        print("Failed to load ATN model. Exiting training.")
        return

    atn_model.eval()
    print("Debug: Before freezing ATN model parameters.") # Debug print before freezing
    for param in atn_model.parameters():
        param.requires_grad = False # Freeze ATN model parameters
    print("Debug: After freezing ATN model parameters.") # Debug print after freezing


    # 4. 掩码加载 (如果使用) - 注意：掩码当前仅用于损失计算，不影响数据流
    # 对于阶段一的特征攻击，区域掩码可能不直接适用，除非你在特征空间定义掩码区域
    # 如果 cfg.mask.use_region_mask 为 True，你需要实现特征空间的掩码逻辑
    decision_mask = None
    attention_mask = None
    if cfg.mask.use_region_mask:
        print("Warning: use_region_mask is enabled. Implement feature-space mask loading/generation.")
        # TODO: Load or generate feature-space masks if needed for feature attack loss
        pass # Placeholder for mask loading


    # 5. 加载检查点 (如果存在)
    start_epoch = 0
    if cfg.training.resume_checkpoint:
        checkpoint_path = cfg.training.resume_checkpoint
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
        generator.train()
        discriminator.train()

        for i, batch_data in enumerate(dataloader):
            # batch_data shape: (B, C, T, H, W) - original image data
            real_x = batch_data.to(device) # Real image data
            print(f"Debug: Epoch {epoch}, Batch {i} - Loaded real_x, shape: {real_x.shape}") # Debug print after loading batch

            # --- Stage 1 Training Logic (Feature Attack) ---

            # Debug print before calling get_atn_outputs
            print(f"Debug: Epoch {epoch}, Batch {i} - Before get_atn_outputs, real_x shape: {real_x.shape}")

            # 1. Get original features from ATN (ATN takes original image data)
            with torch.no_grad():
                 original_atn_outputs = get_atn_outputs(
                      atn_model,
                      real_x, # Pass original image data (B, C, T, H, W)
                      return_features=True,
                      return_decision=False,
                      return_attention=False
                  )
            original_features = original_atn_outputs.get('features') # Shape: (B, 128, N_feat, H_feat, W_feat) approx

            # Debug print after calling get_atn_outputs
            if original_features is not None:
                print(f"Debug: Epoch {epoch}, Batch {i} - After get_atn_outputs, original_features shape: {original_features.shape}")
            else:
                print(f"Debug: Epoch {epoch}, Batch {i} - original_features is None. Skipping batch.")
                continue # Skip if features are None

            # Ensure original_features requires grad for Generator training
            original_features_for_gen = original_features.clone().detach().requires_grad_(True)

            # 2. Generator generates delta in the feature space
            # Debug print before calling generator
            print(f"Debug: Epoch {epoch}, Batch {i} - Before generator forward, original_features_for_gen shape: {original_features_for_gen.shape}")
            delta = generator(original_features_for_gen)
            print(f"Debug: Epoch {epoch}, Batch {i} - After generator forward, delta shape: {delta.shape}")

            # 3. Calculate adversarial features
            adversarial_features = original_features_for_gen + delta
            print(f"Debug: Epoch {epoch}, Batch {i} - Calculated adversarial_features, shape: {adversarial_features.shape}")

            # For Discriminator training, we need to detach features
            original_features_detached = original_features.detach()
            adversarial_features_detached = adversarial_features.detach()
            print(f"Debug: Epoch {epoch}, Batch {i} - Detached features for Discriminator training.")

            #############################
            # Train Discriminator (on features)
            #############################
            optimizer_D.zero_grad()
            print(f"Debug: Epoch {epoch}, Batch {i} - Discriminator optimizer zero_grad.")

            # Debug print before Discriminator forward pass (real)
            print(f"Debug: Epoch {epoch}, Batch {i} - Before Discriminator forward (real), input shape: {original_features_detached.shape}")
            real_D_output = discriminator(original_features_detached)
            print(f"Debug: Epoch {epoch}, Batch {i} - After Discriminator forward (real), output shape: {real_D_output.shape}")

            # Debug print before Discriminator forward pass (fake)
            print(f"Debug: Epoch {epoch}, Batch {i} - Before Discriminator forward (fake), input shape: {adversarial_features_detached.shape}")
            fake_D_output = discriminator(adversarial_features_detached) # Discriminator input is adversarial features
            print(f"Debug: Epoch {epoch}, Batch {i} - After Discriminator forward (fake), output shape: {fake_D_output.shape}")

            # 计算判别器损失
            # discriminator_loss 方法现在返回总损失、真实损失和伪造损失
            loss_D, loss_D_real, loss_D_fake = gan_losses.discriminator_loss(real_D_output, fake_D_output)
            print(f"Debug: Epoch {epoch}, Batch {i} - Calculated loss_D: {loss_D.item():.4f} (Real: {loss_D_real.item():.4f}, Fake: {loss_D_fake.item():.4f})")

            loss_D.backward()
            print(f"Debug: Epoch {epoch}, Batch {i} - Discriminator backward pass completed.")
            optimizer_D.step()
            print(f"Debug: Epoch {epoch}, Batch {i} - Discriminator optimizer step completed.")

            #############################
            # Train Generator (on features)
            #############################
            optimizer_G.zero_grad()
            print(f"Debug: Epoch {epoch}, Batch {i} - Generator optimizer zero_grad.")

            # Debug print before Generator forward pass (for GAN loss)
            print(f"Debug: Epoch {epoch}, Batch {i} - Before Discriminator forward (for G train), input shape: {adversarial_features.shape}")
            fake_D_output_for_G_train = discriminator(adversarial_features)
            print(f"Debug: Epoch {epoch}, Batch {i} - After Discriminator forward (for G train), output shape: {fake_D_output_for_G_train.shape}")

            # GAN loss for Generator (Generator wants D to output 1 for fake data)
            loss_G_gan = gan_losses.generator_loss(fake_D_output_for_G_train)
            print(f"Debug: Epoch {epoch}, Batch {i} - Calculated loss_G_gan: {loss_G_gan.item():.4f}")

            # Attack loss for Generator (Generator wants to perturb ATN outputs)
            # In Stage 1, attack is on features. attack_losses.get_generator_attack_loss handles this.
            # Debug print before calculating attack loss
            print(f"Debug: Epoch {epoch}, Batch {i} - Before calculating loss_G_attack.")
            loss_G_attack = attack_losses.get_generator_attack_loss(
                {'features': original_features_detached}, # Pass original features (detached for loss calculation)
                {'features': adversarial_features}, # Pass adversarial features (requires grad)
                decision_loss_type=cfg.losses.decision_loss_type, # This will be used as feature loss type in Stage 1
                attention_loss_type=cfg.losses.attention_loss_type, # Ignored in Stage 1
                decision_region_mask=decision_mask, # May or may not be used depending on feature attack logic
                attention_region_mask=attention_mask, # Ignored in Stage 1
                train_stage=cfg.training.train_stage # Pass current stage (1)
            )
            print(f"Debug: Epoch {epoch}, Batch {i} - Calculated loss_G_attack (Feature): {loss_G_attack.item():.4f}")

            # Regularization losses
            l2_reg = sum(p.pow(2).sum() for p in generator.parameters() if p.requires_grad)
            loss_G_reg_l2 = cfg.regularization.lambda_l2_penalty * l2_reg
            print(f"Debug: Epoch {epoch}, Batch {i} - Calculated loss_G_reg_l2: {loss_G_reg_l2.item():.4f}")

            # TV loss is on the delta output, which is in feature space
            # delta shape: (B, 128, N_feat, W_feat, H_feat)
            # Debug print before calculating TV loss
            print(f"Debug: Epoch {epoch}, Batch {i} - Before calculating loss_G_reg_tv, delta shape: {delta.shape}")
            loss_G_reg_tv = cfg.regularization.lambda_tv * total_variation_loss(delta)
            print(f"Debug: Epoch {epoch}, Batch {i} - Calculated loss_G_reg_tv: {loss_G_reg_tv.item():.4f}")


            # Total Generator loss
            # Read gan_loss_weight from config
            gan_loss_weight = cfg.losses.get('gan_loss_weight', 1.0) # Default to 1.0 if not in config
            print(f"Debug: Epoch {epoch}, Batch {i} - Using gan_loss_weight: {gan_loss_weight}") # Debug print

            # Use the gan_loss_weight in total loss calculation
            loss_G = gan_loss_weight * loss_G_gan + loss_G_attack + loss_G_reg_l2 + loss_G_reg_tv
            print(f"Debug: Epoch {epoch}, Batch {i} - Calculated total loss_G: {loss_G.item():.4f}")

            loss_G.backward()
            print(f"Debug: Epoch {epoch}, Batch {i} - Generator backward pass completed.")
            optimizer_G.step()
            print(f"Debug: Epoch {epoch}, Batch {i} - Generator optimizer step completed.")

            #############################
            # Logging and Visualization
            #############################
            if global_step % cfg.logging.log_interval == 0:
                print(f"Debug: Epoch {epoch}, Batch {i} - Logging metrics...")
                writer.add_scalar('Loss/Generator_GAN', loss_G_gan.item(), global_step)
                writer.add_scalar('Loss/Generator_Attack (Feature)', loss_G_attack.item(), global_step) # Log as Feature Attack
                writer.add_scalar('Loss/Generator_Regularization_L2', loss_G_reg_l2.item(), global_step)
                writer.add_scalar('Loss/Generator_Regularization_TV', loss_G_reg_tv.item(), global_step)
                writer.add_scalar('Loss/Generator_Total', loss_G.item(), global_step)
                writer.add_scalar('Loss/Discriminator_Real', loss_D_real.item(), global_step)
                writer.add_scalar('Loss/Discriminator_Fake', loss_D_fake.item(), global_step)
                writer.add_scalar('Loss/Discriminator_Total', loss_D.item(), global_step)

                # Log perturbation norms (calculate on delta in feature space)
                if delta is not None and delta.numel() > 0:
                    with torch.no_grad():
                         # linf_norm and l2_norm expect (B, C, N, W, H) or (B, C, N, H, W)
                         # delta shape is (B, 128, N_feat, W_feat, H_feat) approx, match expected shape
                         # Ensure delta shape is (B, C, N, H, W) for norm calculation
                         # Based on Generator forward, delta shape is (B, 128, N, W, H)
                         # Need to permute for norm functions that expect (B, C, N, H, W)
                         delta_for_norm = delta.permute(0, 1, 2, 4, 3) # (B, 128, N, W, H) -> (B, 128, N, H, W)
                         writer.add_scalar('Perturbation/Feature_Linf_Norm', linf_norm(delta_for_norm).mean().item(), global_step) # Log as Feature Norm
                         writer.add_scalar('Perturbation/Feature_L2_Norm', l2_norm(delta_for_norm).mean().item(), global_step) # Log as Feature Norm


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
                 print(f"Debug: Epoch {epoch}, Batch {i} - Visualizing samples...")
                 # Visualize samples and outputs
                 # Pass original image, original features, feature delta, and adversarial features
                 print(f"Debug: visualize_samples_and_outputs inputs - real_x shape: {real_x.shape}, numel: {real_x.numel()}") # Debug print
                 print(f"Debug: visualize_samples_and_outputs inputs - original_features shape: {original_features.shape if original_features is not None else 'None'}, numel: {original_features.numel() if original_features is not None else 0}") # Debug print
                 print(f"Debug: visualize_samples_and_outputs inputs - delta shape: {delta.shape}, numel: {delta.numel()}") # Debug print
                 print(f"Debug: visualize_samples_and_outputs inputs - adversarial_features shape: {adversarial_features.shape}, numel: {adversarial_features.numel()}") # Debug print
                 visualize_samples_and_outputs(
                             writer,
                     real_x.detach().cpu(),         # original_image
                     original_features.detach().cpu(), # <-- Pass original_features here
                     delta.detach().cpu(),          # feature_delta
                     adversarial_features.detach().cpu(), # adversarial_features
                     None,                          # original_decision_map
                     None,                          # adversarial_decision_map
                     global_step,
                     num_samples=cfg.logging.num_vis_samples,
                     sequence_step_to_vis=cfg.logging.sequence_step_to_vis
                 )
                 print(f"Debug: Epoch {epoch}, Batch {i} - Sample visualization completed.")

                 # Visualize attention maps (None in Stage 1)
                 # The visualize_attention_maps function should already handle train_stage == 2 internally
                 # It will check if attention tensors are None.
                 print(f"Debug: Epoch {epoch}, Batch {i} - Visualizing attention maps (if available)...")
                 visualize_attention_maps(
                            writer,
                     None, # Original attention matrix (None in Stage 1)
                     None, # Adversarial attention matrix (None in Stage 1)
                          global_step,
                     num_samples=cfg.logging.num_vis_samples,
                     num_heads_to_vis=cfg.logging.num_vis_heads
                      )
                 print(f"Debug: Epoch {epoch}, Batch {i} - Attention map visualization completed.")


            global_step += 1
            print(f"DEBUG_STEP_CHECK: global_step is now {global_step}") # Add this line
            print(f"Debug: Epoch {epoch}, Batch {i} - Batch processing completed. Global step: {global_step}")

        # End of Epoch Evaluation and Checkpoint Saving
        print(f"Debug: Epoch {epoch} - End of epoch, checking evaluation and checkpoint intervals.")
        if (epoch + 1) % cfg.training.eval_interval == 0:
                print(f"\nEvaluating at end of epoch {epoch+1}...")
                # Evaluation logic might need adjustment for Stage 1
                # evaluate_model expects original_atn_outputs, adversarial_atn_outputs
                # In stage 1, these will only contain 'features'
                # eval_utils needs to handle evaluation based only on features for Stage 1
                eval_results = evaluate_model(
                    generator,
                    discriminator,
                    atn_model, # Pass ATN model to get features during evaluation
                    dataloader, # Use training dataloader for simplicity in eval_utils example, use a separate eval_dataloader in practice
                    device,
                    cfg,
                    current_train_stage=cfg.training.train_stage
                )
                print("--- Evaluation Results ---")
                for metric_name, value in eval_results.items():
                     writer.add_scalar(f'Evaluation/{metric_name}', value, global_step)
                     print(f"  {metric_name}: {value:.4f}")
                print("------------------------")


        # Save checkpoint
        if (epoch + 1) % cfg.logging.save_interval == 0 or (epoch + 1) == cfg.training.num_epochs:
                print(f"Debug: Epoch {epoch} - Saving checkpoint...")
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
                    'cfg': dict(cfg)
                }, checkpoint_path)
                print(f"Debug: Epoch {epoch} - Checkpoint saved.")

    writer.close()
    print("Training finished.")

if __name__ == "__main__":
    # Define default config path
    DEFAULT_CONFIG_PATH = 'configs/default_config.yaml' # Ensure this is the path to your default config

    # Parse command line arguments and load/merge config
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
    parser.add_argument('--model.generator.in_channels', type=int, help='Generator input channels (should match feature channels)')
    parser.add_argument('--model.generator.out_channels', type=int, help='Generator output channels (should match feature channels)')
    parser.add_argument('--model.discriminator.in_channels', type=int, help='Discriminator input channels (should match feature channels)')
    parser.add_argument('--data.channels', type=int, help='Data channels (original image channels)')
    parser.add_argument('--data.sequence_length', type=int, help='Data sequence length')
    parser.add_argument('--data.height', type=int, help='Data height')
    parser.add_argument('--data.width', type=int, help='Data width')
    parser.add_argument('--model.atn.atn_in_channels', type=int, help='ATN expected input channels')
    parser.add_argument('--model.atn.atn_sequence_length', type=int, help='ATN expected input sequence length')
    parser.add_argument('--model.atn.atn_height', type=int, help='ATN expected input height')
    parser.add_argument('--model.atn.atn_width', type=int, help='ATN expected input width')
    parser.add_argument('--model.atn.atn_num_heads', type=int, help='ATN number of attention heads')
    parser.add_argument('--losses.decision_loss_type', type=str, help='Type of decision loss')
    parser.add_argument('--losses.attention_loss_type', type=str, help='Type of attention loss')
    parser.add_argument('--losses.topk_k', type=int, help='K for Top-K attention loss/eval')
    parser.add_argument('--regularization.lambda_l2_penalty', type=float, help='L2 penalty weight')
    parser.add_argument('--regularization.lambda_tv', type=float, help='TV loss weight')
    parser.add_argument('--mask.use_region_mask', action=argparse.BooleanOptionalAction, help='Use region masks')


    cmd_args, unknown = parser.parse_known_args()

    # Load configuration using the config utility
    # Pass cmd_args to parse_args_and_config for merging
    cfg = parse_args_and_config(DEFAULT_CONFIG_PATH, task_config_arg='config')

    if cfg is not None:
        # Ensure Generator/Discriminator channels are set based on expected feature channels (128)
        # Override config if not set, or if set incorrectly for feature attack
        # Use 128 as the feature channel size based on IrisXeonNet output.
        feature_channels = 128
        if not hasattr(cfg.model.generator, 'in_channels') or cfg.model.generator.in_channels != feature_channels:
            print(f"Setting Generator in_channels to {feature_channels} (feature channels).")
            cfg.model.generator.in_channels = feature_channels
        if not hasattr(cfg.model.generator, 'out_channels') or cfg.model.generator.out_channels != feature_channels:
            print(f"Setting Generator out_channels to {feature_channels} (feature channels).")
            cfg.model.generator.out_channels = feature_channels
        if not hasattr(cfg.model.discriminator, 'in_channels') or cfg.model.discriminator.in_channels != feature_channels:
             print(f"Setting Discriminator in_channels to {feature_channels} (feature channels).")
             cfg.model.discriminator.in_channels = feature_channels

        # Print final configuration for confirmation
        import json
        print("--- Final Configuration ---")
        # Convert EasyDict to dict for json dump
        print(json.dumps(dict(cfg), indent=4))
        print("--------------------------")

        # Start training
        train(cfg)

    else:
        print("Failed to load configuration. Exiting.") 