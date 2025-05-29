import torch
import torch.nn as nn
from typing import Dict, Optional, Any
import os
import traceback # Import traceback

# 导入实际使用的模型组件
try:
    # 确保这些导入路径正确，根据您的项目结构
    from IrisBabel.nn.Transformers import IrisAttentionTransformers
    from IrisBabel.nn.CNN import IrisTriggerNet
    # Import IrisXeonNet
    from IrisBabel.nn.CNN import IrisXeonNet # Import IrisXeonNet
    print("Successfully imported IrisAttentionTransformers, IrisTriggerNet, and IrisXeonNet.")
    # Placeholder classes in case import fails - allows graceful failure during development
    _AttentionTransformer = IrisAttentionTransformers
    _TriggerNet = IrisTriggerNet
    _XeonNet = IrisXeonNet # Placeholder for IrisXeonNet
except ImportError as e:
    print(f"FATAL: Error importing actual ATN model components: {e}")
    print("Please ensure the 'IrisBabel' directory and its submodules are correctly placed in the SeqAdvGAN project root and that necessary dependencies are installed.")
    # Define dummy classes to prevent NameError if import fails
    class _AttentionTransformer(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__(); print("Dummy AttentionTransformer used due to import error.")
        def forward(self, x): 
             # Simulate output shape (B, H', W', 128) after squeeze(1) based on test.py
             # Assuming input x is (B, 128, T, H', W') and T=1 after Perception Layer
             # The test.py logic with squeeze(1) suggests T=1
             # Need to know H', W' from IrisXeonNet output
             # Let's assume IrisXeonNet output shape is (B, 128, 1, 64, 64)
             # After AttentionTransformer, it might be (B, 1, 64*64, 128) -> squeeze(1) -> (B, 4096, 128)
             # This is complex, let's return a plausible shape that TriggerNet might accept
             # Based on test.py, AttentionTransformer output fed to TriggerNet seems to be (B, T, H, W, C)
             # If IrisXeonNet output is (B, 128, T, H', W')
             # AttentionTransformer input should be (B, 128, T, H', W')
             # AttentionTransformer output shape is unclear but used as second input to TriggerNet
             # TriggerNet first input is data (B, T, H, W, C)
             # Let's assume AttentionTransformer output matches TriggerNet second input shape (B, T, H', W', C)
             # This contradicts BatchNorm3d(128). This is still highly uncertain.
             # Let's simulate a feature shape that could be an intermediate output
             print("Warning: Dummy AttentionTransformer forward pass returning dummy features.")
             return torch.randn(*x.shape[:-3], 128, *x.shape[-2:], device=x.device) # Simulate (B, 128, T, H', W') output if input is (B, 128, T, H', W')

    class _TriggerNet(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__(); print("Dummy TriggerNet used due to import error.")
        def forward(self, x, weight_matrix): 
            # Simulate output shape based on test.py squeeze(0) -> (H', W', 8)
            # Assuming input x is (B, T, H, W, C) and weight_matrix is AttentionTransformer output
            # Let's assume TriggerNet output is (B, H', W', 8)
            print("Warning: Dummy TriggerNet forward pass returning dummy output.")
            # Need H', W' from somewhere, potentially config or derived from input x
            # Assuming x is (B, T, H, W, C), let's use H, W as H', W'
            return torch.randn(x.shape[0], x.shape[2], x.shape[3], 8, device=x.device) # Simulate (B, H, W, 8)

    class _XeonNet(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__(); print("Dummy IrisXeonNet used due to import error.")
        def forward(self, x): 
            # Simulate output shape (B, 128, T, H', W')
            # Input x is (B, 3, T, H, W)
            # Need to know the downsampling factor for H, W
            # Assuming 4x downsampling for simplicity, and T remains same
            print("Warning: Dummy IrisXeonNet forward pass returning dummy features.")
            return torch.randn(x.shape[0], 128, x.shape[2], x.shape[3]//4, x.shape[4]//4, device=x.device) # Simulate (B, 128, T, H/4, W/4)

# 删除或注释掉旧的 IrisBabelModel 导入
# try:
#     from IrisBabel.IrisBabelModel import IrisBabelModel
#     print("Successfully imported IrisBabelModel from IrisBabel.")
# except ImportError as e:
#     print(f"FATAL: Error importing IrisBabelModel: {e}")
#     print("Please ensure the 'IrisBabel' directory and its '__init__.py' files are correctly placed in the SeqAdvGAN project root.")
#     IrisBabelModel = None


# 更新 load_atn_model 函数，加载 AttentionTransformer, TriggerNet 和 IrisXeonNet
def load_atn_model(cfg: Any, device: torch.device) -> Optional[Dict[str, nn.Module]]:
    """
    加载并初始化 IrisXeonNet, IrisAttentionTransformers 和 IrisTriggerNet 模型。
    """
    # Check if all required classes were imported successfully
    if _XeonNet is None or _AttentionTransformer is None or _TriggerNet is None:
        print("Cannot load ATN models due to previous import errors.")
        return None
    
    # Initialize models
    print("Initializing IrisXeonNet, IrisAttentionTransformers and IrisTriggerNet models...")
    try:
        # Initialize IrisXeonNet
        # IrisXeonNet instantiation based on IrisBabel/nn/CNN/IrisXeonNet.py __init__
        # Seems to only have num_classes=1000 parameter, which might not be relevant here.
        # Let's instantiate without arguments for now, assuming default behavior.
        # Note: If IrisXeonNet __init__ requires image dimensions or other config, add them here.
        xeon_net = _XeonNet().to(device)
        print("IrisXeonNet initialized.")

        # Initialize IrisAttentionTransformers
        # Instantiation based on test.py: axis_transformers = IrisAttentionTransformers()
        # Seems to have optional inject_model parameter. Instantiate without it for now.
        # Note: If IrisAttentionTransformers __init__ requires feature dimensions or other config, add them here.
        attention_transformer_model = _AttentionTransformer().to(device)
        print("IrisAttentionTransformers initialized.")

        # Initialize IrisTriggerNet
        # Instantiation based on test.py: trigger_net = IrisTriggerNet()
        # Seems to have no parameters in __init__.
        # Note: If IrisTriggerNet __init__ requires feature/attention dimensions or other config, add them here.
        trigger_net_model = _TriggerNet().to(device)
        print("IrisTriggerNet initialized.")

    except Exception as e:
         print(f"Error initializing ATN model components: {e}")
         traceback.print_exc() # Print traceback for detailed error info
         return None


    # Load pre-trained weights
    print("Loading ATN model weights...")
    # From config, get weight paths
    # Need perception_layer_path, attention_transformer_path, and trigger_net_path
    if not hasattr(cfg.model.atn, 'perception_layer_path') or \
       not hasattr(cfg.model.atn, 'attention_transformer_path') or \
       not hasattr(cfg.model.atn, 'trigger_net_path'):
        print("Error: ATN model paths (perception_layer_path, attention_transformer_path, or trigger_net_path) are missing in the config.")
        return None

    perception_layer_path = cfg.model.atn.perception_layer_path
    attention_transformer_path = cfg.model.atn.attention_transformer_path
    trigger_net_path = cfg.model.atn.trigger_net_path

    # Load IrisXeonNet weights
    print(f"Loading IrisXeonNet weights from: {perception_layer_path}")
    if os.path.exists(perception_layer_path):
        try:
            # Loading weights, using weights_only=False and strict=False as seen in test.py
            xeon_net.load_state_dict(
                torch.load(perception_layer_path, map_location=device, weights_only=False),
                strict=False
            )
            print("IrisXeonNet weights loaded.")
        except Exception as e:
             print(f"Warning: Error loading IrisXeonNet weights from {perception_layer_path}: {e}. Model will use random initialization.")
             traceback.print_exc()
    else:
        print(f"Warning: IrisXeonNet weights not found at {perception_layer_path}. Model will use random initialization.")

    # Load AttentionTransformer weights
    print(f"Loading AttentionTransformer weights from: {attention_transformer_path}")
    if os.path.exists(attention_transformer_path):
        try:
            # Loading weights, using weights_only=False and strict=False as seen in test.py
            attention_transformer_model.load_state_dict(
                torch.load(attention_transformer_path, map_location=device, weights_only=False),
                strict=False # According to test.py loading
            )
            print("AttentionTransformer weights loaded.")
        except Exception as e:
             print(f"Warning: Error loading AttentionTransformer weights from {attention_transformer_path}: {e}. Model will use random initialization.")
             traceback.print_exc()
    else:
        print(f"Warning: AttentionTransformer weights not found at {attention_transformer_path}. Model will use random initialization.")

    # Load TriggerNet weights
    print(f"Loading TriggerNet weights from: {trigger_net_path}")
    if os.path.exists(trigger_net_path):
        try:
            # Loading weights, using weights_only=False and strict=False as seen in test.py
            trigger_net_model.load_state_dict(
                torch.load(trigger_net_path, map_location=device, weights_only=False),
                strict=False # According to test.py loading
            )
            print("TriggerNet weights loaded.")
        except Exception as e:
             print(f"Warning: Error loading TriggerNet weights from {trigger_net_path}: {e}. Model will use random initialization.")
             traceback.print_exc()
    else:
        print(f"Warning: TriggerNet weights not found at {trigger_net_path}. Model will use random initialization.")

    # Set to evaluation mode and freeze parameters
    xeon_net.eval()
    attention_transformer_model.eval()
    trigger_net_model.eval()
    for param in xeon_net.parameters():
        param.requires_grad = False
    for param in attention_transformer_model.parameters():
        param.requires_grad = False
    for param in trigger_net_model.parameters():
        param.requires_grad = False
    print("ATN model components set to eval mode and frozen.")

    # Return model dictionary including xeon_net
    return {
        'xeon_net': xeon_net,
        'attention_transformer': attention_transformer_model,
        'trigger_net': trigger_net_model
    }

# Updated get_atn_outputs function, using the full pipeline: Image -> XeonNet -> AttentionTransformer + TriggerNet
# It receives the loaded model dictionary.
# Input input_data shape (B, C, T, H, W) where C=3 (original image channels)
def get_atn_outputs(
    atn_model_dict: Dict[str, nn.Module],
    input_data: torch.Tensor, # This should be the raw image tensor
    cfg: Any,
    device: torch.device # Add device parameter
) -> Dict[str, Optional[torch.Tensor]]:
    """
    根据数据流 Image -> XeonNet -> AttentionTransformer + TriggerNet 获取模型输出。
    输入 input_data 形状: (B, 3, T, H, W)
    返回 IrisXeonNet 输出的特征、AttentionTransformer 输出以及 TriggerNet 最终输出。
    """
    xeon_net = atn_model_dict.get('xeon_net')
    attention_transformer = atn_model_dict.get('attention_transformer')
    trigger_net = atn_model_dict.get('trigger_net')

    if xeon_net is None or attention_transformer is None or trigger_net is None:
        print("Error: Required ATN model components not found in the provided dictionary.")
        return {
            'trigger_output': None,
            'attention_features': None,
            'features': None,
            'decision': None,
            'attention': None
        }

    # ATN models are frozen (requires_grad=False) in load_atn_model.
    # We do NOT use torch.no_grad() here for adversarial_x_for_G_train
    # so that gradients can flow back to the Generator.
    # However, for original_images (which might be processed once and reused),
    # or if this function is called during evaluation, torch.no_grad() is appropriate.
    # To handle both cases, the caller should manage torch.no_grad().
    # This function performs the forward pass assuming the models are in the correct mode.

    try:
        # Ensure models are on the correct device
        input_data = input_data.to(device)

        # Convert to float if necessary (assuming ATN models expect float32)
        input_data_float = input_data.float()
        
        print(f"DEBUG: get_atn_outputs - shape of input_data: {input_data.shape}, dtype: {input_data.dtype}")
        print(f"DEBUG: get_atn_outputs - shape of input_data_float: {input_data_float.shape}, dtype: {input_data_float.dtype}")

        # --- 1. XeonNet: Extract Perception Features ---
        # input_data_float IS the image
        features = xeon_net(input_data_float) # <-- Error occurs here
        # print("DEBUG: XeonNet output features shape:", features.shape)

        # 2. Pass features through IrisAttentionTransformers
        # Input: Features (B, 128, T', H', W')
        # Output: attention_features (shape needs confirmation, but used as weight_matrix in TriggerNet)
        # Assuming AttentionTransformer takes 5D features and outputs 5D features for now.
        attention_features = attention_transformer(features)
        # print("DEBUG: AttentionTransformer output shape:", attention_features.shape)

        # 3. Pass features and attention_features through IrisTriggerNet
        # TriggerNet signature: forward(self, x, weight_matrix)
        # Assuming x is features (B, 128, T', H', W')
        # And weight_matrix is attention_features.
        trigger_output = trigger_net(features, attention_features) # Output shape needs confirmation (e.g., B, num_classes or B, H'', W'', num_classes)
        # print("DEBUG: TriggerNet output shape:", trigger_output.shape)

        # Return the key outputs
        return {
            'trigger_output': trigger_output, # TriggerNet's final output (attack target)
            'attention_features': attention_features, # AttentionTransformer's output
            'features': features, # Output from IrisXeonNet (Perception Layer)
            'decision': None, # Old output, mark as None
            'attention': None # Old output, mark as None
        }

    except Exception as e:
        print(f"[ERROR] get_atn_outputs - Error during model forward pass: {e}")
        traceback.print_exc()
        # Return None for all outputs if error occurs
        return {
            'trigger_output': None,
            'attention_features': None,
            'features': None,
            'decision': None,
            'attention': None
        }
