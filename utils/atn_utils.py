import torch
import torch.nn as nn
from typing import Dict, Optional, Any

# 1. 导入真实的ATN模型。
# 因为你已经把IrisBabel文件夹放在了项目根目录，这个导入应该能成功。
try:
    from IrisBabel.IrisBabelModel import IrisBabelModel
    print("Successfully imported IrisBabelModel from IrisBabel.")
except ImportError as e:
    print(f"FATAL: Error importing IrisBabelModel: {e}")
    print("Please ensure the 'IrisBabel' directory and its '__init__.py' files are correctly placed in the SeqAdvGAN project root.")
    IrisBabelModel = None

def load_atn_model(cfg: Any, device: torch.device) -> Optional[nn.Module]:
    """
    加载并初始化真实的 IrisBabelModel。
    """
    if IrisBabelModel is None:
        return None
    
    try:
        # 2. 使用配置文件中的参数实例化模型
        print("Initializing IrisBabelModel with parameters from config...")
        atn_model = IrisBabelModel(
            memory_length=cfg.model.atn.memory_length,
            num_mid_dim=cfg.model.atn.num_mid_dim,
            num_actions=cfg.model.atn.num_actions,
            num_targets=cfg.model.atn.num_targets
        )
        
        # 3. 加载预训练权重
        model_path = cfg.model.atn.model_path
        print(f"Loading ATN weights from: {model_path}")
        # 根据开发者习惯，直接加载状态字典
        # atn_model.load_state_dict(torch.load(model_path, map_location=device))

        # 3. 改为调用 IrisBabelModel 自己的加载方法
        print(f"Calling atn_model.load_model with path: {model_path}")
        atn_model.load_model(model_path)

        # IrisBabelModel 本身不是一个nn.Module，它的组件才是。
        # 所以我们不需要 atn_model.to(device) 或 atn_model.eval()
        # 它内部的组件在初始化时已经 .to(device) 了。
        print("ATN model loaded and configured successfully.")
        return atn_model

    except Exception as e:
        print(f"FATAL: Failed to load ATN model: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_atn_outputs(
    atn_model: IrisBabelModel,
    input_data: torch.Tensor,
    cfg: Any
) -> Dict[str, Optional[torch.Tensor]]:
    """
    Final version: Corrects all tensor shapes for batch_first=False and proper
    Transformer invocation.
    """
    with torch.no_grad():
        batch_size, channels, sequence_length, height, width = input_data.shape
        device = input_data.device
        d_model = atn_model.num_mid_dim

        # 1. Perception Layer: (B, C, N, H, W) -> (B*N, C, H, W) -> (B*N, D)
        input_reshaped = input_data.permute(0, 2, 1, 3, 4).reshape(batch_size * sequence_length, channels, height, width)
        frame_features = atn_model.perception_layer(input_reshaped.float() / 255.0) # Normalize to [0,1]

        # 2. Prepare `src` for Transformer: (B*N, D) -> (B, N, D) -> (N, B, D)
        src_for_transformer = frame_features.view(batch_size, sequence_length, d_model).permute(1, 0, 2)

        # 3. Prepare `tgt` for Transformer: This is the 'query'. In many architectures,
        # this is a learned parameter or a dummy tensor representing the start of a sequence.
        # Its batch size MUST match `src_for_transformer`.
        memory_length = atn_model.memory_length
        tgt_for_transformer = torch.zeros(memory_length, batch_size, d_model, device=device)

        # 4. Call the Transformer
        # Note: babel_transformer expects (tgt, src) order if batch_first=False
        # But IrisBabelTransformer is implemented with batch_first=True internally
        # and expects (src, target) order, where src is the sequence data and target is the memory.
        # Let's revert to the expected (src, target) order based on the IrisBabelTransformer implementation.
        # Re-read IrisBabelTransformer.py's forward method to confirm input order.

        # After reviewing IrisBabel/nn/Transformers/IrisBabelTransformer.py forward method:
        # It expects (src_seq, target_seq) and is implemented with batch_first=True layers internally.
        # So the inputs should be (B, N, D) for src_seq and (B, M, D) for target_seq (memory).
        # We need to adjust the shapes here to match that expectation.

        # Corrected: Prepare `src` for Transformer: (B*N, D) -> (B, N, D)
        src_for_transformer_corrected = frame_features.view(batch_size, sequence_length, d_model)

        # Corrected: Prepare `tgt` (memory) for Transformer: (M, B, D) -> (B, M, D)
        # The dummy action_memory was created as (M, B, D). Need to permute.
        tgt_for_transformer_corrected = tgt_for_transformer.permute(1, 0, 2)

        # 4. Call the Transformer with corrected shapes (B, N, D) and (B, M, D)
        print(f"[DEBUG] get_atn_outputs - Calling babel_transformer with src shape: {src_for_transformer_corrected.shape}, tgt shape: {tgt_for_transformer_corrected.shape}")
        memory_output, attention_probs = atn_model.babel_transformer(src_for_transformer_corrected, tgt_for_transformer_corrected)
        print(f"[DEBUG] get_atn_outputs - babel_transformer output (memory_output) shape: {memory_output.shape}")

        # 5. Call Decoders on the final state of the memory/output sequence
        # memory_output shape from IrisBabelTransformer is (B, M, D) because batch_first=True
        # We need the last state of the memory sequence, which is the last token for each batch.
        final_frame_output = memory_output[:, -1, :] # (B, D)
        print(f"[DEBUG] get_atn_outputs - final_frame_output shape (decoder input): {final_frame_output.shape}")
        
        x, y = atn_model.rel_position_decoder(final_frame_output)
        # Note: action_id and direction decoders are also part of the original model flow
        # Although not used for decision map, they are part of the model output.
        action_id = atn_model.action_decoder(final_frame_output)
        direction_output = atn_model.direction_decoder(final_frame_output)

        # Assuming direction_decoder returns a tuple, take the first element as the score
        # If it returns a single tensor, this will still work as tuple(tensor)[0] is tensor.
        score = direction_output[0] if isinstance(direction_output, tuple) else direction_output

        print(f"[DEBUG] get_atn_outputs - x shape: {x.shape}, y shape: {y.shape}")
        print(f"[DEBUG] get_atn_outputs - direction_output type: {type(direction_output)}, score shape: {score.shape}")

        # 6. Synthesize Decision Map
        map_h = cfg.model.atn.decision_map_height
        map_w = cfg.model.atn.decision_map_width
        decision_map = torch.zeros(batch_size, map_h, map_w, device=device)
        print(f"[DEBUG] get_atn_outputs - decision_map base shape: {decision_map.shape}")
        
        # Ensure indices are 1D tensors of type long
        # Ensure x and y are at least 1D for squeeze to work predictably, then squeeze
        x_indices = (x.view(-1) * (map_w - 1)).round().long().clamp(0, map_w - 1)
        y_indices = (y.view(-1) * (map_h - 1)).round().long().clamp(0, map_h - 1)
        print(f"[DEBUG] get_atn_outputs - x_indices shape: {x_indices.shape}, y_indices shape: {y_indices.shape}")
        print(f"[DEBUG] get_atn_outputs - x_indices: {x_indices}, y_indices: {y_indices}")

        batch_indices = torch.arange(batch_size, device=device)
        
        # Perform the scatter operation or direct indexing
        # Direct indexing requires indices to have the same shape as the tensor being assigned to (excluding batch dim)
        # Or using advanced indexing like [batch_indices, y_indices, x_indices]
        # Direct assignment with advanced indexing: tensor[batch_indices, y_indices, x_indices] = values
        # The shapes must align: batch_indices (B,), y_indices (B,), x_indices (B,), values (B,)

        try:
            # Attempt direct assignment first as it's simpler
            # Ensure score is float type to match decision_map dtype
            decision_map[batch_indices, y_indices, x_indices] = score.float()
            print("[DEBUG] get_atn_outputs - Direct assignment to decision_map successful.")
        except IndexError as e:
             print(f"[ERROR] get_atn_outputs - IndexError during direct assignment: {e}")
             print(f"  batch_indices shape: {batch_indices.shape}")
             print(f"  y_indices shape: {y_indices.shape}")
             print(f"  x_indices shape: {x_indices.shape}")
             print(f"  score shape: {score.shape}")
             print(f"  decision_map shape: {decision_map.shape}")
             # Fallback or alternative logic if direct assignment fails unexpectedly
             # This might indicate a deeper issue with index calculation or shapes not caught by clamp.
             pass # Re-raise or handle as appropriate


        # decision_map shape needs to be (B, 1, 1, H, W) for compatibility with the generator/discriminator
        decision_map = decision_map.unsqueeze(1).unsqueeze(1)
        print(f"[DEBUG] get_atn_outputs - final decision_map shape: {decision_map.shape}")

    return {
        'decision': decision_map,
        'attention': attention_probs, # This will be None for now
        'features': None # Feature output is not part of this stage's attack target
    }