import torch
import torch.nn as nn

# 假设IrisBabelTransformer和PositionalEncoding在这个路径下
# 如果路径不对，请根据你的实际情况修改
from IrisBabel.nn.Transformers.IrisBabelTransformer import IrisBabelTransformer
from IrisBabel.nn.Transformers.PositionalEncoding import PositionalEncoding

def run_isolated_test():
    """
    一个独立的、最小化的测试用例，只用于测试 IrisBabelTransformer。
    """
    print("--- Starting Isolated Transformer Test ---")
    
    # --- 1. 定义测试参数 ---
    batch_size = 1
    sequence_length = 5
    d_model = 512
    nhead = 8 # 我们期望的正确头数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Test Parameters: B={batch_size}, N={sequence_length}, D={d_model}, Heads={nhead}, Device={device}")

    # --- 2. 实例化被测试的模块 ---
    try:
        # 确保Transformer的初始化与我们的期望一致
        # 这里的 num_decoder_layers 参数名在 IrisBabelTransformer 中是 num_decoder_layer
        transformer = IrisBabelTransformer(input_dim=d_model, num_decoder_layer=6).to(device)
        # 强制确认其内部头数设置
        # 注意：IrisBabelTransformer 的 __init__ 中已经设置了 nhead=8 给 TransformerDecoderLayer
        # 并且将 self.num_decoder_head 设置为 8。这里不再需要强制替换整个 decoder。
        # 我们保留 __init__ 中的设置，只确保 IrisBabelTransformer 使用了正确的 num_decoder_layer 参数名。
        
        print("IrisBabelTransformer instantiated successfully with expected nhead=8.")
    except Exception as e:
        print(f"Failed to instantiate transformer: {e}")
        return

    # --- 3. 创建符合原生 (batch_first=False) 格式的模拟输入 ---
    # 根据 IrisBabelTransformer 的 forward 方法，它期望 src_seq 和 target_seq 的初始形状是 (B, N, D)
    # 然后在 forward 内部将 src_seq 调整为 (N, B, D) 传给 decoder (尽管我们稍后会移除这个调整)
    # 但是，为了匹配 IrisBabelTransformer forward 的直接输入期望，我们先创建 (B, N, D) 形状的输入。
    # 在 IrisBabelTransformer 的 forward 方法中，它将输入 src_seq 的形状 (B, N, D) permute 成 (N, B, D)。
    # 并且其内部的 TransformerDecoderLayer 使用 batch_first=True，期望 (B, N, D) 的输入。
    # 这两个地方存在矛盾。让我们按照 TransformerDecoderLayer 的期望来创建输入 (B, N, D)。
    # 同时，我们之后需要修改 IrisBabelTransformer 的 forward，移除那个 (B, N, D) -> (N, B, D) 的 permute 和 unsqueeze(0) 逻辑。

    # 模拟输入形状: (B, N, D) -> (批次大小, 序列长度, 特征维度)
    src_tensor = torch.randn(batch_size, sequence_length, d_model).to(device)
    tgt_tensor = torch.randn(batch_size, cfg.model.atn.memory_length, d_model).to(device) # target_seq 形状应与 action_memory 调整后形状一致
    # 从之前的 utils/atn_utils.py 模拟 action_memory 的形状是 (memory_length, batch_size, num_mid_dim)
    # IrisBabelTransformer 的 forward 期望 target_seq 是 (B, N_target, D) 如果 batch_first=True
    # 让我们模拟 target_seq 形状为 (batch_size, memory_length, d_model)
    tgt_tensor = torch.randn(batch_size, cfg.model.atn.memory_length, d_model).to(device)

    print(f"Created src tensor with shape: {src_tensor.shape}")
    print(f"Created tgt tensor with shape: {tgt_tensor.shape}")

    # --- 4. 执行前向传播 ---
    try:
        print("Calling transformer.forward(src_tensor, tgt_tensor)...")
        
        # IrisBabelTransformer forward 方法只返回 decoder 的输出
        # output 形状应与 target_seq 形状相同: (B, N_target, D)
        output = transformer(src_tensor, tgt_tensor)
        
        print("--- Isolated Test PASSED! ---")
        print(f"Transformer output shape: {output.shape}")

    except RuntimeError as e:
        print("--- Isolated Test FAILED! ---")
        print("The RuntimeError was successfully reproduced in an isolated environment.")
        print("Error message:")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print("--- Isolated Test FAILED with unexpected error! ---")
        print("Error message:")
        import traceback
        traceback.print_exc()

def run_isolated_test_corrected():
    """
    一个独立的、最小化的测试用例，只用于测试 IrisBabelTransformer。
    """
    print("--- Starting Isolated Transformer Test (Corrected) ---")
    
    # --- 1. 定义测试参数 ---
    batch_size = 1
    sequence_length = 5
    d_model = 512
    nhead = 8 # 我们期望的正确头数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 确保 cfg 可用以获取 memory_length
    from utils.config_utils import parse_args_and_config
    cfg = parse_args_and_config(default_config_path='configs/stage2_config.yaml')
    if cfg is None:
         print("Failed to load config for memory_length. Isolated test aborted.")
         return
    memory_length = cfg.model.atn.memory_length
    
    print(f"Test Parameters: B={batch_size}, N={sequence_length}, D={d_model}, Heads={nhead}, MemoryLen={memory_length}, Device={device}")

    # --- 2. 实例化被测试的模块 ---
    try:
        # 确保Transformer的初始化与我们的期望一致
        # 这里的 num_decoder_layers 参数名在 IrisBabelTransformer 中是 num_decoder_layer
        # 并且 IrisBabelTransformer 的 input_dim 默认是 4096，需要传入 512
        transformer = IrisBabelTransformer(input_dim=d_model, num_decoder_layer=6).to(device)
        
        # 强制确认其内部头数设置是不必要的，__init__ 中已经硬编码了 8。
        # 移除对 decoder 的不必要的替换
        
        print("IrisBabelTransformer instantiated successfully with expected input_dim=512 and nhead=8.")
    except Exception as e:
        print(f"Failed to instantiate transformer: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 3. 创建符合 TransformerDecoderLayer (batch_first=True) 期望格式的模拟输入 ---
    # src_seq 形状: (B, N, D) -> (批次大小, 源序列长度, 特征维度)
    # target_seq 形状: (B, N_target, D) -> (批次大小, 目标序列长度, 特征维度)
    # target_seq 的序列长度对应 action memory 的长度 (memory_length)
    src_tensor = torch.randn(batch_size, sequence_length, d_model).to(device)
    tgt_tensor = torch.randn(batch_size, memory_length, d_model).to(device)
    
    print(f"Created src tensor with shape: {src_tensor.shape}")
    print(f"Created tgt tensor with shape: {tgt_tensor.shape}")

    # --- 4. 执行前向传播 ---
    try:
        print("Calling transformer.forward(src_tensor, tgt_tensor)...")
        
        # IrisBabelTransformer forward 方法只返回 decoder 的输出
        # output 形状应与 target_seq 形状相同: (B, N_target, D)
        output = transformer(src_tensor, tgt_tensor)
        
        print("--- Isolated Test PASSED! ---")
        print(f"Transformer output shape: {output.shape}")

    except RuntimeError as e:
        print("--- Isolated Test FAILED! ---")
        print("The RuntimeError was successfully reproduced in an isolated environment.")
        print("Error message:")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print("--- Isolated Test FAILED with unexpected error! ---")
        print("Error message:")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    run_isolated_test_corrected() 