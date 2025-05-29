import torch
import torch.nn as nn
from typing import Dict, Optional, Any
import os
import traceback # 导入 traceback 用于错误追踪

# 从 IrisBabel 导入实际使用的模型组件
try:
    # 确保这些导入路径正确，根据您的项目结构
    from IrisBabel.nn.Transformers import IrisAttentionTransformers
    from IrisBabel.nn.CNN import IrisTriggerNet
    # 导入 IrisXeonNet (感知层)
    from IrisBabel.nn.CNN import IrisXeonNet # 导入 IrisXeonNet
    print("成功导入 IrisAttentionTransformers, IrisTriggerNet 和 IrisXeonNet。")
    # 导入失败时的占位符类 - 允许在开发期间优雅失败
    _AttentionTransformer = IrisAttentionTransformers
    _TriggerNet = IrisTriggerNet
    _XeonNet = IrisXeonNet # IrisXeonNet 的占位符
except ImportError as e:
    print(f"致命错误：导入实际 ATN 模型组件失败：{e}")
    print("请确保 'IrisBabel' 目录及其子模块正确放置在 SeqAdvGAN 项目根目录下，并且已安装必要的依赖项。")
    # 定义虚拟（Dummy）类以防止导入失败时出现 NameError
    class _AttentionTransformer(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__(); print("由于导入错误，使用了虚拟 AttentionTransformer。")
        def forward(self, x):
             # 警告：虚拟 AttentionTransformer forward 方法返回虚拟特征。
             print("警告：虚拟 AttentionTransformer forward 方法返回虚拟特征。")
             # 模拟一个合理的输出形状 (B, 128, T, H', W')
             if x.ndim == 5:
                 return torch.randn(x.shape[0], 128, x.shape[2], x.shape[3]//4, x.shape[4]//4, device=x.device)
             else:
                 return torch.empty(0) # 返回空张量或根据需要调整

    class _TriggerNet(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__(); print("由于导入错误，使用了虚拟 TriggerNet。")
        def forward(self, x, weight_matrix):
            # 警告：虚拟 TriggerNet forward 方法返回虚拟输出。
            print("警告：虚拟 TriggerNet forward 方法返回虚拟输出。")
            # 模拟一个合理的输出形状 (B, H', W', 8) 或 (B, 8)
            if x.ndim == 5:
                 return torch.randn(x.shape[0], x.shape[3]//4, x.shape[4]//4, 8, device=x.device) # 模拟 (B, H'/4, W'/4, 8)
            elif x.ndim == 2: # 如果输入是 (B, features)
                 return torch.randn(x.shape[0], 8, device=x.device) # 模拟 (B, 8)
            else:
                 return torch.empty(0) # 返回空张量或根据需要调整

    class _XeonNet(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__(); print("由于导入错误，使用了虚拟 IrisXeonNet。")
        def forward(self, x):
            # 模拟输出形状 (B, 128, T, H', W')
            # 输入 x 是 (B, 3, T, H, W)
            # 假设下采样率为 4x
            print("警告：虚拟 IrisXeonNet forward 方法返回虚拟特征。")
            if x.ndim == 5:
                return torch.randn(x.shape[0], 128, x.shape[2], x.shape[3]//4, x.shape[4]//4, device=x.device) # 模拟 (B, 128, T, H/4, W/4)
            else:
                 return torch.empty(0) # 返回空张量或根据需要调整

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
    # 检查是否所有必需的类都成功导入
    if _XeonNet is None or _AttentionTransformer is None or _TriggerNet is None:
        print("由于先前的导入错误，无法加载 ATN 模型。")
        return None
    
    # 初始化模型
    print("正在初始化 IrisXeonNet, IrisAttentionTransformers 和 IrisTriggerNet 模型...")
    try:
        # 初始化 IrisXeonNet (感知层)
        # IrisXeonNet 实例化（假设使用默认参数）
        xeon_net = _XeonNet().to(device)
        print("IrisXeonNet 初始化完成。")

        # 初始化 IrisAttentionTransformers (注意力模块)
        # 实例化（假设使用默认参数）
        attention_transformer_model = _AttentionTransformer().to(device)
        print("IrisAttentionTransformers 初始化完成。")

        # 初始化 IrisTriggerNet (决策模块)
        # 实例化（假设使用默认参数）
        trigger_net_model = _TriggerNet().to(device)
        print("IrisTriggerNet 初始化完成。")

    except Exception as e:
         print(f"初始化 ATN 模型组件时出错：{e}")
         traceback.print_exc() # 打印详细的错误信息
         return None


    # 加载预训练权重
    print("正在加载 ATN 模型权重...")
    # 从配置中获取权重路径
    # 需要 perception_layer_path, attention_transformer_path 和 trigger_net_path
    if not hasattr(cfg.model.atn, 'perception_layer_path') or \
       not hasattr(cfg.model.atn, 'attention_transformer_path') or \
       not hasattr(cfg.model.atn, 'trigger_net_path'):
        print("错误：配置中缺少 ATN 模型路径 (perception_layer_path, attention_transformer_path 或 trigger_net_path)。")
        return None

    perception_layer_path = cfg.model.atn.perception_layer_path
    attention_transformer_path = cfg.model.atn.attention_transformer_path
    trigger_net_path = cfg.model.atn.trigger_net_path

    # 加载 IrisXeonNet 权重
    print(f"正在从 {perception_layer_path} 加载 IrisXeonNet 权重")
    if os.path.exists(perception_layer_path):
        try:
            # 加载权重，使用 weights_only=False 和 strict=False (根据 test.py 中的加载方式)
            xeon_net.load_state_dict(
                torch.load(perception_layer_path, map_location=device, weights_only=False),
                strict=False
            )
            print("IrisXeonNet 权重加载成功。")
        except Exception as e:
             print(f"警告：从 {perception_layer_path} 加载 IrisXeonNet 权重时出错：{e}。模型将使用随机初始化。")
             traceback.print_exc()
    else:
        print(f"警告：未找到 IrisXeonNet 权重文件 {perception_layer_path}。模型将使用随机初始化。")

    # 加载 AttentionTransformer 权重
    print(f"正在从 {attention_transformer_path} 加载 AttentionTransformer 权重")
    if os.path.exists(attention_transformer_path):
        try:
            # 加载权重，使用 weights_only=False 和 strict=False (根据 test.py 中的加载方式)
            attention_transformer_model.load_state_dict(
                torch.load(attention_transformer_path, map_location=device, weights_only=False),
                strict=False # 根据 test.py 中的加载方式
            )
            print("AttentionTransformer 权重加载成功。")
        except Exception as e:
             print(f"警告：从 {attention_transformer_path} 加载 AttentionTransformer 权重时出错：{e}。模型将使用随机初始化。")
             traceback.print_exc()
    else:
        print(f"警告：未找到 AttentionTransformer 权重文件 {attention_transformer_path}。模型将使用随机初始化。")

    # 加载 TriggerNet 权重
    print(f"正在从 {trigger_net_path} 加载 TriggerNet 权重")
    if os.path.exists(trigger_net_path):
        try:
            # 加载权重，使用 weights_only=False 和 strict=False (根据 test.py 中的加载方式)
            trigger_net_model.load_state_dict(
                torch.load(trigger_net_path, map_location=device, weights_only=False),
                strict=False # 根据 test.py 中的加载方式
            )
            print("TriggerNet 权重加载成功。")
        except Exception as e:
             print(f"警告：从 {trigger_net_path} 加载 TriggerNet 权重时出错：{e}。模型将使用随机初始化。")
             traceback.print_exc()
    else:
        print(f"警告：未找到 TriggerNet 权重文件 {trigger_net_path}。模型将使用随机初始化。")

    # 设置为评估模式并冻结参数
    xeon_net.eval()
    attention_transformer_model.eval()
    trigger_net_model.eval()
    for param in xeon_net.parameters():
        param.requires_grad = False
    for param in attention_transformer_model.parameters():
        param.requires_grad = False
    for param in trigger_net_model.parameters():
        param.requires_grad = False
    print("ATN 模型组件已设置为评估模式并冻结参数。")

    # 返回包含 xeon_net 的模型字典
    return {
        'xeon_net': xeon_net,
        'attention_transformer': attention_transformer_model,
        'trigger_net': trigger_net_model
    }

# 获取 ATN 模型输出的函数，使用完整的数据流：图像 -> XeonNet -> AttentionTransformer + TriggerNet
# 它接收已加载的模型字典。
# 输入 input_data 形状 (B, C, T, H, W)，其中 C=3 (原始图像通道)
def get_atn_outputs(
    atn_model_dict: Dict[str, nn.Module],
    input_data: torch.Tensor, # 这应该是原始图像张量
    cfg: Any,
    device: torch.device # 添加 device 参数
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
        print("错误：提供的字典中未找到必需的 ATN 模型组件。")
        return {
            'trigger_output': None,
            'attention_features': None,
            'features': None,
            'decision': None, # 旧输出，标记为 None
            'attention': None # 旧输出，标记为 None
        }

    # ATN 模型的参数在 load_atn_model 中已被冻结 (requires_grad=False)。
    # 对于对抗样本 (adversarial_x_for_G)，我们不使用 torch.no_grad() 以便梯度可以流回生成器。
    # 但是，对于原始图像 (real_x)，或者如果在评估阶段调用此函数，则 torch.no_grad() 是合适的。
    # 为了处理这两种情况，调用方应该自行管理 torch.no_grad()。
    # 此函数假设模型处于正确的模式下进行前向传播。

    try:
        # 确保输入数据在正确的设备上
        input_data = input_data.to(device)

        # 如果需要，转换为 float (假设 ATN 模型期望 float32)
        input_data_float = input_data.float()
        
        # 移除了调试打印语句
        # print(f"DEBUG: get_atn_outputs - shape of input_data: {input_data.shape}, dtype: {input_data.dtype}")
        # print(f"DEBUG: get_atn_outputs - shape of input_data_float: {input_data_float.shape}, dtype: {input_data_float.dtype}")

        # --- 1. XeonNet: 提取感知特征 ---
        # input_data_float 是图像
        features = xeon_net(input_data_float) # <-- 错误曾发生在这里
        # print("DEBUG: XeonNet 输出特征形状:", features.shape) # 移除了调试打印

        # --- 2. 通过 IrisAttentionTransformers 传递特征 ---
        # 输入：特征 (B, 128, T', H', W')
        # 输出：attention_features (形状需要确认，但用作 TriggerNet 的 weight_matrix)
        # 假设 AttentionTransformer 接收 5D 特征并输出 5D 特征。
        attention_features = attention_transformer(features)
        # print("DEBUG: AttentionTransformer 输出形状:", attention_features.shape) # 移除了调试打印

        # --- 3. 通过 IrisTriggerNet 传递特征和注意力特征 ---
        # TriggerNet 函数签名：forward(self, x, weight_matrix)
        # 假设 x 是特征 (B, 128, T', H', W')
        # weight_matrix 是 attention_features。
        trigger_output = trigger_net(features, attention_features) # 输出形状需要确认 (例如，B, num_classes 或 B, H'', W'', num_classes)
        # print("DEBUG: TriggerNet 输出形状:", trigger_output.shape) # 移除了调试打印

        # 返回关键输出
        return {
            'trigger_output': trigger_output, # TriggerNet 的最终输出 (攻击目标)
            'attention_features': attention_features, # AttentionTransformer 的输出
            'features': features, # 来自 IrisXeonNet (感知层) 的输出
            'decision': None, # 旧输出，标记为 None
            'attention': None # 旧输出，标记为 None
        }

    except Exception as e:
        print(f"[错误] get_atn_outputs - 模型前向传播期间出错：{e}")
        traceback.print_exc()
        # 如果发生错误，所有输出都返回 None
        return {
            'trigger_output': None,
            'attention_features': None,
            'features': None,
            'decision': None,
            'attention': None
        }
