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


# 更新 load_atn_model 函数，加载 AttentionTransformer 和 IrisTriggerNet
def load_atn_model(cfg: Any, device: torch.device) -> Optional[Dict[str, nn.Module]]:
    """
    加载并初始化 IrisAttentionTransformers 和 IrisTriggerNet 模型。
    Note: IrisXeonNet 是 AttentionTransformer 和 TriggerNet 内部的组件，
          它们将各自加载并使用自己的 XeonNet 实例。
          因此此处不再需要单独初始化和加载 IrisXeonNet。
    """
    # 检查是否所有必需的类都成功导入
    if _AttentionTransformer is None or _TriggerNet is None:
        print("由于先前的导入错误，无法加载 ATN 模型。")
        return None

    # 初始化模型
    print("正在初始化 IrisAttentionTransformers 和 IrisTriggerNet 模型...")
    try:
        # 初始化 IrisAttentionTransformers (注意力模块)
        attention_transformer_model = _AttentionTransformer().to(device)
        print("IrisAttentionTransformers 初始化完成。")

        # 初始化 IrisTriggerNet (决策模块)
        trigger_net_model = _TriggerNet().to(device)
        print("IrisTriggerNet 初始化完成。")

    except Exception as e:
         print(f"初始化 ATN 模型组件时出错：{e}")
         traceback.print_exc() # 打印详细的错误信息
         return None


    # 加载预训练权重
    print("正在加载 ATN 模型权重...")
    # 从配置中获取权重路径
    # 需要 attention_transformer_path 和 trigger_net_path
    # perception_layer_path 不再需要单独加载，因为 XeonNet 在内部
    if not hasattr(cfg.model.atn, 'attention_transformer_path') or \
       not hasattr(cfg.model.atn, 'trigger_net_path'):
        print("错误：配置中缺少 ATN 模型路径 (attention_transformer_path 或 trigger_net_path)。")
        return None

    attention_transformer_path = cfg.model.atn.attention_transformer_path
    trigger_net_path = cfg.model.atn.trigger_net_path

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
    attention_transformer_model.eval()
    trigger_net_model.eval()
    for model in [attention_transformer_model, trigger_net_model]:
        for param in model.parameters():
            param.requires_grad = False
    print("ATN 模型组件已设置为评估模式并冻结参数。")

    # 返回包含 attention_transformer 和 trigger_net 的模型字典
    return {
        'attention_transformer': attention_transformer_model,
        'trigger_net': trigger_net_model
    }

# 获取 ATN 模型输出的函数，使用正确的数据流：图像 -> AttentionTransformer -> Attention_Features, Image + Attention_Features -> TriggerNet
# 它接收已加载的模型字典。
# 输入 input_data 形状 (B, C, T, H, W)，其中 C=3 (原始图像通道)
def get_atn_outputs(
    atn_model_dict: Dict[str, nn.Module],
    input_data: torch.Tensor, # 这应该是原始图像张量
    cfg: Any,
    device: torch.device # 添加 device 参数
) -> Dict[str, Optional[torch.Tensor]]:
    """
    根据数据流 Image -> AttentionTransformer -> Attention_Features, Image + Attention_Features -> TriggerNet 获取模型输出。
    输入 input_data 形状: (B, 3, T, H, W)
    返回 AttentionTransformer 输出的注意力特征以及 TriggerNet 最终输出。
    """
    try:
        # 确保输入数据在正确的设备上
        input_data = input_data.to(device).float() # 确保数据类型和设备正确

        # --- 1. 通过 IrisAttentionTransformers 传递输入数据 ---
        # 输入：input_data (B, 3, T, H, W)
        # 输出：attention_features (形状取决于具体实现，但用于 TriggerNet 的 weight_matrix)
        attention_transformer = atn_model_dict.get('attention_transformer')
        if attention_transformer is None:
             raise ValueError("AttentionTransformer not found in atn_model_dict")
        attention_features = attention_transformer(input_data)
        # print("DEBUG: AttentionTransformer 输出形状:", attention_features.shape) # 移除了调试打印

        # --- 2. 通过 IrisTriggerNet 传递原始输入数据和注意力特征 ---
        # TriggerNet 函数签名：forward(self, x, weight_matrix)
        # 假设 x 是原始图像 input_data (B, 3, T, H, W)
        # weight_matrix 是 attention_features。
        trigger_net = atn_model_dict.get('trigger_net')
        if trigger_net is None:
             raise ValueError("TriggerNet not found in atn_model_dict")
        trigger_output = trigger_net(input_data, attention_features) # 输出形状需要确认 (例如，B, num_classes或 B, H'', W'', num_classes)
        # print("DEBUG: TriggerNet 输出形状:", trigger_output.shape) # 移除了调试打印


        # 返回关键输出
        return {
            'trigger_output': trigger_output, # TriggerNet 的最终输出 (攻击目标)
            'attention_features': attention_features, # AttentionTransformer 的输出
            'features': None, # 不再单独输出 XeonNet 特征
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
