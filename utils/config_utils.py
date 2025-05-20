import yaml
import argparse
import os
from easydict import EasyDict as edict

def load_config(config_path):
    """
    加载 YAML 配置文件。
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return edict(config) # 使用 EasyDict 转换为可以通过点访问的对象

def merge_configs(base, new):
    """
    递归合并两个 EasyDict 配置。
    """
    for k, v in new.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            base[k] = merge_configs(base[k], v)
        else:
            base[k] = v
    return base

def parse_args_and_config(default_config_path, task_config_arg='config'):
    """
    解析命令行参数并加载/合并配置。
    优先级：命令行参数 > 特定任务配置 > 默认配置。
    """
    parser = argparse.ArgumentParser(description='SeqAdvGAN Script')
    
    # 添加一个参数用于指定任务配置文件
    parser.add_argument(f'--{task_config_arg}', type=str, 
                        help='Path to the task-specific YAML configuration file. Overrides default config.')
    
    # 临时解析命令行，只获取 config 文件路径
    temp_args, unknown = parser.parse_known_args()

    # 1. 加载默认配置
    try:
        cfg = load_config(default_config_path)
    except FileNotFoundError as e:
        print(e)
        print("Please ensure default_config.yaml exists.")
        return None

    # 2. 加载特定任务配置 (如果提供了)
    if getattr(temp_args, task_config_arg) and os.path.exists(getattr(temp_args, task_config_arg)):
        print(f"Loading task-specific configuration from: {getattr(temp_args, task_config_arg)}")
        task_cfg = load_config(getattr(temp_args, task_config_arg))
        # 递归合并配置
        cfg = merge_configs(cfg, task_cfg)
    elif getattr(temp_args, task_config_arg): # 提供了config路径但文件不存在
        print(f"Warning: Configuration file not found at {getattr(temp_args, task_config_arg)}. Using default/existing config.")

    # 3. 重新定义解析器，包含所有可能的命令行参数
    # 这一步比较繁琐，需要手动添加所有你希望从命令行直接覆盖的参数
    # 一个更通用的方法是动态地从 cfg 中解析参数，但这超出了当前请求范围。
    # 为了简化，我们手动添加一些常用参数，并使用 unknown 参数来捕获未定义的。
    # **注意：你需要根据实际 YAML 文件中的参数手动在此处添加对应的 parser.add_argument 调用**
    # **并且需要处理嵌套结构的参数名（例如 'data.video_path'）**
    
    # 清空旧的解析器
    parser = argparse.ArgumentParser(description='SeqAdvGAN Script')
    parser.add_argument(f'--{task_config_arg}', type=str, help='Path to task config') # 再次添加 config 参数

    # **手动添加你希望从命令行覆盖的参数，名称与 YAML 结构对应**
    # 例如:
    parser.add_argument('--data.video_path', type=str, help='Path to the real video data')
    parser.add_argument('--data.level_json_path', type=str, help='Path to the level definition JSON file')
    parser.add_argument('--model.atn.model_path', type=str, help='Path to the ATN model or feature head weights.')
    parser.add_argument('--training.batch_size', type=int, help='Batch size for training')
    parser.add_argument('--training.num_epochs', type=int, help='Number of training epochs')
    parser.add_argument('--training.lr_g', type=float, help='Learning rate for generator')
    parser.add_argument('--training.lr_d', type=float, help='Learning rate for discriminator')
    parser.add_argument('--logging.log_dir', type=str, help='Directory for TensorBoard logs')
    parser.add_argument('--logging.checkpoint_dir', type=str, help='Directory to save model checkpoints')
    parser.add_argument('--model.discriminator.type', type=str, choices=['cnn', 'patchgan'], help='Type of discriminator')
    parser.add_argument('--mask.use_region_mask', action=argparse.BooleanOptionalAction, help='Use region masks for attack loss calculation')
    parser.add_argument('--losses.attack_loss_type', type=str, help='Type of attack loss for generator')
    parser.add_argument('--losses.decision_loss_type', type=str, help='Type of decision loss')
    parser.add_argument('--losses.attention_loss_type', type=str, help='Type of attention loss')
    parser.add_argument('--losses.topk_k', type=int, help='K value for Top-K attention loss and evaluation')
    parser.add_argument('--regularization.lambda_tv', type=float, help='Weight for total variation loss')
    parser.add_argument('--regularization.lambda_l2_penalty', type=float, help='Weight for L2 penalty on generator parameters')
    parser.add_argument('--evaluation.success_threshold', type=float, help='Threshold for attack success in evaluation')
    parser.add_argument('--evaluation.success_criterion', type=str, help='Criterion for attack success in evaluation')


    # 解析所有参数，包括未知的
    cmd_args, unknown_args = parser.parse_known_args()

    if unknown_args:
        print(f"Warning: Unknown command line arguments: {unknown_args}")

    # 将命令行参数合并到 cfg 中，覆盖配置文件中的值
    # 使用点号分割参数名，映射到嵌套字典
    for arg_name, arg_value in vars(cmd_args).items():
        if arg_value is not None and arg_name != task_config_arg:
            keys = arg_name.split('.')
            current_cfg = cfg
            for i, key in enumerate(keys):
                if i == len(keys) - 1:
                    if isinstance(current_cfg, dict):
                        current_cfg[key] = arg_value
                    elif isinstance(current_cfg, edict):
                         current_cfg[key] = arg_value
                    else:
                        print(f"Warning: Could not set config value for {arg_name}. Invalid structure.")
                else:
                    if isinstance(current_cfg, dict) and key in current_cfg and isinstance(current_cfg[key], dict):
                        current_cfg = current_cfg[key]
                    elif isinstance(current_cfg, edict) and key in current_cfg and isinstance(current_cfg[key], edict):
                         current_cfg = current_cfg[key]
                    else:
                         # 如果结构不存在，尝试创建它 (仅限 EasyDict)
                         if isinstance(current_cfg, edict):
                              current_cfg[key] = edict()
                              current_cfg = current_cfg[key]
                         else:
                              print(f"Warning: Could not navigate config structure for {arg_name}. Missing or invalid intermediate key: {key}")
                              break # Stop processing this argument


    return cfg
