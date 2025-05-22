import yaml
import argparse
import os
from easydict import EasyDict as edict
from typing import Any # 导入 Any 用于表示任意类型

def load_config(config_path: str) -> edict:
    """
    加载指定路径的 YAML 配置文件。

    Args:
        config_path (str): YAML 配置文件的路径。

    Returns:
        edict: 加载并转换为 EasyDict 对象的配置字典。

    Raises:
        FileNotFoundError: 如果指定的配置文件路径不存在。
    """
    # 检查配置文件是否存在
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
        
    # 打开并加载 YAML 文件
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # 将加载的字典转换为 EasyDict 对象，支持通过点号访问属性
    return edict(config)

def merge_configs(base: edict, new: edict) -> edict:
    """
    递归合并两个 EasyDict 配置对象。
    新配置中的值会覆盖基础配置中的同名值。
    如果遇到嵌套字典，则递归合并。

    Args:
        base (edict): 作为基础的 EasyDict 配置对象。
        new (edict): 包含要合并的新配置的 EasyDict 对象。

    Returns:
        edict: 合并后的 EasyDict 配置对象。
    """
    # 遍历新配置的键值对
    for k, v in new.items():
        # 如果值是字典且基础配置中也存在同名的字典，则递归合并
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            base[k] = merge_configs(base[k], edict(v)) # 确保递归时也使用 EasyDict
        else:
            # 否则直接更新或添加键值对
            base[k] = v
    return base

def parse_args_and_config(default_config_path: str, task_config_arg: str = 'config') -> edict | None:
    """
    解析命令行参数并加载/合并配置。
    合并优先级：命令行参数 > 特定任务配置文件 > 默认配置文件。

    首先加载默认配置，然后根据命令行参数指定的路径加载特定任务配置并合并，
    最后解析所有命令行参数并用它们覆盖合并后的配置。

    Args:
        default_config_path (str): 默认 YAML 配置文件的路径。
        task_config_arg (str): 用于指定特定任务配置文件的命令行参数名称 (例如 'config')。

    Returns:
        edict | None: 加载并合并后的配置对象，如果加载默认配置失败则返回 None。
    """
    # 1. 创建临时的 ArgumentParser，只用于解析特定任务配置文件的路径
    parser = argparse.ArgumentParser(description='SeqAdvGAN Script')
    
    # 添加用于指定任务配置文件的命令行参数
    # 例如: --config configs/stage1_config.yaml
    parser.add_argument(f'--{task_config_arg}', type=str, 
                        help='Path to the task-specific YAML configuration file. Overrides default config.')
    
    # 临时解析命令行参数，只获取特定任务配置文件的路径，并捕获其他未知参数
    # unknown 变量会是一个列表，包含所有未被临时解析器处理的参数
    temp_args, unknown = parser.parse_known_args()

    # 2. 加载默认配置
    try:
        cfg = load_config(default_config_path)
        # 转换为 EasyDict 以便后续合并和点号访问
        cfg = edict(cfg)
    except FileNotFoundError as e:
        # 如果默认配置找不到，打印错误并返回 None
        print(e)
        print("Please ensure default_config.yaml exists.")
        return None

    # 3. 加载并合并特定任务配置 (如果命令行参数提供了路径且文件存在)
    task_config_path = getattr(temp_args, task_config_arg, None)
    if task_config_path and os.path.exists(task_config_path):
        print(f"Loading task-specific configuration from: {task_config_path}")
        try:
            task_cfg = load_config(task_config_path)
            # 递归合并特定任务配置到默认配置中
            cfg = merge_configs(cfg, task_cfg)
        except Exception as e:
             # 如果加载或合并任务配置出错，打印警告，继续使用当前已加载的配置
             print(f"Warning: Error loading or merging task configuration file {task_config_path}: {e}. Using current config.")
    elif task_config_path: # 提供了config路径但文件不存在
        # 如果提供了任务配置路径但文件不存在，打印警告
        print(f"Warning: Configuration file not found at {task_config_path}. Using default/existing config.")

    # 4. 重新构建 ArgumentParser，添加所有你希望通过命令行覆盖的参数
    # 这一步相对手动，需要根据 YAML 配置文件中的结构定义对应的命令行参数。
    # 参数名使用点号表示嵌套结构，例如 'data.video_path' 对应 cfg.data.video_path。
    # 一个更自动化的方法是从 EasyDict cfg 对象动态生成命令行参数，但这会增加复杂性。
    # 为了简化，我们在此手动添加一些常用的、可能需要从命令行覆盖的参数。
    
    # 清空之前的解析器（或者创建一个新的）
    parser = argparse.ArgumentParser(description='SeqAdvGAN Script')
    # 再次添加特定任务配置文件的参数，以便 help 消息中显示
    parser.add_argument(f'--{task_config_arg}', type=str, help='Path to task config') # 再次添加 config 参数

    # **在此处手动添加你希望从命令行覆盖的参数**
    # 参数名称应与 YAML 配置中的层级结构用点号连接对应。
    # 示例 (需要根据您的实际配置文件内容进行添加和调整):
    # 例如，对于 YAML 中的:
    # data:
    #   video_path: xxx
    # 对应命令行参数 --data.video_path
    # training:
    #   batch_size: 32
    # 对应命令行参数 --training.batch_size

    # 常见参数示例：
    parser.add_argument('--data.video_path', type=str, help='Path to the real video data')
    parser.add_argument('--data.level_json_path', type=str, help='Path to the level definition JSON file')
    parser.add_argument('--data.use_mock_data', action=argparse.BooleanOptionalAction, help='Whether to use mock data instead of real video data')
    parser.add_argument('--data.mock_num_samples', type=int, help='Number of mock samples to generate if using mock data')
    parser.add_argument('--model.atn.model_path', type=str, help='Path to the ATN model or feature head weights.')
    parser.add_argument('--model.discriminator.type', type=str, choices=['cnn', 'patchgan'], help='Type of discriminator')
    parser.add_argument('--training.batch_size', type=int, help='Batch size for training')
    parser.add_argument('--training.num_epochs', type=int, help='Number of training epochs')
    parser.add_argument('--training.lr_g', type=float, help='Learning rate for generator')
    parser.add_argument('--training.lr_d', type=float, help='Learning rate for discriminator')
    parser.add_argument('--training.save_interval', type=int, help='Save checkpoint every N epochs')
    parser.add_argument('--training.eval_interval', type=int, help='Evaluate model every N epochs')
    parser.add_argument('--logging.log_dir', type=str, help='Directory for TensorBoard logs')
    parser.add_argument('--logging.checkpoint_dir', type=str, help='Directory to save model checkpoints') # 注意：train_generator 中硬编码使用了 log_dir/checkpoints，这里作为示例保留
    parser.add_argument('--logging.vis_interval', type=int, help='Visualize samples every N global steps')
    parser.add_argument('--logging.num_vis_samples', type=int, help='Number of samples to visualize')
    parser.add_argument('--logging.sequence_step_to_vis', type=int, help='Sequence step to visualize (0-indexed)')
    parser.add_argument('--losses.gan_type', type=str, choices=['bce', 'lsgan'], help='Type of GAN loss')
    parser.add_argument('--losses.gan_loss_weight', type=float, help='Weight for GAN loss in generator total loss')
    # 阶段1 特有参数 (特征攻击) 或 阶段2 通用参数 (决策攻击)
    parser.add_argument('--losses.decision_loss_type', type=str, choices=['mse', 'l1'], help='Type of decision loss (Stage 1: Feature Attack, Stage 2: Decision Map Attack)')
    parser.add_argument('--losses.decision_loss_weight', type=float, help='Weight for decision loss (Stage 1: Feature Attack, Stage 2: Decision Map Attack)')
    # 阶段2 特有参数 (注意力攻击)
    parser.add_argument('--losses.attention_loss_type', type=str, choices=['mse', 'l1', 'kl', 'js', 'topk', 'cosine', 'none'], help='Type of attention loss (Stage 2 only)')
    parser.add_argument('--losses.attention_loss_weight', type=float, help='Weight for attention loss (Stage 2 only)')
    parser.add_argument('--losses.topk_k', type=int, help='K value for Top-K attention loss/evaluation')
    parser.add_argument('--regularization.lambda_l_inf', type=float, help='Weight for L-infinity regularization')
    parser.add_argument('--regularization.lambda_l2', type=float, help='Weight for L2 regularization')
    parser.add_argument('--regularization.lambda_tv', type=float, help='Weight for total variation loss')
    parser.add_argument('--regularization.lambda_l2_penalty', type=float, help='Weight for L2 penalty on generator parameters')
    parser.add_argument('--evaluation.eval_interval', type=int, help='Evaluate model every N epochs')
    parser.add_argument('--evaluation.num_eval_batches', type=int, help='Number of batches to use for evaluation')
    parser.add_argument('--evaluation.success_threshold', type=float, help='Threshold for attack success in evaluation')
    parser.add_argument('--evaluation.success_criterion', type=str, choices=['mse_diff_threshold', 'mean_change_threshold', 'topk_value_drop', 'topk_position_change'], help='Criterion for attack success in evaluation')
    parser.add_argument('--evaluation.attention_success_criterion', type=str, choices=['mse_diff_threshold', 'mean_change_threshold', 'topk_value_drop', 'topk_position_change', 'none'], help='Criterion for attention attack success in evaluation (Stage 2 only)')
    parser.add_argument('--evaluation.attention_success_threshold', type=float, help='Threshold for attention attack success in evaluation (Stage 2 only)')


    # 解析所有参数，包括之前未知的参数 (它们现在应该能被新parser识别)
    # 使用 parse_args() 确保所有已定义的参数都被解析，如果仍有未知参数会报错
    # 如果需要允许未知参数，继续使用 parse_known_args() 并处理 unknown_args
    cmd_args = parser.parse_args(unknown) # 将之前捕获的 unknown 参数传递给新的 parser

    # 将命令行参数的值合并到 cfg 中，覆盖配置文件中的相应值
    # 使用点号分割的参数名来导航和设置 EasyDict 中的嵌套属性
    for arg_name, arg_value in vars(cmd_args).items():
        # 只有当命令行参数的值不是其默认值 (None) 且不是用于指定 config 文件本身时才进行覆盖
        # 注意：argparse 默认值是 None，但如果用户显式提供了 None (例如对于可选参数)，这里也会处理
        # 更严谨的检查是判断 arg_value 是否与 cfg 中对应路径的当前值不同
        # 但此处简化处理：只要命令行提供了非 None 的值，就进行覆盖 (除了 config 参数本身)
        if arg_value is not None and arg_name != task_config_arg:
            keys = arg_name.split('.')
            current_cfg = cfg
            # 遍历参数名中的键，逐层深入 EasyDict 结构
            for i, key in enumerate(keys):
                # 如果是最后一层键，直接设置值
                if i == len(keys) - 1:
                    # 检查当前层是否是字典类型，以便设置键值
                    if isinstance(current_cfg, (dict, edict)):
                         current_cfg[key] = arg_value
                    else:
                        # 如果当前层不是字典，说明配置结构与参数名不符，打印警告并跳过此参数
                        print(f"Warning: Could not set config value for {arg_name}. Invalid structure at key '{key}'. Current structure type: {type(current_cfg)}.")
                else:
                    # 如果不是最后一层键，尝试进入下一层嵌套结构
                    # 检查下一层键是否存在且对应值是字典类型 (或 EasyDict)
                    if isinstance(current_cfg, (dict, edict)) and key in current_cfg and isinstance(current_cfg[key], (dict, edict)):
                         # 进入下一层
                         current_cfg = current_cfg[key]
                    else:
                         # 如果下一层结构不存在或不是字典类型，尝试创建它 (仅限 EasyDict)
                         # 如果当前层不是 EasyDict，则无法创建嵌套结构，打印警告并跳过此参数
                         if isinstance(current_cfg, edict):
                              # 创建缺失的嵌套 EasyDict 结构
                              current_cfg[key] = edict()
                              # 进入新创建的 EasyDict
                              current_cfg = current_cfg[key]
                         else:
                              # 如果当前层不是 EasyDict，无法创建嵌套结构，打印警告并跳过
                              print(f"Warning: Could not navigate config structure for {arg_name}. Missing or invalid intermediate key: {key}. Current structure type: {type(current_cfg)}.")
                              break # 停止处理当前命令行参数

    # 返回最终合并和覆盖后的配置对象
    return cfg
