import torch
from torch.utils.data import Dataset, DataLoader, Subset # Import DataLoader and Subset
import cv2
import os
import json
import numpy as np
import torchvision.transforms as transforms # Import transforms
import random # Import random for potential seed setting
from typing import Optional, Tuple, Any, Dict, List # Import type hints: Optional, Tuple, Any, Dict, List

from IrisArknights import calculate_transform_matrix, transform_image, Level

# Define a custom collate function to filter out None values
def collate_fn_skip_none(batch):
    """
    Custom collate function that filters out None values from the batch
    and uses default_collate for the remaining items. Handles empty batches.
    """
    # Filter out None values from the batch
    batch = [item for item in batch if item is not None]
    
    # If the batch is empty after filtering, return an empty tensor.
    # Returning a simple empty tensor like torch.empty(0) prevents the TypeError in DataLoader.
    if not batch:
        import torch # Ensure torch is available
        return torch.empty(0)
        
    # Use default_collate on the valid samples
    import torch.utils.data._utils.collate
    return torch.utils.data._utils.collate.default_collate(batch)

# Define worker initialization function for multiprocessing DataLoaders
def worker_init_fn(worker_id: int) -> None:
    """
    初始化 cv2.VideoCapture 实例，使其与 DataLoader 的每个 worker 进程关联。
    这确保每个 worker 都有一个独立的视频文件句柄，避免多进程读写同一个文件句柄的问题。
    同时，为保证可复现性（如果需要），可以在此设置 worker 的随机种子。

    Args:
        worker_id (int): 当前 worker 进程的唯一 ID。
    """
    # The actual VideoCapture initialization will now happen lazily in __getitem__
    # when num_workers=0. For num_workers > 0, this function would typically
    # initialize dataset.cap = cv2.VideoCapture(dataset.video_path).
    # Given that worker_init_fn didn't seem to run for num_workers=0, we are
    # temporarily bypassing its primary responsibility for this debug case.
    pass # Keep the function definition, but no video opening logic here for now


class GameVideoDataset(Dataset):
    """
    用于加载游戏视频和关卡 JSON 数据，并进行透视变换的数据集类。
    每个样本是一个包含 sequence_length 帧的视频序列。
    支持对提取的帧进行透视变换、Resize 和其他自定义 transform。
    使用 worker_init_fn 确保多进程数据加载时每个 worker 有独立的视频文件句柄。
    """
    def __init__(self,
                 video_path: str,
                 level_json_path: str,
                 sequence_length: int,
                 transform: Optional[Any] = None,
                 target_height: Optional[int] = None,
                 target_width: Optional[int] = None,
                 device: torch.device = torch.device('cpu')
                ):
        """
        初始化游戏视频数据集实例。

        Args:
            video_path (str): 视频文件的路径。
            level_json_path (str): 关卡 JSON 文件的路径，用于计算透视变换矩阵。期望 JSON 格式符合 IrisArknights 工具的要求。
            sequence_length (int): 每个样本包含的帧序列长度。例如，sequence_length=16 表示每个样本是连续的 16 帧。
            transform (Optional[Any], optional): 应用于每个帧的可选的 PyTorch transform 或 transform 组合。默认为 None。
                                               这个 transform 应该期望输入为 (C, H, W) 的 float 张量 (值在 [0, 1])，并返回相同格式的张量。
            target_height (Optional[int], optional): 如果指定，将透视变换后的帧缩放到此高度。默认为 None。
            target_width (Optional[int], optional): 如果指定，将透视变换后的帧缩放到此宽度。默认为 None。
                                              如果只指定了其中一个，Resize 将不生效。如果两者都指定，优先使用它们。
            device (torch.device, optional): 用于张量操作的设备。默认为 torch.device('cpu')。
        Raises:
            IOError: 如果无法打开视频文件进行初始化 (获取总帧数)。
        """
        # 存储初始化参数
        self.video_path: str = video_path
        self.level_json_path: str = level_json_path
        self.sequence_length: int = sequence_length
        self.transform: Optional[Any] = transform # Optional external transform
        self.target_height: Optional[int] = target_height
        self.target_width: Optional[int] = target_width
        self.device: torch.device = device # Store the device for tensor operations
        
        # cv2.VideoCapture 实例：将在每个 worker 进程中由 worker_init_fn 初始化，避免多进程冲突。
        # 在主进程 __init__ 中将其初始化为 None。
        # 如果 num_workers = 0 (单进程)，则直接在此处初始化 VideoCapture
        self.cap: Optional[cv2.VideoCapture] = None # Initialize to None

        # --- 1. 初始化关卡和变换矩阵 ---
        # 尝试从关卡 JSON 文件计算透视变换矩阵和逆变换矩阵。
        # 这些矩阵用于将原始视频帧的特定区域（如游戏区域）变换到标准视图。
        try:
            self.level: Level # 用于存储解析后的关卡对象 Type hint for level
            self.transform_matrix: np.ndarray # 透视变换矩阵 Type hint for transform matrix
            self.inverse_transform_matrix: np.ndarray # 逆透视变换矩阵 Type hint for inverse transform matrix
            # calculate_transform_matrix 是外部工具函数，期望返回 level 对象和 numpy 变换矩阵
            self.level, self.transform_matrix, self.inverse_transform_matrix = calculate_transform_matrix(self.level_json_path)
            # Debug 日志：为 {} 初始化关卡和变换矩阵成功。
            # print(f"Debug: Initialized level and transform matrix for {self.level_json_path}") # Debug print
        except Exception as e:
            # 错误信息：从 {} 初始化关卡和变换矩阵失败：{}。依赖变换的步骤将失败。
            print(f"Error: Failed to initialize level and transform matrix from {self.level_json_path}: {e}")
            # 如果初始化失败，将相关属性设置为 None，并在 __getitem__ 中进行检查
            self.level = None
            self.transform_matrix = None
            self.inverse_transform_matrix = None
            # 注意：如果变换矩阵失败，后续尝试进行 transform_image 将会出错，__getitem__ 会捕获并返回 None

        # --- 2. 加载视频并获取帧数 (仅在 __init__ 中获取总帧数，不在此时保持cap对象) ---
        # 在主进程 __init__ 中打开视频一次以获取总帧数（确定数据集大小），然后立即释放。
        # 实际的视频读取将在 __getitem__ 中由每个 worker 独立完成，使用各自的 self.cap 实例。
        cap_init = cv2.VideoCapture(self.video_path)
        if not cap_init.isOpened():
            # 错误信息：无法打开视频文件进行初始化：{}。
            raise IOError(f"Cannot open video file for initialization: {video_path}")
        # 获取视频的总帧数 (cv2.CAP_PROP_FRAME_COUNT 返回浮点数，需要转换为整数)
        self.total_frames: int = int(cap_init.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_init.release() # 立即释放视频捕获对象，避免资源泄露

        # 确认数据集初始化成功日志：初始化数据集成功，视频：{}，总帧数：{}。
        # print(f"Initialized GameVideoDataset for {self.video_path}, total frames: {self.total_frames}") # Removed verbose initialization print
        
        # --- 3. 创建 Resize transform 如果需要 ---
        # 如果指定了目标高度和宽度，则创建一个 Resize transform
        self.resize_transform: Optional[transforms.Resize] = None # Type hint and initialize to None
        # 确保 target_height 和 target_width 都不是 None 且大于 0
        if self.target_height is not None and self.target_width is not None and self.target_height > 0 and self.target_width > 0:
             # 使用 transforms.Resize 创建一个图像缩放 transform
             # 注意：对于图像，通常使用双线性 (bilinear) 或双三次 (bicubic) 插值
             # 默认的插值方法取决于 torchvision 版本，通常是 bilinear。
             # 可以通过 interpolation 参数明确指定，例如 transforms.InterpolationMode.BILINEAR
             # 例如: transforms.Resize((self.target_height, self.target_width), interpolation=transforms.InterpolationMode.BILINEAR)
             self.resize_transform = transforms.Resize((self.target_height, self.target_width)) # Use interpolation='bilinear' or 'bicubic' if resizing images
        # else: 如果 target_height 或 target_width 未指定或无效，则不进行 Resize


    def __len__(self) -> int:
        """
        返回数据集中的样本总数。
        样本数 = 总帧数 - 序列长度 + 1 (可以构成完整 sequence_length 帧序列的起始帧数)。
        如果总帧数不足以构成一个完整的序列，则返回 0。
        """
        # max(0, ...) 确保返回非负数
        return max(0, self.total_frames - self.sequence_length + 1)

    def __getitem__(self, idx: int) -> Optional[torch.Tensor]: # Return type hint: Optional[torch.Tensor] for returning None
        """
        获取指定索引 (起始帧索引) 的视频帧序列样本。

        Args:
            idx (int): 样本的起始帧索引 (0-indexed)。此索引对应于视频的全局帧索引。

        Returns:
            Optional[torch.Tensor]: 包含 sequence_length 帧的张量，形状为 (C, T, H, W)，
                                    数据类型为 float，像素值在 [0, 1] 范围内 (RGB 格式)。
                                    如果读取或处理过程中发生错误 (如视频未打开、读取帧失败、透视变换失败)，返回 None。
        """
        # Debug 日志：__getitem__ Called for index {idx}
        # print(f"Debug: __getitem__ called for index {idx}") # Debug print at start
        
        # Using VideoCapture instance initialized in worker_init_fn for multiprocessing.
        # For num_workers=0 (single process), worker_init_fn might not be reliably called.
        # Implement lazy initialization here if self.cap is None (covers num_workers=0 case primarily).
        if self.cap is None:
            # Debug print: Attempting lazy VideoCapture initialization in __getitem__
            print(f"Debug: __getitem__ index {idx} - Performing lazy VideoCapture initialization for {self.video_path}")
            self.cap = cv2.VideoCapture(self.video_path)
            if self.cap.isOpened():
                # print(f"Debug: __getitem__ index {idx} - Lazy VideoCapture initialization successful.") # Removed debug print
                pass # Added pass to fix linter error
            else:
                print(f"Error: __getitem__ index {idx} - Lazy VideoCapture initialization failed.")

        # 检查 self.cap 是否已初始化且打开
        if self.cap is None or not self.cap.isOpened():
            # 如果 worker_init_fn 没有正确执行（例如视频文件不存在或损坏），这里会检测到
            # 错误信息：__getitem__ 索引 {} - VideoCapture 未初始化或未打开。跳过此样本。
            print(f"Error: __getitem__ index {idx} - VideoCapture not initialized or not opened. Skipping sample.")
            return None # 返回 None 表示此序列无效，DataLoader 会自动跳过 None 样本 (需要默认的 collate_fn 或自定义)

        # 设置视频帧的起始位置 (使用基于 0 的索引 'idx')
        # cv2.CAP_PROP_POS_FRAMES 属性用于设置下一个要读取的帧的索引
        # print(f"Debug: __getitem__ index {idx} - Before cap.set(POS_FRAMES, {idx})") # Debug print before set
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        # print(f"Debug: __getitem__ index {idx} - After cap.set(POS_FRAMES, {idx})") # Debug print after set
        
        sequence_frames: List[torch.Tensor] = [] # 用于存储读取和处理后的帧张量列表 Type hint for list of tensors
        # Debug 日志：__getitem__ 索引 {} - Starting frame read loop for {self.sequence_length} frames.
        # print(f"Debug: __getitem__ index {idx} - Starting frame read loop for {self.sequence_length} frames.") # Debug print before loop

        # --- 遍历并读取 sequence_length 帧 --- #
        for i in range(self.sequence_length):
            # Debug 日志：__getitem__ 索引 {}，帧 {} - Before cap.read()
            # print(f"Debug: __getitem__ index {idx}, frame {i} - Before cap.read()") # Debug print before read
            # 读取当前帧。ret 是一个布尔值表示是否成功读取，frame 是读取到的帧 (numpy 数组)。
            ret, frame = self.cap.read()
            # Debug 日志：__getitem__ 索引 {}，帧 {} read, ret: {}
            # print(f"Debug: __getitem__ index {idx}, frame {i} read, ret: {ret}") # Debug print after reading frame
            
            # 检查帧是否成功读取，并且 frame 不是 None (读取失败时 frame 可能为 None)
            if not ret or frame is None:
                # 如果读取失败，可能是文件结束、读取错误或文件损坏
                # 警告信息：__getitem__ 索引 {}，帧 {} - 无法读取全局索引为 {} 的帧。返回 None。
                print(f"Warning: __getitem__ index {idx}, frame {i} - Could not read frame at global index {idx + i}. Sequence index {idx}. Returning None.")
                # 返回 None 表示此序列无效，DataLoader 会自动跳过 None 样本 (需要默认的 collate_fn 或自定义)
                return None

            # --- 对当前帧进行预处理 (透视变换, Resize, 颜色空间转换, numpy转Tensor, 归一化) ---
            
            # 对当前帧应用透视变换 (如果变换矩阵已初始化)
            try:
                # Debug 日志：__getitem__ 索引 {}，帧 {} - Before transform_image.
                # print(f"Debug: __getitem__ index {idx}, frame {i} - Before transform_image.") # Debug print before transform
                # 检查 level 和 transform_matrix 是否已成功初始化 (__init__ 中)
                if self.level is None or self.transform_matrix is None:
                     # 错误信息：__getitem__ 索引 {}，帧 {} - 关卡或变换矩阵未初始化。跳过此样本。
                     print(f"Error: __getitem__ index {idx}, frame {i} - Level or transform matrix not initialized. Skipping sample.")
                     return None # 停止处理此序列并返回 None，因为它依赖于变换矩阵

                # 应用透视变换到当前帧。返回的是 numpy 数组。
                processed_frame = transform_image(self.level, self.transform_matrix, frame)
                # Debug 日志：__getitem__ 索引 {}，帧 {} - After transform_image, shape: {}, dtype: {}
                # print(f"Debug: __getitem__ index {idx}, frame {i} - After transform_image, shape: {processed_frame.shape}, dtype: {processed_frame.dtype}") # Debug print after transform_image
            except Exception as e:
                # 错误信息：__getitem__ 索引 {}，帧 {} - 透视变换期间出错：{}。返回 None。
                print(f"Error: __getitem__ index {idx}, frame {i} - Error during transform_image: {e}. Returning None.")
                return None # 停止处理此序列并返回 None

            
            # 转换为 PyTorch 兼容的格式 (H, W, C) 和数据类型
            # OpenCV 读取的图像是 BGR 格式的 numpy 数组 (H, W, C)，值在 [0, 255]
            # 1. 将颜色空间从 BGR (cv2 默认) 转换为 RGB
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            # Debugging: Check the numpy array before converting to torch tensor
            if not isinstance(processed_frame, np.ndarray):
                print(f"Error: __getitem__ index {idx}, frame {i} - processed_frame is not a numpy array. Type: {type(processed_frame)}")
                return None
            if processed_frame.size == 0:
                print(f"Error: __getitem__ index {idx}, frame {i} - processed_frame is an empty numpy array. Shape: {processed_frame.shape}")
                return None
            # Check the number of channels after color conversion
            if processed_frame.ndim != 3 or processed_frame.shape[2] != 3:
                # Add a specific warning if channels are not 3
                print(f"Warning: __getitem__ index {idx}, frame {i} - processed_frame has unexpected channels after BGR2RGB conversion. Expected 3, got {processed_frame.shape[2] if processed_frame.ndim == 3 else processed_frame.ndim}. Shape: {processed_frame.shape}")
                # Depending on desired behavior, you might return None or try to handle non-3 channels
                # For now, let's return None if channels are not 3, as the models expect 3 channels.
                return None # Return None if channel count is not 3

            # Ensure the numpy array is contiguous and writable before converting to tensor
            # This can help with multiprocessing issues
            if processed_frame is not None:
                processed_frame = np.ascontiguousarray(processed_frame)
                # Add a copy operation to potentially help with multiprocessing/serialization issues
                processed_frame = processed_frame.copy()
            else:
                 print(f"Warning: __getitem__ index {idx}, frame {i} - processed_frame is None after transform_image. Skipping tensor conversion.")
                 sequence_frames.append(None) # Append None if processing failed
                 continue # Skip to next frame in sequence

            # Convert numpy array (H, W, C) to torch tensor (C, H, W)
            # OpenCV reads as BGR, so convert to RGB
            # processed_frame is (H, W, C) after transform_image (which uses cv2.resize, preserving channel dim)
            # Permute to (C, H, W)
            frame_tensor = torch.from_numpy(processed_frame).permute(2, 0, 1).float()

            # 应用 Resize transform 如果存在且已初始化
            if self.resize_transform:
                 try:
                    # Debug 日志：__getitem__ 索引 {}，帧 {} - Before resize_transform, input shape: {}
                    # print(f"Debug: __getitem__ index {idx}, frame {i} - Before resize_transform, input shape: {processed_frame.shape}") # Debug print before resize
                    # resize_transform 期望 (C, H, W) 张量并返回 (C, H_new, W_new)
                    frame_tensor = self.resize_transform(frame_tensor)
                    # Debug 日志：__getitem__ 索引 {}，帧 {} - After resize_transform, shape: {}
                    # print(f"Debug: __getitem__ index {idx}, frame {i} - After resize_transform, shape: {frame_tensor.shape}") # Debug print after resize
                 except Exception as e:
                    # 错误信息：__getitem__ 索引 {}，帧 {} - Resize 变换期间出错：{}。返回 None。
                    print(f"Error: __getitem__ index {idx}, frame {i} - Error during resize_transform: {e}. Returning None.")
                    return None # 停止处理并返回 None

            # 归一化像素值从 [0, 255] 范围到 [0, 1] 范围
            # 确保数据类型是 float 才能进行除法
            if frame_tensor.max() > 1.0 + 1e-6: # 避免对已经归一化的数据再次归一化
                 frame_tensor = frame_tensor / 255.0
            # Debug 日志：__getitem__ 索引 {}，帧 {} - After normalization.
            # print(f"Debug: __getitem__ index {idx}, frame {i} - After normalization.") # Debug print after normalization

            # 应用可选的外部 transform (例如数据增强，如果用户在 DataLoader 中提供了)
            # 这个 transform 应该作用在单个帧 (C, H, W) 张量上
            if self.transform:
                try:
                    # Debug 日志：__getitem__ 索引 {}，帧 {} - Before external transform.
                    # print(f"Debug: __getitem__ index {idx}, frame {i} - Before external transform.") # Debug print before external transform
                    # 外部 transform 应期望输入为 (C, H, W) float 张量并返回相同格式的张量
                    frame_tensor = self.transform(frame_tensor)
                    # Debug 日志：__getitem__ 索引 {}，帧 {} - After external transform.
                    # print(f"Debug: __getitem__ index {idx}, frame {i} - After external transform.") # Debug print after external transform
                except Exception as e:
                    # 错误信息：__getitem__ 索引 {}，帧 {} - 外部 transform 期间出错：{}。返回 None。
                    # print(f"Error: __getitem__ index {idx}, frame {i} - Error during external transform: {e}. Returning None.")
                    return None # 停止处理并返回 None

            # 将处理好的帧张量添加到序列列表中
            sequence_frames.append(frame_tensor)

            # Debug print: Check shape and max value of each frame before stacking
            # print(f"Debug: __getitem__ index {idx}, frame {i} - frame_tensor shape: {frame_tensor.shape}, max value: {frame_tensor.max().item()}") # Removed debug print

        # 将序列帧列表堆叠成一个张量 (T, C, H, W)，其中 T = sequence_length
        # 在堆叠之前，再次检查 sequence_frames 列表是否包含有效数量的张量
        if len(sequence_frames) != self.sequence_length or not all(torch.is_tensor(f) for f in sequence_frames):
             print(f"Error: __getitem__ index {idx} - Not all frames in sequence_frames are valid tensors. Expected {self.sequence_length} tensors, got {len(sequence_frames)} tensors and {sum(1 for f in sequence_frames if not torch.is_tensor(f))} non-tensors. Returning None.")
             return None # Return None if the list doesn't contain the expected number of tensors

        try:
             sequence_tensor = torch.stack(sequence_frames, dim=0) # Stack the list of tensors into a single tensor (T, C, H, W)
             # Debug 日志：__getitem__ 索引 {} - Stacked sequence, shape before final permute: {}
             # print(f"Debug: __getitem__ index {idx} - Stacked sequence, shape before final permute: {sequence_tensor.shape}") # Debug print

             # Permute dimension to match test.py expected shape (T, H, W, C)
             # Input frames are currently (T, C, H, W)
             sequence_tensor = sequence_tensor.permute(0, 2, 3, 1) # (T, C, H, W) -> (T, H, W, C)
             # Debug 日志：__getitem__ 索引 {} - Stacked and permuted sequence, final shape: {} 理应是 (C, T, H, W) 才能匹配后续模型输入 (B, C, T, H, W)
             # print(f"Debug: __getitem__ index {idx} - Stacked and permuted sequence, final shape: {sequence_tensor.shape}") # Debug print before return

             # TODO: If data set needs to return masks (decision area mask, attention area mask, etc.), load/generate masks here and return them
             # Depending on the specific needs of the project, you might need to load or generate corresponding masks for each sample.
             # If multiple items need to be returned, DataLoader's collate_fn needs to be adjusted accordingly.
             # Currently only returns image sequence tensor
             # return sequence_tensor, decision_mask, attention_mask
             return sequence_tensor # Only return image sequence tensor for now
        except Exception as e:
             print(f"Error: __getitem__ index {idx} - Error stacking sequence frames: {e}. Returning None.")
             return None # Return None if stacking fails

    # __del__ 方法在对象被销毁时调用，但对于多进程数据加载，VideoCapture 句柄应该由 worker_init_fn 管理和释放。
    # 在 worker_init_fn 中初始化的 self.cap 实例会在 worker 进程结束时自动清理。

def create_mock_dataloader( batch_size: int, num_samples: int, sequence_length: int, channels: int,height: int, width: int,shuffle: bool = True, num_workers: int = 0) -> DataLoader: # Type hint for DataLoader
    """
    创建一个返回模拟数据的数据加载器（DataLoader）。
    主要用于在没有真实视频数据时，快速测试模型的训练和评估流程。
    """
    # 创建一个模拟数据集实例
    mock_dataset = MockDataset(num_samples, sequence_length, channels, height, width)
    # 创建一个 DataLoader 实例，使用 mock_dataset 和指定的参数
    return DataLoader(mock_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)