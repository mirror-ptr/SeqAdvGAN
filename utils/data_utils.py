import torch
from torch.utils.data import Dataset
import cv2
import os
import json
import numpy as np
import torchvision.transforms as transforms # Import transforms

from IrisArknights import calculate_transform_matrix, transform_image, Level

# Define the worker initialization function
def worker_init_fn(worker_id):
    """
    Initializes cv2.VideoCapture for each worker process.
    """
    # Get the dataset instance for the current worker
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset # dataset is the instance of GameVideoDataset

    # Create a VideoCapture instance for this worker
    print(f"Debug: Worker {worker_id} initializing VideoCapture for {dataset.video_path}")
    dataset.cap = cv2.VideoCapture(dataset.video_path)
    
    if not dataset.cap.isOpened():
        print(f"Error: Worker {worker_id} failed to open video file: {dataset.video_path}")
        # Depending on desired behavior, could raise an error or handle gracefully
        # For now, let it proceed, but subsequent read calls will fail.

    # Optional: set a unique seed for each worker if needed for randomness
    # np.random.seed(worker_info.seed)
    # random.seed(worker_info.seed)


class GameVideoDataset(Dataset):
    def __init__(self, video_path, level_json_path, sequence_length, transform=None, target_height=None, target_width=None):
        self.video_path = video_path
        self.level_json_path = level_json_path
        self.sequence_length = sequence_length
        self.transform = transform # Optional external transform
        self.target_height = target_height
        self.target_width = target_width
        # self.cap will be initialized by worker_init_fn in each worker process
        self.cap = None # Initialize to None

        # 1. 初始化关卡和变换矩阵
        # assuming calculate_transform_matrix and Level are available and work correctly
        try:
            self.level, self.transform_matrix, self.inverse_transform_matrix = calculate_transform_matrix(self.level_json_path)
            print(f"Debug: Initialized level and transform matrix for {self.level_json_path}")
        except Exception as e:
            print(f"Error: Failed to initialize level and transform matrix from {self.level_json_path}: {e}")
            # Depending on desired behavior, might raise an error or set flags
            self.level = None
            self.transform_matrix = None
            self.inverse_transform_matrix = None

        # 2. 加载视频并获取帧数 (仅在 __init__ 中获取总帧数，不在此时保持cap对象)
        # 在 __init__ 中打开视频一次以获取总帧数，然后立即释放。
        # 实际的视频读取将在 __getitem__ 中每个worker独立完成。
        cap_init = cv2.VideoCapture(self.video_path)
        if not cap_init.isOpened():
            raise IOError(f"Cannot open video file for initialization: {video_path}")
        self.total_frames = int(cap_init.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_init.release() # 立即释放视频捕获对象

        print(f"Initialized GameVideoDataset for {self.video_path}, total frames: {self.total_frames}")
        
        # 创建一个 Resize transform 如果需要
        self.resize_transform = None
        if self.target_height is not None and self.target_width is not None:
             self.resize_transform = transforms.Resize((self.target_height, self.target_width)) # Use interpolation='bilinear' or 'bicubic' if resizing images


    def __len__(self):
        # 确保有足够的帧来形成一个序列
        return max(0, self.total_frames - self.sequence_length + 1)

    def __getitem__(self, idx):
        # print(f"Debug: __getitem__ called for index {idx}") # Debug print at start
        
        # Use the VideoCapture instance initialized by worker_init_fn
        if self.cap is None or not self.cap.isOpened():
            # This should ideally not happen if worker_init_fn runs correctly
            print(f"Error: __getitem__ index {idx} - VideoCapture not initialized or not opened.")
            return None

        # 设置视频帧的起始位置
        # print(f"Debug: __getitem__ index {idx} - Setting frame position to {idx}") # Debug print before set
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        # print(f"Debug: __getitem__ index {idx} - Frame position set.") # Debug print after set
        
        sequence_frames = []
        # print(f"Debug: __getitem__ index {idx} - Starting frame read loop for {self.sequence_length} frames.") # Debug print before loop

        for _ in range(self.sequence_length):
            # print(f"Debug: __getitem__ index {idx}, frame {_} - Before cap.read()") # Debug print before read
            ret, frame = self.cap.read()
            # print(f"Debug: __getitem__ index {idx}, frame {_} read, ret: {ret}") # Debug print after reading frame
            if not ret:
                # 如果读取失败，可能是文件结束或读取错误
                print(f"Warning: __getitem__ index {idx}, frame {_} - Could not read frame at global index {idx + _}. Sequence index {idx}.")
                # Instead of recursive call, return None to signal an invalid sample
                return None # 返回 None 表示此序列无效
            
            # 对当前帧进行透视变换
            try:
                # print(f"Debug: __getitem__ index {idx}, frame {_} - Before transform_image.") # Debug print before transform
                # Check if transform matrix was initialized successfully
                if self.level is None or self.transform_matrix is None:
                     print(f"Error: __getitem__ index {idx}, frame {_} - Level or transform matrix not initialized.")
                     return None

                processed_frame = transform_image(self.level, self.transform_matrix, frame)
                # print(f"Debug: __getitem__ index {idx}, frame {_} - After transform_image, shape: {processed_frame.shape}, dtype: {processed_frame.dtype}") # Debug print after transform_image
            except Exception as e:
                print(f"Error: __getitem__ index {idx}, frame {_} - Error during transform_image: {e}")
                return None # Stop processing this sequence on error

            
            # 转换为 PyTorch 兼容的格式 (C, H, W) 和数据类型
            # Convert from BGR (cv2 default) to RGB
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB) 
            # Convert numpy array to torch tensor
            processed_frame = torch.from_numpy(processed_frame).permute(2, 0, 1).float() # HWC to CHW, to float
            # print(f"Debug: __getitem__ index {idx}, frame {_} - After color convert/tensor/permute, shape: {processed_frame.shape}, dtype: {processed_frame.dtype}") # Debug print after conversion and permute

            # 应用 Resize transform 如果存在
            if self.resize_transform:
                 try:
                    # print(f"Debug: __getitem__ index {idx}, frame {_} - Before resize_transform, input shape: {processed_frame.shape}") # Debug print before resize
                    processed_frame = self.resize_transform(processed_frame)
                    # print(f"Debug: __getitem__ index {idx}, frame {_} - After resize_transform, shape: {processed_frame.shape}") # Debug print after resize
                 except Exception as e:
                    print(f"Error: __getitem__ index {idx}, frame {_} - Error during resize_transform: {e}")
                    return None # Stop processing on error

            # 归一化到 [0, 1]
            processed_frame = processed_frame / 255.0
            # print(f"Debug: __getitem__ index {idx}, frame {_} - After normalization.") # Debug print after normalization

            # 应用可选的外部 transform
            if self.transform:
                try:
                    # print(f"Debug: __getitem__ index {idx}, frame {_} - Before external transform.") # Debug print before external transform
                    processed_frame = self.transform(processed_frame)
                    # print(f"Debug: __getitem__ index {idx}, frame {_} - After external transform.") # Debug print after external transform
                except Exception as e:
                    print(f"Error: __getitem__ index {idx}, frame {_} - Error during external transform: {e}")
                    return None # Stop processing on error

            sequence_frames.append(processed_frame)

        # 将序列帧堆叠成一个张量 (T, C, H, W)
        sequence_tensor = torch.stack(sequence_frames, dim=0)
        
        # Permute to (C, T, H, W) for consistency with model input expectation
        # Need (C, N, H, W) where N=T
        sequence_tensor = sequence_tensor.permute(1, 0, 2, 3) # (C, T, H, W)
        # print(f"Debug: __getitem__ index {idx} - Stacked and permuted sequence, final shape: {sequence_tensor.shape}") # Debug print before return

        # TODO: If using masks, load/generate masks here and return them
        # return sequence_tensor, decision_mask, attention_mask
        return sequence_tensor # Only return features for now

    # No need for __del__ as cap is managed by worker_init_fn
    # def __del__(self):
    #     # Release video capture object
    #     if hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
    #         self.cap.release()

# 占位符：创建模拟数据加载器（保留用于测试）
def create_mock_dataloader(batch_size, num_samples, sequence_length, channels, height, width, shuffle=True, num_workers=0):
    """
    创建一个模拟数据的 DataLoader。
    """
    class MockDataset(Dataset):
        def __init__(self, num_samples, sequence_length, channels, height, width):
            self.num_samples = num_samples
            self.sequence_length = sequence_length
            self.channels = channels
            self.height = height
            self.width = width

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            # 模拟数据形状: (C, N, H, W)
            # 确保数据范围在 [0, 1] 以模拟归一化后的图像或特征
            mock_data = torch.randn(self.channels, self.sequence_length, self.height, self.width) * 0.1 + 0.5 # centered around 0.5
            mock_data = torch.clamp(mock_data, 0, 1)
            
            # TODO: 如果需要模拟掩码，在这里生成并返回
            # decision_mask = torch.ones(self.height, self.width)
            # attention_mask = torch.ones(4, self.sequence_length, self.sequence_length) # Assuming 4 heads
            # return mock_data, decision_mask, attention_mask

            return mock_data

    mock_dataset = MockDataset(num_samples, sequence_length, channels, height, width)
    # Pass worker_init_fn to DataLoader when using multiple workers for real data
    # For mock data, worker_init_fn is not strictly necessary unless mock data generation is complex.
    dataloader = DataLoader(mock_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader