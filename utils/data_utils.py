import torch
from torch.utils.data import Dataset
import cv2
import os
import json
import numpy as np
import torchvision.transforms as transforms # Import transforms

from IrisArknights import calculate_transform_matrix, transform_image, Level

class GameVideoDataset(Dataset):
    def __init__(self, video_path, level_json_path, sequence_length, transform=None, target_height=None, target_width=None):
        self.video_path = video_path
        self.level_json_path = level_json_path
        self.sequence_length = sequence_length
        self.transform = transform # Optional external transform
        self.target_height = target_height
        self.target_width = target_width

        # 1. 初始化关卡和变换矩阵
        # assuming calculate_transform_matrix and Level are available and work correctly
        self.level, self.transform_matrix, self.inverse_transform_matrix = calculate_transform_matrix(self.level_json_path)

        # 2. 加载视频并获取帧数
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 创建一个 Resize transform 如果需要
        self.resize_transform = None
        if self.target_height is not None and self.target_width is not None:
             self.resize_transform = transforms.Resize((self.target_height, self.target_width))


    def __len__(self):
        # 确保有足够的帧来形成一个序列
        return max(0, self.total_frames - self.sequence_length + 1)

    def __getitem__(self, idx):
        # 设置视频帧的起始位置
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        
        sequence_frames = []
        for _ in range(self.sequence_length):
            ret, frame = self.cap.read()
            if not ret:
                # 如果读取失败，尝试获取下一序列
                print(f"Warning: Could not read frame at index {idx + _}. Attempting next sequence.")
                return self.__getitem__(idx + 1) if idx + 1 < len(self) else None # 尝试获取下一序列
            
            # 对当前帧进行透视变换
            # 调用 transform_image 函数，传入 level 实例, 变换矩阵 M, 和当前帧
            processed_frame = transform_image(self.level, self.transform_matrix, frame)
            
            # 转换为 PyTorch 兼容的格式 (C, H, W) 和数据类型
            # Convert from BGR (cv2 default) to RGB
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB) 
            # Convert numpy array to torch tensor
            processed_frame = torch.from_numpy(processed_frame).permute(2, 0, 1).float() # HWC to CHW, to float

            # 应用 Resize transform 如果存在
            if self.resize_transform:
                 processed_frame = self.resize_transform(processed_frame)

            # 归一化到 [0, 1]
            processed_frame = processed_frame / 255.0

            # 应用可选的外部 transform
            if self.transform:
                processed_frame = self.transform(processed_frame)

            sequence_frames.append(processed_frame)

        # 将序列帧堆叠成一个张量 (T, C, H, W)
        sequence_tensor = torch.stack(sequence_frames, dim=0)
        
        # 你的模型可能期望 (B, C, T, H, W) 格式，这里需要调整
        # Dataset __getitem__ 应该返回单个样本，所以是 (T, C, H, W)
        # DataLoader 会堆叠成 (B, T, C, H, W)
        # 如果你的模型期望 (B, C, T, H, W)，你可能需要在 DataLoader 的 collate_fn 中处理，
        # 或者在训练循环中 permute 张量。
        # 这里假设 DataLoader 返回 (B, T, C, H, W) 并需要在训练循环中转换为 (B, C, T, H, W)

        # Add sequence dimension after batching in DataLoader
        # Return (C, T, H, W) or (T, C, H, W)? Let's return (C, T, H, W)
        # No, DataLoader expects samples to be stacked. Let's return (T, C, H, W)
        # Permute in the training loop if (B, C, T, H, W) is needed.

        # Permute to (C, T, H, W) for consistency with model input expectation
        # The generator input is (B, C, N, W, H) - wait, let's check Generator __init__
        # Generator expects (B, 128, N, W, H)
        # So the input to the generator should be (B, Channels, Sequence_Length, Width, Height)
        # Let's adjust the dataset output to (Channels, Sequence_Length, Height, Width)
        # Original: (T, C, H, W) -> permute(1, 0, 2, 3) -> (C, T, H, W)
        # Need (C, N, H, W) where N=T
        sequence_tensor = sequence_tensor.permute(1, 0, 2, 3) # (C, T, H, W)

        # TODO: If using masks, load/generate masks here and return them
        # return sequence_tensor, decision_mask, attention_mask
        return sequence_tensor # Only return features for now

    def __del__(self):
        # 释放视频捕获对象
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

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
    dataloader = DataLoader(mock_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader