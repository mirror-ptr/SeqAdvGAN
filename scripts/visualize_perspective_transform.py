import sys
import os
import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 将项目根目录添加到 Python 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.data_utils import GameVideoDataset

def visualize_perspective_transform(video_path, level_json_path, sequence_length, target_height, target_width, num_batches_to_vis=1, visualize_original=False, display_delay_ms=1):
    """
    可视化 GameVideoDataset 应用透视变换后的视频帧。

    Args:
        video_path (str): 视频文件路径。
        level_json_path (str): 关卡 JSON 文件路径。
        sequence_length (int): 每个数据样本的序列长度。
        target_height (int): 变换后图像的目标高度。
        target_width (int): 变换后图像的目标宽度。
        num_batches_to_vis (int): 可视化多少个批次的数据。
        visualize_original (bool): 是否同时可视化原始帧 (需要修改 GameVideoDataset 来返回原始帧)。
        display_delay_ms (int): OpenCV imshow 的显示延迟（毫秒）。越小越快，0表示无限等待按键。
    """
    print(f"Loading video from {video_path}")
    print(f"Loading level JSON from {level_json_path}")

    try:
        # 初始化数据集
        dataset = GameVideoDataset(
            video_path=video_path,
            level_json_path=level_json_path,
            sequence_length=sequence_length,
            target_height=target_height, # Use the passed target height
            target_width=target_width,   # Use the passed target width
        )

        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0) # Use batch_size=1 for simple frame-by-frame vis

        cv2.namedWindow('Transformed Frame', cv2.WINDOW_NORMAL)
        if visualize_original:
            cv2.namedWindow('Original Frame', cv2.WINDOW_NORMAL)

        for batch_idx, data in enumerate(dataloader):
            if batch_idx >= num_batches_to_vis:
                break

            # Assuming data is (C, N, H, W) for transformed frames
            # data[0] is the batch, data[0][0] is the first sample in the batch
            # data[0][0][:, seq_idx, :, :] for a specific sequence step
            
            # For visualization, we'll take the first frame of the sequence for simplicity
            # Convert PyTorch tensor (C, N, H, W) to OpenCV format (H, W, C)
            # And scale back to 0-255 for display
            transformed_sequence_tensor = data[0].squeeze(0) # Remove batch dim: (C, N, H, W)

            # Take the first frame of the sequence for display
            transformed_frame_to_display = transformed_sequence_tensor[:, 0, :, :].permute(1, 2, 0).cpu().numpy()
            transformed_frame_to_display = (transformed_frame_to_display * 255).astype(np.uint8)
            
            # Convert RGB to BGR for OpenCV display
            transformed_frame_to_display = cv2.cvtColor(transformed_frame_to_display, cv2.COLOR_RGB2BGR)

            cv2.imshow('Transformed Frame', transformed_frame_to_display)

            # Optional: Visualize original frame (requires GameVideoDataset to return it)
            # You would need to modify GameVideoDataset's __getitem__ to return original_frame as well
            # if visualize_original and len(data) > 1:
            #     original_frame = data[1].squeeze(0).permute(1, 2, 0).cpu().numpy()
            #     original_frame = (original_frame * 255).astype(np.uint8)
            #     original_frame = cv2.cvtColor(original_frame, cv2.COLOR_RGB2BGR)
            #     cv2.imshow('Original Frame', original_frame)


            key = cv2.waitKey(display_delay_ms) & 0xFF
            if key == ord('q'): # Press 'q' to quit
                break

            if key == 27: # Esc key
                break

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        print("Please ensure the video_path and level_json_path are correct.")
    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # 清理 OpenCV 窗口
        cv2.destroyAllWindows()
        if 'dataset' in locals() and hasattr(dataset, 'cap') and dataset.cap.isOpened():
            dataset.cap.release() # Ensure video capture is released

if __name__ == "__main__":
    test_video_path = 'data/videos/1392e8264d9904bfffe142421b784cbd.mp4'
    test_level_json_path = 'resources/level/hard_15-01-obt-hard-level_hard_15-01.json'
    test_sequence_length = 5
    
    # --- Recommended changes for faster debugging ---
    test_target_height = 1080 # Reduce resolution for faster processing
    test_target_width = 1920  # Reduce resolution for faster processing
    test_num_batches_to_vis = 99999999 # Visualize fewer batches
    display_delay = 1 # Milliseconds to wait, smaller means faster playback if processing allows
    # ------------------------------------------------

    visualize_perspective_transform(
        video_path=test_video_path,
        level_json_path=test_level_json_path,
        sequence_length=test_sequence_length,
        target_height=test_target_height,
        target_width=test_target_width,
        num_batches_to_vis=test_num_batches_to_vis,
        visualize_original=False, # Set to True if GameVideoDataset is modified to return original frame
        display_delay_ms=display_delay
    )