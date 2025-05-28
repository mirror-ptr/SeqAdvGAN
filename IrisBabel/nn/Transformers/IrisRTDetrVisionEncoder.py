import torch
from torch import nn
from transformers import AutoImageProcessor, RTDetrV2Model
import os
# 导入 torchvision 库
import torchvision.transforms as T
from PIL import Image

class IrisRTDetrVisionEncoder(nn.Module):
    def __init__(self, output_dim=4096, device=torch.device("cpu")):
        super(IrisRTDetrVisionEncoder, self).__init__()
        self.device = device
        current_path = os.path.realpath(__file__)
        directory_path = os.path.dirname(current_path)
        self.image_processor = AutoImageProcessor.from_pretrained(directory_path + "/pretrained/rtdetr_v2_r50vd")
        self.model = RTDetrV2Model.from_pretrained(directory_path + "/pretrained/rtdetr_v2_r50vd")
        self.model.to(self.device)
        self.projection = nn.Linear(256, output_dim)

    def forward(self, image):
        # 将输入的 PyTorch 张量 (B, C, H, W) 转换为 PIL Image 列表
        pil_images = []
        # 确保输入的张量是 uint8 类型，适合转为 PIL Image
        # 如果输入是浮点类型，可能需要先转换为 [0, 255] 范围并转为 uint8
        # 假设这里的输入已经是 [0, 255] 的 uint8 张量，因为我们在 test_integration.py 中做了处理
        if image.dtype != torch.uint8:
            # 根据需要添加处理浮点输入的逻辑，例如: image = (image * 255).byte()
            # 为了冒烟测试，我们假设输入是 uint8
            print(f"Warning: Expected uint8 tensor for PIL conversion, but got {image.dtype}")
            # 尝试转换，如果不是 uint8 可能仍会出错
            image = (image * 255).byte() # 强制转换为 [0, 255] uint8，基于之前在 test_integration.py 中的调整
            
        to_pil = T.ToPILImage()
        for i in range(image.shape[0]):
            # 提取每个批次的单张图像 (C, H, W)
            single_image_tensor = image[i, :, :, :]
            # 将张量转换为 PIL Image
            # 注意：PIL Image 期望 (H, W, C) 或 (H, W) 格式，ToTensor 默认转为 (C, H, W)
            # ToPILImage 也期望 (C, H, W) 或 (H, W) 或 (H, W, C)，取决于模式。
            # 对于彩色图像 (C=3)，ToPILImage 期望 (C, H, W)。
            pil_image = to_pil(single_image_tensor)
            pil_images.append(pil_image)

        # 将 PIL Image 列表传递给图像处理器
        # image = self.image_processor(image, return_tensors="pt")
        processed_inputs = self.image_processor(images=pil_images, return_tensors="pt")

        # 图像处理器返回的是一个字典，包含处理后的张量
        processed_inputs.to(self.device)
        
        # output = self.model(**image)
        output = self.model(**processed_inputs)
        last_hidden_state = output.last_hidden_state
        pooled_last_hidden_state = torch.mean(last_hidden_state, dim=1)
        final_output = self.projection(pooled_last_hidden_state)
        return final_output