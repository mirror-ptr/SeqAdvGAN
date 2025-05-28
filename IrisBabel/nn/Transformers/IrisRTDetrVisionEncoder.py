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
        # 确保输入的张量适合转为 PIL Image。
        # RTDetr 图像处理器通常期望 uint8 类型的图像 (0-255 范围)。
        # 如果输入是 float32 (0-1 范围)，则需要转换。
        # 我们假定输入到此模块的是 float32 且范围为 [0, 1]。
        if image.dtype != torch.uint8:
            # 将 float [0, 1] 转换为 uint8 [0, 255]
            # 使用 .mul(255) 和 .byte() 进行转换
            # clamp 确保值在 [0, 1] 范围内，防止因精度问题导致 > 1
            image = torch.clamp(image, 0, 1).mul(255).byte()
            # print(f"Converted input to uint8: {image.dtype}") # Optional debug print

        to_pil = T.ToPILImage()
        pil_images = []
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