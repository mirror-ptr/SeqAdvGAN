import torch
from torch import nn
from transformers import AutoImageProcessor, RTDetrV2Model
import os

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
        image = self.image_processor(image, return_tensors="pt")
        image.to(self.device)
        output = self.model(**image)
        last_hidden_state = output.last_hidden_state
        pooled_last_hidden_state = torch.mean(last_hidden_state, dim=1)
        final_output = self.projection(pooled_last_hidden_state)
        return final_output