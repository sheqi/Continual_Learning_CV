import json
from PIL import Image
import torch
from torchvision import transforms

from efficientnet_pytorch import EfficientNet

def efficientnet(class_num):
    model = EfficientNet.from_pretrained('efficientnet-b0',num_classes=class_num)
    return model

