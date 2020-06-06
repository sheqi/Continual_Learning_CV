from efficientnet_pytorch import EfficientNet

def efficientnet(class_num):
    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=class_num)
    return model