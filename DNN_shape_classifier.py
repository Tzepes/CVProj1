import torch
import torch.nn as nn
from torchvision import models

def get_shape_model(num_classes=10):
    model = models.resnet18(pretrained=False)
    
    # Modify the first conv layer to accept 1-channel grayscale input
    model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Replace the final classifier layer for 6 shape classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model
