from torchvision.models import resnet18
import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)
