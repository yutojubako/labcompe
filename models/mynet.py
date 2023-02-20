import torch.nn as nn
import torch.nn.functional as F

class Mynet(nn.Module):
    def __init__(self):
        super(Mynet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels=12, kernel_size=3)
        
        self.drop = nn.Dropout(0.25)
        self.batch1 = nn.BatchNorm2d(6)
        self.batch2 = nn.BatchNorm2d(12)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(in_features=12*5*5,out_features=10)

    
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv1(x)
        x = self.batch1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.maxpool(x)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        return x