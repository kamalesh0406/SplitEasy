import torch
import torchvision.models as models
import torch.nn as nn 
import densenet_cifar

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x 

class SplitA(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
    def forward(self, x):
        output = self.features(x)
        return output

class SplitB(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = densenet_cifar.densenet121()
    def forward(self, x):
        x = self.features(x)
        return x

class SplitC(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Linear(1024, 100)
    def forward(self, x):
        x = self.features(x)
        return x