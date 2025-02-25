import torch
import torch.nn as nn

class SimpleEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(128),
            nn.GELU(),
            
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.GELU(),
            
            nn.MaxPool2d(2),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layers(x)
        # print(x.shape)
        x = self.adaptive_pool(x)
        # print(x.shape)
        x = self.fc(x)
        # print(x.shape)
        return x
