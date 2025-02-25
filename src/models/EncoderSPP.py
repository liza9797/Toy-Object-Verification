import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderSPP(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2) ### -> [batch, 64, 16, 16]
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.MaxPool2d(2) ### -> [batch, 128, 8, 8]
        )
 
        self.block3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=1, padding=0, dilation=1),
            nn.BatchNorm2d(256),
            nn.GELU(), 
            nn.MaxPool2d(2) ### -> [batch, 128, 4, 4]
        )

        self.fc = nn.Sequential(
            nn.Linear(256 + 1024 + 2048, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.Sigmoid()
        )

    def forward(self, x):
        ### Get feature maps
        out1 = self.block1(x)    
        out2 = self.block2(out1) 
        out3 = self.block3(out2)

        ### Apply Adaptive pooling
        pool1 = F.adaptive_max_pool2d(out1, (4, 4)).view(x.size(0), -1) 
        pool2 = F.adaptive_max_pool2d(out2, (2, 2)).view(x.size(0), -1)  
        pool3 = F.adaptive_max_pool2d(out3, (1, 1)).view(x.size(0), -1)  

        #### Concat pooled features
        features = torch.cat([pool1, pool2, pool3], dim=1)        
        # print(features.shape, )

        ### Final
        out = self.fc(features)                                         
        return out
