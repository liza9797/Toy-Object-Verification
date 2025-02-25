import torch
import torch.nn as nn

class CoAttentionLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, y):
        ### x: [B, N (Nuber of Patches in x), D], y: [B, M (Nuber of Patches in y), D]
        attn = torch.bmm(self.linear(x), y.transpose(1, 2))
        attn = self.softmax(attn)
        x_co = torch.bmm(attn, y)
        return x_co

class CoAttentionModel(nn.Module):
    def __init__(self, feature_dim=128, num_patches=16):
        super().__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), 
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, feature_dim, kernel_size=3, padding=1), 
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  ### Number of patches = 16
        )
        self.num_patches = num_patches  
        
        ### Co-Attention
        self.co_attn = CoAttentionLayer(feature_dim)
        
        ### Self-Attention
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        ### For stacked embeddings
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * (num_patches * 2), 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),
            # nn.Sigmoid()
        )
    
    def forward(self, img1, img2):
        ### Extract features 
        feat1 = self.cnn(img1) 
        feat2 = self.cnn(img2)
        # print(feat1.shape)
        
        B, D, H, W = feat1.shape
        N = H * W  ### Number of patches
        
        ### Transform feture maps into sequences
        feat1 = feat1.view(B, D, N).transpose(1, 2)
        feat2 = feat2.view(B, D, N).transpose(1, 2) #[Batch size, Nuber of patcher, Dimensionality]
        # print(feat1.shape)
        
        ### Apply co-attention to both images
        ### Here: we make attention on one image 
        ### based on feature maps from another one
        feat1_co = self.co_attn(feat1, feat2)
        feat2_co = self.co_attn(feat2, feat1)
        
        ### Fusion (as residul)
        fused1 = feat1 + feat1_co
        fused2 = feat2 + feat2_co
        
        ### Concat patches
        fused = torch.cat([fused1, fused2], dim=1)
        # print(fused.shape)
        
        ### For now each patch has no info about other pathes 
        ### from the same image. We apply self-attention to exchange 
        ### spatial information inside the image
        fused = fused.transpose(0, 1) 
        fused = self.transformer(fused)
        fused = fused.transpose(0, 1)
        
        ### Final
        fused = fused.flatten(start_dim=1)
        # print(fused.shape)
        out = self.classifier(fused)
        return out
