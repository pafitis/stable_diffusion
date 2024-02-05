import torch
from torch import nn
from torch import functional as F
from attention import SelfAttention

class VAE_ResidualBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x: (Batch_size, In_channels, Height, Width)
        
        residual = x
        x = self.groupnorm_1(x)
        
        x = F.SiLU(x)
        
        x = self.conv_1(x)
        
        x = self.groupnorm_2(x)
        
        x= F.SiLU(x)
        
        x = self.conv_2(x)
        
        return x + self.residual_layer(residual)
    
    
class VAE_AttentionBlock(nn.Module):
    
    def __init__(self, channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch_size, Features, Height, Width)
        residual = x
        
        n, c, h, w = x.shape
        # (Batch_size, Features, Height, Width) -> (Batch_size, Features, Height * Width)
        x = x.view(n, c, h*w)
        # (Batch_size, Features, Height * Width) -> (Height * Width, Batch_size, Features)
        x = x.transpose(-1, -2)
        
        x = self.attention(x)
        
        # go back to original dims
        x = x.transpose(-1, -2)
        x = x.view((n,c,h,w))
        
        return x + residual
    
class VAE_Decoder(nn.Sequential):
    
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            
            VAE_ResidualBlock(512, 512),
            VAE_AttentionBlock(512),
            
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            
            # (Batch_size, Features, Height/8, Width/8) -> (Batch_size, Features, Height/4, Width/4)
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            
            # (Batch_size, Features, Height/4, Width/4) -> (Batch_size, Features, Height/2, Width/2)
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),
            
            # (Batch_size, Features, Height/2, Width/2) -> (Batch_size, Features, Height, Width)
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),
            
            # group features by 32
            # 128 features will be split into 32 groups and normalised within-group
            nn.GroupNorm(32, 128),
            
            nn.SiLU(),
            # (Batch_size, 128, Height, Width) -> (Batch_size, 3, Height, Width)
            nn.Conv2d(128, 3, kernel_size=3, padding=1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # reverse mysterious scaling
        x /= 0.18215
        
        for module in self:
            x = module(x)
            
        return x