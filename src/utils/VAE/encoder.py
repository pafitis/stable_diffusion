import torch
from torch import nn
from torch.nn import functional as F

from decoder import VAE_AttentionBlock, VAE_ResidualBlock



class VAE_Encoder(nn.Sequential):
    
    def __init__(self):
        super().__init__(
            
            # Reduce image size but increase number of features
            # Pixels hold more information
            
            # (Batch_size, Channel, Height, Width) -> (Batch_Size, 128, Height, Width)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),
            
            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height/2, Width/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            VAE_ResidualBlock(128, 256),
            VAE_ResidualBlock(256, 256),

            # (Batch_Size, 256, Height/2, Width/2) -> (Batch_Size, 256, Height/4, Width/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            VAE_ResidualBlock(256, 512),
            VAE_ResidualBlock(512, 512),
            
            # (Batch_Size, 512, Height/4, Width/4) -> (Batch_Size, 512, Height/8, Width/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            
            # (Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512, 512),
            
            # normalisation + activation function
            # (Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/8, Width/8)
            nn.GroupNorm(32, 512),
            nn.SiLU(),
            
            # (Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 8, Height/8, Width/8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
        )
        
    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: (Batch_Size, Channel, Height, Width)
        # noise: (Batch_Size, Out_Channels, Height/8, Width/8)
        # output of encoder is the mu, sigma of the latent space gaussian distribution
        for module in self:
            if getattr(module, 'stride', None) == (2, 2): # catch for asymmetrical padding when stride=2
                x = F.pad(x, (0, 1, 0, 1)) # (padding_left, padding_right, padding_top, padding_bottom)
                
            x = module(x)
        
        # (Batch_size, 8, Height/8, Width/8) -> two tensors of (Batch_size, 4, Height/8, Width/8)
        mean, log_variance = torch.chunk(x, 2, dim=1)
        log_variance = torch.clamp(log_variance, -30, 20) # bind log_variance within range 
        variance = torch.exp(log_variance) # get variance by exp(log(variance))
        std = torch.sqrt(variance)
        
        # To go from Z ~N(0,1) to X~N(mu, sigma)
        x = mean + std * noise
        # mysterious scaling from paper
        x *= 0.18215
        pass
    