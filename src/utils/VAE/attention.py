import torch
from torch import nn
from torch import functional as F

class SelfAttention(nn.Module):
    
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
        
    def forward(self, x: torch.Tensor, causal_mask=False):
        # x: (Batch_Size, Seq_len, Dim)
        
        batch_size, seq_length, d_embed = x.shape
        interim_shape = (batch_size, seq_length, self.n_heads, self.d_head)
        
        # (Batch_Size, Seq_len, Dim) -> (Batch_Size, Seq_len, 3 * Dim) -> 3 tensors of (Batch_Size, Seq_len, Dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        
        # (Batch_Size, Seq_len, Dim) -> (Batch_Size, Seq_len, H, Dim/H) -> (Batch_Size, H, Seq_len, Dim/H)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)
        
        # (Batch_Size, H, Seq_len, Dim/H)
        weight = q @ k.transpose(-1, -2)
        
        if causal_mask:
            # upper triangular mask of 1s
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)
            
        weight /= torch.sqrt(self.d_head)
        weight = F.softmax(weight, dim=1)
        
        output = weight @ v
        output = output.reshape((batch_size, seq_length, d_embed))
        output = self.out_proj(output)
        
        return output
        
            