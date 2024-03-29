import torch
from torch import nn
from torch.nn import functional as F

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
        
class CrossAttention(nn.Module):
    
    def __init__(self, n_heads: int, d_embed: int, d_cross: int, in_proj_bias=True, out_proj_bias=True):
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
    
    def forward(self, latent, context):
        # latent: (Batch_Size, Seq_len_Q, Dim_Q)
        # context: (Batch_Size, Seq_len_KV, Dim_KV) = (Batch_Size, 77, 768)
        input_shape = latent.shape
        batch_size, sequence_length, d_embed = input_shape
        
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)
        
        q = self.q_proj(latent)
        k = self.k_proj(context)
        v = self.v_proj(context)
        
        q = q.view(interim_shape).transpose(1,2)
        k = k.view(interim_shape).transpose(1,2)
        v = v.view(interim_shape).transpose(1,2)
        
        weight = q @ k.transpose(-1, -2)
        weight /= torch.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)
        
        output = weight @ v
        output = output.transpose(1,2)
        output = self.out_proj(output)
        
        return output
        
        
        