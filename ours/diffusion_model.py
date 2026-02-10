# diffusion_model.py 
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import math
from diffusion_model_utils import *


class TabEncoder(nn.Module):
    """Simplified TabEncoder for VTFS project"""
    def __init__(self, tab_len, hidden_size, dropout=0.1, num_layers=2):
        super(TabEncoder, self).__init__()
        self.tab_len = tab_len
        self.hidden_size = hidden_size
        
        self.layers = nn.ModuleList([
            nn.Linear(tab_len if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        
        x = self.norm(x)
        return x


class TransformerDM(nn.Module):
    """
    Transformer Diffusion Model - consistent with DIFFT
    """
    def __init__(self, in_channels, t_channels=256, hidden_channels=512, context_channels=128,
                 depth=1, n_heads=8, dropout=0., tab_len=267, out_channels=None):
        super(TransformerDM, self).__init__()
        self.in_channels = in_channels
        d_head = hidden_channels // n_heads
        self.t_channels = t_channels
        
        # Condition encoder
        self.cond_tab_encoder = TabEncoder(tab_len, hidden_channels, dropout, 4)

        # Input projection
        self.proj_in = nn.Linear(in_channels, hidden_channels, bias=False)
        self.cond_proj_in = nn.Linear(context_channels, context_channels, bias=False)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(hidden_channels, n_heads, d_head, dropout=dropout, context_dim=context_channels)
                for _ in range(depth)]
        )

        self.norm = nn.LayerNorm(hidden_channels)

        # Output projection
        if out_channels is None:
            self.proj_out = zero_module(nn.Linear(hidden_channels, in_channels, bias=False))
        else:
            self.num_cls = out_channels
            self.proj_out = zero_module(nn.Linear(hidden_channels, out_channels, bias=False))

        # Time embedding
        self.map_noise = PositionalEmbedding(t_channels)
        self.map_layer0 = nn.Linear(in_features=t_channels, out_features=hidden_channels)
        self.map_layer1 = nn.Linear(in_features=hidden_channels, out_features=hidden_channels)
        
        # Structured sparse attention mask
        self.sparse_mask = None
    
    def set_sparse_mask(self, mask):
        """
        Set structured sparse attention mask
        
        Args:
            mask: torch tensor, shape [seq_len, seq_len], additive mask (0=connected, -inf=masked)
        """
        self.sparse_mask = mask
        print(f'TransformerDM: Sparse mask set, shape={mask.shape}')

    def forward(self, x, t, cond=None):
        # Process condition information
        if cond is not None:
            cond = self.cond_tab_encoder(cond)
            if cond.dim() == 2:
                cond = cond.unsqueeze(1)

        # Time embedding
        t_emb = self.map_noise(t)[:, None]
        t_emb = F.silu(self.map_layer0(t_emb))
        t_emb = F.silu(self.map_layer1(t_emb))

        # Project input
        x = self.proj_in(x)
        if cond is not None:
            cond = self.cond_proj_in(cond)

        # Pass through Transformer blocks
        for block in self.transformer_blocks:
            x = block(x, t_emb, context=cond, attn_mask=self.sparse_mask)
        
        # Output
        x = self.norm(x)
        x = self.proj_out(x)
        return x
