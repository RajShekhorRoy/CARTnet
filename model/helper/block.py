import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .rotary import RotaryEmbedding

class SwiGLUFusion(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()

        self.proj = nn.Linear(in_dim, hidden_dim * 2)
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        x = F.silu(gate) * x
        return self.out(x)

class RecycleAttentionBlock(nn.Module):
    def __init__(self, hidden_dim=128, feat_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear( hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, 1)
        )
        self.hint_proj = nn.Sequential(
            nn.Linear(1,hidden_dim),  # [direct_template, propagated_template]
            nn.ELU(),
            nn.Linear(hidden_dim,hidden_dim)
        )
        self.rotary_embeddings = RotaryEmbedding(dim=hidden_dim)

        self.fusion = nn.Sequential(
            nn.Linear(feat_dim + hidden_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(0.1)
        )
        self.fusion_GLU = nn.Sequential(
            SwiGLUFusion(feat_dim + hidden_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1)
        )
        self.mlp = nn.Sequential(
            nn.Linear( hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, 1)
        )
        self.SwiGLUFusion_g =SwiGLUFusion(in_dim=1, hidden_dim=1)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, V, x3, x4, x5):
        # V:    [B, L, H]
        # x3:   [B, L, F]
        # x4:   [B, L, L]
        # hint: [B, L, H]
        hint = self.hint_proj(x5)
        Q = V + hint
        K = V + hint
        V = V+ hint

        query_layer, key_layer = self.rotary_embeddings(Q, K)

        scale = math.sqrt(query_layer.size(-1))
        attn_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2)
        ) / scale

        x4_longer = x4
        L = x4.shape[-1]
        idx = torch.arange(L, device=x4_longer.device)
        mask = (idx[None, :] - idx[:, None]).abs() <= 5
        x4_longer_5 = x4_longer.masked_fill(mask, 0)

        attn_scores = attn_scores +  x4.unsqueeze(dim=1) +x4_longer_5.unsqueeze(dim=1)
        attn_weights = F.softmax(attn_scores, dim=-1)

        attn_out = torch.matmul(attn_weights, V)

        fused = torch.cat([x3, attn_out, V], dim=-1)
        update = self.fusion(fused)

        # ResNet-style residual update
        V = self.norm(V + update)
        # x5  = self.mlp(V) +self.SwiGLUFusion_g(x5)
        x5  = self.mlp(V) + x5
        #added this for stabiliy in uq
        # x5 =  torch.sigmoid(x5)
        return x5
