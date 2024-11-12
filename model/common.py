import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import softmax

from einops import rearrange


class ResidualBlock(nn.Module):
    def __init__(self, fn):
        super().__init__()

        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()

        self.dim = dim
        self.heads = heads

        # Linear layers for queries, keys, and values
        # out shape
        # batches    -           -
        # 1       1*width       dim
        self.queries = nn.Linear(dim, dim)
        self.keys = nn.Linear(dim, dim)
        self.values = nn.Linear(dim, dim)

        # Dropout for attention weights
        self.attention_drop = nn.Dropout(dropout)
        # Projection layer to map concatenated head outputs to the output dimension
        self.projection = nn.Linear(dim, dim)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # output shape: 1 10 41 4
        # Rearrange queries, keys, and values for multi-head attention
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.heads)

        # Calculate attention scores (energy) with scaled dot-product attention
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.masked_fill_(~mask, fill_value)

        # Scale energy values and apply softmax to get attention weights
        scaling = self.dim ** (1 / 2)
        attention = softmax(energy / scaling, dim=-1)
        attention = self.attention_drop(attention)

        # Calculate weighted sum of values based on attention
        out = torch.einsum('bhal, bhlv -> bhav ', attention, values)
        out = rearrange(out, "b h n d -> b n (h d)")

        # Apply final linear projection to combine head outputs
        out = self.projection(out)

        return out


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_out):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_out),
            nn.Linear(expansion * emb_size, emb_size),
        )


class EncoderBlock(nn.Sequential):
    def __init__(self, dim, heads=10, drop=0.5, expansion=4, forward_drop=0.5):
        super().__init__(
            ResidualBlock(
                nn.Sequential(
                    nn.LayerNorm(dim),
                    MultiHeadAttention(dim, heads, drop),
                    nn.Dropout(drop)
                )
            ),
            ResidualBlock(
                nn.Sequential(
                    nn.LayerNorm(dim),
                    FeedForwardBlock(dim, expansion=expansion, drop_out=forward_drop),
                    nn.Dropout(drop)
                )
            )
        )


class TransformerModule(nn.Sequential):
    """Transformer module composed of multiple encoder blocks."""
    def __init__(self, dim, heads, depth):
        super().__init__(*[EncoderBlock(dim, heads) for _ in range(depth)])
