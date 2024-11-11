import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import softmax

from einops import rearrange
from einops.layers.torch import Rearrange, Reduce


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


class FCModule(nn.Sequential):
    """Fully connected module."""
    def __init__(self, in_channels: int, num_classes: int, hid_channels: int = 32, dropout: float = 0.5):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_channels, hid_channels * 8),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hid_channels * 8, hid_channels),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hid_channels, num_classes)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        x = self.fc(x)
        return x


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


class ConvModule(nn.Module):
    def __init__(self, emb_size=40, inner_channels=40):
        super().__init__()

        self.input = nn.Sequential(
            nn.Conv2d(1, inner_channels, (1, 25), (1, 1)),
            nn.Conv2d(inner_channels, inner_channels, (62, 1), (1, 1)),
            nn.BatchNorm2d(inner_channels),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 5)),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(inner_channels, emb_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.input(x)
        x = self.projection(x)

        return x


class TransformerModule(nn.Sequential):
    """Transformer module composed of multiple encoder blocks."""
    def __init__(self, dim, heads, depth):
        super().__init__(*[EncoderBlock(dim, heads) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(7240, 256),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)

        return out


class Conformer(nn.Sequential):
    def __init__(self, emb_size=40, inner_channels=40, heads=10, depth=6, n_classes=4, **kwargs):
        super().__init__(
            ConvModule(emb_size, inner_channels),
            TransformerModule(emb_size, heads, depth),
            ClassificationHead(emb_size, n_classes)
        )
