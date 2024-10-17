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

        # out shape
        # batches    -           -
        # 1       1*width       dim
        self.queries = nn.Linear(dim, dim)
        self.keys = nn.Linear(dim, dim)
        self.values = nn.Linear(dim, dim)

        self.attention_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(dim, dim)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # batches    n         (h d)      batches   heads    n     d
        # 1       1*width       dim  ->      1      heads  width   dim/heads
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.heads)

        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.masked_fill_(~mask, fill_value)
        scaling = self.dim ** (1 / 2)
        attention = softmax(energy / scaling, dim=-1)
        attention = self.attention_drop(attention)
        out = torch.einsum('bhal, bhlv -> bhav ', attention, values)
        out = rearrange(out, "b h n d -> b n (h d)")

        out = self.projection(out)

        return out


class FeedForwardBlock(nn.Sequential):
    def __init__(self, dim, expansion, drop_out):
        super().__init__(
            nn.Linear(dim, expansion * dim),
            nn.GELU(),
            nn.Dropout(drop_out),
            nn.Linear(expansion * dim, dim),
        )


class FCModule(nn.Sequential):
    def __init__(self, dim, n_classes):
        super().__init__()

        self.cov = nn.Sequential(
            nn.Conv1d(190, 1, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )

        self.cls_head = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(dim),
            nn.Linear(dim, n_classes)
        )

        self.cls_head_fc = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(dim),
            nn.Linear(dim, 32),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(32, n_classes)
        )

        self.fc = nn.Sequential(
            nn.Linear(1640, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)

        return out


class EncoderBlock(nn.Sequential):
    def __init__(self, dim, heads=5, drop=0.5, expansion=4, forward_drop=0.5):
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
    def __init__(self, dim):
        super().__init__()

        self.input = nn.Sequential(
            # output shape:
            # batches channels  eeg_channels    width
            # 1       dim          62           width
            nn.Conv2d(1, dim, (1, 25), (1, 1)),
            # output shape:
            # batches channels  eeg_channels    width
            # 1       dim           1           width
            nn.Conv2d(dim, dim, (62, 1), (1, 1)),
            nn.BatchNorm2d(dim),
            nn.ELU(),
            # output shape:
            # batches channels  eeg_channels    width
            # 1       dim            1          width
            nn.AvgPool2d((1, 75), (1, 15)),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            # output shape:
            # batches channels  eeg_channels    width
            # 1       token_dim     1           width
            nn.Conv2d(dim, dim, (1, 1), stride=(1, 1)),
            # output shape:
            # batches    -           -
            # 1       1*width       dim
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.input(x)
        x = self.projection(x)

        return x


class TransformerModule(nn.Sequential):
    def __init__(self, dim, depth):
        super().__init__(*[EncoderBlock(dim) for _ in range(depth)])


class ViT(nn.Sequential):
    def __init__(self, dim=40, depth=6, classes=4):
        super().__init__(
            ConvModule(dim),
            TransformerModule(dim, depth),
            FCModule(dim, classes)
        )
