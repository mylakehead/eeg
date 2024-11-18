import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import softmax

from einops import rearrange
from einops.layers.torch import Rearrange


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
    """Multi-head attention module."""
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
        """Forward pass for the multi-head attention module.
        
        Args:
            x (Tensor): The input tensor.
            mask (Tensor, optional): The mask tensor. Defaults to None.
        """
        # output shape: 1 10 41 4
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.heads)

        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            # energy.mask_fill(~mask, fill_value)
            energy.masked_fill_(~mask, fill_value)
        scaling = self.dim ** (1 / 2)
        attention = softmax(energy / scaling, dim=-1)
        attention = self.attention_drop(attention)
        out = torch.einsum('bhal, bhlv -> bhav ', attention, values)
        out = rearrange(out, "b h n d -> b n (h d)")

        out = self.projection(out)

        return out


class FeedForwardBlock(nn.Sequential):
    """Feed-forward block module."""
    def __init__(self, dim, expansion, drop_out):
        super().__init__(
            nn.Linear(dim, expansion * dim),
            nn.GELU(),
            nn.Dropout(drop_out),
            nn.Linear(expansion * dim, dim),
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
    def __init__(self, input_channels, block_size, dim):
        super().__init__()

        self.input = nn.Sequential(
            nn.Conv2d(input_channels, dim, (1, 1)),
            nn.Conv2d(dim, dim, (block_size, 1)),
            nn.BatchNorm2d(dim),
            nn.ELU(),
            # nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(dim, dim, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.input(x)
        x = self.projection(x)

        return x


class TransformerModule(nn.Sequential):
    """Transformer module."""
    def __init__(self, dim, heads, depth):
        super().__init__(*[EncoderBlock(dim, heads) for _ in range(depth)])


class Conformer(nn.Sequential):
    """Conformer feature model."""
    def __init__(self, input_channels, block_size, dim, heads, depth, classes):
        super().__init__()

        self.input_channels = input_channels
        self.block_size = block_size

        self.conv = ConvModule(input_channels, block_size, dim)
        self.transformer = TransformerModule(dim, heads, depth)
        self.fc = FCModule(self.feature_dim(), classes)

    def feature_dim(self):
        with torch.no_grad():
            mock_eeg = torch.zeros(1, self.input_channels, self.block_size, 62)
            mock_eeg = self.conv(mock_eeg)
            mock_eeg = self.transformer(mock_eeg)
            mock_eeg = mock_eeg.flatten(start_dim=1)

            return mock_eeg.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.transformer(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)

        return x
