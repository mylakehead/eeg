import torch.nn as nn
from torch import Tensor

from einops.layers.torch import Rearrange

from .common import TransformerModule


class ConvModule(nn.Module):
    def __init__(self, emb_size=40, inner_channels=40):
        super().__init__()

        self.input = nn.Sequential(
            nn.Conv2d(1, inner_channels, (1, 25), (1, 1)),
            nn.Conv2d(inner_channels, inner_channels, (62, 1), (1, 1)),
            nn.BatchNorm2d(inner_channels),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),
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


class ClassificationHead(nn.Sequential):
    def __init__(self, n_classes):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(280, 32),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)

        return out


class Conformer(nn.Sequential):
    def __init__(self, emb_size=40, inner_channels=40, heads=10, depth=6, n_classes=4):
        super().__init__(
            ConvModule(emb_size, inner_channels),
            TransformerModule(emb_size, heads, depth),
            ClassificationHead(n_classes)
        )
