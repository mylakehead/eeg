import torch
import torch.nn as nn
from torch import Tensor

from einops.layers.torch import Rearrange

from .common import TransformerModule


class FCModule(nn.Sequential):
    def __init__(self, in_channels: int, num_classes: int, hid_channels: int = 32, dropout: float = 0.5):
        super().__init__()

        self.fc = nn.Sequential(
            # nn.Linear(in_channels, hid_channels * 8),
            nn.Linear(in_channels, 128),
            nn.ELU(),
            nn.Dropout(dropout),
            # nn.Linear(hid_channels * 8, hid_channels),
            nn.Linear(128, 32),
            nn.ELU(),
            nn.Dropout(dropout),
            # nn.Linear(hid_channels, num_classes)
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        x = self.fc(x)
        return x


class ConvModule(nn.Module):
    def __init__(self, input_channels, sample_length, inner_channels, dropout=0.5):
        super().__init__()

        self.input = nn.Sequential(
            nn.Conv2d(input_channels, inner_channels, (1, 1), (1, 1)),
            # nn.Conv2d(inner_channels, inner_channels, (62, 1), (1, 1)),
            # nn.Dropout(0.5),
            # features(channels) normalization
            # nn.BatchNorm2d(inner_channels),
            # nn.ELU(),
            # input shape:  batch_size, dim, 1, sequence - 24
            # output shape: batch_size, dim, 1, (sequence - 99)/15 + 1
            # nn.AvgPool2d((1, 3), (1, 1)),
            # nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            # nn.Conv2d(inner_channels, inner_channels, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.input(x)
        x = self.projection(x)

        return x


class ConformerFeature(nn.Sequential):
    def __init__(self, input_channels, sample_length, inner_channels=40, heads=10, depth=6, classes=4):
        super().__init__()

        self.sample_length = sample_length

        self.conv = ConvModule(input_channels, sample_length, inner_channels)
        self.transformer = TransformerModule(inner_channels, heads, depth)
        self.fc = FCModule(self.feature_dim(), classes)

    def feature_dim(self):
        with torch.no_grad():
            mock_eeg = torch.zeros(1, 10, 62, 1)
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
