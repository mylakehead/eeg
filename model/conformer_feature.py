import torch
import torch.nn as nn
from torch import Tensor

from einops.layers.torch import Rearrange

from .common import TransformerModule


class FCModule(nn.Sequential):
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


class TemporalAttention(nn.Module):
    def __init__(self, out_frames):
        super(TemporalAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool3d(output_size=(out_frames, 1, 1))

    def forward(self, x):
        x = self.avg_pool(x)

        x = torch.sigmoid(x)

        return x


class ConvModule(nn.Module):
    def __init__(self, input_channels, sample_length, inner_channels, dropout=0.5):
        super().__init__()

        self.input = nn.Sequential(
            # 5 samples 62
            # nn.Conv2d(input_channels, inner_channels, (10, 1), (10, 1)),
            # 62 samples 5
            # nn.Conv2d(input_channels, inner_channels, (1, 5), (1, 5)),
            # samples 62 5
            nn.Conv2d(input_channels, inner_channels, (1, 1), (1, 1)),
            nn.BatchNorm2d(inner_channels),
            nn.ELU(),
            # nn.AvgPool2d((1, 75), (1, 15)),
            nn.Dropout(dropout),
        )

        self.projection = nn.Sequential(
            # nn.Conv2d(dim, dim, (1, 1), stride=(1, 1)),
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
            # mock_eeg = torch.zeros(1, 5, self.sample_length, 62)
            # mock_eeg = torch.zeros(1, 62, self.sample_length, 5)
            mock_eeg = torch.zeros(1, self.sample_length, 62, 5)
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
