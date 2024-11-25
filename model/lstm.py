import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self,
                 num_electrodes: int = 32,
                 hid_channels: int = 64,
                 num_classes: int = 2):
        super(LSTM, self).__init__()

        self.num_electrodes = num_electrodes
        self.hid_channels = hid_channels
        self.num_classes = num_classes

        self.gru_layer = nn.LSTM(input_size=num_electrodes,
                                 hidden_size=hid_channels,
                                 num_layers=2,
                                 bias=True,
                                 batch_first=True)

        self.out = nn.Linear(hid_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)

        r_out, (_, _) = self.gru_layer(x, None)
        r_out = F.dropout(r_out, 0.3)
        x = self.out(r_out[:, -1, :])  # choose r_out at the last time step
        return x