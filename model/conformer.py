"""
CNN + Transformer + FC

Copyright:
    MIT License

    Copyright © 2024 Lakehead University, Large Scale Data Analytics Group Project

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software
    and associated documentation files (the "Software"), to deal in the Software without restriction,
    including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
    subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial
    portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
    LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Authors:
    Kang Hong, XingJian Han, Minh Anh Nguyen
    hongkang@hongkang.name, xhan15@lakeheadu.ca, mnguyen9@lakeheadu.ca

Date:
    Created: 2024-10-02
    Last Modified: 2024-11-24
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import softmax

from einops import rearrange
from einops.layers.torch import Rearrange


class ResidualBlock(nn.Module):
    """
    A PyTorch implementation of a residual block.

    This class represents a residual block commonly used in residual networks (ResNets).
    A residual block introduces a skip connection that adds the input (residual) to the output
    of the block, which helps alleviate the vanishing gradient problem and facilitates the
    training of very deep networks.

    Args:
        fn (nn.Module): A PyTorch module representing the transformation function
                        (e.g., convolutional layers, batch normalization, etc.)
                        applied to the input.
    """
    def __init__(self, fn):
        super().__init__()

        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res

        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module for self-attention mechanisms.

    Multi-head attention allows the model to jointly attend to information from different representation
    subspaces, improving its ability to capture dependencies between elements in the input sequence.

    Args:
        dim (int): The dimensionality of the input and output features.
        heads (int): The number of attention heads.
        dropout (float): Dropout rate for attention scores to prevent overfitting.
    """
    def __init__(self, dim, heads, dropout):
        super().__init__()

        self.dim = dim
        self.heads = heads

        # Linear layers to generate queries, keys, and values
        self.queries = nn.Linear(dim, dim)
        self.keys = nn.Linear(dim, dim)
        self.values = nn.Linear(dim, dim)

        self.attention_drop = nn.Dropout(dropout)

        # Linear layer to project the concatenated output back to the original dimension
        self.projection = nn.Linear(dim, dim)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """
        Forward pass for multi-head attention.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, dim).
            mask (Tensor, optional): Mask tensor to apply (e.g., for causal masking).
        """
        # Generate queries, keys, and values and reshape for multiple heads
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.heads)

        # Compute scaled dot-product attention scores
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)

        # Apply mask if provided
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            # energy.mask_fill(~mask, fill_value)
            energy.masked_fill_(~mask, fill_value)
        scaling = self.dim ** (1 / 2)
        attention = softmax(energy / scaling, dim=-1)
        attention = self.attention_drop(attention)

        # Compute the weighted sum of values
        out = torch.einsum('bhal, bhlv -> bhav ', attention, values)

        # Reshape the output back to the original format
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
    """
    Fully Connected (FC) module for classification tasks.

    This module implements a fully connected feedforward neural network. It includes
    multiple layers with activation functions and dropout for regularization, making
    it suitable for tasks such as classification.

    Args:
        in_channels (int): Number of input features (channels).
        num_classes (int): Number of output classes (size of the final output layer).
        hid_channels (int, optional): Number of hidden units in the intermediate layers. Default is 32.
        dropout (float, optional): Dropout rate for regularization. Default is 0.5.
    """
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
    """
    A convolutional module for feature extraction and dimensionality projection.

    This module is designed to process 2D inputs using convolutional layers,
    followed by a projection step to reshape the output into a format suitable
    for further processing (e.g., feeding into a transformer or fully connected layers).

    Args:
        input_channels (int): Number of input channels for the convolutional layers.
        block_size (int): Kernel size for the second convolutional layer in the height dimension.
        dim (int): Number of output channels (features) for the convolutional layers.
    """
    def __init__(self, input_channels, block_size, dim):
        super().__init__()

        self.input = nn.Sequential(
            nn.Conv2d(input_channels, dim, (1, 1)),
            nn.Conv2d(dim, dim, (block_size, 1)),
            nn.BatchNorm2d(dim),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
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
    """
    Conformer feature model for processing EEG data.

    The Conformer model combines convolutional layers for local feature extraction,
    a transformer module for global context modeling, and a fully connected (FC) layer
    for classification. This architecture is well-suited for tasks like EEG data analysis
    and sequence classification.

    Args:
        input_channels (int): Number of input channels (e.g., EEG sensor channels).
        block_size (int): Block size for the convolutional module (e.g., height of input).
        dim (int): Dimensionality of the feature space.
        heads (int): Number of attention heads in the transformer.
        depth (int): Number of layers in the transformer.
        classes (int): Number of output classes for classification.
    """
    def __init__(self, input_channels, block_size, dim, heads, depth, classes):
        super().__init__()

        self.input_channels = input_channels
        self.block_size = block_size

        # Convolutional module for feature extraction
        self.conv = ConvModule(input_channels, block_size, dim)
        # Transformer module for global context modeling
        self.transformer = TransformerModule(dim, heads, depth)
        # Fully connected module for classification
        self.fc = FCModule(self.feature_dim(), classes)

    def feature_dim(self):
        """
        Dynamically computes the flattened feature dimensionality after the convolution
        and transformer modules. Uses a mock tensor with the same dimensions as the input.

        Returns:
        int: The feature dimensionality after flattening.
        """
        with torch.no_grad():
            mock_eeg = torch.zeros(1, self.input_channels, self.block_size, 62)
            mock_eeg = self.conv(mock_eeg)
            mock_eeg = self.transformer(mock_eeg)
            mock_eeg = mock_eeg.flatten(start_dim=1)

            return mock_eeg.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Conformer model.

        Args:
            x (Tensor): Input tensor of shape `(batch_size, input_channels, height, width)`.
        """
        x = self.conv(x)
        x = self.transformer(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)

        return x
