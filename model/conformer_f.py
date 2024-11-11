"""
Module Name: Conformer Model for EEG Data Processing

Description:
    This module defines a Conformer-based neural network architecture for EEG data processing, composed of
    convolutional, transformer, and fully connected layers. The model is designed to learn complex patterns
    in EEG data through a combination of spatial feature extraction and temporal modeling. The core components
    include Residual Connections, Multi-Head Attention, Feed-Forward blocks, and Encoder blocks, each contributing
    to an effective deep learning architecture for time-series classification tasks.

Classes:
    - ResidualBlock: Adds a residual connection to any specified function or module, enabling effective
      gradient flow through the network.
    - MultiHeadAttention: Implements the multi-head attention mechanism to allow the model to focus on
      different parts of the input sequence.
    - FeedForwardBlock: A two-layer feed-forward network with GELU activation and dropout, commonly used
      in transformer architectures.
    - FCModule: A fully connected module for mapping features to the desired number of output classes.
    - EncoderBlock: A transformer encoder block that applies multi-head attention and feed-forward layers
      with residual connections.
    - ConvModule: A convolutional module for initial EEG data processing, including feature extraction
      through convolutional layers, normalization, and activation.
    - TransformerModule: A multi-layer transformer module composed of stacked encoder blocks to enable
      deep sequence modeling.
    - Conformer: A high-level model that combines convolutional, transformer, and fully connected modules
      for end-to-end EEG data classification.

Usage:
    This module is intended for EEG-based machine learning applications, particularly those requiring
    time-series classification. The `Conformer` class serves as the main entry point for creating an
    end-to-end model, which can be used in training and inference tasks on EEG datasets.

License:
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
    Created: 2024-10-17
    Last Modified: 2024-11-02
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import softmax

from einops import rearrange
from einops.layers.torch import Rearrange


class ResidualBlock(nn.Module):
    """
    A residual block module that adds a residual connection to a given function.

    This class is useful in neural networks where residual connections are needed to allow
    gradients to bypass one or more layers, facilitating more effective training of deeper
    networks.

    Attributes:
        fn (callable): The function or module to which the residual connection will be added.
    """
    def __init__(self, fn):
        """
        Initializes the ResidualBlock with a specified function.

        :param fn: A callable function or module that will be applied to the input and
                   combined with the residual connection.
        :type fn: callable
        """
        super().__init__()

        self.fn = fn

    def forward(self, x, **kwargs):
        """
        Performs the forward pass, adding the input (residual) to the output of the function.

        :param x: The input tensor.
        :param kwargs: Additional arguments to pass to the function.
        :return: The output tensor after applying the function and adding the residual connection.
        :rtype: Tensor
        """
        res = x
        x = self.fn(x, **kwargs)
        x += res

        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module.

    This module implements the multi-head attention mechanism, which allows the model to focus on different
    parts of the input sequence with multiple attention heads. Each head independently performs attention
    and the results are concatenated and linearly transformed to produce the final output.

    Attributes:
        dim (int): The dimensionality of the input and output.
        heads (int): The number of attention heads.
        queries (nn.Linear): Linear layer to generate query vectors.
        keys (nn.Linear): Linear layer to generate key vectors.
        values (nn.Linear): Linear layer to generate value vectors.
        attention_drop (nn.Dropout): Dropout layer applied to attention weights.
        projection (nn.Linear): Linear layer to project the concatenated heads' output to the output dimension.
    """
    def __init__(self, dim, heads, dropout):
        """
        Initializes the MultiHeadAttention module.

        :param dim: The dimensionality of the input and output.
        :type dim: int
        :param heads: The number of attention heads.
        :type heads: int
        :param dropout: Dropout rate to apply to the attention weights.
        :type dropout: float

        This module allows for multi-head attention, where multiple attention heads are used
        to focus on different parts of the input sequence. Each head independently computes
        attention, and their outputs are concatenated and projected to produce the final output.
        """
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
        """
        Forward pass for the multi-head attention module.

        :param x: The input tensor of shape (batch, sequence length, dim).
        :type x: Tensor
        :param mask: The mask tensor to prevent attention to certain positions.
                     Shape should be broadcastable to (batch, heads, query length, key length).
                     Defaults to None.
        :type mask: Tensor, optional
        :return: The output tensor after applying multi-head attention, of shape (batch, sequence length, dim).
        :rtype: Tensor

        This method computes attention scores, applies the attention weights to the values,
        and then combines the outputs from each head into a single tensor. It optionally
        applies a mask to prevent attention to certain positions.
        """
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
    """Feed-forward block module."""
    def __init__(self, dim, expansion, drop_out):
        """
        Initializes the FeedForwardBlock module.

        :param dim: The input and output dimensionality of the block.
        :type dim: int
        :param expansion: Expansion factor for the hidden layer dimensionality.
        :type expansion: int
        :param drop_out: Dropout rate to apply after the activation layer.
        :type drop_out: float

        This feed-forward block consists of two linear layers with a GELU activation and dropout in between.
        The first linear layer expands the input dimensionality by the specified `expansion` factor, and the
        second linear layer projects it back to the original dimension. This block is commonly used in
        transformer architectures.
        """
        super().__init__(
            nn.Linear(dim, expansion * dim),
            nn.GELU(),
            nn.Dropout(drop_out),
            nn.Linear(expansion * dim, dim),
        )


class FCModule(nn.Sequential):
    """Fully connected module."""
    def __init__(self, in_channels: int, num_classes: int, hid_channels: int = 32, dropout: float = 0.5):
        """
        Initializes the FCModule.

        :param in_channels: The number of input channels.
        :type in_channels: int
        :param num_classes: The number of output classes.
        :type num_classes: int
        :param hid_channels: The number of hidden channels in the intermediate layers. Defaults to 32.
        :type hid_channels: int
        :param dropout: Dropout rate to apply after each activation layer. Defaults to 0.5.
        :type dropout: float

        This module consists of a sequence of fully connected layers with ELU activations and dropout.
        The architecture includes three linear layers:
            - The first layer expands the input to `hid_channels * 8` channels.
            - The second layer reduces the dimensionality back to `hid_channels`.
            - The final layer outputs the desired number of classes.

        The dropout is applied after each activation layer to prevent overfitting.
        """
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
        """
        Forward pass for the FCModule.

        :param x: The input tensor.
        :type x: Tensor
        :return: The output tensor after applying the fully connected layers.
        :rtype: Tensor

        The input tensor is first reshaped to ensure a consistent shape, then passed through the
        fully connected layers defined in `self.fc`.
        """
        x = x.contiguous().view(x.size(0), -1)
        x = self.fc(x)
        return x


class EncoderBlock(nn.Sequential):
    def __init__(self, dim, heads=10, drop=0.5, expansion=4, forward_drop=0.5):
        """
        Initializes the EncoderBlock.

        :param dim: The dimensionality of the input and output.
        :type dim: int
        :param heads: The number of attention heads in the multi-head attention module. Defaults to 10.
        :type heads: int
        :param drop: Dropout rate to apply in attention and feed-forward blocks. Defaults to 0.5.
        :type drop: float
        :param expansion: Expansion factor for the hidden layer dimensionality in the feed-forward block. Defaults to 4.
        :type expansion: int
        :param forward_drop: Dropout rate to apply in the feed-forward block. Defaults to 0.5.
        :type forward_drop: float

        This encoder block consists of two main submodules:
            1. A multi-head attention layer with residual connections and layer normalization.
            2. A feed-forward layer with residual connections and layer normalization.

        Each submodule is encapsulated within a `ResidualBlock`, which adds the residual connection to
        the output of each block, following a transformer-like architecture. Dropout is applied after each
        layer to prevent overfitting.
        """
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
    Convolutional module for EEG data processing.

    This module applies a sequence of convolutional, normalization, and activation layers to process
    EEG data. It reduces the spatial dimensions and projects the features for further processing.

    Attributes:
        input (nn.Sequential): A sequence of layers that applies initial convolutions, normalization,
                               and activation to the input.
        projection (nn.Sequential): A sequence of layers that projects the output features to a desired
                                    shape for further processing.
    """
    def __init__(self, channels, block_size, dim, dropout):
        """
        Initializes the ConvModule.

        :param channels: The number of input channels (e.g., EEG channels).
        :type channels: int
        :param block_size: The kernel size for the main convolution block.
        :type block_size: int
        :param dim: The number of output channels after initial convolution.
        :type dim: int

        The module consists of two main parts:
            1. `input`: A sequence of layers that applies convolutions, batch normalization, and ELU activation.
               - The initial Conv2D layer reshapes the input channels to the specified `dim`.
               - A Conv2D layer with kernel size `(block_size, 1)` reduces the spatial dimension.
               - Batch normalization and ELU activation are applied to the output.
            2. `projection`: A sequence that projects the output features and rearranges the shape.
               - A Conv2D layer with kernel size `(1, 1)` refines the output.
               - The output is reshaped using a custom rearrange function for compatibility with subsequent layers.
        """
        super().__init__()

        # Initial processing layers for input features
        self.input = nn.Sequential(
            # input shape:  batch_size, 1,   eeg_channels, sequence
            # output shape: batch_size, dim, eeg_channels, sequence - 24
            #nn.Conv2d(channels, dim, (1, 1)),
            # input shape:  batch_size, dim, eeg_channels, sequence - 24
            # output shape: batch_size, dim, 1,            sequence - 24
            #nn.Conv2d(dim, dim, (block_size, 1)),
            #nn.Dropout(dropout),
            # features(channels) normalization
            nn.BatchNorm2d(dim),
            nn.ELU(),
            # input shape:  batch_size, dim, 1, sequence - 24
            # output shape: batch_size, dim, 1, (sequence - 99)/15 + 1
            #nn.AvgPool2d((1, 1)),
            nn.Dropout(dropout),
        )

        # Projection layers for further processing
        self.projection = nn.Sequential(
            # input shape:  batch_size, dim, 1, (sequence - 99)/15 + 1
            # output shape: batch_size, dim, 1, (sequence - 99)/15 + 1
            # nn.Conv2d(dim, dim, (1, 1), stride=(1, 1)),
            # input shape:  batch_size, dim, 1, (sequence - 99)/15 + 1
            # output shape: batch_size, 1 * (sequence - 99)/15 + 1, dim
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the ConvModule.

        :param x: The input tensor with shape (batch_size, channels, eeg_channels, sequence).
        :type x: Tensor
        :return: The output tensor after processing, reshaped for further processing.
        :rtype: Tensor

        The input tensor is processed by the `input` layers, followed by the `projection` layers
        to generate the final output.
        """
        x = self.input(x)
        x = self.projection(x)

        return x


class TransformerModule(nn.Sequential):
    """Transformer module composed of multiple encoder blocks."""
    def __init__(self, dim, heads, depth):
        """
        Initializes the TransformerModule.

        :param dim: The dimensionality of the input and output of each encoder block.
        :type dim: int
        :param heads: The number of attention heads in each encoder block.
        :type heads: int
        :param depth: The number of encoder blocks to stack in the transformer module.
        :type depth: int

        This module consists of a stack of `EncoderBlock` layers, each of which applies multi-head
        attention and feed-forward transformations with residual connections. The depth parameter
        controls the number of encoder blocks stacked, enabling the transformer to learn complex
        representations by increasing depth.
        """
        super().__init__(*[EncoderBlock(dim, heads) for _ in range(depth)])


class ConformerF(nn.Sequential):
    """Conformer model combining convolutional, transformer, and fully connected layers."""
    def __init__(self, channels, block_size, dim=40, heads=10, depth=6, classes=4, dropout_1=0.5):
        """
        Initializes the ConformerFeature model.

        :param channels: The number of input channels (e.g., EEG channels).
        :type channels: int
        :param block_size: The kernel size for the main convolution block in `ConvModule`.
        :type block_size: int
        :param dim: The dimensionality of the feature space used in the model. Defaults to 40.
        :type dim: int
        :param heads: The number of attention heads in each transformer encoder block. Defaults to 10.
        :type heads: int
        :param depth: The number of encoder blocks in the transformer module. Defaults to 6.
        :type depth: int
        :param classes: The number of output classes for classification. Defaults to 4.
        :type classes: int

        The ConformerFeature model consists of three main components:
            1. `ConvModule`: A convolutional module to process input data and extract initial features.
            2. `TransformerModule`: A transformer module that further processes features through
               a series of encoder blocks.
            3. `FCModule`: A fully connected layer that maps the flattened transformer output to
               the specified number of output classes.
        """
        super().__init__()
        self.block_size = block_size

        self.conv = ConvModule(channels, block_size, dim, dropout_1)
        self.transformer = TransformerModule(dim, heads, depth)
        self.fc = FCModule(self.feature_dim(), classes)

    def feature_dim(self):
        """
        Computes the feature dimensionality after applying convolution and transformer layers.

        This method creates a mock input tensor with the expected input shape and passes it
        through the convolutional and transformer modules to determine the flattened feature
        dimensionality.

        :return: The computed feature dimensionality after processing through convolution and transformer layers.
        :rtype: int
        """
        with torch.no_grad():
            mock_eeg = torch.zeros(1, 5, self.block_size, 62)
            mock_eeg = self.conv(mock_eeg)
            mock_eeg = self.transformer(mock_eeg)
            mock_eeg = mock_eeg.flatten(start_dim=1)

            return mock_eeg.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the ConformerFeature model.

        :param x: The input tensor with shape (batch_size, channels, height, width).
        :type x: torch.Tensor
        :return: The output tensor after classification, with shape (batch_size, classes).
        :rtype: torch.Tensor

        The input tensor is processed by the convolutional and transformer modules, then flattened
        and passed through the fully connected layer for classification.
        """
        x = self.conv(x)
        x = self.transformer(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)

        return x
