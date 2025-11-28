"""
Shared Conv1D backbone for discrete-action agents (QR-DQN, Categorical SAC, etc.).

Transforms an input sequence of shape (batch, seq_len, num_features) into a
latent feature vector used by various heads (policy, Q-value, quantile).
"""

from typing import List
import torch
import torch.nn as nn


class ConvBackbone(nn.Module):
    """
    Conv1D + FC backbone that maps (batch, seq, features) to a latent vector.

    Steps:
        1) Permute to (batch, features, seq) for Conv1d.
        2) Conv1d with configurable filters/kernel.
        3) Flatten.
        4) Two fully-connected layers.
    """

    def __init__(
        self,
        sequence_length: int,
        num_features: int,
        conv_filters: int,
        conv_kernel: int,
        fc_layers: List[int],
        activation: str,
        dropout: float = 0.0,
    ):
        super().__init__()

        if activation.lower() == "gelu":
            self.activation = nn.GELU()
        elif activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.conv1d = nn.Conv1d(
            in_channels=num_features,
            out_channels=conv_filters,
            kernel_size=conv_kernel,
            padding=0,
        )

        conv_output_length = sequence_length - conv_kernel + 1
        flatten_size = conv_filters * conv_output_length

        layers = []
        input_size = flatten_size
        for fc_size in fc_layers:
            layers.append(nn.Linear(input_size, fc_size))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_size = fc_size

        self.fc = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, num_features)
        Returns:
            latent: Tensor of shape (batch, fc_layers[-1])
        """
        # (batch, seq, features) -> (batch, features, seq)
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = self.activation(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x
