"""
Critic Network for TraderNet-CRv2 PPO Agent

The Critic network estimates state values for advantage calculation.
Maps state observations to a single value estimate V(s).
"""

import torch
import torch.nn as nn
from typing import Tuple

from config.config import (
    SEQUENCE_LENGTH,
    NUM_FEATURES,
    NETWORK_PARAMS,
    NETWORK_INIT_PARAMS
)


class CriticNetwork(nn.Module):
    """
    Critic network for PPO agent that estimates state values.

    Architecture:
        - Conv1D layer for temporal feature extraction
        - Flatten
        - Two fully connected hidden layers
        - Output layer with linear activation (single value)

    Input: (batch_size, sequence_length, num_features) = (batch, 12, 28)
    Output: (batch_size, 1) = (batch, 1) - state value estimates
    """

    def __init__(
        self,
        sequence_length: int = SEQUENCE_LENGTH,
        num_features: int = NUM_FEATURES,
        conv_filters: int = NETWORK_PARAMS['conv_filters'],
        conv_kernel: int = NETWORK_PARAMS['conv_kernel'],
        fc_layers: list = NETWORK_PARAMS['fc_layers'],
        activation: str = NETWORK_PARAMS['activation'],
        dropout: float = NETWORK_PARAMS['dropout']
    ):
        """
        Initialize Critic network.

        Args:
            sequence_length: Number of timesteps in observation (default: 12)
            num_features: Number of features per timestep (default: 28)
            conv_filters: Number of Conv1D output channels (default: 32)
            conv_kernel: Conv1D kernel size (default: 3)
            fc_layers: List of FC layer sizes (default: [256, 256])
            activation: Activation function name (default: 'gelu')
            dropout: Dropout rate (default: 0.0)
        """
        super(CriticNetwork, self).__init__()

        self.sequence_length = sequence_length
        self.num_features = num_features
        self.conv_filters = conv_filters
        self.conv_kernel = conv_kernel
        self.fc_layers = fc_layers
        self.dropout = dropout

        # Set activation function
        if activation.lower() == 'gelu':
            self.activation = nn.GELU()
        elif activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation.lower() == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Conv1D layer
        # Input: (batch, num_features, sequence_length)
        # Output: (batch, conv_filters, sequence_length - kernel_size + 1)
        self.conv1d = nn.Conv1d(
            in_channels=num_features,
            out_channels=conv_filters,
            kernel_size=conv_kernel,
            padding=0  # No padding - reduces sequence length
        )

        # Calculate flattened size after conv
        conv_output_length = sequence_length - conv_kernel + 1
        flatten_size = conv_filters * conv_output_length

        # Fully connected layers
        fc_input_size = flatten_size
        self.fc_hidden = nn.ModuleList()

        for fc_size in fc_layers:
            self.fc_hidden.append(nn.Linear(fc_input_size, fc_size))
            fc_input_size = fc_size

        # Dropout layer (if enabled)
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None

        # Output layer for value estimate (single neuron)
        self.fc_output = nn.Linear(fc_input_size, 1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize network weights.

        Conv and hidden layers use Xavier/Kaiming initialization.
        Output layer uses Uniform[-init_range, +init_range] for value head.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear) and module != self.fc_output:
                # Hidden layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # Special initialization for value output head
        init_range = NETWORK_INIT_PARAMS['value_head_init_range']
        nn.init.uniform_(self.fc_output.weight, -init_range, init_range)
        if self.fc_output.bias is not None:
            nn.init.uniform_(self.fc_output.bias, -init_range, init_range)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Critic network.

        Args:
            x: State observation tensor
               Shape: (batch_size, sequence_length, num_features)

        Returns:
            State value estimates
            Shape: (batch_size, 1)
        """
        batch_size = x.shape[0]

        # Transpose for Conv1D: (batch, seq, features) -> (batch, features, seq)
        x = x.transpose(1, 2)  # (batch, num_features, sequence_length)

        # Conv1D + activation
        x = self.conv1d(x)  # (batch, conv_filters, conv_output_length)
        x = self.activation(x)

        # Flatten
        x = x.reshape(batch_size, -1)  # (batch, flatten_size)

        # Fully connected layers
        for fc_layer in self.fc_hidden:
            x = fc_layer(x)
            x = self.activation(x)
            if self.dropout_layer is not None:
                x = self.dropout_layer(x)

        # Output layer (linear activation - no softmax)
        value = self.fc_output(x)  # (batch, 1)

        return value

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get value estimate for a single state.

        Args:
            state: Single state observation (sequence_length, num_features)

        Returns:
            value: State value estimate (scalar)
        """
        # Add batch dimension if needed
        if state.dim() == 2:
            state = state.unsqueeze(0)  # (1, sequence_length, num_features)

        # Get value estimate
        value = self.forward(state)  # (1, 1)

        return value.squeeze()  # Scalar

    def evaluate_states(self, states: torch.Tensor) -> torch.Tensor:
        """
        Evaluate value estimates for a batch of states.
        Used during PPO training.

        Args:
            states: Batch of state observations (batch, sequence_length, num_features)

        Returns:
            values: State value estimates (batch, 1)
        """
        return self.forward(states)


if __name__ == '__main__':
    """Test CriticNetwork with random input."""
    print("=" * 80)
    print("Testing CriticNetwork")
    print("=" * 80)

    # Create network
    critic = CriticNetwork()
    print(f"\nNetwork Architecture:")
    print(critic)

    # Count parameters
    total_params = sum(p.numel() for p in critic.parameters())
    trainable_params = sum(p.numel() for p in critic.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test with random input
    batch_size = 32
    test_input = torch.randn(batch_size, SEQUENCE_LENGTH, NUM_FEATURES)
    print(f"\nInput shape: {test_input.shape}")

    # Forward pass
    values = critic(test_input)
    print(f"Output shape: {values.shape}")
    print(f"Output (first 5 samples): {values[:5].squeeze()}")
    print(f"Value range: [{values.min().item():.4f}, {values.max().item():.4f}]")
    print(f"Mean value: {values.mean().item():.4f}")

    # Test get_value (single state)
    single_state = torch.randn(SEQUENCE_LENGTH, NUM_FEATURES)
    value = critic.get_value(single_state)
    print(f"\nSingle state value: {value.item():.4f}")

    # Test evaluate_states
    batch_values = critic.evaluate_states(test_input)
    print(f"\nEvaluate states:")
    print(f"  Values shape: {batch_values.shape}")
    print(f"  Mean value: {batch_values.mean().item():.4f}")
    print(f"  Std value: {batch_values.std().item():.4f}")

    # Test output layer initialization
    init_range = NETWORK_INIT_PARAMS['value_head_init_range']
    print(f"\nOutput layer weight stats:")
    print(f"  Weight range: [{critic.fc_output.weight.min().item():.6f}, "
          f"{critic.fc_output.weight.max().item():.6f}]")
    print(f"  Expected range: [-{init_range}, {init_range}]")
    print(f"  Weight mean: {critic.fc_output.weight.mean().item():.6f}")

    print("\n" + "=" * 80)
    print("CriticNetwork test completed successfully!")
    print("=" * 80)
