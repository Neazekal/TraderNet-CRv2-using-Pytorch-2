"""
Critic Network for TraderNet-CRv2 discrete-action agents.

Uses the shared ConvBackbone, then a value head to estimate V(s). This backbone
can also be reused to build Q/quantile heads for QR-DQN or Categorical SAC.
"""

import torch
import torch.nn as nn
from typing import Tuple

from config.config import (
    SEQUENCE_LENGTH,
    NUM_FEATURES,
    NETWORK_PARAMS,
    NETWORK_INIT_PARAMS,
)
from agents.networks.backbone import ConvBackbone


class CriticNetwork(nn.Module):
    """
    Critic network that estimates state values for discrete-action agents.

    Input: (batch_size, sequence_length, num_features) = (batch, 12, 21)
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
        super(CriticNetwork, self).__init__()

        self.backbone = ConvBackbone(
            sequence_length=sequence_length,
            num_features=num_features,
            conv_filters=conv_filters,
            conv_kernel=conv_kernel,
            fc_layers=fc_layers,
            activation=activation,
            dropout=dropout,
        )

        self.value_head = nn.Linear(fc_layers[-1], 1)
        nn.init.uniform_(
            self.value_head.weight,
            -NETWORK_INIT_PARAMS['value_head_init_range'],
            NETWORK_INIT_PARAMS['value_head_init_range'],
        )
        nn.init.constant_(self.value_head.bias, 0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute state value.

        Args:
            state: Tensor of shape (batch_size, sequence_length, num_features)

        Returns:
            value: Tensor of shape (batch_size, 1)
        """
        latent = self.backbone(state)
        value = self.value_head(latent)
        return value

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get a single state value estimate.

        Args:
            state: Tensor of shape (sequence_length, num_features) or batch

        Returns:
            value: Tensor scalar if input was single sample, else (batch, 1)
        """
        if state.dim() == 2:
            state = state.unsqueeze(0)
        value = self.forward(state)
        return value.squeeze()

    def evaluate_states(self, states: torch.Tensor) -> torch.Tensor:
        """
        Evaluate state values for a batch of states.
        """
        return self.forward(states)
