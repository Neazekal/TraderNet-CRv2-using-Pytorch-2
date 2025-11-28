"""
Actor Network for TraderNet-CRv2 discrete-action agents.

Uses the shared ConvBackbone, then a categorical policy head with Softmax to
produce action probabilities. Suitable for categorical SAC or simple policy
gradient baselines; can also act as the policy head while QR-DQN uses its own
quantile head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from config.config import (
    SEQUENCE_LENGTH,
    NUM_FEATURES,
    NUM_ACTIONS,
    NETWORK_PARAMS,
)
from agents.networks.backbone import ConvBackbone
from agents.networks.heads import CategoricalPolicyHead


class ActorNetwork(nn.Module):
    """
    Actor network that outputs action probabilities for discrete-action agents.

    Input: (batch_size, sequence_length, num_features) = (batch, 12, 21)
    Output: (batch_size, num_actions) = (batch, 3) - action probabilities
    """

    def __init__(
        self,
        sequence_length: int = SEQUENCE_LENGTH,
        num_features: int = NUM_FEATURES,
        num_actions: int = NUM_ACTIONS,
        conv_filters: int = NETWORK_PARAMS['conv_filters'],
        conv_kernel: int = NETWORK_PARAMS['conv_kernel'],
        fc_layers: list = NETWORK_PARAMS['fc_layers'],
        activation: str = NETWORK_PARAMS['activation'],
        dropout: float = NETWORK_PARAMS['dropout']
    ):
        super(ActorNetwork, self).__init__()

        self.backbone = ConvBackbone(
            sequence_length=sequence_length,
            num_features=num_features,
            conv_filters=conv_filters,
            conv_kernel=conv_kernel,
            fc_layers=fc_layers,
            activation=activation,
            dropout=dropout,
        )
        self.policy_head = CategoricalPolicyHead(fc_layers[-1], num_actions)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute action probabilities.

        Args:
            state: Tensor of shape (batch_size, sequence_length, num_features)

        Returns:
            action_probs: Tensor of shape (batch_size, num_actions)
        """
        latent = self.backbone(state)
        logits = self.policy_head(latent)
        action_probs = F.softmax(logits, dim=-1)
        return action_probs

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[int, torch.Tensor]:
        """
        Get an action and its log probability for a single state.

        Args:
            state: Tensor of shape (sequence_length, num_features)
            deterministic: If True, pick argmax; else sample from distribution

        Returns:
            action: int
            log_prob: tensor scalar
        """
        if state.dim() == 2:
            state = state.unsqueeze(0)  # Add batch dimension
        action_probs = self.forward(state)
        dist = torch.distributions.Categorical(action_probs)
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
            log_prob = dist.log_prob(action)
        else:
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob.squeeze()

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probabilities and entropy for given actions (for training).

        Args:
            states: Tensor of shape (batch_size, sequence_length, num_features)
            actions: Tensor of shape (batch_size,)

        Returns:
            log_probs: Tensor of shape (batch_size,)
            entropy: Tensor scalar (mean entropy)
        """
        action_probs = self.forward(states)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        return log_probs, entropy
