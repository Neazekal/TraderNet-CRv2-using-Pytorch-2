"""
TraderNet-CRv2 PyTorch Agents Module

This module contains the PPO agent implementation and neural network architectures.
"""

from agents.networks.actor import ActorNetwork
from agents.networks.critic import CriticNetwork

__all__ = ['ActorNetwork', 'CriticNetwork']
