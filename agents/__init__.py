"""
TraderNet-CRv2 PyTorch Agents Module

This module contains neural network backbones and agents for QR-DQN and Categorical SAC.
"""

from agents.networks.actor import ActorNetwork
from agents.networks.critic import CriticNetwork

__all__ = ['ActorNetwork', 'CriticNetwork']
