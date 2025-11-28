"""
Neural Network Architectures for TraderNet-CRv2

Contains Actor and Critic networks for PPO agent.
"""

from agents.networks.actor import ActorNetwork
from agents.networks.critic import CriticNetwork

__all__ = ['ActorNetwork', 'CriticNetwork']
