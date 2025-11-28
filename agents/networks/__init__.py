"""
Neural Network Architectures for TraderNet-CRv2

Contains Actor and Critic networks usable by QR-DQN and Categorical SAC.
"""

from agents.networks.actor import ActorNetwork
from agents.networks.critic import CriticNetwork

__all__ = ['ActorNetwork', 'CriticNetwork']
