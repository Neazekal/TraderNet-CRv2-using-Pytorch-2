"""
Head modules for discrete-action RL agents.

- QuantileHead: outputs quantile values for QR-DQN.
- QValueHead: outputs Q-values per action (useful for SAC critics).
- CategoricalPolicyHead: outputs logits over actions.
"""

import torch.nn as nn


class QuantileHead(nn.Module):
    def __init__(self, input_dim: int, num_actions: int, num_quantiles: int):
        super().__init__()
        self.num_actions = num_actions
        self.num_quantiles = num_quantiles
        self.linear = nn.Linear(input_dim, num_actions * num_quantiles)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        out = self.linear(x)
        return out.view(-1, self.num_actions, self.num_quantiles)


class QValueHead(nn.Module):
    def __init__(self, input_dim: int, num_actions: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_actions)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        return self.linear(x)


class CategoricalPolicyHead(nn.Module):
    def __init__(self, input_dim: int, num_actions: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_actions)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        return self.linear(x)  # logits (no softmax applied here)
