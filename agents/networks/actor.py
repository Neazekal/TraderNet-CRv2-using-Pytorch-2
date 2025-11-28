"""
Actor Network for TraderNet-CRv2 discrete-action agents.

The Actor network outputs action probabilities for the trading policy.
Maps state observations to a probability distribution over actions (used by categorical SAC and can serve as a policy head for other discrete methods).
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
    NETWORK_INIT_PARAMS
)


class ActorNetwork(nn.Module):
    """
    Actor network that outputs action probabilities for discrete-action agents.

    Architecture:
        - Conv1D layer for temporal feature extraction
        - Flatten
        - Two fully connected hidden layers
        - Output layer with Softmax activation

    Input: (batch_size, sequence_length, num_features) = (batch, 12, 28)
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
        """
        Initialize Actor network.

        Args:
            sequence_length: Number of timesteps in observation (default: 12)
            num_features: Number of features per timestep (default: 28)
            num_actions: Number of discrete actions (default: 3)
            conv_filters: Number of Conv1D output channels (default: 32)
            conv_kernel: Conv1D kernel size (default: 3)
            fc_layers: List of FC layer sizes (default: [256, 256])
            activation: Activation function name (default: 'gelu')
            dropout: Dropout rate (default: 0.0)
        """
        super(ActorNetwork, self).__init__()

        self.sequence_length = sequence_length
        self.num_features = num_features
        self.num_actions = num_actions
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

        # Output layer for action probabilities
        self.fc_output = nn.Linear(fc_input_size, num_actions)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier/Kaiming initialization."""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Actor network.

        Args:
            x: State observation tensor
               Shape: (batch_size, sequence_length, num_features)

        Returns:
            Action probabilities
            Shape: (batch_size, num_actions)
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

        # Output layer
        x = self.fc_output(x)  # (batch, num_actions)

        # Softmax to get action probabilities
        action_probs = F.softmax(x, dim=-1)  # (batch, num_actions)

        return action_probs

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[int, torch.Tensor]:
        """
        Sample an action from the policy.

        Args:
            state: Single state observation (sequence_length, num_features)
            deterministic: If True, return argmax action (greedy)
                          If False, sample from distribution (stochastic)

        Returns:
            action: Selected action index (0, 1, or 2)
            log_prob: Log probability of the selected action
        """
        # Add batch dimension if needed
        if state.dim() == 2:
            state = state.unsqueeze(0)  # (1, sequence_length, num_features)

        # Get action probabilities
        action_probs = self.forward(state)  # (1, num_actions)

        if deterministic:
            # Greedy action selection
            action = torch.argmax(action_probs, dim=-1)
        else:
            # Stochastic action selection
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()

        # Calculate log probability
        epsilon = NETWORK_INIT_PARAMS['log_prob_epsilon']
        log_prob = torch.log(action_probs.squeeze(0)[action] + epsilon)

        return action.item(), log_prob

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probabilities and entropy for given state-action pairs.
        Useful for policy-gradient style training (e.g., categorical SAC).

        Args:
            states: Batch of state observations (batch, sequence_length, num_features)
            actions: Batch of actions taken (batch,)

        Returns:
            log_probs: Log probabilities of actions (batch,)
            entropy: Entropy of action distributions (batch,)
        """
        # Get action probabilities
        action_probs = self.forward(states)  # (batch, num_actions)

        # Create categorical distribution
        dist = torch.distributions.Categorical(action_probs)

        # Calculate log probabilities for the given actions
        log_probs = dist.log_prob(actions)  # (batch,)

        # Calculate entropy
        entropy = dist.entropy()  # (batch,)

        return log_probs, entropy


if __name__ == '__main__':
    """Test ActorNetwork with random input."""
    print("=" * 80)
    print("Testing ActorNetwork")
    print("=" * 80)

    # Create network
    actor = ActorNetwork()
    print(f"\nNetwork Architecture:")
    print(actor)

    # Count parameters
    total_params = sum(p.numel() for p in actor.parameters())
    trainable_params = sum(p.numel() for p in actor.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test with random input
    batch_size = 32
    test_input = torch.randn(batch_size, SEQUENCE_LENGTH, NUM_FEATURES)
    print(f"\nInput shape: {test_input.shape}")

    # Forward pass
    action_probs = actor(test_input)
    print(f"Output shape: {action_probs.shape}")
    print(f"Output (first sample): {action_probs[0]}")
    print(f"Sum of probabilities: {action_probs[0].sum().item():.6f} (should be ~1.0)")

    # Test get_action (single state)
    single_state = torch.randn(SEQUENCE_LENGTH, NUM_FEATURES)
    action_det, log_prob_det = actor.get_action(single_state, deterministic=True)
    action_stoch, log_prob_stoch = actor.get_action(single_state, deterministic=False)
    print(f"\nDeterministic action: {action_det}, log_prob: {log_prob_det.item():.4f}")
    print(f"Stochastic action: {action_stoch}, log_prob: {log_prob_stoch.item():.4f}")

    # Test evaluate_actions
    actions = torch.randint(0, NUM_ACTIONS, (batch_size,))
    log_probs, entropy = actor.evaluate_actions(test_input, actions)
    print(f"\nEvaluate actions:")
    print(f"  Log probs shape: {log_probs.shape}")
    print(f"  Entropy shape: {entropy.shape}")
    print(f"  Mean entropy: {entropy.mean().item():.4f}")

    print("\n" + "=" * 80)
    print("ActorNetwork test completed successfully!")
    print("=" * 80)
