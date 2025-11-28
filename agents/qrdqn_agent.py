"""
QR-DQN Agent for Distributional Deep Q-Learning.

Implements Quantile Regression DQN (QR-DQN) for distributional value learning.
Maintains a distribution over Q-values using quantile regression with Huber loss.

Paper: Dabney et al. "Distributional Reinforcement Learning with Quantile Regression"
       (ICML 2017)

Key Features:
- Distributional value learning: Q-value distribution via quantiles
- Prioritized Experience Replay: Rare important transitions sampled more often
- Target network: Separate network updated every N steps for stability
- Quantile Huber loss: Smooth loss function for quantile regression
- Epsilon-greedy exploration: Adaptive exploration strategy
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path

from config.config import (
    QR_DQN_PARAMS,
    NETWORK_PARAMS,
    NUM_ACTIONS,
    NUM_FEATURES,
    SEQUENCE_LENGTH,
    OBS_SHAPE,
    ACTION_NAMES,
    AGENT_TRAINING_PARAMS,
)
from agents.networks.actor import ActorNetwork
from agents.networks.critic import CriticNetwork
from agents.buffers.replay_buffer import ReplayBuffer


class QRDQNAgent:
    """
    Quantile Regression DQN agent for discrete action spaces.

    Learns a distributional value function using quantile regression.
    Distribution is represented as tau_quantiles: Q_tau(s,a) for tau in [0,1].

    Architecture:
    - Shared Conv1D + FC backbone (inherited from ActorNetwork structure)
    - Quantile regression head: (num_actions, num_quantiles)
    - Target network: Copy of main network updated periodically
    - Replay buffer: Prioritized sampling of experiences
    """

    def __init__(
        self,
        num_actions: int = NUM_ACTIONS,
        num_quantiles: int = 51,
        learning_rate: float = 0.0005,
        gamma: float = 0.99,
        huber_kappa: float = 1.0,
        replay_buffer_size: int = 500_000,
        batch_size: int = 128,
        target_update_interval: int = 2000,
        device: torch.device = torch.device('cpu'),
    ):
        """
        Initialize QR-DQN agent.

        Args:
            num_actions: Number of discrete actions
            num_quantiles: Number of quantiles for distribution
            learning_rate: Adam optimizer learning rate
            gamma: Discount factor
            huber_kappa: Huber loss threshold
            replay_buffer_size: Maximum replay buffer capacity
            batch_size: Training batch size
            target_update_interval: Steps between target network updates
            device: torch device for computation
        """
        self.num_actions = num_actions
        self.num_quantiles = num_quantiles
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.huber_kappa = huber_kappa
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.device = device

        # Training state
        self.total_steps = 0
        self.update_count = 0

        # Networks
        self._init_networks()

        # Optimizer
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=learning_rate,
        )

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=replay_buffer_size,
            alpha=0.6,
            beta_start=0.4,
            beta_frames=500_000,
            device=device,
        )

        # Quantile embedding for IQN-style operation
        self.tau_quantiles = torch.linspace(
            0, 1, num_quantiles + 2, device=device
        )[1:-1]  # Remove edges

    def _init_networks(self):
        """Initialize Q-network and target Q-network."""
        # Q-network with quantile output head
        self.q_network = QuantileQNetwork(
            num_actions=self.num_actions,
            num_quantiles=self.num_quantiles,
            device=self.device,
        ).to(self.device)

        # Target network (copy of Q-network)
        self.target_q_network = QuantileQNetwork(
            num_actions=self.num_actions,
            num_quantiles=self.num_quantiles,
            device=self.device,
        ).to(self.device)

        # Initialize target network with same weights
        self._update_target_network(tau=1.0)  # Hard copy

        # Freeze target network
        for param in self.target_q_network.parameters():
            param.requires_grad = False

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current observation (shape: (12, 21))
            epsilon: Exploration probability

        Returns:
            Action index (0, 1, or 2)
        """
        # Epsilon-greedy exploration
        if np.random.rand() < epsilon:
            return np.random.randint(self.num_actions)

        # Greedy action: argmax mean of quantile distribution
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            quantiles = self.q_network(state_tensor)  # (1, num_actions, num_quantiles)
            q_values = quantiles.mean(dim=2)  # (1, num_actions)
            action = q_values.argmax(dim=1).item()

        return action

    def add_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """
        Add experience to replay buffer.

        Args:
            state: Current observation
            action: Action taken
            reward: Reward received
            next_state: Next observation
            done: Episode termination flag
        """
        self.replay_buffer.add(state, action, reward, next_state, done)
        self.total_steps += 1

    def train_step(self) -> Optional[Dict]:
        """
        Perform one training step.

        Returns:
            Dictionary with training metrics, or None if buffer not ready
        """
        # Check if buffer has enough samples
        if not self.replay_buffer.is_ready(self.batch_size):
            return None

        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)

        # Compute loss and update
        loss, td_errors = self._compute_loss(batch)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        grad_clip_norm = AGENT_TRAINING_PARAMS['gradient_clip_norm']
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=grad_clip_norm)
        self.optimizer.step()

        # Update priorities in replay buffer
        td_errors_cpu = td_errors.detach().cpu().numpy()
        self.replay_buffer.update_priorities(batch['indices'], td_errors_cpu)

        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_interval == 0:
            self._update_target_network(tau=1.0)  # Hard update

        # Collect metrics
        metrics = {
            'loss': loss.item(),
            'mean_td_error': td_errors.mean().item(),
            'max_td_error': td_errors.max().item(),
            'buffer_size': len(self.replay_buffer),
        }

        return metrics

    def _compute_loss(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute quantile regression loss.

        Uses Huber loss with tau-weighted quantile regression.

        Args:
            batch: Batch from replay buffer

        Returns:
            Tuple of (loss, td_errors)
        """
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones'].float()
        weights = batch['weights']

        # Get current quantiles: Z(s,a) for taken actions
        quantiles = self.q_network(states)  # (batch, num_actions, num_quantiles)
        action_quantiles = quantiles[range(len(states)), actions]  # (batch, num_quantiles)

        # Get next quantiles: Z(s',a') for greedy actions
        with torch.no_grad():
            next_quantiles = self.target_q_network(next_states)  # (batch, num_actions, num_quantiles)
            next_q_values = next_quantiles.mean(dim=2)  # (batch, num_actions)
            next_actions = next_q_values.argmax(dim=1)  # (batch,)
            next_action_quantiles = next_quantiles[range(len(next_states)), next_actions]  # (batch, num_quantiles)

        # Compute target quantiles
        # Z_target(s,a) = r + gamma * (1 - done) * Z(s', argmax_a' Q(s',a'))
        target_quantiles = (
            rewards.unsqueeze(1) +
            self.gamma * (1 - dones.unsqueeze(1)) * next_action_quantiles
        )  # (batch, num_quantiles)

        # Quantile Huber loss
        # For each tau, minimize |tau - (z_target < z)| * huber(z_target - z)
        td_errors = target_quantiles - action_quantiles  # (batch, num_quantiles)
        huber_loss = self._huber_loss(td_errors)  # (batch, num_quantiles)

        # Quantile weights: tau - (1 if error < 0 else 0)
        tau = self.tau_quantiles.unsqueeze(0)  # (1, num_quantiles)
        quantile_weights = torch.abs(tau - (td_errors < 0).float())  # (batch, num_quantiles)

        # Apply quantile weights and take mean
        loss = (quantile_weights * huber_loss).mean(dim=1)  # (batch,)

        # Apply importance sampling weights and take mean
        loss = (weights * loss).mean()

        # Compute TD error for priority update (use max quantile difference)
        td_errors_for_priority = td_errors.abs().max(dim=1)[0]  # (batch,)

        return loss, td_errors_for_priority

    def _huber_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Huber loss element-wise.

        Args:
            x: Tensor of values

        Returns:
            Huber loss applied element-wise
        """
        huber = torch.where(
            x.abs() <= self.huber_kappa,
            0.5 * x ** 2,
            self.huber_kappa * (x.abs() - 0.5 * self.huber_kappa),
        )
        return huber

    def _update_target_network(self, tau: float = 0.005):
        """
        Update target network weights.

        Args:
            tau: Soft update coefficient (0=no update, 1=hard copy)
        """
        for param, target_param in zip(
            self.q_network.parameters(),
            self.target_q_network.parameters(),
        ):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )

    def save_checkpoint(self, filepath: Path):
        """
        Save agent checkpoint.

        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            'q_network': self.q_network.state_dict(),
            'target_q_network': self.target_q_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'update_count': self.update_count,
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: Path):
        """
        Load agent checkpoint.

        Args:
            filepath: Path to load checkpoint from
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_q_network.load_state_dict(checkpoint['target_q_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.total_steps = checkpoint['total_steps']
        self.update_count = checkpoint['update_count']

    def get_metrics(self) -> Dict:
        """Return current agent metrics."""
        return {
            'total_steps': self.total_steps,
            'update_count': self.update_count,
            'buffer_stats': self.replay_buffer.get_stats(),
        }


class QuantileQNetwork(nn.Module):
    """
    Quantile Regression Q-Network for QR-DQN.

    Architecture:
    - Shared backbone: Conv1D + FC layers
    - Output head: (num_actions, num_quantiles) for quantile values
    """

    def __init__(
        self,
        num_actions: int = 3,
        num_quantiles: int = 51,
        device: torch.device = torch.device('cpu'),
    ):
        """Initialize quantile Q-network."""
        super().__init__()

        self.num_actions = num_actions
        self.num_quantiles = num_quantiles
        self.device = device

        # Shared backbone (from ActorNetwork design)
        self.backbone = QuantileQNetworkBackbone()

        # Quantile head: output shape (num_actions * num_quantiles)
        self.quantile_head = nn.Linear(256, num_actions * num_quantiles)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state: Input state (batch, 12, 21)

        Returns:
            Quantiles (batch, num_actions, num_quantiles)
        """
        features = self.backbone(state)  # (batch, 256)
        quantiles_flat = self.quantile_head(features)  # (batch, num_actions * num_quantiles)
        quantiles = quantiles_flat.view(-1, self.num_actions, self.num_quantiles)
        return quantiles


class QuantileQNetworkBackbone(nn.Module):
    """Shared backbone for Q-network (Conv1D + FC layers)."""

    def __init__(self):
        super().__init__()

        # Conv1D layer - use NUM_FEATURES from config
        self.conv1d = nn.Conv1d(
            in_channels=NUM_FEATURES,
            out_channels=32,
            kernel_size=3,
            padding=1,
        )

        # Flattened size: 32 channels * SEQUENCE_LENGTH timesteps
        self.flatten_size = 32 * SEQUENCE_LENGTH

        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, 256)
        self.fc2 = nn.Linear(256, 256)

        # Activation
        self.activation = nn.GELU()

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for module in [self.conv1d, self.fc1, self.fc2]:
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state: (batch, 12, 21) - 12 timesteps, 21 features

        Returns:
            Features (batch, 256)
        """
        # Conv1d expects (batch, channels, seq_length)
        x = state.transpose(1, 2)  # (batch, 12, 21) → (batch, 21, 12)
        x = self.conv1d(x)  # (batch, 21, 12) → (batch, 32, 12)
        x = self.activation(x)

        # Flatten
        x = x.view(x.size(0), -1)  # (batch, 384)

        # FC layers
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)

        return x


if __name__ == '__main__':
    """Test QR-DQN agent."""
    import sys
    sys.path.insert(0, '.')

    # Create agent
    agent = QRDQNAgent()
    print(f"QR-DQN Agent created")
    print(f"  Q-Network params: {sum(p.numel() for p in agent.q_network.parameters()):,}")
    print(f"  Action space: {agent.num_actions}")
    print(f"  Quantiles: {agent.num_quantiles}")

    # Simulate some experiences
    print("\nAdding experiences to buffer...")
    for i in range(50):
        state = np.random.randn(12, 21).astype(np.float32)
        action = np.random.randint(0, agent.num_actions)
        reward = np.random.randn()
        next_state = np.random.randn(12, 21).astype(np.float32)
        done = np.random.rand() < 0.1

        agent.add_experience(state, action, reward, next_state, done)

    print(f"Buffer size: {len(agent.replay_buffer)}")

    # Test training
    print("\nTraining...")
    for step in range(10):
        metrics = agent.train_step()
        if metrics:
            print(f"Step {step}: Loss={metrics['loss']:.4f}, "
                  f"TD-error={metrics['mean_td_error']:.4f}")

    # Test action selection
    print("\nAction selection...")
    state = np.random.randn(12, 21).astype(np.float32)
    action = agent.select_action(state, epsilon=0.0)
    print(f"Selected action: {action} ({ACTION_NAMES[action]})")

    print("\n✓ QR-DQN agent test passed!")
