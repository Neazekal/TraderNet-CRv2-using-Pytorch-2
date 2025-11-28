"""
Categorical Soft Actor-Critic (Cat-SAC) Agent.

Implements Soft Actor-Critic with categorical policy for discrete action spaces.
Combines maximum entropy RL with off-policy learning for stable exploration.

Paper: Haarnoja et al. "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL"
       (ICML 2018)

Key Features:
- Soft Actor-Critic framework: policy + value learning with entropy regularization
- Categorical policy: discrete action probabilities from Actor network
- Twin Q-networks: Two critics to reduce overestimation bias
- Entropy temperature auto-tuning: Dynamic alpha adjustment
- Prioritized Experience Replay: Rare important transitions
- Soft target updates: Gradual network target updates (tau=0.005)
- Multi-GPU support: Automatic DataParallel wrapping when multiple GPUs available
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
from pathlib import Path

from config.config import (
    CATEGORICAL_SAC_PARAMS,
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
from utils.distributed import (
    get_device, wrap_model_multi_gpu, unwrap_model,
    get_available_gpus, print_gpu_info
)


class CategoricalSACAgent:
    """
    Categorical Soft Actor-Critic agent for discrete actions.

    Learns a policy and value function using entropy-regularized maximum expected
    reward objective. The policy is categorical (discrete action probabilities).

    Architecture:
    - Actor: Categorical policy network (outputs action probabilities)
    - Critic: Twin Q-networks for off-policy learning with entropy regularization
    - Entropy temperature: Auto-tuned for target entropy level
    - Replay buffer: Prioritized sampling of experiences
    """

    def __init__(
        self,
        num_actions: int = NUM_ACTIONS,
        learning_rate: float = 0.0005,
        gamma: float = 0.99,
        tau: float = 0.005,
        entropy_target: float = -1.0,
        alpha_init: float = 0.2,
        batch_size: int = 256,
        replay_buffer_size: int = 500_000,
        target_update_interval: int = 1,
        device: torch.device = None,
        gpu_ids: Optional[List[int]] = None,
        use_multi_gpu: bool = True,
    ):
        """
        Initialize Categorical SAC agent.

        Args:
            num_actions: Number of discrete actions
            learning_rate: Adam optimizer learning rate
            gamma: Discount factor
            tau: Soft update coefficient (0.005 = gentle updates)
            entropy_target: Target entropy for auto alpha tuning
            alpha_init: Initial entropy temperature
            batch_size: Training batch size
            replay_buffer_size: Maximum replay buffer capacity
            target_update_interval: Steps between target network updates
            device: torch device for computation (auto-detected if None)
            gpu_ids: List of GPU IDs for multi-GPU training (auto-detected if None)
            use_multi_gpu: Whether to use multiple GPUs if available
        """
        # Auto-detect device and GPUs if not specified
        if device is None:
            self.device, detected_gpus = get_device(prefer_multi_gpu=use_multi_gpu)
        else:
            self.device = device
            detected_gpus = get_available_gpus() if use_multi_gpu else []
        
        # Set GPU IDs for multi-GPU training
        self.gpu_ids = gpu_ids if gpu_ids is not None else detected_gpus
        self.use_multi_gpu = use_multi_gpu and len(self.gpu_ids) > 1
        
        if self.use_multi_gpu:
            print(f"Categorical SAC: Multi-GPU training enabled on GPUs {self.gpu_ids}")
        
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval

        # Training state
        self.total_steps = 0
        self.update_count = 0

        # Entropy target and temperature
        self.target_entropy = entropy_target
        self.log_alpha = torch.tensor(np.log(alpha_init), device=self.device, requires_grad=True)

        # Networks
        self._init_networks()

        # Optimizers - use unwrapped model parameters for DataParallel
        self.actor_optimizer = optim.Adam(
            self._get_actor().parameters(),
            lr=learning_rate,
        )

        self.q1_optimizer = optim.Adam(
            self._get_q1_network().parameters(),
            lr=learning_rate,
        )

        self.q2_optimizer = optim.Adam(
            self._get_q2_network().parameters(),
            lr=learning_rate,
        )

        self.alpha_optimizer = optim.Adam(
            [self.log_alpha],
            lr=learning_rate,
        )

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=replay_buffer_size,
            alpha=0.6,
            beta_start=0.4,
            beta_frames=500_000,
            device=self.device,
        )

    def _get_actor(self) -> nn.Module:
        """Get the underlying actor network (unwrapped if DataParallel)."""
        return unwrap_model(self.actor)
    
    def _get_q1_network(self) -> nn.Module:
        """Get the underlying Q1 network (unwrapped if DataParallel)."""
        return unwrap_model(self.q1_network)
    
    def _get_q2_network(self) -> nn.Module:
        """Get the underlying Q2 network (unwrapped if DataParallel)."""
        return unwrap_model(self.q2_network)
    
    def _get_target_q1_network(self) -> nn.Module:
        """Get the underlying target Q1 network (unwrapped if DataParallel)."""
        return unwrap_model(self.target_q1_network)
    
    def _get_target_q2_network(self) -> nn.Module:
        """Get the underlying target Q2 network (unwrapped if DataParallel)."""
        return unwrap_model(self.target_q2_network)

    def _init_networks(self):
        """Initialize actor and critic networks with optional multi-GPU support."""
        # Actor: Outputs categorical policy logits
        self.actor = ActorNetwork().to(self.device)

        # Twin Q-networks with shared backbone but separate heads
        self.q1_network = SACQNetwork().to(self.device)
        self.q2_network = SACQNetwork().to(self.device)

        # Target Q-networks (copy for soft updates)
        self.target_q1_network = SACQNetwork().to(self.device)
        self.target_q2_network = SACQNetwork().to(self.device)

        # Wrap with DataParallel if multi-GPU is enabled
        if self.use_multi_gpu:
            self.actor = wrap_model_multi_gpu(self.actor, self.gpu_ids)
            self.q1_network = wrap_model_multi_gpu(self.q1_network, self.gpu_ids)
            self.q2_network = wrap_model_multi_gpu(self.q2_network, self.gpu_ids)
            self.target_q1_network = wrap_model_multi_gpu(self.target_q1_network, self.gpu_ids)
            self.target_q2_network = wrap_model_multi_gpu(self.target_q2_network, self.gpu_ids)

        # Initialize target networks
        self._hard_update_target_networks()

        # Freeze target networks
        for param in self._get_target_q1_network().parameters():
            param.requires_grad = False
        for param in self._get_target_q2_network().parameters():
            param.requires_grad = False

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> int:
        """
        Select action using policy.

        Args:
            state: Current observation (shape: (12, 21))
            deterministic: Use max probability instead of sampling

        Returns:
            Action index (0, 1, or 2)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_probs = self.actor(state_tensor)  # (1, num_actions)

            if deterministic:
                # Greedy: max probability action
                action = action_probs.argmax(dim=1).item()
            else:
                # Stochastic: sample from categorical distribution
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample().item()

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

        Updates actor, Q-networks, and entropy temperature.

        Returns:
            Dictionary with training metrics, or None if buffer not ready
        """
        # Check if buffer has enough samples
        if not self.replay_buffer.is_ready(self.batch_size):
            return None

        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)

        # Update Q-networks
        q1_loss, q2_loss, td_errors = self._update_q_networks(batch)

        # Update actor
        actor_loss, entropy = self._update_actor(batch)

        # Update entropy temperature
        alpha_loss = self._update_alpha(entropy)

        # Update target networks (soft update)
        self.update_count += 1
        if self.update_count % self.target_update_interval == 0:
            self._soft_update_target_networks()

        # Update priorities in replay buffer
        td_errors_cpu = td_errors.detach().cpu().numpy()
        self.replay_buffer.update_priorities(batch['indices'], td_errors_cpu)

        # Collect metrics
        metrics = {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item() if alpha_loss is not None else 0.0,
            'alpha': self.log_alpha.exp().item(),
            'entropy': entropy.item(),
            'mean_td_error': td_errors.mean().item(),
            'buffer_size': len(self.replay_buffer),
        }

        return metrics

    def _update_q_networks(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Update twin Q-networks.

        Args:
            batch: Batch from replay buffer

        Returns:
            Tuple of (q1_loss, q2_loss, td_errors)
        """
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones'].float()
        weights = batch['weights']

        # Get current Q-values for taken actions
        q1_values = self.q1_network(states)  # (batch, num_actions)
        q2_values = self.q2_network(states)  # (batch, num_actions)

        q1_action = q1_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # (batch,)
        q2_action = q2_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # (batch,)

        # Compute target Q-values
        with torch.no_grad():
            # Get next action probabilities and values
            next_action_probs = self.actor(next_states)  # (batch, num_actions)

            # Get next Q-values from target networks
            next_q1 = self.target_q1_network(next_states)  # (batch, num_actions)
            next_q2 = self.target_q2_network(next_states)  # (batch, num_actions)
            next_q_min = torch.min(next_q1, next_q2)  # (batch, num_actions)

            # Entropy regularized value: V(s') = E_a[Q(s',a) - alpha * log pi(a|s')]
            log_eps = AGENT_TRAINING_PARAMS['log_epsilon']
            next_log_probs = torch.log(next_action_probs + log_eps)  # (batch, num_actions)
            alpha = self.log_alpha.exp()
            next_value = (next_action_probs * (next_q_min - alpha * next_log_probs)).sum(dim=1)  # (batch,)

            # Target Q-value
            target_q = rewards + self.gamma * (1 - dones) * next_value  # (batch,)

        # Q-network losses
        q1_loss = F.mse_loss(q1_action, target_q, reduction='none')  # (batch,)
        q2_loss = F.mse_loss(q2_action, target_q, reduction='none')  # (batch,)

        # Apply importance sampling weights
        q1_loss = (weights * q1_loss).mean()
        q2_loss = (weights * q2_loss).mean()

        # Update Q1
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        grad_clip_norm = AGENT_TRAINING_PARAMS['gradient_clip_norm']
        torch.nn.utils.clip_grad_norm_(self.q1_network.parameters(), max_norm=grad_clip_norm)
        self.q1_optimizer.step()

        # Update Q2
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q2_network.parameters(), max_norm=grad_clip_norm)
        self.q2_optimizer.step()

        # Compute TD errors for priority update
        with torch.no_grad():
            td_errors = torch.abs(q1_action - target_q) + torch.abs(q2_action - target_q)

        return q1_loss, q2_loss, td_errors

    def _update_actor(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update actor (policy) network.

        Args:
            batch: Batch from replay buffer

        Returns:
            Tuple of (actor_loss, entropy)
        """
        states = batch['states']
        weights = batch['weights']

        # Get policy
        action_probs = self.actor(states)  # (batch, num_actions)
        log_eps = AGENT_TRAINING_PARAMS['log_epsilon']
        log_action_probs = torch.log(action_probs + log_eps)

        # Get Q-values
        q1_values = self.q1_network(states)
        q2_values = self.q2_network(states)
        q_min = torch.min(q1_values, q2_values)  # (batch, num_actions)

        # Entropy regularized loss
        alpha = self.log_alpha.exp().detach()
        actor_loss = (action_probs * (alpha * log_action_probs - q_min)).sum(dim=1)  # (batch,)

        # Apply importance sampling weights
        actor_loss = (weights * actor_loss).mean()

        # Entropy
        entropy = -(action_probs * log_action_probs).sum(dim=1).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        grad_clip_norm = AGENT_TRAINING_PARAMS['gradient_clip_norm']
        torch.nn.utils.clip_grad_norm_(self._get_actor().parameters(), max_norm=grad_clip_norm)
        self.actor_optimizer.step()

        return actor_loss, entropy

    def _update_alpha(self, entropy: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Update entropy temperature (alpha).

        Args:
            entropy: Current policy entropy

        Returns:
            Alpha loss, or None
        """
        # Target entropy: maximize policy entropy towards target
        # Loss = -log(alpha) * (entropy + target_entropy)
        alpha_loss = -(self.log_alpha * (entropy + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        return alpha_loss

    def _soft_update_target_networks(self):
        """Soft update target networks using tau."""
        # Use unwrapped models for parameter iteration (handles DataParallel)
        # Q1 target
        for param, target_param in zip(
            self._get_q1_network().parameters(),
            self._get_target_q1_network().parameters(),
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        # Q2 target
        for param, target_param in zip(
            self._get_q2_network().parameters(),
            self._get_target_q2_network().parameters(),
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def _hard_update_target_networks(self):
        """Hard copy target networks."""
        # Use unwrapped models for parameter iteration (handles DataParallel)
        for param, target_param in zip(
            self._get_q1_network().parameters(),
            self._get_target_q1_network().parameters(),
        ):
            target_param.data.copy_(param.data)

        for param, target_param in zip(
            self._get_q2_network().parameters(),
            self._get_target_q2_network().parameters(),
        ):
            target_param.data.copy_(param.data)

    def save_checkpoint(self, filepath: Path):
        """
        Save agent checkpoint.

        Args:
            filepath: Path to save checkpoint
        """
        # Save unwrapped model state dicts (handles DataParallel)
        checkpoint = {
            'actor': self._get_actor().state_dict(),
            'q1_network': self._get_q1_network().state_dict(),
            'q2_network': self._get_q2_network().state_dict(),
            'target_q1_network': self._get_target_q1_network().state_dict(),
            'target_q2_network': self._get_target_q2_network().state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'q1_optimizer': self.q1_optimizer.state_dict(),
            'q2_optimizer': self.q2_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
            'log_alpha': self.log_alpha.data,
            'total_steps': self.total_steps,
            'update_count': self.update_count,
            'use_multi_gpu': self.use_multi_gpu,
            'gpu_ids': self.gpu_ids,
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: Path):
        """
        Load agent checkpoint.

        Args:
            filepath: Path to load checkpoint from
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        # Load into unwrapped models (handles DataParallel)
        self._get_actor().load_state_dict(checkpoint['actor'])
        self._get_q1_network().load_state_dict(checkpoint['q1_network'])
        self._get_q2_network().load_state_dict(checkpoint['q2_network'])
        self._get_target_q1_network().load_state_dict(checkpoint['target_q1_network'])
        self._get_target_q2_network().load_state_dict(checkpoint['target_q2_network'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.q1_optimizer.load_state_dict(checkpoint['q1_optimizer'])
        self.q2_optimizer.load_state_dict(checkpoint['q2_optimizer'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        self.log_alpha.data = checkpoint['log_alpha']
        self.total_steps = checkpoint['total_steps']
        self.update_count = checkpoint['update_count']

    def get_metrics(self) -> Dict:
        """Return current agent metrics."""
        return {
            'total_steps': self.total_steps,
            'update_count': self.update_count,
            'alpha': self.log_alpha.exp().item(),
            'buffer_stats': self.replay_buffer.get_stats(),
        }


class SACQNetwork(nn.Module):
    """
    Q-Network for Soft Actor-Critic (SAC).

    Architecture:
    - Shared backbone: Conv1D + FC layers
    - Output head: (num_actions,) for Q-values of each action
    """

    def __init__(self, num_actions: int = 3):
        """Initialize Q-network."""
        super().__init__()

        # Shared backbone
        self.backbone = SACQNetworkBackbone()

        # Q-value head
        self.q_head = nn.Linear(256, num_actions)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state: Input state (batch, 12, 21)

        Returns:
            Q-values (batch, num_actions)
        """
        features = self.backbone(state)  # (batch, 256)
        q_values = self.q_head(features)  # (batch, num_actions)
        return q_values


class SACQNetworkBackbone(nn.Module):
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
    """Test Categorical SAC agent."""
    import sys
    sys.path.insert(0, '.')

    # Create agent
    agent = CategoricalSACAgent()
    print(f"Categorical SAC Agent created")
    print(f"  Actor params: {sum(p.numel() for p in agent.actor.parameters()):,}")
    print(f"  Q1 params: {sum(p.numel() for p in agent.q1_network.parameters()):,}")
    print(f"  Q2 params: {sum(p.numel() for p in agent.q2_network.parameters()):,}")
    print(f"  Action space: {agent.num_actions}")
    print(f"  Initial alpha: {agent.log_alpha.exp().item():.4f}")

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

    # Test action selection
    print("\nAction selection...")
    state = np.random.randn(12, 21).astype(np.float32)
    action_stochastic = agent.select_action(state, deterministic=False)
    action_deterministic = agent.select_action(state, deterministic=True)
    print(f"Stochastic action: {action_stochastic} ({ACTION_NAMES[action_stochastic]})")
    print(f"Deterministic action: {action_deterministic} ({ACTION_NAMES[action_deterministic]})")

    print("\n✓ Categorical SAC agent test passed!")
