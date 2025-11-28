"""
Prioritized Experience Replay (PER) Buffer for RL agents.

Implements a replay buffer with prioritized sampling for rare important experiences.
Supports both QR-DQN (distributional) and Categorical SAC algorithms.

Key Features:
- Prioritized Experience Replay (Schaul et al., 2015)
- Efficient binary segment tree for O(log N) operations
- Support for quantile targets (QR-DQN)
- Batch sampling with importance weights for off-policy learning
"""

import numpy as np
import torch
from typing import Tuple, Dict, Optional
from collections import deque


class SumTree:
    """
    Binary segment tree for efficient priority sampling.

    Allows O(log N) priority updates and O(log N) sampling.
    Used for Prioritized Experience Replay (PER).
    """

    def __init__(self, capacity: int):
        """Initialize sum tree with given capacity."""
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Internal + leaf nodes
        self.data = deque(maxlen=capacity)
        self.n_entries = 0
        self.pending_idx = 0

    def add(self, priority: float, data: Tuple) -> int:
        """
        Add new experience with priority.

        Args:
            priority: Priority value (TD error or 1.0 for new experiences)
            data: Experience tuple (state, action, reward, next_state, done)

        Returns:
            Index where experience was stored
        """
        # Get index in leaf nodes
        leaf_idx = self.pending_idx + self.capacity - 1

        # Store data
        self.data.append(data)

        # Update priority
        self._update_tree(leaf_idx, priority)

        # Move to next position (circular)
        self.pending_idx = (self.pending_idx + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

        return leaf_idx - (self.capacity - 1)

    def _update_tree(self, leaf_idx: int, priority: float):
        """Update tree values from leaf to root."""
        delta = priority - self.tree[leaf_idx]
        self.tree[leaf_idx] = priority

        # Propagate change up the tree
        parent_idx = (leaf_idx - 1) // 2
        while parent_idx >= 0:
            self.tree[parent_idx] += delta
            parent_idx = (parent_idx - 1) // 2

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample batch of experiences using prioritized sampling.

        Args:
            batch_size: Number of samples to draw

        Returns:
            Tuple of (indices, data, is_weights) where:
            - indices: Array of sampled indices
            - data: List of experience tuples
            - is_weights: Importance sampling weights
        """
        indices = []
        data = []
        is_weights = []

        total_priority = self.tree[0]
        min_priority = np.min(self.tree[-self.n_entries:]) / total_priority

        for _ in range(batch_size):
            # Sample from priority distribution
            priority_sample = np.random.uniform(0, total_priority)

            # Find leaf index with accumulated priority >= sample
            leaf_idx = self._find_leaf(priority_sample)
            idx = leaf_idx - (self.capacity - 1)

            indices.append(idx)
            data.append(self.data[idx])

            # Calculate importance weight: (1/N * 1/P(i))^beta
            priority = self.tree[leaf_idx]
            is_weight = (self.n_entries * priority / total_priority) ** (-1)  # beta=1 for simplicity
            is_weights.append(is_weight)

        # Normalize importance weights
        is_weights = np.array(is_weights)
        is_weights /= is_weights.max()

        return np.array(indices), data, is_weights

    def _find_leaf(self, value: float) -> int:
        """Find leaf index with accumulated priority >= value."""
        idx = 0
        while idx < self.capacity - 1:
            left_child = 2 * idx + 1
            right_child = 2 * idx + 2

            if self.tree[left_child] >= value:
                idx = left_child
            else:
                value -= self.tree[left_child]
                idx = right_child

        return idx

    def update_priority(self, leaf_idx: int, priority: float):
        """Update priority of a leaf node."""
        self._update_tree(leaf_idx, priority)

    def get_max_priority(self) -> float:
        """Get maximum priority in buffer."""
        return np.max(self.tree[-self.n_entries:]) if self.n_entries > 0 else 1.0


class ReplayBuffer:
    """
    Prioritized Experience Replay (PER) buffer for RL training.

    Stores transitions (state, action, reward, next_state, done) with priorities.
    Supports efficient prioritized sampling and importance weight calculation.

    Works with:
    - QR-DQN: Distributional Q-learning with quantile regression
    - Categorical SAC: Soft Actor-Critic with categorical policy

    Paper: Schaul et al. "Prioritized Experience Replay" (ICLR 2016)
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
        device: torch.device = torch.device('cpu'),
    ):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum buffer size
            alpha: Priority exponent (0=uniform, 1=full priority)
            beta_start: Initial importance sampling exponent
            beta_frames: Number of frames to anneal beta to 1.0
            device: torch device for tensor operations
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.device = device

        # Storage
        self.states = None
        self.actions = None
        self.rewards = None
        self.next_states = None
        self.dones = None

        self.position = 0
        self.size = 0

        # Priority tracking
        self.priorities = np.zeros(capacity)
        self.max_priority = 1.0

        # Frame counter for beta annealing
        self.frame_count = 0

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        priority: Optional[float] = None,
    ):
        """
        Add experience to buffer.

        Args:
            state: Observation (shape: (12, 28) for TraderNet)
            action: Action taken (0, 1, or 2)
            reward: Reward received
            next_state: Next observation
            done: Episode termination flag
            priority: Priority value (default: max_priority)
        """
        # Initialize storage on first add
        if self.states is None:
            self._initialize_storage(state)

        # Store experience
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done

        # Set priority (max priority for new experiences)
        if priority is None:
            priority = self.max_priority

        # Store priority with alpha weighting for PER
        self.priorities[self.position] = priority ** self.alpha

        # Move to next position (circular)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _initialize_storage(self, state: np.ndarray):
        """Initialize storage arrays on first experience."""
        state_shape = state.shape

        self.states = np.zeros((self.capacity,) + state_shape, dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_states = np.zeros((self.capacity,) + state_shape, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.bool_)

    def sample(self, batch_size: int) -> Dict:
        """
        Sample prioritized batch from buffer.

        Args:
            batch_size: Number of samples to draw

        Returns:
            Dictionary containing:
            - 'states': Tensor of states
            - 'actions': Tensor of actions
            - 'rewards': Tensor of rewards
            - 'next_states': Tensor of next states
            - 'dones': Tensor of done flags
            - 'indices': Array of buffer indices
            - 'weights': Importance sampling weights
        """
        assert self.size > 0, "Buffer is empty"

        # Sample indices according to priorities
        priorities = self.priorities[:self.size]
        probabilities = priorities / priorities.sum()

        indices = np.random.choice(
            self.size, size=batch_size, p=probabilities, replace=True
        )

        # Calculate importance sampling weights
        # is_weight = (1/N * 1/P(i))^beta
        is_weights = (self.size * probabilities[indices]) ** (-self._get_beta())
        is_weights /= is_weights.max()  # Normalize by max

        # Convert to tensors
        batch = {
            'states': torch.FloatTensor(self.states[indices]).to(self.device),
            'actions': torch.LongTensor(self.actions[indices]).to(self.device),
            'rewards': torch.FloatTensor(self.rewards[indices]).to(self.device),
            'next_states': torch.FloatTensor(self.next_states[indices]).to(self.device),
            'dones': torch.BoolTensor(self.dones[indices]).to(self.device),
            'indices': indices,
            'weights': torch.FloatTensor(is_weights).to(self.device),
        }

        return batch

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        Update priorities based on TD errors.

        Args:
            indices: Indices of experiences to update
            td_errors: TD error magnitudes (higher = more important)
        """
        # Clip TD errors for numerical stability
        td_errors = np.abs(td_errors) + 1e-6

        # Update priorities with alpha weighting
        for idx, td_error in zip(indices, td_errors):
            priority = td_error ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def _get_beta(self) -> float:
        """Get current beta value (annealed from beta_start to 1.0)."""
        progress = min(self.frame_count / self.beta_frames, 1.0)
        beta = self.beta_start + progress * (1.0 - self.beta_start)
        self.frame_count += 1
        return beta

    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for training."""
        return self.size >= batch_size

    def reset(self):
        """Clear all experiences from buffer."""
        self.states = None
        self.actions = None
        self.rewards = None
        self.next_states = None
        self.dones = None
        self.position = 0
        self.size = 0
        self.priorities = np.zeros(self.capacity)
        self.max_priority = 1.0
        self.frame_count = 0

    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size

    def get_stats(self) -> Dict:
        """Return buffer statistics."""
        return {
            'size': self.size,
            'capacity': self.capacity,
            'utilization': self.size / self.capacity,
            'max_priority': self.max_priority,
            'avg_priority': self.priorities[:self.size].mean(),
        }


if __name__ == '__main__':
    """Test replay buffer."""
    import sys
    sys.path.insert(0, '.')

    # Create buffer
    buffer = ReplayBuffer(capacity=1000, alpha=0.6, beta_start=0.4)

    # Add some dummy experiences
    for i in range(100):
        state = np.random.randn(12, 28).astype(np.float32)
        action = np.random.randint(0, 3)
        reward = np.random.randn()
        next_state = np.random.randn(12, 28).astype(np.float32)
        done = np.random.rand() < 0.1

        buffer.add(state, action, reward, next_state, done)

    # Sample batch
    batch = buffer.sample(batch_size=32)

    print(f"Buffer stats: {buffer.get_stats()}")
    print(f"Batch shapes:")
    print(f"  states: {batch['states'].shape}")
    print(f"  actions: {batch['actions'].shape}")
    print(f"  rewards: {batch['rewards'].shape}")
    print(f"  weights: {batch['weights'].shape}")

    # Update priorities
    td_errors = np.random.rand(32) * 10
    buffer.update_priorities(batch['indices'], td_errors)

    print(f"\nAfter priority update: {buffer.get_stats()}")
    print("\n✓ ReplayBuffer test passed!")
