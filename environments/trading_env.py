"""
Trading Environment Module.

Implements a Gymnasium-compatible trading environment for cryptocurrency trading
using reinforcement learning. The environment simulates trading with:
- State: Sliding window of technical features (12 timesteps x 21 features)
- Actions: BUY (0), SELL (1), HOLD (2)
- Rewards: Based on future price movements (MarketLimitOrder)
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config.config import (
    FEATURES, SEQUENCE_LENGTH, HORIZON, FEES,
    NUM_ACTIONS, ACTION_BUY, ACTION_SELL, ACTION_HOLD, ACTION_NAMES
)
from environments.rewards.base import BaseRewardFunction
from environments.rewards.market_limit import MarketLimitOrderReward


class TradingEnv(gym.Env):
    """
    Gymnasium environment for cryptocurrency trading.
    
    The agent observes a window of past market data (technical indicators)
    and decides to BUY, SELL, or HOLD. Rewards are based on the potential
    profit/loss within a future horizon window.
    
    Observation Space:
        Box of shape (sequence_length, num_features) = (12, 19)
        Values are Min-Max scaled to [0, 1]
    
    Action Space:
        Discrete(3): BUY=0, SELL=1, HOLD=2
    
    Episode:
        Runs through the entire dataset sequentially.
        Done when reaching the end (no more valid steps for reward calculation).
    
    Example:
        env = TradingEnv(sequences, highs, lows, closes)
        obs, info = env.reset()
        
        for _ in range(1000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break
    """
    
    metadata = {'render_modes': ['human']}
    
    # Action constants (from config)
    BUY = ACTION_BUY
    SELL = ACTION_SELL
    HOLD = ACTION_HOLD
    ACTION_NAMES = ACTION_NAMES
    
    def __init__(
        self,
        sequences: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        reward_function: Optional[BaseRewardFunction] = None,
        horizon: int = HORIZON,
        fees: float = FEES,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the trading environment.
        
        Args:
            sequences: Preprocessed feature sequences, shape (num_samples, seq_len, num_features)
            highs: High prices aligned with sequences
            lows: Low prices aligned with sequences
            closes: Close prices aligned with sequences
            reward_function: Custom reward function (default: MarketLimitOrderReward)
            horizon: Lookahead horizon for reward calculation
            fees: Transaction fee percentage
            render_mode: Rendering mode ('human' or None)
        """
        super().__init__()
        
        # Store data
        self.sequences = sequences.astype(np.float32)
        self.highs = highs.astype(np.float32)
        self.lows = lows.astype(np.float32)
        self.closes = closes.astype(np.float32)
        
        # Environment parameters
        self.horizon = horizon
        self.fees = fees
        self.render_mode = render_mode
        
        # Data dimensions
        self.num_samples, self.seq_length, self.num_features = sequences.shape
        
        # Create reward function if not provided
        if reward_function is not None:
            self.reward_function = reward_function
        else:
            self.reward_function = MarketLimitOrderReward(
                highs=self.highs,
                lows=self.lows,
                closes=self.closes,
                horizon=self.horizon,
                fees=self.fees
            )
        
        # Maximum valid step (limited by reward function horizon)
        self.max_step = len(self.reward_function) - 1
        
        # Define spaces
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.seq_length, self.num_features),
            dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(NUM_ACTIONS)  # BUY, SELL, HOLD
        
        # State tracking
        self.current_step = 0
        self.total_reward = 0.0
        self.action_history = []
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (not used currently)
            
        Returns:
            Tuple of (observation, info_dict)
        """
        super().reset(seed=seed)
        
        # Reset state
        self.current_step = 0
        self.total_reward = 0.0
        self.action_history = []
        
        # Get initial observation
        observation = self._get_observation()
        
        info = {
            'step': self.current_step,
            'close_price': self.closes[self.current_step],
            'total_reward': self.total_reward
        }
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take (0=BUY, 1=SELL, 2=HOLD)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}. Must be in [0, 1, 2]")
        
        # Get reward for current step and action
        reward = self.reward_function.get_reward(self.current_step, action)
        
        # Update tracking
        self.total_reward += reward
        self.action_history.append(action)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        terminated = self.current_step > self.max_step
        truncated = False
        
        # Get next observation (or zeros if terminated)
        if terminated:
            observation = np.zeros((self.seq_length, self.num_features), dtype=np.float32)
        else:
            observation = self._get_observation()
        
        # Build info dict
        info = {
            'step': self.current_step,
            'action': self.ACTION_NAMES[action],
            'reward': reward,
            'total_reward': self.total_reward,
            'close_price': self.closes[min(self.current_step, len(self.closes) - 1)]
        }
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Get the current observation (state).
        
        Returns:
            Feature sequence of shape (seq_length, num_features)
        """
        return self.sequences[self.current_step].copy()
    
    def render(self) -> None:
        """Render the current state (optional)."""
        if self.render_mode == 'human':
            print(f"Step: {self.current_step}/{self.max_step}")
            print(f"Close Price: {self.closes[self.current_step]:.2f}")
            print(f"Total Reward: {self.total_reward:.4f}")
            if self.action_history:
                print(f"Last Action: {self.ACTION_NAMES[self.action_history[-1]]}")
            print("-" * 40)
    
    def get_action_distribution(self) -> Dict[str, float]:
        """
        Get the distribution of actions taken in current episode.
        
        Returns:
            Dictionary with action percentages
        """
        if not self.action_history:
            return {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 0.0}
        
        total = len(self.action_history)
        return {
            'BUY': self.action_history.count(self.BUY) / total,
            'SELL': self.action_history.count(self.SELL) / total,
            'HOLD': self.action_history.count(self.HOLD) / total
        }
    
    @property
    def episode_length(self) -> int:
        """Maximum episode length."""
        return self.max_step + 1


def create_trading_env(
    processed_csv_path: str,
    **kwargs
) -> TradingEnv:
    """
    Create a TradingEnv from a processed dataset CSV.
    
    Args:
        processed_csv_path: Path to processed dataset CSV
        **kwargs: Additional arguments passed to TradingEnv
        
    Returns:
        Configured TradingEnv instance
    """
    from data.datasets.utils import load_processed_dataset, create_sequences, get_price_data
    
    # Load data
    df = load_processed_dataset(processed_csv_path)
    
    # Create sequences and get price data
    sequences = create_sequences(df, SEQUENCE_LENGTH, FEATURES)
    highs, lows, closes = get_price_data(df, SEQUENCE_LENGTH)
    
    # Create reward function
    reward_fn = MarketLimitOrderReward(
        highs=highs,
        lows=lows,
        closes=closes,
        horizon=kwargs.get('horizon', HORIZON),
        fees=kwargs.get('fees', FEES)
    )
    
    return TradingEnv(
        sequences=sequences,
        highs=highs,
        lows=lows,
        closes=closes,
        reward_function=reward_fn,
        **kwargs
    )


if __name__ == '__main__':
    # Test the environment
    from data.datasets.utils import prepare_training_data
    
    print("Loading data...")
    data = prepare_training_data('data/datasets/BTC_processed.csv')
    
    print("\nCreating environment...")
    env = TradingEnv(
        sequences=data['train']['sequences'],
        highs=data['train']['highs'],
        lows=data['train']['lows'],
        closes=data['train']['closes']
    )
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Episode length: {env.episode_length}")
    
    # Run a few steps
    print("\nRunning test episode...")
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    
    total_reward = 0
    for i in range(100):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated:
            break
    
    print(f"\nAfter 100 steps:")
    print(f"  Total reward: {total_reward:.4f}")
    print(f"  Action distribution: {env.get_action_distribution()}")
    
    # Test full episode
    print("\nRunning full episode with random actions...")
    obs, _ = env.reset()
    episode_reward = 0
    steps = 0
    
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        steps += 1
        
        if terminated:
            break
    
    print(f"Episode finished after {steps} steps")
    print(f"Total episode reward: {episode_reward:.4f}")
    print(f"Action distribution: {env.get_action_distribution()}")
