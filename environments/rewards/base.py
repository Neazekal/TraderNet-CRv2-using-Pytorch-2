"""
Base Reward Function Module.

Provides abstract base class for reward functions used in the trading environment.
All reward functions compute rewards for BUY, SELL, and HOLD actions based on
future price movements within a lookahead horizon.
"""

import numpy as np
from abc import ABC, abstractmethod

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import ACTION_BUY, ACTION_SELL, ACTION_HOLD, HORIZON, FEES


class BaseRewardFunction(ABC):
    """
    Abstract base class for reward functions.
    
    Reward functions compute the reward for each action (BUY, SELL, HOLD)
    based on future price movements within a horizon window.
    
    The reward is typically based on log returns to ensure:
    - Rewards can be summed for cumulative returns
    - Scale is consistent across different price levels
    - Stationarity for learning
    
    Args:
        highs: Array of high prices
        lows: Array of low prices
        closes: Array of close prices
        horizon: Number of future timesteps to look ahead (K)
        fees: Transaction fee percentage (e.g., 0.01 for 1%)
    """
    
    # Action indices (from config)
    BUY = ACTION_BUY
    SELL = ACTION_SELL
    HOLD = ACTION_HOLD
    
    def __init__(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        horizon: int = HORIZON,
        fees: float = FEES
    ):
        """
        Initialize reward function with price data.
        
        Args:
            highs: High prices array
            lows: Low prices array
            closes: Close prices array
            horizon: Lookahead horizon for reward calculation
            fees: Transaction fee percentage
        """
        self.highs = highs.astype(np.float32)
        self.lows = lows.astype(np.float32)
        self.closes = closes.astype(np.float32)
        self.horizon = horizon
        self.fees = fees
        
        # Precompute fee adjustment (applied to BUY and SELL)
        # fee_adjustment = log((1 - fees) / (1 + fees))
        # For 1% fees: log(0.99/1.01) ≈ -0.0199
        self.fee_adjustment = np.log((1 - fees) / (1 + fees))
        
        # Precompute all rewards for efficiency
        self._rewards = self._build_rewards()
    
    @property
    def num_steps(self) -> int:
        """Number of valid timesteps for trading."""
        return len(self._rewards)
    
    def _build_rewards(self) -> np.ndarray:
        """
        Precompute rewards for all timesteps and actions.
        
        Returns:
            Array of shape (num_steps, 3) with rewards for [BUY, SELL, HOLD]
        """
        num_steps = len(self.closes) - self.horizon
        
        if num_steps <= 0:
            raise ValueError(
                f"Not enough data for horizon={self.horizon}. "
                f"Data length: {len(self.closes)}"
            )
        
        rewards = np.zeros((num_steps, 3), dtype=np.float32)
        
        for i in range(num_steps):
            rewards[i] = self._compute_step_rewards(i)
        
        return rewards
    
    @abstractmethod
    def _compute_step_rewards(self, step: int) -> np.ndarray:
        """
        Compute rewards for a single timestep.
        
        Args:
            step: Current timestep index
            
        Returns:
            Array of [buy_reward, sell_reward, hold_reward]
        """
        pass
    
    def get_reward(self, step: int, action: int) -> float:
        """
        Get reward for a specific step and action.
        
        Args:
            step: Current timestep index
            action: Action taken (0=BUY, 1=SELL, 2=HOLD)
            
        Returns:
            Reward value
        """
        if step >= self.num_steps:
            raise IndexError(f"Step {step} out of range. Max: {self.num_steps - 1}")
        
        return self._rewards[step, action]
    
    def get_all_rewards(self, step: int) -> np.ndarray:
        """
        Get rewards for all actions at a specific step.
        
        Args:
            step: Current timestep index
            
        Returns:
            Array of [buy_reward, sell_reward, hold_reward]
        """
        if step >= self.num_steps:
            raise IndexError(f"Step {step} out of range. Max: {self.num_steps - 1}")
        
        return self._rewards[step].copy()
    
    def __call__(self, step: int, action: int) -> float:
        """Shorthand for get_reward()."""
        return self.get_reward(step, action)
    
    def __len__(self) -> int:
        """Number of valid timesteps."""
        return self.num_steps
