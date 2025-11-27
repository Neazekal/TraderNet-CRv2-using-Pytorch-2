"""
Position-Based Trading Environment Module.

A more realistic trading environment that tracks positions:
- FLAT: No position open
- LONG: Bought, waiting to sell
- SHORT: Sold, waiting to buy back

This simulates real trading where you must close a position to realize profit/loss.
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


class PositionTradingEnv(gym.Env):
    """
    Position-based trading environment for realistic trading simulation.
    
    Unlike TradingEnv which treats each step independently, this environment
    tracks open positions and calculates actual P&L when positions are closed.
    
    Position States:
        FLAT (0):  No position - can open LONG or SHORT
        LONG (1):  Holding buy position - can HOLD or SELL to close
        SHORT (2): Holding sell position - can HOLD or BUY to close
    
    Actions:
        BUY (0):  Open LONG (if FLAT) or Close SHORT (if SHORT)
        SELL (1): Open SHORT (if FLAT) or Close LONG (if LONG)
        HOLD (2): Maintain current position
    
    Rewards:
        - Opening position: 0 (or small negative for fees)
        - Holding position: unrealized P&L (optional) or 0
        - Closing position: realized P&L with fees
    
    Example:
        env = PositionTradingEnv(sequences, highs, lows, closes)
        obs, info = env.reset()
        
        # Open long position
        obs, reward, _, _, info = env.step(0)  # BUY
        print(info['position'])  # 'LONG'
        
        # Hold position
        obs, reward, _, _, info = env.step(2)  # HOLD
        print(info['unrealized_pnl'])  # Current unrealized P&L
        
        # Close position
        obs, reward, _, _, info = env.step(1)  # SELL
        print(reward)  # Realized profit/loss
    """
    
    metadata = {'render_modes': ['human']}
    
    # Position states
    FLAT = 0
    LONG = 1
    SHORT = 2
    POSITION_NAMES = {0: 'FLAT', 1: 'LONG', 2: 'SHORT'}
    
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
        fees: float = FEES,
        reward_on_hold: str = 'zero',  # 'zero', 'unrealized', 'small_negative'
        render_mode: Optional[str] = None
    ):
        """
        Initialize the position-based trading environment.
        
        Args:
            sequences: Preprocessed feature sequences, shape (num_samples, seq_len, num_features)
            highs: High prices aligned with sequences
            lows: Low prices aligned with sequences
            closes: Close prices aligned with sequences
            fees: Transaction fee percentage (applied on open and close)
            reward_on_hold: How to reward holding ('zero', 'unrealized', 'small_negative')
            render_mode: Rendering mode ('human' or None)
        """
        super().__init__()
        
        # Store data
        self.sequences = sequences.astype(np.float32)
        self.highs = highs.astype(np.float32)
        self.lows = lows.astype(np.float32)
        self.closes = closes.astype(np.float32)
        
        # Environment parameters
        self.fees = fees
        self.reward_on_hold = reward_on_hold
        self.render_mode = render_mode
        
        # Data dimensions
        self.num_samples, self.seq_length, self.num_features = sequences.shape
        self.max_step = self.num_samples - 1
        
        # Define spaces
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.seq_length, self.num_features),
            dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(NUM_ACTIONS)  # BUY, SELL, HOLD
        
        # Position tracking (initialized in reset)
        self.current_step = 0
        self.position = self.FLAT
        self.entry_price = 0.0
        self.entry_step = 0
        
        # Statistics tracking
        self.total_reward = 0.0
        self.realized_pnl = 0.0
        self.num_trades = 0
        self.winning_trades = 0
        self.action_history = []
        self.trade_history = []
        
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
        
        # Reset position
        self.current_step = 0
        self.position = self.FLAT
        self.entry_price = 0.0
        self.entry_step = 0
        
        # Reset statistics
        self.total_reward = 0.0
        self.realized_pnl = 0.0
        self.num_trades = 0
        self.winning_trades = 0
        self.action_history = []
        self.trade_history = []
        
        # Get initial observation
        observation = self._get_observation()
        
        info = self._get_info()
        
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
        
        current_price = self.closes[self.current_step]
        reward = 0.0
        trade_closed = False
        
        # Process action based on current position
        if self.position == self.FLAT:
            # No position - can open LONG or SHORT
            if action == self.BUY:
                # Open LONG position
                self.position = self.LONG
                self.entry_price = current_price
                self.entry_step = self.current_step
                reward = 0.0  # No reward on open (fees deducted on close)
                
            elif action == self.SELL:
                # Open SHORT position
                self.position = self.SHORT
                self.entry_price = current_price
                self.entry_step = self.current_step
                reward = 0.0
                
            else:  # HOLD
                reward = self._get_hold_reward()
                
        elif self.position == self.LONG:
            # Holding LONG - can HOLD or SELL to close
            if action == self.SELL:
                # Close LONG position
                reward = self._calculate_long_pnl(current_price)
                trade_closed = True
                self._record_trade('LONG', reward)
                self.position = self.FLAT
                self.entry_price = 0.0
                
            elif action == self.BUY:
                # Already LONG, treat as HOLD
                reward = self._get_hold_reward(current_price)
                
            else:  # HOLD
                reward = self._get_hold_reward(current_price)
                
        elif self.position == self.SHORT:
            # Holding SHORT - can HOLD or BUY to close
            if action == self.BUY:
                # Close SHORT position
                reward = self._calculate_short_pnl(current_price)
                trade_closed = True
                self._record_trade('SHORT', reward)
                self.position = self.FLAT
                self.entry_price = 0.0
                
            elif action == self.SELL:
                # Already SHORT, treat as HOLD
                reward = self._get_hold_reward(current_price)
                
            else:  # HOLD
                reward = self._get_hold_reward(current_price)
        
        # Update tracking
        self.total_reward += reward
        self.action_history.append(action)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        terminated = self.current_step > self.max_step
        truncated = False
        
        # Force close position at end of episode
        if terminated and self.position != self.FLAT:
            final_price = self.closes[min(self.current_step, len(self.closes) - 1)]
            if self.position == self.LONG:
                final_reward = self._calculate_long_pnl(final_price)
                self._record_trade('LONG', final_reward)
            else:  # SHORT
                final_reward = self._calculate_short_pnl(final_price)
                self._record_trade('SHORT', final_reward)
            self.total_reward += final_reward
            self.position = self.FLAT
        
        # Get next observation
        if terminated:
            observation = np.zeros((self.seq_length, self.num_features), dtype=np.float32)
        else:
            observation = self._get_observation()
        
        info = self._get_info()
        info['action'] = self.ACTION_NAMES[action]
        info['reward'] = reward
        info['trade_closed'] = trade_closed
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get the current observation (state)."""
        return self.sequences[self.current_step].copy()
    
    def _get_info(self) -> Dict[str, Any]:
        """Get current info dictionary."""
        current_price = self.closes[min(self.current_step, len(self.closes) - 1)]
        
        info = {
            'step': self.current_step,
            'position': self.POSITION_NAMES[self.position],
            'entry_price': self.entry_price,
            'current_price': current_price,
            'total_reward': self.total_reward,
            'realized_pnl': self.realized_pnl,
            'num_trades': self.num_trades,
            'win_rate': self.winning_trades / max(self.num_trades, 1),
        }
        
        # Add unrealized P&L if in position
        if self.position == self.LONG:
            info['unrealized_pnl'] = self._calculate_long_pnl(current_price)
        elif self.position == self.SHORT:
            info['unrealized_pnl'] = self._calculate_short_pnl(current_price)
        else:
            info['unrealized_pnl'] = 0.0
        
        return info
    
    def _calculate_long_pnl(self, exit_price: float) -> float:
        """
        Calculate P&L for closing a LONG position.
        
        P&L = log(exit_price / entry_price) - 2 * fees
        (fees on both entry and exit)
        """
        if self.entry_price <= 0:
            return 0.0
        
        gross_pnl = np.log(exit_price / self.entry_price)
        fee_cost = 2 * self.fees  # Entry + exit fees
        net_pnl = gross_pnl - fee_cost
        
        return float(net_pnl)
    
    def _calculate_short_pnl(self, exit_price: float) -> float:
        """
        Calculate P&L for closing a SHORT position.
        
        P&L = log(entry_price / exit_price) - 2 * fees
        (profit when price goes down)
        """
        if self.entry_price <= 0:
            return 0.0
        
        gross_pnl = np.log(self.entry_price / exit_price)
        fee_cost = 2 * self.fees
        net_pnl = gross_pnl - fee_cost
        
        return float(net_pnl)
    
    def _get_hold_reward(self, current_price: float = None) -> float:
        """Get reward for holding position."""
        if self.reward_on_hold == 'zero':
            return 0.0
        elif self.reward_on_hold == 'small_negative':
            return -0.0001  # Small penalty to encourage action
        elif self.reward_on_hold == 'unrealized':
            # Return unrealized P&L change (not implemented yet)
            return 0.0
        return 0.0
    
    def _record_trade(self, trade_type: str, pnl: float) -> None:
        """Record a completed trade."""
        self.num_trades += 1
        self.realized_pnl += pnl
        
        if pnl > 0:
            self.winning_trades += 1
        
        self.trade_history.append({
            'type': trade_type,
            'entry_step': self.entry_step,
            'exit_step': self.current_step,
            'entry_price': self.entry_price,
            'exit_price': self.closes[self.current_step],
            'pnl': pnl,
            'holding_period': self.current_step - self.entry_step
        })
    
    def render(self) -> None:
        """Render the current state."""
        if self.render_mode == 'human':
            current_price = self.closes[min(self.current_step, len(self.closes) - 1)]
            print(f"Step: {self.current_step}/{self.max_step}")
            print(f"Position: {self.POSITION_NAMES[self.position]}")
            print(f"Current Price: {current_price:.2f}")
            if self.position != self.FLAT:
                print(f"Entry Price: {self.entry_price:.2f}")
                if self.position == self.LONG:
                    print(f"Unrealized P&L: {self._calculate_long_pnl(current_price):.4f}")
                else:
                    print(f"Unrealized P&L: {self._calculate_short_pnl(current_price):.4f}")
            print(f"Realized P&L: {self.realized_pnl:.4f}")
            print(f"Trades: {self.num_trades} (Win rate: {self.winning_trades/max(self.num_trades,1):.1%})")
            print("-" * 40)
    
    def get_trade_statistics(self) -> Dict[str, Any]:
        """Get comprehensive trade statistics."""
        if not self.trade_history:
            return {
                'num_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_pnl': 0.0,
                'avg_holding_period': 0.0,
            }
        
        pnls = [t['pnl'] for t in self.trade_history]
        holding_periods = [t['holding_period'] for t in self.trade_history]
        
        return {
            'num_trades': self.num_trades,
            'win_rate': self.winning_trades / self.num_trades,
            'total_pnl': self.realized_pnl,
            'avg_pnl': np.mean(pnls),
            'max_pnl': np.max(pnls),
            'min_pnl': np.min(pnls),
            'avg_holding_period': np.mean(holding_periods),
            'long_trades': sum(1 for t in self.trade_history if t['type'] == 'LONG'),
            'short_trades': sum(1 for t in self.trade_history if t['type'] == 'SHORT'),
        }
    
    def get_action_distribution(self) -> Dict[str, float]:
        """Get the distribution of actions taken."""
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


def create_position_trading_env(
    processed_csv_path: str,
    **kwargs
) -> PositionTradingEnv:
    """
    Create a PositionTradingEnv from a processed dataset CSV.
    
    Args:
        processed_csv_path: Path to processed dataset CSV
        **kwargs: Additional arguments passed to PositionTradingEnv
        
    Returns:
        Configured PositionTradingEnv instance
    """
    from data.datasets.utils import load_processed_dataset, create_sequences, get_price_data
    
    # Load data
    df = load_processed_dataset(processed_csv_path)
    
    # Create sequences and get price data
    sequences = create_sequences(df, SEQUENCE_LENGTH, FEATURES)
    highs, lows, closes = get_price_data(df, SEQUENCE_LENGTH)
    
    return PositionTradingEnv(
        sequences=sequences,
        highs=highs,
        lows=lows,
        closes=closes,
        **kwargs
    )


if __name__ == '__main__':
    # Test the position-based environment
    from data.datasets.utils import prepare_training_data
    
    print("Loading data...")
    data = prepare_training_data('data/datasets/BTC_processed.csv')
    
    print("\nCreating position-based environment...")
    env = PositionTradingEnv(
        sequences=data['train']['sequences'],
        highs=data['train']['highs'],
        lows=data['train']['lows'],
        closes=data['train']['closes']
    )
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Episode length: {env.episode_length}")
    
    # Test trading sequence
    print("\n--- Testing Trade Sequence ---")
    obs, info = env.reset()
    print(f"Initial position: {info['position']}")
    
    # Open LONG
    obs, reward, _, _, info = env.step(0)  # BUY
    print(f"After BUY: position={info['position']}, reward={reward:.4f}")
    
    # Hold
    for _ in range(5):
        obs, reward, _, _, info = env.step(2)  # HOLD
    print(f"After 5 HOLDs: position={info['position']}, unrealized_pnl={info['unrealized_pnl']:.4f}")
    
    # Close LONG
    obs, reward, _, _, info = env.step(1)  # SELL
    print(f"After SELL: position={info['position']}, reward={reward:.4f}")
    
    # Run full episode with simple strategy
    print("\n--- Running Full Episode ---")
    obs, _ = env.reset()
    
    for step in range(min(1000, env.episode_length)):
        # Simple strategy: alternate between trades
        if env.position == env.FLAT:
            action = env.BUY if step % 100 < 50 else env.SELL
        elif env.position == env.LONG:
            action = env.SELL if np.random.random() < 0.05 else env.HOLD
        else:  # SHORT
            action = env.BUY if np.random.random() < 0.05 else env.HOLD
        
        obs, reward, terminated, _, info = env.step(action)
        
        if terminated:
            break
    
    stats = env.get_trade_statistics()
    print(f"\nTrade Statistics:")
    print(f"  Trades: {stats['num_trades']}")
    print(f"  Win Rate: {stats['win_rate']:.1%}")
    print(f"  Total P&L: {stats['total_pnl']:.4f}")
    print(f"  Avg P&L per trade: {stats['avg_pnl']:.4f}")
    print(f"  Avg Holding Period: {stats['avg_holding_period']:.1f} steps")
