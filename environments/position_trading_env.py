"""
Position-Based Trading Environment Module.

A more realistic trading environment that tracks positions with capital management:
- FLAT: No position open
- LONG: Bought, waiting to sell
- SHORT: Sold, waiting to buy back

Features:
- Position tracking (FLAT/LONG/SHORT)
- Capital management (balance, position sizing)
- Leverage support for futures trading
- Risk per trade control
- Actual P&L in both percentage and dollar amounts
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config.config import (
    FEATURES, SEQUENCE_LENGTH, FEES,
    NUM_ACTIONS, ACTION_BUY, ACTION_SELL, ACTION_HOLD, ACTION_NAMES,
    INITIAL_CAPITAL, RISK_PER_TRADE, LEVERAGE, MAX_POSITION_SIZE
)


class PositionTradingEnv(gym.Env):
    """
    Position-based trading environment with capital management.
    
    Simulates realistic futures/spot trading with:
    - Position tracking (FLAT/LONG/SHORT)
    - Capital and balance management
    - Leverage support
    - Risk-based position sizing
    - Actual P&L calculation in dollars
    
    Position States:
        FLAT (0):  No position - can open LONG or SHORT
        LONG (1):  Holding buy position - can HOLD or SELL to close
        SHORT (2): Holding sell position - can HOLD or BUY to close
    
    Actions:
        BUY (0):  Open LONG (if FLAT) or Close SHORT (if SHORT)
        SELL (1): Open SHORT (if FLAT) or Close LONG (if LONG)
        HOLD (2): Maintain current position
    
    Capital Management:
        - initial_capital: Starting balance (default: $10,000)
        - risk_per_trade: % of capital to risk per trade (default: 2%)
        - leverage: Position multiplier for futures (default: 10x)
        - Position size = (capital * risk_per_trade) * leverage
    
    Rewards:
        - Returns log returns for RL training (scale-invariant)
        - info dict contains actual dollar P&L
    
    Example:
        env = PositionTradingEnv(
            sequences, highs, lows, closes,
            initial_capital=10000,
            risk_per_trade=0.02,
            leverage=10
        )
        obs, info = env.reset()
        print(info['balance'])  # $10,000
        
        obs, reward, _, _, info = env.step(0)  # BUY
        print(info['position_value'])  # $2,000 (2% * 10x leverage)
        
        obs, reward, _, _, info = env.step(1)  # SELL (close)
        print(info['pnl_dollars'])  # Actual profit/loss
        print(info['balance'])  # Updated balance
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
        initial_capital: float = INITIAL_CAPITAL,
        risk_per_trade: float = RISK_PER_TRADE,
        leverage: float = LEVERAGE,
        max_position_size: float = MAX_POSITION_SIZE,
        fees: float = FEES,
        reward_on_hold: str = 'zero',
        render_mode: Optional[str] = None
    ):
        """
        Initialize the position-based trading environment.
        
        Args:
            sequences: Preprocessed feature sequences, shape (num_samples, seq_len, num_features)
            highs: High prices aligned with sequences
            lows: Low prices aligned with sequences
            closes: Close prices aligned with sequences
            initial_capital: Starting capital in USDT (default: 10000)
            risk_per_trade: Fraction of capital to risk per trade (default: 0.02 = 2%)
            leverage: Leverage multiplier (default: 10 for futures)
            max_position_size: Max fraction of capital for position (default: 0.5 = 50%)
            fees: Transaction fee percentage per trade (default: 0.001 = 0.1%)
            reward_on_hold: Reward type when holding ('zero', 'unrealized', 'small_negative')
            render_mode: Rendering mode ('human' or None)
        """
        super().__init__()
        
        # Store data
        self.sequences = sequences.astype(np.float32)
        self.highs = highs.astype(np.float32)
        self.lows = lows.astype(np.float32)
        self.closes = closes.astype(np.float32)
        
        # Capital management parameters
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.leverage = leverage
        self.max_position_size = max_position_size
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
        
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        
        # Initialize state variables (will be reset in reset())
        self._init_state_variables()
    
    def _init_state_variables(self) -> None:
        """Initialize all state variables."""
        # Step tracking
        self.current_step = 0
        
        # Position tracking
        self.position = self.FLAT
        self.entry_price = 0.0
        self.entry_step = 0
        self.position_size = 0.0      # Size in USDT
        self.position_quantity = 0.0  # Quantity of asset
        
        # Capital tracking
        self.balance = self.initial_capital
        self.equity = self.initial_capital  # Balance + unrealized P&L
        
        # Statistics tracking
        self.total_reward = 0.0
        self.realized_pnl = 0.0
        self.realized_pnl_dollars = 0.0
        self.num_trades = 0
        self.winning_trades = 0
        self.max_drawdown = 0.0
        self.peak_equity = self.initial_capital
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
        
        # Reset all state variables
        self._init_state_variables()
        
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
        pnl_dollars = 0.0
        trade_closed = False
        
        # Process action based on current position
        if self.position == self.FLAT:
            # No position - can open LONG or SHORT
            if action == self.BUY:
                # Open LONG position
                self._open_position(self.LONG, current_price)
                reward = 0.0
                
            elif action == self.SELL:
                # Open SHORT position
                self._open_position(self.SHORT, current_price)
                reward = 0.0
                
            else:  # HOLD
                reward = self._get_hold_reward()
                
        elif self.position == self.LONG:
            # Holding LONG - can HOLD or SELL to close
            if action == self.SELL:
                # Close LONG position
                reward, pnl_dollars = self._close_position(current_price)
                trade_closed = True
                
            elif action == self.BUY:
                # Already LONG, treat as HOLD
                reward = self._get_hold_reward(current_price)
                
            else:  # HOLD
                reward = self._get_hold_reward(current_price)
                
        elif self.position == self.SHORT:
            # Holding SHORT - can HOLD or BUY to close
            if action == self.BUY:
                # Close SHORT position
                reward, pnl_dollars = self._close_position(current_price)
                trade_closed = True
                
            elif action == self.SELL:
                # Already SHORT, treat as HOLD
                reward = self._get_hold_reward(current_price)
                
            else:  # HOLD
                reward = self._get_hold_reward(current_price)
        
        # Update equity (balance + unrealized P&L)
        self._update_equity(current_price)
        
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
            final_reward, final_pnl_dollars = self._close_position(final_price)
            self.total_reward += final_reward
        
        # Get next observation
        if terminated:
            observation = np.zeros((self.seq_length, self.num_features), dtype=np.float32)
        else:
            observation = self._get_observation()
        
        info = self._get_info()
        info['action'] = self.ACTION_NAMES[action]
        info['reward'] = reward
        info['pnl_dollars'] = pnl_dollars
        info['trade_closed'] = trade_closed
        
        return observation, reward, terminated, truncated, info
    
    def _open_position(self, position_type: int, price: float) -> None:
        """
        Open a new position (LONG or SHORT).
        
        Position size is calculated as:
        - Base size = balance * risk_per_trade
        - With leverage = base_size * leverage
        - Capped at max_position_size * balance
        """
        self.position = position_type
        self.entry_price = price
        self.entry_step = self.current_step
        
        # Calculate position size
        base_size = self.balance * self.risk_per_trade
        leveraged_size = base_size * self.leverage
        max_size = self.balance * self.max_position_size * self.leverage
        
        self.position_size = min(leveraged_size, max_size)
        self.position_quantity = self.position_size / price
    
    def _close_position(self, exit_price: float) -> Tuple[float, float]:
        """
        Close current position and calculate P&L.
        
        Returns:
            Tuple of (log_return, pnl_dollars)
        """
        if self.position == self.LONG:
            log_return = self._calculate_long_pnl(exit_price)
        else:  # SHORT
            log_return = self._calculate_short_pnl(exit_price)
        
        # Convert log return to percentage return
        pct_return = np.exp(log_return) - 1
        
        # Calculate dollar P&L
        pnl_dollars = self.position_size * pct_return
        
        # Update balance
        self.balance += pnl_dollars
        
        # Record trade
        self._record_trade(
            'LONG' if self.position == self.LONG else 'SHORT',
            log_return,
            pnl_dollars,
            exit_price
        )
        
        # Reset position
        self.position = self.FLAT
        self.entry_price = 0.0
        self.position_size = 0.0
        self.position_quantity = 0.0
        
        return log_return, pnl_dollars
    
    def _update_equity(self, current_price: float) -> None:
        """Update equity (balance + unrealized P&L) and track drawdown."""
        unrealized_pnl = 0.0
        
        if self.position == self.LONG:
            pct_return = (current_price - self.entry_price) / self.entry_price
            unrealized_pnl = self.position_size * pct_return
        elif self.position == self.SHORT:
            pct_return = (self.entry_price - current_price) / self.entry_price
            unrealized_pnl = self.position_size * pct_return
        
        self.equity = self.balance + unrealized_pnl
        
        # Track peak and drawdown
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        
        drawdown = (self.peak_equity - self.equity) / self.peak_equity
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
    
    def _get_observation(self) -> np.ndarray:
        """Get the current observation (state)."""
        return self.sequences[self.current_step].copy()
    
    def _get_info(self) -> Dict[str, Any]:
        """Get current info dictionary."""
        current_price = self.closes[min(self.current_step, len(self.closes) - 1)]
        
        # Calculate unrealized P&L
        unrealized_pnl = 0.0
        unrealized_pnl_dollars = 0.0
        
        if self.position == self.LONG:
            unrealized_pnl = self._calculate_long_pnl(current_price)
            pct_return = (current_price - self.entry_price) / self.entry_price
            unrealized_pnl_dollars = self.position_size * pct_return
        elif self.position == self.SHORT:
            unrealized_pnl = self._calculate_short_pnl(current_price)
            pct_return = (self.entry_price - current_price) / self.entry_price
            unrealized_pnl_dollars = self.position_size * pct_return
        
        info = {
            'step': self.current_step,
            'position': self.POSITION_NAMES[self.position],
            'entry_price': self.entry_price,
            'current_price': current_price,
            # Capital info
            'balance': self.balance,
            'equity': self.equity,
            'position_size': self.position_size,
            'position_quantity': self.position_quantity,
            # P&L info
            'unrealized_pnl': unrealized_pnl,
            'unrealized_pnl_dollars': unrealized_pnl_dollars,
            'realized_pnl': self.realized_pnl,
            'realized_pnl_dollars': self.realized_pnl_dollars,
            'total_reward': self.total_reward,
            # Statistics
            'num_trades': self.num_trades,
            'win_rate': self.winning_trades / max(self.num_trades, 1),
            'roi': (self.balance - self.initial_capital) / self.initial_capital,
            'max_drawdown': self.max_drawdown,
        }
        
        return info
    
    def _calculate_long_pnl(self, exit_price: float) -> float:
        """
        Calculate log return P&L for closing a LONG position.
        
        P&L = log(exit_price / entry_price) - 2 * fees
        """
        if self.entry_price <= 0:
            return 0.0
        
        gross_pnl = np.log(exit_price / self.entry_price)
        fee_cost = 2 * self.fees
        net_pnl = gross_pnl - fee_cost
        
        return float(net_pnl)
    
    def _calculate_short_pnl(self, exit_price: float) -> float:
        """
        Calculate log return P&L for closing a SHORT position.
        
        P&L = log(entry_price / exit_price) - 2 * fees
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
    
    def _record_trade(self, trade_type: str, pnl: float, pnl_dollars: float, exit_price: float) -> None:
        """Record a completed trade."""
        self.num_trades += 1
        self.realized_pnl += pnl
        self.realized_pnl_dollars += pnl_dollars
        
        if pnl > 0:
            self.winning_trades += 1
        
        self.trade_history.append({
            'type': trade_type,
            'entry_step': self.entry_step,
            'exit_step': self.current_step,
            'entry_price': self.entry_price,
            'exit_price': exit_price,
            'position_size': self.position_size,
            'pnl': pnl,
            'pnl_dollars': pnl_dollars,
            'holding_period': self.current_step - self.entry_step
        })
    
    def render(self) -> None:
        """Render the current state."""
        if self.render_mode == 'human':
            current_price = self.closes[min(self.current_step, len(self.closes) - 1)]
            print(f"Step: {self.current_step}/{self.max_step}")
            print(f"Balance: ${self.balance:,.2f} | Equity: ${self.equity:,.2f}")
            print(f"Position: {self.POSITION_NAMES[self.position]}")
            print(f"Current Price: ${current_price:,.2f}")
            if self.position != self.FLAT:
                print(f"Entry Price: ${self.entry_price:,.2f}")
                print(f"Position Size: ${self.position_size:,.2f}")
                pct_return = (current_price - self.entry_price) / self.entry_price
                if self.position == self.SHORT:
                    pct_return = -pct_return
                unrealized = self.position_size * pct_return
                print(f"Unrealized P&L: ${unrealized:,.2f} ({pct_return:.2%})")
            print(f"Realized P&L: ${self.realized_pnl_dollars:,.2f}")
            print(f"ROI: {(self.balance - self.initial_capital) / self.initial_capital:.2%}")
            print(f"Trades: {self.num_trades} (Win rate: {self.winning_trades/max(self.num_trades,1):.1%})")
            print(f"Max Drawdown: {self.max_drawdown:.2%}")
            print("-" * 50)
    
    def get_trade_statistics(self) -> Dict[str, Any]:
        """Get comprehensive trade statistics."""
        if not self.trade_history:
            return {
                'num_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'total_pnl_dollars': 0.0,
                'avg_pnl': 0.0,
                'avg_pnl_dollars': 0.0,
                'avg_holding_period': 0.0,
                'final_balance': self.balance,
                'roi': 0.0,
                'max_drawdown': self.max_drawdown,
            }
        
        pnls = [t['pnl'] for t in self.trade_history]
        pnls_dollars = [t['pnl_dollars'] for t in self.trade_history]
        holding_periods = [t['holding_period'] for t in self.trade_history]
        
        return {
            'num_trades': self.num_trades,
            'win_rate': self.winning_trades / self.num_trades,
            'total_pnl': self.realized_pnl,
            'total_pnl_dollars': self.realized_pnl_dollars,
            'avg_pnl': np.mean(pnls),
            'avg_pnl_dollars': np.mean(pnls_dollars),
            'max_pnl': np.max(pnls),
            'min_pnl': np.min(pnls),
            'max_pnl_dollars': np.max(pnls_dollars),
            'min_pnl_dollars': np.min(pnls_dollars),
            'avg_holding_period': np.mean(holding_periods),
            'long_trades': sum(1 for t in self.trade_history if t['type'] == 'LONG'),
            'short_trades': sum(1 for t in self.trade_history if t['type'] == 'SHORT'),
            'final_balance': self.balance,
            'roi': (self.balance - self.initial_capital) / self.initial_capital,
            'max_drawdown': self.max_drawdown,
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
    # Test the position-based environment with capital management
    from data.datasets.utils import prepare_training_data
    
    print("Loading data...")
    data = prepare_training_data('data/datasets/BTC_processed.csv')
    
    print("\nCreating position-based environment with capital management...")
    env = PositionTradingEnv(
        sequences=data['train']['sequences'],
        highs=data['train']['highs'],
        lows=data['train']['lows'],
        closes=data['train']['closes'],
        initial_capital=10000,
        risk_per_trade=0.02,
        leverage=10
    )
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Episode length: {env.episode_length}")
    print(f"Initial capital: ${env.initial_capital:,.2f}")
    print(f"Risk per trade: {env.risk_per_trade:.1%}")
    print(f"Leverage: {env.leverage}x")
    
    # Test trading sequence
    print("\n--- Testing Trade Sequence ---")
    obs, info = env.reset()
    print(f"Initial balance: ${info['balance']:,.2f}")
    
    # Open LONG
    obs, reward, _, _, info = env.step(0)  # BUY
    print(f"After BUY:")
    print(f"  Position: {info['position']}")
    print(f"  Position size: ${info['position_size']:,.2f}")
    print(f"  Entry price: ${info['entry_price']:,.2f}")
    
    # Hold
    for _ in range(5):
        obs, reward, _, _, info = env.step(2)  # HOLD
    print(f"After 5 HOLDs:")
    print(f"  Unrealized P&L: ${info['unrealized_pnl_dollars']:,.2f}")
    print(f"  Equity: ${info['equity']:,.2f}")
    
    # Close LONG
    obs, reward, _, _, info = env.step(1)  # SELL
    print(f"After SELL (close):")
    print(f"  Position: {info['position']}")
    print(f"  P&L: ${info['pnl_dollars']:,.2f}")
    print(f"  Balance: ${info['balance']:,.2f}")
    
    # Run longer test
    print("\n--- Running 1000-step Episode ---")
    obs, _ = env.reset()
    
    for step in range(min(1000, env.episode_length)):
        # Simple strategy
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
    print(f"  Total P&L: ${stats['total_pnl_dollars']:,.2f}")
    print(f"  Avg P&L per trade: ${stats['avg_pnl_dollars']:,.2f}")
    print(f"  Final Balance: ${stats['final_balance']:,.2f}")
    print(f"  ROI: {stats['roi']:.2%}")
    print(f"  Max Drawdown: {stats['max_drawdown']:.2%}")
    print(f"  Avg Holding Period: {stats['avg_holding_period']:.1f} steps")
