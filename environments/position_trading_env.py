"""
Position-Based Trading Environment Module.

A realistic trading environment with instant position flipping:
- FLAT: No position open
- LONG: Holding long position
- SHORT: Holding short position

Features:
- Position tracking (FLAT/LONG/SHORT)
- Instant position flipping (LONG→SHORT or SHORT→LONG in 1 step)
- Capital management (balance, position sizing)
- Leverage support for futures trading
- Risk per trade control (isolated margin)
- Stop-Loss and Take-Profit auto-close
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
    NUM_ACTIONS, ACTION_LONG, ACTION_SHORT, ACTION_FLAT, ACTION_NAMES,
    POSITION_LONG, POSITION_SHORT, POSITION_FLAT, POSITION_NAMES,
    ACTION_TO_POSITION,
    INITIAL_CAPITAL, RISK_PER_TRADE, LEVERAGE, MAX_POSITION_SIZE,
    STOP_LOSS, TAKE_PROFIT,
    DRAWDOWN_PENALTY_ENABLED, DRAWDOWN_PENALTY_THRESHOLD,
    DRAWDOWN_PENALTY_SCALE, DRAWDOWN_PENALTY_MAX
)


class PositionTradingEnv(gym.Env):
    """
    Position-based trading environment with instant flip capability.
    
    Simulates realistic futures trading with:
    - Position tracking (FLAT/LONG/SHORT)
    - Instant position flipping (no waiting)
    - Capital and balance management
    - Leverage support (isolated margin)
    - Risk-based position sizing
    - Stop-Loss and Take-Profit auto-close
    - Actual P&L calculation in dollars
    
    Position Values (intuitive for calculations):
        FLAT (0):   No position open - neutral
        LONG (+1):  Holding long position - bullish
        SHORT (-1): Holding short position - bearish
        
    The sign indicates market direction:
        - Positive (+1) = bullish/long
        - Negative (-1) = bearish/short  
        - Zero (0) = neutral/flat
        
    This allows math-friendly calculations:
        pnl = position * price_change
    
    Actions (Gymnasium Discrete(3)):
        0 = LONG:  Go/Stay LONG - Opens LONG if FLAT, keeps if LONG, flips if SHORT
        1 = SHORT: Go/Stay SHORT - Opens SHORT if FLAT, keeps if SHORT, flips if LONG
        2 = FLAT:  Go FLAT - Closes any open position
    
    Action Matrix:
        Current | LONG action | SHORT action | FLAT action
        --------|-------------|--------------|-------------
        FLAT    | Open LONG   | Open SHORT   | Do nothing
        LONG    | Keep LONG   | Flip→SHORT   | Close LONG
        SHORT   | Flip→LONG   | Keep SHORT   | Close SHORT
    
    Exit Conditions (checked each step):
        1. Agent action (FLAT or flip to opposite)
        2. Take-Profit triggered (price reaches +TP%)
        3. Stop-Loss triggered (price reaches -SL%)
    
    Capital Management:
        - initial_capital: Starting balance (default: $10,000)
        - risk_per_trade: % of capital to risk per trade (default: 2%)
        - leverage: Position multiplier for futures (default: 10x)
        - Isolated margin: max loss per trade = risk_per_trade * capital
    
    Rewards:
        - Manual close: actual log return P&L
        - TP triggered: +take_profit log return
        - SL triggered: -stop_loss log return
    
    Example:
        env = PositionTradingEnv(
            sequences, highs, lows, closes,
            initial_capital=10000,
            risk_per_trade=0.02,
            leverage=10,
            stop_loss=0.02,
            take_profit=0.04
        )
        obs, info = env.reset()
        
        obs, reward, _, _, info = env.step(0)  # LONG action -> Open LONG
        obs, reward, _, _, info = env.step(1)  # SHORT action -> Flip to SHORT (instant!)
        obs, reward, _, _, info = env.step(2)  # FLAT action -> Close SHORT
    """
    
    metadata = {'render_modes': ['human']}
    
    # Position values (intuitive: +1 long, -1 short, 0 flat)
    FLAT = POSITION_FLAT    # 0
    LONG = POSITION_LONG    # +1
    SHORT = POSITION_SHORT  # -1
    POSITION_NAMES = POSITION_NAMES  # {1: 'LONG', -1: 'SHORT', 0: 'FLAT'}
    
    # Exit reasons
    EXIT_MANUAL = 'manual'
    EXIT_FLIP = 'flip'
    EXIT_STOP_LOSS = 'stop_loss'
    EXIT_TAKE_PROFIT = 'take_profit'
    EXIT_END_EPISODE = 'end_episode'
    
    # Action constants (Gymnasium indices)
    ACT_LONG = ACTION_LONG    # 0
    ACT_SHORT = ACTION_SHORT  # 1
    ACT_FLAT = ACTION_FLAT    # 2
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
        stop_loss: float = STOP_LOSS,
        take_profit: float = TAKE_PROFIT,
        fees: float = FEES,
        reward_on_hold: str = 'zero',
        drawdown_penalty_enabled: bool = DRAWDOWN_PENALTY_ENABLED,
        drawdown_penalty_threshold: float = DRAWDOWN_PENALTY_THRESHOLD,
        drawdown_penalty_scale: float = DRAWDOWN_PENALTY_SCALE,
        drawdown_penalty_max: float = DRAWDOWN_PENALTY_MAX,
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
            stop_loss: Stop-loss threshold (default: 0.02 = 2%)
            take_profit: Take-profit threshold (default: 0.04 = 4%)
            fees: Transaction fee percentage per trade (default: 0.001 = 0.1%)
            reward_on_hold: Reward type when holding ('zero', 'unrealized', 'small_negative')
            drawdown_penalty_enabled: Whether to apply drawdown penalty (default: True)
            drawdown_penalty_threshold: Drawdown % to start penalizing (default: 0.05 = 5%)
            drawdown_penalty_scale: Penalty multiplier (default: 0.5)
            drawdown_penalty_max: Maximum penalty per step (default: 0.1)
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
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.fees = fees
        self.reward_on_hold = reward_on_hold
        self.render_mode = render_mode
        
        # Drawdown penalty parameters
        self.drawdown_penalty_enabled = drawdown_penalty_enabled
        self.drawdown_penalty_threshold = drawdown_penalty_threshold
        self.drawdown_penalty_scale = drawdown_penalty_scale
        self.drawdown_penalty_max = drawdown_penalty_max
        
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
        self.sl_price = 0.0           # Stop-loss trigger price
        self.tp_price = 0.0           # Take-profit trigger price
        
        # Capital tracking
        self.balance = self.initial_capital
        self.equity = self.initial_capital  # Balance + unrealized P&L
        
        # Statistics tracking
        self.total_reward = 0.0
        self.realized_pnl = 0.0
        self.realized_pnl_dollars = 0.0
        self.num_trades = 0
        self.winning_trades = 0
        self.sl_triggered_count = 0
        self.tp_triggered_count = 0
        self.manual_close_count = 0
        self.flip_count = 0
        self.max_drawdown = 0.0
        self.peak_equity = self.initial_capital
        self.total_drawdown_penalty = 0.0  # Total drawdown penalty accumulated
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
        
        Order of operations:
        1. Check if SL/TP triggered (using high/low prices)
        2. If not triggered, process agent's action
        3. Update equity and tracking
        
        Action Matrix:
            Current | LONG(0)     | SHORT(1)    | FLAT(2)
            --------|-------------|-------------|------------
            FLAT    | Open LONG   | Open SHORT  | Do nothing
            LONG    | Keep LONG   | Flip→SHORT  | Close LONG
            SHORT   | Flip→LONG   | Keep SHORT  | Close SHORT
        
        Args:
            action: Action to take (0=LONG, 1=SHORT, 2=FLAT)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}. Must be in [0, 1, 2]")
        
        current_price = self.closes[self.current_step]
        current_high = self.highs[self.current_step]
        current_low = self.lows[self.current_step]
        
        reward = 0.0
        pnl_dollars = 0.0
        trade_closed = False
        exit_reason = None
        flipped = False
        
        # Check SL/TP triggers FIRST (before processing action)
        if self.position != self.FLAT:
            sl_tp_result = self._check_sl_tp(current_high, current_low)
            if sl_tp_result is not None:
                reward, pnl_dollars, exit_reason = sl_tp_result
                trade_closed = True
        
        # Process agent action only if position wasn't closed by SL/TP
        if not trade_closed:
            if self.position == self.FLAT:
                # FLAT: Can open LONG or SHORT, or stay FLAT
                if action == self.ACT_LONG:
                    self._open_position(self.LONG, current_price)
                    reward = 0.0
                elif action == self.ACT_SHORT:
                    self._open_position(self.SHORT, current_price)
                    reward = 0.0
                else:  # ACT_FLAT
                    reward = 0.0  # Do nothing
                    
            elif self.position == self.LONG:
                if action == self.ACT_SHORT:
                    # Flip LONG → SHORT (close LONG, open SHORT)
                    reward, pnl_dollars = self._close_position(current_price, self.EXIT_FLIP)
                    self._open_position(self.SHORT, current_price)
                    trade_closed = True
                    exit_reason = self.EXIT_FLIP
                    flipped = True
                elif action == self.ACT_FLAT:
                    # Close LONG, go FLAT
                    reward, pnl_dollars = self._close_position(current_price, self.EXIT_MANUAL)
                    trade_closed = True
                    exit_reason = self.EXIT_MANUAL
                else:  # ACT_LONG
                    reward = self._get_hold_reward(current_price)  # Keep LONG
                    
            elif self.position == self.SHORT:
                if action == self.ACT_LONG:
                    # Flip SHORT → LONG (close SHORT, open LONG)
                    reward, pnl_dollars = self._close_position(current_price, self.EXIT_FLIP)
                    self._open_position(self.LONG, current_price)
                    trade_closed = True
                    exit_reason = self.EXIT_FLIP
                    flipped = True
                elif action == self.ACT_FLAT:
                    # Close SHORT, go FLAT
                    reward, pnl_dollars = self._close_position(current_price, self.EXIT_MANUAL)
                    trade_closed = True
                    exit_reason = self.EXIT_MANUAL
                else:  # ACT_SHORT
                    reward = self._get_hold_reward(current_price)  # Keep SHORT
        
        # Update equity (balance + unrealized P&L)
        self._update_equity(current_price)
        
        # Apply drawdown penalty (after equity update)
        drawdown_penalty = self._calculate_drawdown_penalty()
        reward -= drawdown_penalty
        
        # Update tracking
        self.total_reward += reward
        self.total_drawdown_penalty += drawdown_penalty
        self.action_history.append(action)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        terminated = self.current_step > self.max_step
        truncated = False
        
        # Force close position at end of episode
        if terminated and self.position != self.FLAT:
            final_price = self.closes[min(self.current_step, len(self.closes) - 1)]
            final_reward, final_pnl_dollars = self._close_position(final_price, self.EXIT_END_EPISODE)
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
        info['exit_reason'] = exit_reason
        info['flipped'] = flipped
        info['drawdown_penalty'] = drawdown_penalty
        
        return observation, reward, terminated, truncated, info
    
    def _check_sl_tp(self, high: float, low: float) -> Optional[Tuple[float, float, str]]:
        """
        Check if Stop-Loss or Take-Profit is triggered.
        
        Uses high/low prices to check if SL/TP was hit during the candle.
        
        Returns:
            None if not triggered, or (reward, pnl_dollars, exit_reason) tuple
        """
        if self.position == self.LONG:
            # LONG: SL if low <= sl_price, TP if high >= tp_price
            if low <= self.sl_price:
                # Stop-loss triggered - exit at SL price
                reward, pnl_dollars = self._close_position(self.sl_price, self.EXIT_STOP_LOSS)
                return reward, pnl_dollars, self.EXIT_STOP_LOSS
            elif high >= self.tp_price:
                # Take-profit triggered - exit at TP price
                reward, pnl_dollars = self._close_position(self.tp_price, self.EXIT_TAKE_PROFIT)
                return reward, pnl_dollars, self.EXIT_TAKE_PROFIT
                
        elif self.position == self.SHORT:
            # SHORT: SL if high >= sl_price, TP if low <= tp_price
            if high >= self.sl_price:
                # Stop-loss triggered - exit at SL price
                reward, pnl_dollars = self._close_position(self.sl_price, self.EXIT_STOP_LOSS)
                return reward, pnl_dollars, self.EXIT_STOP_LOSS
            elif low <= self.tp_price:
                # Take-profit triggered - exit at TP price
                reward, pnl_dollars = self._close_position(self.tp_price, self.EXIT_TAKE_PROFIT)
                return reward, pnl_dollars, self.EXIT_TAKE_PROFIT
        
        return None
    
    def _open_position(self, position_type: int, price: float) -> None:
        """
        Open a new position (LONG or SHORT) with SL/TP levels.
        
        Position size is calculated based on risk per trade:
        - With isolated margin, max loss = risk_per_trade * balance
        - Position size = (balance * risk_per_trade / stop_loss) * leverage
        """
        self.position = position_type
        self.entry_price = price
        self.entry_step = self.current_step
        
        # Calculate position size based on risk
        # Max loss = balance * risk_per_trade
        # Position size = max_loss / stop_loss * leverage (simplified)
        base_size = self.balance * self.risk_per_trade
        leveraged_size = base_size * self.leverage
        max_size = self.balance * self.max_position_size * self.leverage
        
        self.position_size = min(leveraged_size, max_size)
        self.position_quantity = self.position_size / price
        
        # Set SL/TP prices
        if position_type == self.LONG:
            self.sl_price = price * (1 - self.stop_loss)
            self.tp_price = price * (1 + self.take_profit)
        else:  # SHORT
            self.sl_price = price * (1 + self.stop_loss)
            self.tp_price = price * (1 - self.take_profit)
    
    def _close_position(self, exit_price: float, exit_reason: str = EXIT_MANUAL) -> Tuple[float, float]:
        """
        Close current position and calculate P&L.
        
        Uses unified P&L calculation with position direction:
        P&L = position * log(exit_price / entry_price) - 2 * fees
        
        Args:
            exit_price: Price at which position is closed
            exit_reason: Reason for closing (manual, stop_loss, take_profit, end_episode)
        
        Returns:
            Tuple of (log_return, pnl_dollars)
        """
        # Use unified calculation with position direction
        log_return = self._calculate_pnl(exit_price)
        
        # Convert log return to percentage return
        pct_return = np.exp(log_return) - 1
        
        # Calculate dollar P&L
        pnl_dollars = self.position_size * pct_return
        
        # Update balance
        self.balance += pnl_dollars
        
        # Update exit statistics
        if exit_reason == self.EXIT_STOP_LOSS:
            self.sl_triggered_count += 1
        elif exit_reason == self.EXIT_TAKE_PROFIT:
            self.tp_triggered_count += 1
        elif exit_reason == self.EXIT_MANUAL:
            self.manual_close_count += 1
        elif exit_reason == self.EXIT_FLIP:
            self.flip_count += 1
        
        # Record trade
        self._record_trade(
            self.POSITION_NAMES[self.position],  # 'LONG' or 'SHORT'
            log_return,
            pnl_dollars,
            exit_price,
            exit_reason
        )
        
        # Reset position
        self.position = self.FLAT
        self.entry_price = 0.0
        self.position_size = 0.0
        self.position_quantity = 0.0
        self.sl_price = 0.0
        self.tp_price = 0.0
        
        return log_return, pnl_dollars
    
    def _update_equity(self, current_price: float) -> None:
        """
        Update equity (balance + unrealized P&L) and track drawdown.
        
        Uses position direction for unified calculation:
        unrealized_pnl = position * (current_price - entry_price) / entry_price
        """
        unrealized_pnl = 0.0
        
        if self.position != self.FLAT and self.entry_price > 0:
            # Unified calculation using position direction
            pct_change = (current_price - self.entry_price) / self.entry_price
            pct_return = self.position * pct_change  # position is +1 or -1
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
        
        # Calculate unrealized P&L using position direction
        unrealized_pnl = 0.0
        unrealized_pnl_dollars = 0.0
        
        if self.position != self.FLAT and self.entry_price > 0:
            # Unified calculation using position direction
            log_return = np.log(current_price / self.entry_price)
            unrealized_pnl = self.position * log_return  # position is +1 or -1
            
            pct_change = (current_price - self.entry_price) / self.entry_price
            pct_return = self.position * pct_change
            unrealized_pnl_dollars = self.position_size * pct_return
        
        info = {
            'step': self.current_step,
            'position': self.POSITION_NAMES[self.position],
            'position_value': self.position,  # +1, -1, or 0
            'entry_price': self.entry_price,
            'current_price': current_price,
            # SL/TP info
            'sl_price': self.sl_price,
            'tp_price': self.tp_price,
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
            'current_drawdown': (self.peak_equity - self.equity) / self.peak_equity if self.peak_equity > 0 else 0.0,
            'total_drawdown_penalty': self.total_drawdown_penalty,
            # Exit statistics
            'sl_triggered': self.sl_triggered_count,
            'tp_triggered': self.tp_triggered_count,
            'manual_closes': self.manual_close_count,
            'flips': self.flip_count,
        }
        
        return info
    
    def _calculate_long_pnl(self, exit_price: float) -> float:
        """
        Calculate log return P&L for closing a LONG position.
        
        P&L = log(exit_price / entry_price) - 2 * fees
        
        With position value (+1), can also be computed as:
        P&L = position * log(exit_price / entry_price) - 2 * fees
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
             = -log(exit_price / entry_price) - 2 * fees
        
        With position value (-1), can also be computed as:
        P&L = position * log(exit_price / entry_price) - 2 * fees
        """
        if self.entry_price <= 0:
            return 0.0
        
        gross_pnl = np.log(self.entry_price / exit_price)
        fee_cost = 2 * self.fees
        net_pnl = gross_pnl - fee_cost
        
        return float(net_pnl)
    
    def _calculate_pnl(self, exit_price: float) -> float:
        """
        Calculate P&L using position direction.
        
        Unified formula using position value {+1, -1, 0}:
        P&L = position * log(exit_price / entry_price) - 2 * fees
        
        This works because:
        - LONG (+1):  +1 * log(exit/entry) = profit when price goes up
        - SHORT (-1): -1 * log(exit/entry) = profit when price goes down
        """
        if self.entry_price <= 0 or self.position == self.FLAT:
            return 0.0
        
        log_return = np.log(exit_price / self.entry_price)
        gross_pnl = self.position * log_return  # position is +1 or -1
        fee_cost = 2 * self.fees
        net_pnl = gross_pnl - fee_cost
        
        return float(net_pnl)
    
    def _calculate_drawdown_penalty(self) -> float:
        """
        Calculate drawdown penalty based on current drawdown.
        
        Penalty formula:
        - No penalty if drawdown < threshold
        - Linear penalty: scale * (drawdown - threshold) if drawdown >= threshold
        - Capped at max penalty
        
        This encourages the agent to preserve capital and avoid large drawdowns.
        
        Returns:
            Drawdown penalty (negative value to subtract from reward)
        """
        if not self.drawdown_penalty_enabled:
            return 0.0
        
        # Calculate current drawdown
        if self.peak_equity <= 0:
            return 0.0
        
        current_drawdown = (self.peak_equity - self.equity) / self.peak_equity
        
        # No penalty if below threshold
        if current_drawdown <= self.drawdown_penalty_threshold:
            return 0.0
        
        # Calculate penalty (linear scaling above threshold)
        excess_drawdown = current_drawdown - self.drawdown_penalty_threshold
        penalty = self.drawdown_penalty_scale * excess_drawdown
        
        # Cap at maximum penalty
        penalty = min(penalty, self.drawdown_penalty_max)
        
        return penalty
    
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
    
    def _record_trade(self, trade_type: str, pnl: float, pnl_dollars: float, exit_price: float, exit_reason: str = 'manual') -> None:
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
            'sl_price': self.sl_price,
            'tp_price': self.tp_price,
            'position_size': self.position_size,
            'pnl': pnl,
            'pnl_dollars': pnl_dollars,
            'holding_period': self.current_step - self.entry_step,
            'exit_reason': exit_reason
        })
    
    def render(self) -> None:
        """Render the current state."""
        if self.render_mode == 'human':
            current_price = self.closes[min(self.current_step, len(self.closes) - 1)]
            print(f"Step: {self.current_step}/{self.max_step}")
            print(f"Balance: ${self.balance:,.2f} | Equity: ${self.equity:,.2f}")
            print(f"Position: {self.POSITION_NAMES[self.position]} ({self.position:+d})")
            print(f"Current Price: ${current_price:,.2f}")
            if self.position != self.FLAT:
                print(f"Entry Price: ${self.entry_price:,.2f}")
                print(f"SL: ${self.sl_price:,.2f} | TP: ${self.tp_price:,.2f}")
                print(f"Position Size: ${self.position_size:,.2f}")
                # Unified P&L calculation using position direction
                pct_change = (current_price - self.entry_price) / self.entry_price
                pct_return = self.position * pct_change
                unrealized = self.position_size * pct_return
                print(f"Unrealized P&L: ${unrealized:,.2f} ({pct_return:.2%})")
            print(f"Realized P&L: ${self.realized_pnl_dollars:,.2f}")
            print(f"ROI: {(self.balance - self.initial_capital) / self.initial_capital:.2%}")
            print(f"Trades: {self.num_trades} (Win rate: {self.winning_trades/max(self.num_trades,1):.1%})")
            print(f"Exits - SL: {self.sl_triggered_count} | TP: {self.tp_triggered_count} | Flip: {self.flip_count} | Manual: {self.manual_close_count}")
            print(f"Max Drawdown: {self.max_drawdown:.2%} | Drawdown Penalty: {self.total_drawdown_penalty:.4f}")
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
                'total_drawdown_penalty': self.total_drawdown_penalty,
                'sl_triggered': 0,
                'tp_triggered': 0,
                'manual_closes': 0,
                'flips': 0,
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
            'total_drawdown_penalty': self.total_drawdown_penalty,
            # Exit reason breakdown
            'sl_triggered': self.sl_triggered_count,
            'tp_triggered': self.tp_triggered_count,
            'manual_closes': self.manual_close_count,
            'flips': self.flip_count,
        }
    
    def get_action_distribution(self) -> Dict[str, float]:
        """Get the distribution of actions taken."""
        if not self.action_history:
            return {'LONG': 0.0, 'SHORT': 0.0, 'FLAT': 0.0}
        
        total = len(self.action_history)
        return {
            'LONG': self.action_history.count(self.ACT_LONG) / total,
            'SHORT': self.action_history.count(self.ACT_SHORT) / total,
            'FLAT': self.action_history.count(self.ACT_FLAT) / total
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
    # Test the position-based environment with drawdown penalty
    import numpy as np
    
    print("Creating synthetic test data...")
    # Create synthetic data for testing
    num_samples = 3000
    seq_length = SEQUENCE_LENGTH
    num_features = len(FEATURES)
    
    # Random sequences (normalized 0-1)
    sequences = np.random.rand(num_samples, seq_length, num_features).astype(np.float32)
    
    # Simulate price with trend and volatility
    base_price = 50000.0
    returns = np.random.normal(0, 0.005, num_samples)  # 0.5% hourly volatility
    prices = base_price * np.exp(np.cumsum(returns))
    
    closes = prices.astype(np.float32)
    highs = (prices * (1 + np.abs(np.random.normal(0, 0.003, num_samples)))).astype(np.float32)
    lows = (prices * (1 - np.abs(np.random.normal(0, 0.003, num_samples)))).astype(np.float32)
    
    print("\nCreating environment with drawdown penalty...")
    env = PositionTradingEnv(
        sequences=sequences,
        highs=highs,
        lows=lows,
        closes=closes,
        initial_capital=10000,
        risk_per_trade=0.02,
        leverage=10,
        stop_loss=0.02,    # 2% SL
        take_profit=0.04,  # 4% TP
        drawdown_penalty_enabled=True,
        drawdown_penalty_threshold=0.05,  # 5% threshold
        drawdown_penalty_scale=0.5,
        drawdown_penalty_max=0.1
    )
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Actions (Gymnasium): LONG(0), SHORT(1), FLAT(2)")
    print(f"Position values: LONG(+1), SHORT(-1), FLAT(0)")
    print(f"Initial capital: ${env.initial_capital:,.2f}")
    print(f"Stop-Loss: {env.stop_loss:.1%} | Take-Profit: {env.take_profit:.1%}")
    
    # Test instant flip
    print("\n--- Testing Instant Flip with New Position Values ---")
    obs, info = env.reset()
    print(f"Initial: {info['position']} (value={info['position_value']:+d}), Balance: ${info['balance']:,.2f}")
    
    # LONG action -> Open LONG
    obs, reward, _, _, info = env.step(env.ACT_LONG)
    print(f"After LONG action: {info['position']} (value={info['position_value']:+d}), Entry: ${info['entry_price']:,.2f}")
    
    # SHORT action -> Flip to SHORT (instant!)
    obs, reward, _, _, info = env.step(env.ACT_SHORT)
    print(f"After SHORT action: {info['position']} (value={info['position_value']:+d}), Flipped: {info['flipped']}, P&L: ${info['pnl_dollars']:,.2f}")
    print(f"  New Entry: ${info['entry_price']:,.2f}")
    
    # LONG action -> Flip back to LONG
    obs, reward, _, _, info = env.step(env.ACT_LONG)
    print(f"After LONG action: {info['position']} (value={info['position_value']:+d}), Flipped: {info['flipped']}, P&L: ${info['pnl_dollars']:,.2f}")
    
    # FLAT action -> Close position
    obs, reward, _, _, info = env.step(env.ACT_FLAT)
    print(f"After FLAT action: {info['position']} (value={info['position_value']:+d}), Closed: {info['trade_closed']}")
    
    # Test action matrix
    print("\n--- Testing Action Matrix ---")
    obs, _ = env.reset()
    
    print("From FLAT (0):")
    obs, _, _, _, info = env.step(env.ACT_LONG)
    print(f"  LONG action -> {info['position']} ({info['position_value']:+d})")
    env.reset()
    obs, _, _, _, info = env.step(env.ACT_SHORT)
    print(f"  SHORT action -> {info['position']} ({info['position_value']:+d})")
    env.reset()
    obs, _, _, _, info = env.step(env.ACT_FLAT)
    print(f"  FLAT action -> {info['position']} ({info['position_value']:+d})")
    
    # Run episode with random actions
    print("\n--- Running 2000-step Episode ---")
    obs, _ = env.reset()
    
    for step in range(2000):
        # Random action
        action = np.random.choice([env.ACT_LONG, env.ACT_SHORT, env.ACT_FLAT])
        obs, reward, terminated, _, info = env.step(action)
        
        if terminated:
            break
    
    stats = env.get_trade_statistics()
    print(f"\nTrade Statistics:")
    print(f"  Trades: {stats['num_trades']}")
    print(f"  Win Rate: {stats['win_rate']:.1%}")
    print(f"  Total P&L: ${stats['total_pnl_dollars']:,.2f}")
    print(f"  Final Balance: ${stats['final_balance']:,.2f}")
    print(f"  ROI: {stats['roi']:.2%}")
    print(f"  Max Drawdown: {stats['max_drawdown']:.2%}")
    print(f"  Total Drawdown Penalty: {stats['total_drawdown_penalty']:.4f}")
    print(f"\nExit Breakdown:")
    print(f"  Stop-Loss: {stats['sl_triggered']}")
    print(f"  Take-Profit: {stats['tp_triggered']}")
    print(f"  Flips: {stats['flips']}")
    print(f"  Manual (FLAT): {stats['manual_closes']}")
    print(f"\n  Avg Holding Period: {stats['avg_holding_period']:.1f} steps")
    
    # Show action distribution
    dist = env.get_action_distribution()
    print(f"\nAction Distribution:")
    print(f"  LONG: {dist['LONG']:.1%}")
    print(f"  SHORT: {dist['SHORT']:.1%}")
    print(f"  FLAT: {dist['FLAT']:.1%}")
