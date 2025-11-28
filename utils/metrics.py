"""
Trading metrics calculator for performance evaluation.

Metrics computed:
- Cumulative Return: Total log return
- Sharpe Ratio: Risk-adjusted return
- Sortino Ratio: Downside risk-adjusted return
- Maximum Drawdown: Peak-to-trough decline
- Win Rate: Percentage of profitable trades
- Average Trade Duration: Mean holding time
- Profit Factor: Gross profit / gross loss
- Calmar Ratio: Return / max drawdown
"""

import numpy as np
from typing import Dict, List, Optional

from config.config import METRICS_PARAMS


class TradingMetrics:
    """Calculate and track trading performance metrics."""

    def __init__(self, risk_free_rate: float = None):
        """
        Initialize metrics tracker.

        Args:
            risk_free_rate: Annual risk-free rate for Sharpe/Sortino calculation
                           (default: from METRICS_PARAMS)
        """
        self.risk_free_rate = risk_free_rate if risk_free_rate is not None else METRICS_PARAMS['risk_free_rate']
        self.periods_per_year = METRICS_PARAMS['periods_per_year']
        self.reset()

    def reset(self):
        """Reset all metrics to initial state."""
        self.rewards = []  # Episode returns
        self.pnls = []  # Trade PnL values
        self.trade_wins = []  # Win/loss flag for each trade
        self.trade_durations = []  # Duration of each trade in hours
        self.equity_curve = [10000.0]  # Starting with $10K
        self.peak_equity = 10000.0
        self.drawdowns = []  # Drawdown at each step
        self.all_returns = []  # Return at each step for Sharpe/Sortino

    def update(self, reward: float, pnl: float, trade_info: Optional[Dict] = None):
        """
        Update metrics with step information.

        Args:
            reward: Step reward from environment
            pnl: Trade PnL in dollars
            trade_info: Dictionary with trade metadata (win, duration, etc.)
        """
        self.all_returns.append(reward)

        if trade_info is None:
            trade_info = {}

        # Track trade if trade was closed
        if trade_info.get('trade_closed', False):
            self.pnls.append(pnl)
            self.trade_wins.append(pnl > 0)
            if 'duration' in trade_info:
                self.trade_durations.append(trade_info['duration'])

        # Update equity curve
        current_equity = self.equity_curve[-1] + pnl
        self.equity_curve.append(current_equity)

        # Track peak for drawdown calculation
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        # Calculate drawdown
        drawdown = (self.peak_equity - current_equity) / self.peak_equity if self.peak_equity > 0 else 0
        self.drawdowns.append(drawdown)

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics and return as dictionary.

        Returns:
            Dictionary with all calculated metrics
        """
        metrics = {}

        # Cumulative return
        if len(self.equity_curve) > 1:
            total_return = (self.equity_curve[-1] - self.equity_curve[0]) / self.equity_curve[0]
            metrics['cumulative_return'] = total_return
        else:
            metrics['cumulative_return'] = 0.0

        # Sharpe Ratio
        if len(self.all_returns) > 1:
            returns_array = np.array(self.all_returns)
            daily_rf = self.risk_free_rate / self.periods_per_year
            excess_returns = returns_array - daily_rf

            if np.std(excess_returns) > 0:
                sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(self.periods_per_year)
                metrics['sharpe_ratio'] = float(sharpe)
            else:
                metrics['sharpe_ratio'] = 0.0
        else:
            metrics['sharpe_ratio'] = 0.0

        # Sortino Ratio (only downside volatility)
        if len(self.all_returns) > 1:
            returns_array = np.array(self.all_returns)
            daily_rf = self.risk_free_rate / self.periods_per_year
            excess_returns = returns_array - daily_rf
            downside_returns = np.minimum(excess_returns, 0)

            downside_std = np.sqrt(np.mean(downside_returns ** 2))
            if downside_std > 0:
                sortino = np.mean(excess_returns) / downside_std * np.sqrt(self.periods_per_year)
                metrics['sortino_ratio'] = float(sortino)
            else:
                metrics['sortino_ratio'] = 0.0
        else:
            metrics['sortino_ratio'] = 0.0

        # Maximum Drawdown
        if self.drawdowns:
            metrics['max_drawdown'] = float(np.max(self.drawdowns))
        else:
            metrics['max_drawdown'] = 0.0

        # Win Rate
        if len(self.trade_wins) > 0:
            win_rate = np.mean(self.trade_wins)
            metrics['win_rate'] = float(win_rate)
        else:
            metrics['win_rate'] = 0.0

        # Average Trade Duration
        if len(self.trade_durations) > 0:
            metrics['avg_trade_duration'] = float(np.mean(self.trade_durations))
        else:
            metrics['avg_trade_duration'] = 0.0

        # Profit Factor (gross profit / gross loss)
        if len(self.pnls) > 0:
            gross_profit = sum([p for p in self.pnls if p > 0])
            gross_loss = abs(sum([p for p in self.pnls if p < 0]))

            if gross_loss > 0:
                metrics['profit_factor'] = float(gross_profit / gross_loss)
            else:
                metrics['profit_factor'] = float(gross_profit) if gross_profit > 0 else 0.0
        else:
            metrics['profit_factor'] = 0.0

        # Calmar Ratio (return / max drawdown)
        if metrics['max_drawdown'] > 0:
            calmar = metrics['cumulative_return'] / metrics['max_drawdown']
            metrics['calmar_ratio'] = float(calmar)
        else:
            metrics['calmar_ratio'] = 0.0 if metrics['cumulative_return'] == 0 else float('inf')

        # Total trades
        metrics['total_trades'] = len(self.pnls)

        return metrics

    def summary(self) -> str:
        """
        Return formatted summary of metrics.

        Returns:
            Formatted string with all metrics
        """
        metrics = self.compute()

        summary_str = "=== Trading Metrics ===\n"
        summary_str += f"Cumulative Return: {metrics['cumulative_return']:.2%}\n"
        summary_str += f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
        summary_str += f"Sortino Ratio: {metrics['sortino_ratio']:.2f}\n"
        summary_str += f"Max Drawdown: {metrics['max_drawdown']:.2%}\n"
        summary_str += f"Win Rate: {metrics['win_rate']:.2%}\n"
        summary_str += f"Avg Trade Duration: {metrics['avg_trade_duration']:.1f}h\n"
        summary_str += f"Profit Factor: {metrics['profit_factor']:.2f}\n"
        summary_str += f"Calmar Ratio: {metrics['calmar_ratio']:.2f}\n"
        summary_str += f"Total Trades: {metrics['total_trades']}\n"

        return summary_str


# Alias for compatibility with __init__.py
MetricsCalculator = TradingMetrics
