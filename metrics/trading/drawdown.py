"""
Maximum Drawdown calculation.

Drawdown measures the largest peak-to-trough decline in portfolio value.
Formula: (Peak - Trough) / Peak

Maximum drawdown is the largest drawdown during the period.
Used to assess downside risk and portfolio volatility.
"""

import numpy as np
from typing import Tuple


def calculate_max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Calculate maximum drawdown from equity curve.

    Args:
        equity_curve: Array of portfolio equity values over time

    Returns:
        Maximum drawdown as percentage (0.0 to 1.0)
    """
    if len(equity_curve) < 2:
        return 0.0

    equity_curve = np.asarray(equity_curve)

    # Running maximum (peak)
    running_max = np.maximum.accumulate(equity_curve)

    # Drawdown at each point
    drawdowns = (equity_curve - running_max) / running_max

    # Maximum drawdown (most negative)
    max_drawdown = np.min(drawdowns)

    return float(max(0.0, -max_drawdown))  # Return as positive value


def calculate_max_drawdown_detailed(equity_curve: np.ndarray) -> Tuple[float, int, int]:
    """
    Calculate maximum drawdown with peak and trough indices.

    Args:
        equity_curve: Array of portfolio equity values over time

    Returns:
        Tuple of (max_drawdown, peak_index, trough_index)
    """
    if len(equity_curve) < 2:
        return 0.0, 0, 0

    equity_curve = np.asarray(equity_curve)

    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / running_max

    # Find maximum drawdown
    trough_idx = np.argmin(drawdowns)
    max_dd = drawdowns[trough_idx]

    # Find peak before trough
    peak_idx = np.argmax(equity_curve[:trough_idx + 1])

    return float(max(0.0, -max_dd)), int(peak_idx), int(trough_idx)


def calculate_drawdown_duration(equity_curve: np.ndarray) -> float:
    """
    Calculate average drawdown duration (periods underwater).

    Args:
        equity_curve: Array of portfolio equity values over time

    Returns:
        Average number of periods in drawdown
    """
    if len(equity_curve) < 2:
        return 0.0

    equity_curve = np.asarray(equity_curve)
    running_max = np.maximum.accumulate(equity_curve)

    # Boolean array: True when underwater (below peak)
    underwater = equity_curve < running_max

    # Count consecutive underwater periods
    if not underwater.any():
        return 0.0

    # Get all drawdown periods
    dd_changes = np.diff(underwater.astype(int))
    dd_starts = np.where(dd_changes == 1)[0] + 1
    dd_ends = np.where(dd_changes == -1)[0] + 1

    if len(dd_starts) == 0:
        return 0.0

    durations = dd_ends - dd_starts
    return float(np.mean(durations)) if len(durations) > 0 else 0.0


class DrawdownCalculator:
    """Incremental maximum drawdown calculator."""

    def __init__(self):
        """Initialize drawdown calculator."""
        self.equity_curve = []

    def add_value(self, equity: float):
        """Add equity value to the curve."""
        self.equity_curve.append(equity)

    def calculate_max_drawdown(self) -> float:
        """Calculate current maximum drawdown."""
        if len(self.equity_curve) < 2:
            return 0.0
        return calculate_max_drawdown(np.array(self.equity_curve))

    def calculate_duration(self) -> float:
        """Calculate average drawdown duration."""
        if len(self.equity_curve) < 2:
            return 0.0
        return calculate_drawdown_duration(np.array(self.equity_curve))

    def get_detailed_stats(self) -> dict:
        """Get detailed drawdown statistics."""
        if len(self.equity_curve) < 2:
            return {
                'max_drawdown': 0.0,
                'peak_index': 0,
                'trough_index': 0,
                'peak_value': 0.0,
                'trough_value': 0.0,
                'duration': 0.0,
            }

        equity = np.array(self.equity_curve)
        max_dd, peak_idx, trough_idx = calculate_max_drawdown_detailed(equity)

        return {
            'max_drawdown': max_dd,
            'peak_index': peak_idx,
            'trough_index': trough_idx,
            'peak_value': float(equity[peak_idx]),
            'trough_value': float(equity[trough_idx]),
            'duration': trough_idx - peak_idx,
        }

    def reset(self):
        """Reset calculator."""
        self.equity_curve = []
