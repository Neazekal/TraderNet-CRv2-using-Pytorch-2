"""
Trading metrics module.

Provides functions and classes for calculating trading performance metrics:
- Sharpe Ratio: Risk-adjusted returns
- Sortino Ratio: Downside risk-adjusted returns
- Maximum Drawdown: Peak-to-trough decline
"""

from .sharpe import calculate_sharpe_ratio, SharpeRatioCalculator
from .sortino import calculate_sortino_ratio, SortinoRatioCalculator
from .drawdown import (
    calculate_max_drawdown,
    calculate_max_drawdown_detailed,
    DrawdownCalculator,
)

__all__ = [
    'calculate_sharpe_ratio',
    'SharpeRatioCalculator',
    'calculate_sortino_ratio',
    'SortinoRatioCalculator',
    'calculate_max_drawdown',
    'calculate_max_drawdown_detailed',
    'DrawdownCalculator',
]
