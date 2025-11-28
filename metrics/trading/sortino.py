"""
Sortino Ratio calculation.

Sortino ratio is similar to Sharpe ratio but only considers downside volatility.
Formula: (mean_return - risk_free_rate) / downside_std_return

Only negative returns (losses) are included in volatility calculation.
Higher Sortino ratio indicates better downside risk-adjusted performance.
"""

import numpy as np
from typing import Union


def calculate_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    target_return: float = 0.0,
) -> float:
    """
    Calculate Sortino ratio from returns array.

    Args:
        returns: Array of returns (daily, hourly, etc.)
        risk_free_rate: Annual risk-free rate (default: 0%)
        periods_per_year: Number of periods per year (252 for daily, ~8760 for hourly)
        target_return: Target return threshold (default: 0%, can use risk_free_rate)

    Returns:
        Sortino ratio (annualized)
    """
    if len(returns) < 2:
        return 0.0

    returns = np.asarray(returns)

    # Daily risk-free rate
    daily_rf = risk_free_rate / periods_per_year

    # Excess returns
    excess_returns = returns - daily_rf

    # Downside returns (only negative returns)
    downside_returns = np.minimum(excess_returns, 0)

    # Downside volatility (std of negative returns)
    downside_std = np.sqrt(np.mean(downside_returns ** 2))

    if downside_std == 0:
        # If no downside volatility, return 0 (or could return inf for perfect returns)
        return 0.0

    # Sortino ratio
    mean_excess_return = np.mean(excess_returns)
    sortino = (mean_excess_return / downside_std) * np.sqrt(periods_per_year)

    return float(sortino)


class SortinoRatioCalculator:
    """Incremental Sortino ratio calculator."""

    def __init__(
        self,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
        target_return: float = 0.0,
    ):
        """
        Initialize Sortino ratio calculator.

        Args:
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods per year
            target_return: Target return threshold
        """
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        self.target_return = target_return
        self.returns = []

    def add_return(self, ret: float):
        """Add a return value."""
        self.returns.append(ret)

    def calculate(self) -> float:
        """Calculate current Sortino ratio."""
        return calculate_sortino_ratio(
            np.array(self.returns),
            self.risk_free_rate,
            self.periods_per_year,
            self.target_return,
        )

    def reset(self):
        """Reset calculator."""
        self.returns = []
