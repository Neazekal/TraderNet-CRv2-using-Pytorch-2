"""
Sharpe Ratio calculation.

Sharpe ratio measures risk-adjusted returns.
Formula: (mean_return - risk_free_rate) / std_return

Higher Sharpe ratio indicates better risk-adjusted performance.
"""

import numpy as np
from typing import Union


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate Sharpe ratio from returns array.

    Args:
        returns: Array of returns (daily, hourly, etc.)
        risk_free_rate: Annual risk-free rate (default: 0%)
        periods_per_year: Number of periods per year (252 for daily, ~8760 for hourly)

    Returns:
        Sharpe ratio (annualized)
    """
    if len(returns) < 2:
        return 0.0

    returns = np.asarray(returns)

    # Daily risk-free rate
    daily_rf = risk_free_rate / periods_per_year

    # Excess returns
    excess_returns = returns - daily_rf

    # Calculate Sharpe ratio
    mean_excess_return = np.mean(excess_returns)
    std_excess_return = np.std(excess_returns)

    if std_excess_return == 0:
        return 0.0

    sharpe = (mean_excess_return / std_excess_return) * np.sqrt(periods_per_year)

    return float(sharpe)


class SharpeRatioCalculator:
    """Incremental Sharpe ratio calculator."""

    def __init__(self, risk_free_rate: float = 0.0, periods_per_year: int = 252):
        """
        Initialize Sharpe ratio calculator.

        Args:
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods per year
        """
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        self.returns = []

    def add_return(self, ret: float):
        """Add a return value."""
        self.returns.append(ret)

    def calculate(self) -> float:
        """Calculate current Sharpe ratio."""
        return calculate_sharpe_ratio(
            np.array(self.returns),
            self.risk_free_rate,
            self.periods_per_year,
        )

    def reset(self):
        """Reset calculator."""
        self.returns = []
