"""
Smurf Reward Function.

Modified reward function for training the "Smurf" agent - a conservative
secondary agent that helps prevent over-trading.

The key difference from MarketLimitOrderReward is the HOLD reward:
- Instead of negative penalty, HOLD gets a small positive reward (0.0055)
- This encourages the Smurf agent to prefer holding
- Results in a more conservative trading strategy
"""

import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import HORIZON, FEES, SMURF_HOLD_REWARD
from environments.rewards.base import BaseRewardFunction
from environments.rewards.market_limit import MarketLimitOrderReward


class SmurfReward(BaseRewardFunction):
    """
    Smurf reward function for conservative agent training.
    
    The Smurf agent acts as a "gatekeeper" - if it decides to HOLD,
    the system holds regardless of what the main TraderNet suggests.
    
    Reward modification:
    - BUY and SELL rewards: Same as MarketLimitOrderReward
    - HOLD reward: Fixed positive value (0.0055) instead of negative
    
    This positive HOLD reward makes the agent:
    - More likely to hold during uncertain conditions
    - Less prone to over-trading
    - More conservative overall
    
    Example:
        smurf_reward = SmurfReward(highs, lows, closes, horizon=20, fees=0.01)
        reward = smurf_reward.get_reward(step=100, action=2)  # HOLD reward
    """
    
    # Default hold reward (from config)
    DEFAULT_HOLD_REWARD = SMURF_HOLD_REWARD
    
    def __init__(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        horizon: int = HORIZON,
        fees: float = FEES,
        hold_reward: float = None
    ):
        """
        Initialize Smurf reward function.
        
        Args:
            highs: High prices array
            lows: Low prices array
            closes: Close prices array
            horizon: Lookahead horizon (K hours)
            fees: Transaction fee percentage
            hold_reward: Fixed positive reward for HOLD action (default: 0.0055)
        """
        self.hold_reward_value = hold_reward or self.DEFAULT_HOLD_REWARD
        
        super().__init__(
            highs=highs,
            lows=lows,
            closes=closes,
            horizon=horizon,
            fees=fees
        )
    
    def _compute_step_rewards(self, step: int) -> np.ndarray:
        """
        Compute rewards for BUY, SELL, HOLD at a single timestep.
        
        BUY and SELL are same as MarketLimitOrderReward.
        HOLD is a fixed positive value to encourage conservative behavior.
        
        Args:
            step: Current timestep index
            
        Returns:
            Array of [buy_reward, sell_reward, hold_reward]
        """
        # Current close price (entry price)
        current_close = self.closes[step]
        
        # Future price range within horizon
        horizon_start = step + 1
        horizon_end = step + 1 + self.horizon
        
        max_high = self.highs[horizon_start:horizon_end].max()
        min_low = self.lows[horizon_start:horizon_end].min()
        
        # BUY reward (same as MarketLimitOrder)
        buy_reward = np.log(max_high / current_close) + self.fee_adjustment
        
        # SELL reward (same as MarketLimitOrder)
        sell_reward = np.log(current_close / min_low) + self.fee_adjustment
        
        # HOLD reward: Fixed positive value (key difference from MarketLimitOrder)
        hold_reward = self.hold_reward_value
        
        return np.array([buy_reward, sell_reward, hold_reward], dtype=np.float32)


def create_smurf_from_market_limit(
    market_limit_reward: MarketLimitOrderReward,
    hold_reward: float = SmurfReward.DEFAULT_HOLD_REWARD
) -> SmurfReward:
    """
    Create a SmurfReward from an existing MarketLimitOrderReward.
    
    Useful when you want both reward functions to use the same price data.
    
    Args:
        market_limit_reward: Existing MarketLimitOrderReward instance
        hold_reward: Fixed HOLD reward value
        
    Returns:
        SmurfReward instance with same price data
    """
    return SmurfReward(
        highs=market_limit_reward.highs,
        lows=market_limit_reward.lows,
        closes=market_limit_reward.closes,
        horizon=market_limit_reward.horizon,
        fees=market_limit_reward.fees,
        hold_reward=hold_reward
    )


if __name__ == '__main__':
    # Test and compare reward functions
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from data.datasets.utils import load_processed_dataset, get_price_data
    from config.config import SEQUENCE_LENGTH, HORIZON, FEES
    
    # Load BTC data
    df = load_processed_dataset('data/datasets/BTC_processed.csv')
    highs, lows, closes = get_price_data(df, SEQUENCE_LENGTH)
    
    # Create both reward functions
    market_reward = MarketLimitOrderReward(highs, lows, closes, HORIZON, FEES)
    smurf_reward = SmurfReward(highs, lows, closes, HORIZON, FEES)
    
    print("Comparison of MarketLimitOrder vs Smurf rewards:")
    print("=" * 50)
    
    print(f"\nMarketLimitOrder HOLD reward:")
    print(f"  mean: {market_reward._rewards[:, 2].mean():.4f}")
    print(f"  min:  {market_reward._rewards[:, 2].min():.4f}")
    print(f"  max:  {market_reward._rewards[:, 2].max():.4f}")
    
    print(f"\nSmurf HOLD reward:")
    print(f"  mean: {smurf_reward._rewards[:, 2].mean():.4f}")
    print(f"  min:  {smurf_reward._rewards[:, 2].min():.4f}")
    print(f"  max:  {smurf_reward._rewards[:, 2].max():.4f}")
    
    print(f"\nSample at step 1000:")
    print(f"  MarketLimit: BUY={market_reward(1000, 0):.4f}, SELL={market_reward(1000, 1):.4f}, HOLD={market_reward(1000, 2):.4f}")
    print(f"  Smurf:       BUY={smurf_reward(1000, 0):.4f}, SELL={smurf_reward(1000, 1):.4f}, HOLD={smurf_reward(1000, 2):.4f}")
