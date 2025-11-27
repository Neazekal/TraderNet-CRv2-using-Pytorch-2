"""
Market Limit Order Reward Function.

Implements the main reward function from the TraderNet-CRv2 paper.
Rewards are based on the best possible outcome within a future horizon:
- BUY: Maximum potential profit from buying (max high in horizon)
- SELL: Maximum potential profit from selling (min low in horizon)
- HOLD: Negative of best possible action, capped at 0
"""

import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import HORIZON, FEES
from environments.rewards.base import BaseRewardFunction


class MarketLimitOrderReward(BaseRewardFunction):
    """
    Market Limit Order reward function.
    
    This reward function assumes perfect execution at the best price
    within the horizon window (like a limit order that gets filled).
    
    Reward formulas:
    - BUY:  log(max_high_in_horizon / close_price) + fee_adjustment
    - SELL: log(close_price / min_low_in_horizon) + fee_adjustment
    - HOLD: -max(buy_reward, sell_reward), capped at 0
    
    The HOLD reward is designed to:
    - Be negative when there's a good trading opportunity (penalize inaction)
    - Be zero when neither BUY nor SELL would be profitable
    
    Example:
        reward_fn = MarketLimitOrderReward(highs, lows, closes, horizon=20, fees=0.01)
        reward = reward_fn.get_reward(step=100, action=0)  # BUY reward at step 100
    """
    
    def __init__(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        horizon: int = HORIZON,
        fees: float = FEES
    ):
        """
        Initialize Market Limit Order reward function.
        
        Args:
            highs: High prices array
            lows: Low prices array
            closes: Close prices array
            horizon: Lookahead horizon (K hours)
            fees: Transaction fee percentage (default: 1%)
        """
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
        
        # BUY reward: profit from buying at current close, selling at max high
        # log(max_high / current_close) = potential upside
        buy_reward = np.log(max_high / current_close) + self.fee_adjustment
        
        # SELL reward: profit from selling at current close, buying back at min low
        # log(current_close / min_low) = potential downside profit
        sell_reward = np.log(current_close / min_low) + self.fee_adjustment
        
        # HOLD reward: penalize holding when there's opportunity
        # Negative of best possible reward, but capped at 0
        # (don't reward holding, just don't penalize when no opportunity)
        hold_reward = -max(buy_reward, sell_reward)
        hold_reward = min(hold_reward, 0.0)
        
        return np.array([buy_reward, sell_reward, hold_reward], dtype=np.float32)


if __name__ == '__main__':
    # Test the reward function
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from data.datasets.utils import load_processed_dataset, get_price_data
    from config.config import SEQUENCE_LENGTH, HORIZON, FEES
    
    # Load BTC data
    df = load_processed_dataset('data/datasets/BTC_processed.csv')
    highs, lows, closes = get_price_data(df, SEQUENCE_LENGTH)
    
    # Create reward function
    reward_fn = MarketLimitOrderReward(
        highs=highs,
        lows=lows,
        closes=closes,
        horizon=HORIZON,
        fees=FEES
    )
    
    print(f"Number of valid steps: {len(reward_fn)}")
    print(f"\nReward statistics:")
    print(f"  BUY  - mean: {reward_fn._rewards[:, 0].mean():.4f}, std: {reward_fn._rewards[:, 0].std():.4f}")
    print(f"  SELL - mean: {reward_fn._rewards[:, 1].mean():.4f}, std: {reward_fn._rewards[:, 1].std():.4f}")
    print(f"  HOLD - mean: {reward_fn._rewards[:, 2].mean():.4f}, std: {reward_fn._rewards[:, 2].std():.4f}")
    
    print(f"\nSample rewards at step 1000:")
    print(f"  BUY:  {reward_fn.get_reward(1000, 0):.4f}")
    print(f"  SELL: {reward_fn.get_reward(1000, 1):.4f}")
    print(f"  HOLD: {reward_fn.get_reward(1000, 2):.4f}")
