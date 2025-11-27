"""
Market Regime Detection Module.

Detects market regimes using rolling volatility and trend indicators.
This lightweight approach avoids complex models like HMM while still
providing useful regime classification for the trading agent.

Regimes:
- TRENDING_UP: Strong uptrend with moderate/low volatility
- TRENDING_DOWN: Strong downtrend with moderate/low volatility  
- HIGH_VOLATILITY: High volatility (choppy/uncertain market)
- RANGING: Low volatility, no clear trend (sideways market)

The regime information helps the agent:
1. Adapt strategy based on market conditions
2. Avoid overtrading in ranging markets
3. Be more aggressive in trending markets
4. Be cautious in high volatility periods
"""

import pandas as pd
import numpy as np
from typing import Tuple


class RegimeDetector:
    """
    Market regime detector using volatility and trend analysis.
    
    Uses:
    - Rolling volatility (standard deviation of returns)
    - ADX for trend strength
    - Price vs moving average for trend direction
    
    This is a lightweight alternative to HMM that:
    - Requires no training
    - Has minimal computational overhead
    - Works well for real-time trading
    
    Example:
        detector = RegimeDetector()
        df = detector.detect_regime(df)
        # df now has 'regime' and 'regime_encoded' columns
    """
    
    # Regime constants
    TRENDING_UP = 'TRENDING_UP'
    TRENDING_DOWN = 'TRENDING_DOWN'
    HIGH_VOLATILITY = 'HIGH_VOLATILITY'
    RANGING = 'RANGING'
    
    # Regime encoding for neural network input
    REGIME_ENCODING = {
        TRENDING_UP: 0,
        TRENDING_DOWN: 1,
        HIGH_VOLATILITY: 2,
        RANGING: 3
    }
    
    def __init__(
        self,
        volatility_window: int = 24,
        volatility_high_percentile: float = 75.0,
        volatility_low_percentile: float = 25.0,
        trend_window: int = 50,
        adx_threshold: float = 25.0,
        lookback_for_percentiles: int = 720  # 30 days for 1h timeframe
    ):
        """
        Initialize the regime detector.
        
        Args:
            volatility_window: Window for rolling volatility calculation (hours)
            volatility_high_percentile: Percentile above which volatility is "high"
            volatility_low_percentile: Percentile below which volatility is "low"
            trend_window: Window for trend detection (SMA period)
            adx_threshold: ADX value above which trend is considered strong
            lookback_for_percentiles: Period for calculating volatility percentiles
        """
        self.volatility_window = volatility_window
        self.volatility_high_percentile = volatility_high_percentile
        self.volatility_low_percentile = volatility_low_percentile
        self.trend_window = trend_window
        self.adx_threshold = adx_threshold
        self.lookback_for_percentiles = lookback_for_percentiles
    
    def compute_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute rolling volatility using log returns.
        
        Volatility is measured as the rolling standard deviation of
        log returns, annualized for interpretability.
        
        Args:
            df: DataFrame with 'close' column
            
        Returns:
            DataFrame with 'volatility' column added
        """
        df = df.copy()
        
        # Compute log returns if not present
        if 'close_log_returns' not in df.columns:
            df['close_log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Rolling standard deviation of returns
        df['volatility'] = df['close_log_returns'].rolling(
            window=self.volatility_window, 
            min_periods=self.volatility_window
        ).std()
        
        # Annualize for 1h timeframe (8760 hours per year)
        df['volatility_annualized'] = df['volatility'] * np.sqrt(8760)
        
        return df
    
    def compute_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute trend indicators for regime detection.
        
        Uses:
        - SMA for trend direction (price above/below)
        - ADX for trend strength (if available, else compute simple version)
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with trend indicators added
        """
        df = df.copy()
        
        # Simple Moving Average for trend direction
        df['sma_trend'] = df['close'].rolling(window=self.trend_window).mean()
        
        # Trend direction: 1 = above SMA, -1 = below SMA
        df['trend_direction'] = np.where(df['close'] > df['sma_trend'], 1, -1)
        
        # If ADX not present, compute a simple trend strength measure
        if 'adx' not in df.columns:
            # Use price momentum as proxy for trend strength
            # Higher absolute momentum = stronger trend
            momentum = df['close'].pct_change(self.volatility_window)
            df['trend_strength'] = momentum.abs().rolling(window=self.volatility_window).mean() * 100
        else:
            df['trend_strength'] = df['adx']
        
        return df
    
    def compute_volatility_percentiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute rolling percentiles for volatility classification.
        
        This adaptive approach classifies volatility relative to recent history,
        which is more robust than fixed thresholds across different market periods.
        
        Args:
            df: DataFrame with 'volatility' column
            
        Returns:
            DataFrame with volatility classification columns
        """
        df = df.copy()
        
        # Rolling percentile calculation
        def rolling_percentile(series, window, percentile):
            return series.rolling(window=window, min_periods=window).apply(
                lambda x: np.percentile(x, percentile), raw=True
            )
        
        # Compute rolling thresholds
        df['vol_high_threshold'] = rolling_percentile(
            df['volatility'], 
            self.lookback_for_percentiles, 
            self.volatility_high_percentile
        )
        df['vol_low_threshold'] = rolling_percentile(
            df['volatility'], 
            self.lookback_for_percentiles, 
            self.volatility_low_percentile
        )
        
        # Classify volatility
        df['volatility_class'] = 'MEDIUM'
        df.loc[df['volatility'] >= df['vol_high_threshold'], 'volatility_class'] = 'HIGH'
        df.loc[df['volatility'] <= df['vol_low_threshold'], 'volatility_class'] = 'LOW'
        
        return df
    
    def classify_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify market regime based on volatility and trend.
        
        Classification logic:
        1. HIGH_VOLATILITY: Volatility in top percentile (regardless of trend)
        2. TRENDING_UP: Strong trend (ADX > threshold) + price above SMA
        3. TRENDING_DOWN: Strong trend (ADX > threshold) + price below SMA
        4. RANGING: Low volatility and weak trend
        
        Args:
            df: DataFrame with volatility and trend indicators
            
        Returns:
            DataFrame with 'regime' column
        """
        df = df.copy()
        
        # Initialize regime column
        df['regime'] = self.RANGING  # Default to ranging
        
        # High volatility regime (highest priority)
        high_vol_mask = df['volatility_class'] == 'HIGH'
        df.loc[high_vol_mask, 'regime'] = self.HIGH_VOLATILITY
        
        # Strong trend regimes (only when not high volatility)
        strong_trend_mask = (df['trend_strength'] > self.adx_threshold) & (~high_vol_mask)
        
        # Trending up
        trending_up_mask = strong_trend_mask & (df['trend_direction'] == 1)
        df.loc[trending_up_mask, 'regime'] = self.TRENDING_UP
        
        # Trending down
        trending_down_mask = strong_trend_mask & (df['trend_direction'] == -1)
        df.loc[trending_down_mask, 'regime'] = self.TRENDING_DOWN
        
        # Encode regime for neural network
        df['regime_encoded'] = df['regime'].map(self.REGIME_ENCODING)
        
        return df
    
    def detect_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full regime detection pipeline.
        
        Computes all necessary indicators and classifies regime.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with regime columns added:
            - 'volatility': Rolling volatility
            - 'volatility_annualized': Annualized volatility
            - 'trend_direction': 1 (up) or -1 (down)
            - 'trend_strength': ADX or momentum-based strength
            - 'regime': String regime label
            - 'regime_encoded': Numeric encoding for NN input
        """
        df = self.compute_volatility(df)
        df = self.compute_trend_indicators(df)
        df = self.compute_volatility_percentiles(df)
        df = self.classify_regime(df)
        
        # Clean up intermediate columns
        columns_to_drop = ['sma_trend', 'vol_high_threshold', 'vol_low_threshold', 'volatility_class']
        df = df.drop(columns=[c for c in columns_to_drop if c in df.columns])
        
        return df
    
    def get_regime_stats(self, df: pd.DataFrame) -> dict:
        """
        Get statistics about regime distribution.
        
        Useful for understanding market conditions in the dataset.
        
        Args:
            df: DataFrame with 'regime' column
            
        Returns:
            Dictionary with regime statistics
        """
        if 'regime' not in df.columns:
            raise ValueError("DataFrame must have 'regime' column. Run detect_regime first.")
        
        regime_counts = df['regime'].value_counts()
        regime_pcts = df['regime'].value_counts(normalize=True) * 100
        
        stats = {
            'total_periods': len(df),
            'regime_counts': regime_counts.to_dict(),
            'regime_percentages': regime_pcts.to_dict(),
            'most_common_regime': regime_counts.idxmax(),
            'least_common_regime': regime_counts.idxmin()
        }
        
        return stats


# One-hot encoding utility for neural network input
def one_hot_encode_regime(regime_encoded: pd.Series, num_regimes: int = 4) -> np.ndarray:
    """
    Convert regime encoding to one-hot vectors.
    
    Args:
        regime_encoded: Series of regime encodings (0-3)
        num_regimes: Number of regime classes
        
    Returns:
        numpy array of shape (n_samples, num_regimes)
    """
    n_samples = len(regime_encoded)
    one_hot = np.zeros((n_samples, num_regimes))
    
    # Handle NaN values
    valid_mask = ~pd.isna(regime_encoded)
    valid_indices = regime_encoded[valid_mask].astype(int).values
    
    one_hot[valid_mask, valid_indices] = 1
    
    return one_hot


if __name__ == '__main__':
    # Example usage and testing
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from data.preprocessing.ohlcv import OHLCVPreprocessor
    
    # Load sample data
    ohlcv_preprocessor = OHLCVPreprocessor()
    df = ohlcv_preprocessor.load('data/storage/BTC.csv')
    
    print(f"Loaded {len(df)} rows of BTC data")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # Detect regimes
    detector = RegimeDetector()
    df = detector.detect_regime(df)
    
    # Print regime statistics
    stats = detector.get_regime_stats(df)
    print("\n" + "="*50)
    print("REGIME STATISTICS")
    print("="*50)
    print(f"Total periods: {stats['total_periods']}")
    print(f"\nRegime distribution:")
    for regime, count in stats['regime_counts'].items():
        pct = stats['regime_percentages'][regime]
        print(f"  {regime}: {count} ({pct:.1f}%)")
    print(f"\nMost common: {stats['most_common_regime']}")
    print(f"Least common: {stats['least_common_regime']}")
    
    # Show sample output
    print("\n" + "="*50)
    print("SAMPLE OUTPUT (last 20 rows)")
    print("="*50)
    print(df[['close', 'volatility', 'trend_direction', 'trend_strength', 
              'regime', 'regime_encoded']].tail(20).to_string())
    
    # Test one-hot encoding
    print("\n" + "="*50)
    print("ONE-HOT ENCODING TEST")
    print("="*50)
    one_hot = one_hot_encode_regime(df['regime_encoded'].tail(5))
    print("Last 5 regime encodings:", df['regime_encoded'].tail(5).values)
    print("One-hot encoded:\n", one_hot)
