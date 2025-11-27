"""
Technical Feature Preprocessing Module.

Computes derived features from raw technical indicators:
- Price relative to indicators (close_dema, close_vwap, etc.)
- Second-order differences for volume indicators (adl_diffs2, obv_diffs2)

These derived features capture the relationship between price and indicators,
which is more informative than raw indicator values.
"""

import pandas as pd
import numpy as np


class TechnicalPreprocessor:
    """
    Preprocessor for deriving features from technical indicators.
    
    Transforms raw indicator values into more meaningful features:
    - Relative positions (price vs indicator)
    - Rate of change (differences)
    
    Example:
        preprocessor = TechnicalPreprocessor()
        df = preprocessor.compute_derived_features(df)
    """
    
    def __init__(self):
        """Initialize the technical preprocessor."""
        pass
    
    def compute_price_relative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute price position relative to indicators.
        
        These features show whether price is above/below key levels:
        - close_dema: Price vs DEMA (trend direction)
        - close_vwap: Price vs VWAP (fair value)
        - bband_up_close: Distance to upper band (overbought)
        - close_bband_down: Distance to lower band (oversold)
        
        Args:
            df: DataFrame with close, dema, vwap, bband_up, bband_down
            
        Returns:
            DataFrame with added relative features
        """
        df = df.copy()
        
        # Price relative to DEMA (positive = above trend)
        if 'dema' in df.columns:
            df['close_dema'] = df['close'] - df['dema']
        
        # Price relative to VWAP (positive = above fair value)
        if 'vwap' in df.columns:
            df['close_vwap'] = df['close'] - df['vwap']
        
        # Distance to Bollinger Bands
        if 'bband_up' in df.columns:
            df['bband_up_close'] = df['bband_up'] - df['close']
        
        if 'bband_down' in df.columns:
            df['close_bband_down'] = df['close'] - df['bband_down']
        
        return df
    
    def compute_volume_differences(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute second-order differences for volume indicators.
        
        Using 2nd order differences (acceleration) instead of raw values because:
        - ADL and OBV are cumulative, not stationary
        - 1st diff = rate of change (velocity)
        - 2nd diff = change in rate of change (acceleration)
        
        Args:
            df: DataFrame with adl and obv columns
            
        Returns:
            DataFrame with added difference features
        """
        df = df.copy()
        
        # ADL second-order differences
        if 'adl' in df.columns:
            df['adl_diffs'] = df['adl'].diff()      # 1st order (velocity)
            df['adl_diffs2'] = df['adl_diffs'].diff()  # 2nd order (acceleration)
        
        # OBV second-order differences
        if 'obv' in df.columns:
            df['obv_diffs'] = df['obv'].diff()      # 1st order (velocity)
            df['obv_diffs2'] = df['obv_diffs'].diff()  # 2nd order (acceleration)
        
        return df
    
    def compute_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all derived technical features.
        
        Args:
            df: DataFrame with technical indicators
            
        Returns:
            DataFrame with all derived features added
        """
        df = self.compute_price_relative_features(df)
        df = self.compute_volume_differences(df)
        return df


if __name__ == '__main__':
    # Example usage
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from data.preprocessing.ohlcv import OHLCVPreprocessor
    from analysis.technical.indicators_calc import TechnicalIndicators
    
    # Load OHLCV data
    ohlcv_preprocessor = OHLCVPreprocessor()
    df = ohlcv_preprocessor.load('data/storage/BTC.csv')
    
    # Compute technical indicators
    ta_calc = TechnicalIndicators()
    df = ta_calc.compute_all(df)
    
    # Compute derived features
    tech_preprocessor = TechnicalPreprocessor()
    df = tech_preprocessor.compute_derived_features(df)
    
    print("Derived features:")
    print(df[['close_dema', 'close_vwap', 'bband_up_close', 'close_bband_down', 
              'adl_diffs2', 'obv_diffs2']].tail(10))
