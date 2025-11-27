"""
Funding Rate Preprocessing Module.

Handles preprocessing of funding rate data from Binance Futures:
- Resample 8-hour funding data to match 1-hour OHLCV data
- Merge funding data with OHLCV data

Funding rates are paid every 8 hours (00:00, 08:00, 16:00 UTC).
Positive funding = longs pay shorts (bullish sentiment)
Negative funding = shorts pay longs (bearish sentiment)
"""

import pandas as pd
from pathlib import Path


class FundingPreprocessor:
    """
    Preprocessor for funding rate data.
    
    Keeps only raw funding_rate - the signal itself is already meaningful:
    - Positive funding: Market is over-leveraged long
    - Negative funding: Market is over-leveraged short
    - Near zero: Balanced market
    
    Example:
        preprocessor = FundingPreprocessor()
        df = preprocessor.load_and_preprocess('data/storage/BTC_funding.csv')
    """
    
    def __init__(self):
        """Initialize the funding rate preprocessor."""
        pass
    
    def load(self, filepath: str) -> pd.DataFrame:
        """
        Load raw funding rate data from CSV file.
        
        Args:
            filepath: Path to CSV file with funding rate data
            
        Returns:
            DataFrame with raw funding rate data
        """
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df
    
    def resample_to_hourly(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample 8-hour funding data to 1-hour frequency.
        
        Funding rate is collected every 8 hours but we need hourly data
        to match OHLCV. We forward-fill the funding rate.
        
        Args:
            df: DataFrame with 8-hour funding data
            
        Returns:
            DataFrame with hourly funding data (forward-filled)
        """
        df = df.copy()
        df = df.set_index('timestamp')
        
        # Keep only funding_rate column
        df = df[['funding_rate']]
        
        # Resample to hourly and forward-fill
        df_hourly = df.resample('1h').ffill()
        
        # Reset index
        df_hourly = df_hourly.reset_index()
        
        return df_hourly
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all preprocessing steps to raw funding data.
        
        Steps:
        1. Keep only funding_rate column
        2. Resample to hourly frequency
        
        Args:
            df: DataFrame with raw funding rate data
            
        Returns:
            DataFrame with funding_rate at hourly frequency
        """
        # Validate input
        if 'funding_rate' not in df.columns:
            raise ValueError("Missing required column: funding_rate")
        if 'timestamp' not in df.columns:
            raise ValueError("Missing required column: timestamp")
        
        # Resample to hourly (keeps only funding_rate)
        df = self.resample_to_hourly(df)
        
        return df
    
    def load_and_preprocess(self, filepath: str) -> pd.DataFrame:
        """
        Load raw funding data and apply all preprocessing.
        
        Args:
            filepath: Path to CSV file with funding rate data
            
        Returns:
            DataFrame with funding_rate at hourly frequency
        """
        df = self.load(filepath)
        df = self.preprocess(df)
        return df
    
    def merge_with_ohlcv(self, ohlcv_df: pd.DataFrame, funding_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge funding rate with OHLCV data.
        
        Aligns funding data with OHLCV timestamps using asof merge
        (matches each OHLCV timestamp with the most recent funding data).
        
        Args:
            ohlcv_df: DataFrame with OHLCV data (must have 'timestamp' column)
            funding_df: DataFrame with funding_rate (must have 'timestamp' column)
            
        Returns:
            DataFrame with OHLCV and funding_rate merged
        """
        # Ensure both dataframes are sorted by timestamp
        ohlcv_df = ohlcv_df.sort_values('timestamp').reset_index(drop=True)
        funding_df = funding_df.sort_values('timestamp').reset_index(drop=True)
        
        # Perform asof merge (each OHLCV timestamp gets the most recent funding data)
        merged = pd.merge_asof(
            ohlcv_df,
            funding_df[['timestamp', 'funding_rate']],
            on='timestamp',
            direction='backward'
        )
        
        # Fill any remaining NaN values with 0 (for early data before funding started)
        merged['funding_rate'] = merged['funding_rate'].fillna(0)
        
        return merged


def get_funding_feature_columns() -> list:
    """
    Get list of funding feature column names.
    
    Returns:
        List of funding feature column names
    """
    return ['funding_rate']


if __name__ == '__main__':
    # Example usage
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    preprocessor = FundingPreprocessor()
    
    # Load and preprocess funding data
    funding_df = preprocessor.load_and_preprocess('data/storage/BTC_funding.csv')
    
    print("Funding Data:")
    print("Columns:", funding_df.columns.tolist())
    print("\nShape:", funding_df.shape)
    print("\nSample data:")
    print(funding_df.head(20))
    
    # Example: Merge with OHLCV
    from ohlcv import OHLCVPreprocessor
    
    ohlcv_preprocessor = OHLCVPreprocessor()
    ohlcv_df = ohlcv_preprocessor.load_and_preprocess('data/storage/BTC.csv')
    
    merged_df = preprocessor.merge_with_ohlcv(ohlcv_df, funding_df)
    
    print("\n\nMerged Data:")
    print("Columns:", merged_df.columns.tolist())
    print("\nShape:", merged_df.shape)
    print("\nSample:")
    print(merged_df[['timestamp', 'close', 'funding_rate']].head(20))
