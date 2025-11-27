"""
OHLCV Preprocessing Module.

Handles preprocessing of raw OHLCV data from Binance:
- Calculate log returns for price and volume columns
- Extract hour from timestamp
- Prepare data for technical indicator calculation
"""

import numpy as np
import pandas as pd
from pathlib import Path


class OHLCVPreprocessor:
    """
    Preprocessor for raw OHLCV (Open, High, Low, Close, Volume) data.
    
    Calculates log returns and extracts temporal features from raw candlestick data.
    Log returns are used instead of raw prices because:
    - They are stationary (important for ML models)
    - They can be summed to get cumulative returns
    - They approximate percentage returns for small values
    
    Example:
        preprocessor = OHLCVPreprocessor()
        df = preprocessor.load_and_preprocess('data/storage/BTC.csv')
    """
    
    # Columns expected in raw OHLCV data
    RAW_COLUMNS = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    # Output columns after preprocessing
    LOG_RETURN_COLUMNS = [
        'open_log_returns',
        'high_log_returns', 
        'low_log_returns',
        'close_log_returns',
        'volume_log_returns'
    ]
    
    def __init__(self):
        """Initialize the OHLCV preprocessor."""
        pass
    
    def load(self, filepath: str) -> pd.DataFrame:
        """
        Load raw OHLCV data from CSV file.
        
        Args:
            filepath: Path to CSV file with OHLCV data
            
        Returns:
            DataFrame with raw OHLCV data
        """
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    def calculate_log_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate log returns for OHLCV columns.
        
        Log return = ln(price_t / price_t-1) = ln(price_t) - ln(price_t-1)
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added log return columns
        """
        df = df.copy()
        
        # Calculate log returns for price columns
        df['open_log_returns'] = np.log(df['open']).diff()
        df['high_log_returns'] = np.log(df['high']).diff()
        df['low_log_returns'] = np.log(df['low']).diff()
        df['close_log_returns'] = np.log(df['close']).diff()
        
        # Calculate log returns for volume (handle zero volume)
        df['volume_log_returns'] = np.log(df['volume'].replace(0, np.nan)).diff()
        
        return df
    
    def extract_hour(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract hour from timestamp as a feature.
        
        Hour (0-23) can capture intraday patterns in trading activity.
        
        Args:
            df: DataFrame with timestamp column
            
        Returns:
            DataFrame with added hour column
        """
        df = df.copy()
        df['hour'] = df['timestamp'].dt.hour
        return df
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all preprocessing steps to raw OHLCV data.
        
        Steps:
        1. Calculate log returns for open, high, low, close, volume
        2. Extract hour from timestamp
        
        Args:
            df: DataFrame with raw OHLCV data
            
        Returns:
            DataFrame with preprocessed features
        """
        # Validate input columns
        missing_cols = set(self.RAW_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Apply preprocessing steps
        df = self.calculate_log_returns(df)
        df = self.extract_hour(df)
        
        return df
    
    def load_and_preprocess(self, filepath: str) -> pd.DataFrame:
        """
        Load raw OHLCV data and apply all preprocessing.
        
        Args:
            filepath: Path to CSV file with OHLCV data
            
        Returns:
            DataFrame with preprocessed features
        """
        df = self.load(filepath)
        df = self.preprocess(df)
        return df


if __name__ == '__main__':
    # Example usage
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    preprocessor = OHLCVPreprocessor()
    df = preprocessor.load_and_preprocess('data/storage/BTC.csv')
    
    print("Columns:", df.columns.tolist())
    print("\nShape:", df.shape)
    print("\nSample data:")
    print(df[['timestamp', 'close', 'close_log_returns', 'hour']].head(10))
