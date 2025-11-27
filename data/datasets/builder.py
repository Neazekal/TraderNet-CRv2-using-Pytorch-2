"""
Dataset Builder Module.

Builds the final dataset by combining:
- OHLCV preprocessing (log returns, hour)
- Technical indicators (12 indicators)
- Derived features (relative prices, volume diffs)
- Market regime detection (trending, ranging, high volatility)
- Funding rate features (8 features from Binance Futures)
- Feature scaling (Min-Max to [0, 1])

Output: 28 features ready for model training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from typing import Optional, Tuple
import pickle

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import FEATURES, TA_PARAMS, REGIME_PARAMS, FUNDING_PARAMS
from data.preprocessing.ohlcv import OHLCVPreprocessor
from analysis.technical.indicators_calc import TechnicalIndicators
from data.preprocessing.technical import TechnicalPreprocessor
from data.preprocessing.regime import RegimeDetector
from data.preprocessing.funding import FundingPreprocessor


class DatasetBuilder:
    """
    Builder for creating training-ready datasets.
    
    Combines all preprocessing steps into a single pipeline:
    1. Load raw OHLCV data
    2. Compute log returns and extract hour
    3. Compute technical indicators
    4. Compute derived features
    5. Select final features
    6. Apply Min-Max scaling
    7. Drop NaN rows
    
    Example:
        builder = DatasetBuilder()
        df, scaler = builder.build('data/storage/BTC.csv')
        builder.save('data/datasets/BTC_processed.csv')
    """
    
    def __init__(self, features: list = None, ta_params: dict = None, regime_params: dict = None, funding_params: dict = None):
        """
        Initialize dataset builder.
        
        Args:
            features: List of feature names to include (default: from config)
            ta_params: Technical indicator parameters (default: from config)
            regime_params: Market regime detection parameters (default: from config)
            funding_params: Funding rate processing parameters (default: from config)
        """
        self.features = features or FEATURES
        self.ta_params = ta_params or TA_PARAMS
        self.regime_params = regime_params or REGIME_PARAMS
        self.funding_params = funding_params or FUNDING_PARAMS
        
        # Initialize preprocessors
        self.ohlcv_preprocessor = OHLCVPreprocessor()
        self.ta_calculator = TechnicalIndicators(**self.ta_params)
        self.tech_preprocessor = TechnicalPreprocessor()
        self.regime_detector = RegimeDetector(**self.regime_params)
        self.funding_preprocessor = FundingPreprocessor()
        
        # Scaler (fitted during build)
        self.scaler: Optional[MinMaxScaler] = None
        
        # Processed data
        self.data: Optional[pd.DataFrame] = None
        self.raw_data: Optional[pd.DataFrame] = None
    
    def build(
        self, 
        filepath: str,
        funding_filepath: str = None,
        fit_scaler: bool = True,
        scaler: Optional[MinMaxScaler] = None
    ) -> Tuple[pd.DataFrame, MinMaxScaler]:
        """
        Build processed dataset from raw OHLCV file.
        
        Args:
            filepath: Path to raw OHLCV CSV file
            funding_filepath: Path to funding rate CSV file (auto-detected if None)
            fit_scaler: Whether to fit a new scaler (True for training data)
            scaler: Pre-fitted scaler to use (for test data)
            
        Returns:
            Tuple of (processed DataFrame, fitted scaler)
        """
        print(f"Building dataset from {filepath}...")
        
        # Auto-detect funding file path if not provided
        if funding_filepath is None:
            ohlcv_path = Path(filepath)
            funding_filepath = ohlcv_path.parent / f"{ohlcv_path.stem}_funding.csv"
            if not funding_filepath.exists():
                print(f"  Warning: Funding file not found at {funding_filepath}")
                funding_filepath = None
        
        # Step 1: Load raw data
        print("  Loading raw OHLCV data...")
        df = self.ohlcv_preprocessor.load(filepath)
        self.raw_data = df.copy()
        
        # Step 2: Compute OHLCV features (log returns, hour)
        print("  Computing log returns and hour...")
        df = self.ohlcv_preprocessor.preprocess(df)
        
        # Step 3: Load and merge funding rate data
        if funding_filepath and Path(funding_filepath).exists():
            print(f"  Loading and processing funding rate data from {funding_filepath}...")
            funding_df = self.funding_preprocessor.load_and_preprocess(str(funding_filepath))
            df = self.funding_preprocessor.merge_with_ohlcv(df, funding_df)
            print(f"    Added {len(self.funding_preprocessor.load(str(funding_filepath)))} funding rate records")
        else:
            print("  Skipping funding data (file not found)")
            # Add empty funding columns
            from data.preprocessing.funding import get_funding_feature_columns
            for col in get_funding_feature_columns():
                df[col] = 0.0
        
        # Step 4: Compute technical indicators
        print("  Computing technical indicators...")
        df = self.ta_calculator.compute_all(df)
        
        # Step 5: Compute derived features
        print("  Computing derived features...")
        df = self.tech_preprocessor.compute_derived_features(df)
        
        # Step 6: Detect market regime
        print("  Detecting market regimes...")
        df = self.regime_detector.detect_regime(df)
        
        # Print regime statistics
        stats = self.regime_detector.get_regime_stats(df)
        print(f"    Regime distribution: ", end="")
        for regime, pct in stats['regime_percentages'].items():
            print(f"{regime}={pct:.1f}% ", end="")
        print()
        
        # Step 7: Keep timestamp and required columns for reward calculation
        # Store highs, lows, closes before selecting features
        metadata_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        metadata = df[metadata_cols].copy()
        
        # Step 8: Select final features
        print(f"  Selecting {len(self.features)} features...")
        missing_features = set(self.features) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        feature_df = df[self.features].copy()
        
        # Step 9: Drop NaN rows (from indicator warm-up periods)
        print("  Dropping NaN rows...")
        valid_mask = ~feature_df.isna().any(axis=1)
        feature_df = feature_df[valid_mask]
        metadata = metadata[valid_mask]
        
        print(f"  Dropped {(~valid_mask).sum()} rows with NaN values")
        
        # Step 10: Apply Min-Max scaling
        print("  Applying Min-Max scaling...")
        if fit_scaler:
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_values = self.scaler.fit_transform(feature_df)
        else:
            if scaler is None:
                raise ValueError("Must provide scaler when fit_scaler=False")
            self.scaler = scaler
            scaled_values = self.scaler.transform(feature_df)
        
        # Create final DataFrame
        scaled_df = pd.DataFrame(scaled_values, columns=self.features)
        
        # Add metadata back
        scaled_df = pd.concat([
            metadata.reset_index(drop=True),
            scaled_df.reset_index(drop=True)
        ], axis=1)
        
        self.data = scaled_df
        
        print(f"  Final dataset shape: {self.data.shape}")
        print(f"  Features: {self.features}")
        
        return self.data, self.scaler
    
    def save(self, filepath: str, save_scaler: bool = True) -> None:
        """
        Save processed dataset to CSV.
        
        Args:
            filepath: Path to save CSV file
            save_scaler: Whether to save the scaler as pickle
        """
        if self.data is None:
            raise ValueError("No data to save. Call build() first.")
        
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save dataset
        self.data.to_csv(filepath, index=False)
        print(f"Saved dataset to {filepath}")
        
        # Save scaler
        if save_scaler and self.scaler is not None:
            scaler_path = path.with_suffix('.scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"Saved scaler to {scaler_path}")
    
    def load(self, filepath: str, load_scaler: bool = True) -> pd.DataFrame:
        """
        Load processed dataset from CSV.
        
        Args:
            filepath: Path to CSV file
            load_scaler: Whether to load the scaler
            
        Returns:
            Processed DataFrame
        """
        self.data = pd.read_csv(filepath)
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        print(f"Loaded dataset from {filepath}: {self.data.shape}")
        
        if load_scaler:
            scaler_path = Path(filepath).with_suffix('.scaler.pkl')
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print(f"Loaded scaler from {scaler_path}")
        
        return self.data


def build_all_datasets(
    input_dir: str = 'data/storage/',
    output_dir: str = 'data/datasets/',
    cryptos: list = None
) -> None:
    """
    Build processed datasets for all cryptocurrencies.
    
    Args:
        input_dir: Directory with raw OHLCV CSV files
        output_dir: Directory to save processed datasets
        cryptos: List of crypto names (default: all CSV files in input_dir)
    """
    input_path = Path(input_dir)
    
    if cryptos is None:
        cryptos = [f.stem for f in input_path.glob('*.csv')]
    
    for crypto in cryptos:
        print(f"\n{'='*50}")
        print(f"Processing {crypto}...")
        print(f"{'='*50}")
        
        builder = DatasetBuilder()
        builder.build(f"{input_dir}/{crypto}.csv")
        builder.save(f"{output_dir}/{crypto}_processed.csv")


if __name__ == '__main__':
    # Example: Build dataset for BTC
    # builder = DatasetBuilder()
    # df, scaler = builder.build('data/storage/BTC.csv')
    # builder.save('data/datasets/BTC_processed.csv')
    
    # Build all datasets
    from config.config import SUPPORTED_CRYPTOS
    build_all_datasets(cryptos=list(SUPPORTED_CRYPTOS.keys()))
