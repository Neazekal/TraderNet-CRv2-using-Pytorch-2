"""
Dataset Utilities Module.

Provides utilities for:
- Train/eval data splitting
- Sequence generation for RL environment
- Data loading helpers
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import FEATURES, SEQUENCE_LENGTH, EVAL_HOURS


def train_eval_split(
    df: pd.DataFrame,
    eval_hours: int = EVAL_HOURS,
    features: list = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into training and evaluation sets.
    
    Split strategy (from paper):
    - Training: All data except last eval_hours
    - Evaluation: Last eval_hours (~3 months for 2250 hours)
    
    Args:
        df: Processed DataFrame with features
        eval_hours: Number of hours for evaluation (default: 2250)
        features: Feature columns to include
        
    Returns:
        Tuple of (train_df, eval_df)
    """
    features = features or FEATURES
    
    # Split by index (time-ordered)
    split_idx = len(df) - eval_hours
    
    if split_idx <= 0:
        raise ValueError(f"Not enough data for eval_hours={eval_hours}. "
                        f"Dataset has {len(df)} rows.")
    
    train_df = df.iloc[:split_idx].copy()
    eval_df = df.iloc[split_idx:].copy()
    
    print(f"Train set: {len(train_df)} rows ({train_df['timestamp'].min()} to {train_df['timestamp'].max()})")
    print(f"Eval set: {len(eval_df)} rows ({eval_df['timestamp'].min()} to {eval_df['timestamp'].max()})")
    
    return train_df, eval_df


def create_sequences(
    df: pd.DataFrame,
    sequence_length: int = SEQUENCE_LENGTH,
    features: list = None
) -> np.ndarray:
    """
    Create sequences for RL environment states.
    
    Each state is a window of the last N timesteps.
    Shape: (num_samples, sequence_length, num_features)
    
    Args:
        df: DataFrame with feature columns
        sequence_length: Number of timesteps per sequence (default: 12)
        features: Feature columns to include
        
    Returns:
        3D numpy array of shape (samples, sequence_length, features)
    """
    features = features or FEATURES
    
    # Extract feature values
    data = df[features].values
    
    num_samples = len(data) - sequence_length + 1
    num_features = len(features)
    
    # Create sequences using sliding window
    sequences = np.zeros((num_samples, sequence_length, num_features), dtype=np.float32)
    
    for i in range(num_samples):
        sequences[i] = data[i:i + sequence_length]
    
    print(f"Created {num_samples} sequences of shape ({sequence_length}, {num_features})")
    
    return sequences


def get_price_data(
    df: pd.DataFrame,
    sequence_length: int = SEQUENCE_LENGTH
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract price data aligned with sequences for reward calculation.
    
    Returns highs, lows, closes starting from sequence_length index
    (aligned with the sequences).
    
    Args:
        df: DataFrame with OHLCV columns
        sequence_length: Sequence length to align with
        
    Returns:
        Tuple of (highs, lows, closes) numpy arrays
    """
    # Align with sequences (skip first sequence_length-1 rows)
    start_idx = sequence_length - 1
    
    highs = df['high'].values[start_idx:].astype(np.float32)
    lows = df['low'].values[start_idx:].astype(np.float32)
    closes = df['close'].values[start_idx:].astype(np.float32)
    
    return highs, lows, closes


def load_processed_dataset(
    filepath: str,
    features: list = None
) -> pd.DataFrame:
    """
    Load a processed dataset from CSV.
    
    Args:
        filepath: Path to processed CSV file
        features: Optional list of features to validate
        
    Returns:
        DataFrame with processed data
    """
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    if features:
        missing = set(features) - set(df.columns)
        if missing:
            raise ValueError(f"Missing features in dataset: {missing}")
    
    print(f"Loaded {filepath}: {df.shape}")
    return df


def prepare_training_data(
    filepath: str,
    sequence_length: int = SEQUENCE_LENGTH,
    eval_hours: int = EVAL_HOURS,
    features: list = None
) -> dict:
    """
    Prepare all data needed for training in one call.
    
    Args:
        filepath: Path to processed dataset
        sequence_length: Sequence length for states
        eval_hours: Hours to reserve for evaluation
        features: Feature columns
        
    Returns:
        Dictionary with train/eval sequences and price data
    """
    features = features or FEATURES
    
    # Load dataset
    df = load_processed_dataset(filepath, features)
    
    # Split train/eval
    train_df, eval_df = train_eval_split(df, eval_hours, features)
    
    # Create sequences
    train_sequences = create_sequences(train_df, sequence_length, features)
    eval_sequences = create_sequences(eval_df, sequence_length, features)
    
    # Get price data for rewards
    train_highs, train_lows, train_closes = get_price_data(train_df, sequence_length)
    eval_highs, eval_lows, eval_closes = get_price_data(eval_df, sequence_length)
    
    return {
        'train': {
            'sequences': train_sequences,
            'highs': train_highs,
            'lows': train_lows,
            'closes': train_closes,
            'df': train_df
        },
        'eval': {
            'sequences': eval_sequences,
            'highs': eval_highs,
            'lows': eval_lows,
            'closes': eval_closes,
            'df': eval_df
        }
    }


if __name__ == '__main__':
    # Example usage
    from config.config import SUPPORTED_CRYPTOS
    
    # Load and prepare BTC data
    data = prepare_training_data('data/datasets/BTC_processed.csv')
    
    print("\nTraining data:")
    print(f"  Sequences shape: {data['train']['sequences'].shape}")
    print(f"  Price arrays length: {len(data['train']['closes'])}")
    
    print("\nEvaluation data:")
    print(f"  Sequences shape: {data['eval']['sequences'].shape}")
    print(f"  Price arrays length: {len(data['eval']['closes'])}")
