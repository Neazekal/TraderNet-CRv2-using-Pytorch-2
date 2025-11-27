"""
Data Preprocessing Module.

Contains:
- OHLCVPreprocessor: Load and preprocess raw OHLCV data
- TechnicalPreprocessor: Compute derived technical features
- RegimeDetector: Detect market regimes (trending, ranging, high volatility)
- FundingPreprocessor: Process funding rate data from Binance Futures
"""

from data.preprocessing.ohlcv import OHLCVPreprocessor
from data.preprocessing.technical import TechnicalPreprocessor
from data.preprocessing.regime import RegimeDetector, one_hot_encode_regime
from data.preprocessing.funding import FundingPreprocessor, get_funding_feature_columns

__all__ = [
    "OHLCVPreprocessor",
    "TechnicalPreprocessor", 
    "RegimeDetector",
    "one_hot_encode_regime",
    "FundingPreprocessor",
    "get_funding_feature_columns"
]
