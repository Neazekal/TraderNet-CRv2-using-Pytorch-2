"""
Data Preprocessing Module.

Contains:
- OHLCVPreprocessor: Load and preprocess raw OHLCV data
- TechnicalPreprocessor: Compute derived technical features
- RegimeDetector: Detect market regimes (trending, ranging, high volatility)
"""

from data.preprocessing.ohlcv import OHLCVPreprocessor
from data.preprocessing.technical import TechnicalPreprocessor
from data.preprocessing.regime import RegimeDetector, one_hot_encode_regime

__all__ = [
    "OHLCVPreprocessor",
    "TechnicalPreprocessor", 
    "RegimeDetector",
    "one_hot_encode_regime"
]
