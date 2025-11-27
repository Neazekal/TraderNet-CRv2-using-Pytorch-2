"""
Technical Indicators Module.

Computes technical analysis indicators using the `ta` library.
All indicators follow parameters from the original TraderNet-CRv2 paper.

Indicators implemented:
- Trend: DEMA, MACD, ADX, AROON
- Momentum: RSI, STOCH, CCI
- Volatility: Bollinger Bands
- Volume: VWAP, ADL, OBV
"""

import pandas as pd
import numpy as np
from ta.trend import MACD, ADXIndicator, AroonIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from ta.volume import AccDistIndexIndicator, OnBalanceVolumeIndicator


class TechnicalIndicators:
    """
    Calculator for technical analysis indicators.
    
    Uses the `ta` library to compute standard indicators with parameters
    matching the TraderNet-CRv2 paper specifications.
    
    Example:
        ta_calc = TechnicalIndicators()
        df = ta_calc.compute_all(df)  # df must have OHLCV columns
    """
    
    def __init__(
        self,
        dema_window: int = 15,
        vwap_window: int = 10,
        macd_short: int = 12,
        macd_long: int = 26,
        macd_signal: int = 9,
        rsi_window: int = 14,
        stoch_window: int = 14,
        cci_window: int = 20,
        adx_window: int = 14,
        aroon_window: int = 25,
        bbands_window: int = 20
    ):
        """
        Initialize with indicator parameters.
        
        Args:
            dema_window: Double EMA window (default: 15)
            vwap_window: VWAP rolling window (default: 10)
            macd_short: MACD fast period (default: 12)
            macd_long: MACD slow period (default: 26)
            macd_signal: MACD signal period (default: 9)
            rsi_window: RSI period (default: 14)
            stoch_window: Stochastic period (default: 14)
            cci_window: CCI period (default: 20)
            adx_window: ADX period (default: 14)
            aroon_window: Aroon period (default: 25)
            bbands_window: Bollinger Bands period (default: 20)
        """
        self.dema_window = dema_window
        self.vwap_window = vwap_window
        self.macd_short = macd_short
        self.macd_long = macd_long
        self.macd_signal = macd_signal
        self.rsi_window = rsi_window
        self.stoch_window = stoch_window
        self.cci_window = cci_window
        self.adx_window = adx_window
        self.aroon_window = aroon_window
        self.bbands_window = bbands_window
    
    def compute_dema(self, close: pd.Series) -> pd.Series:
        """
        Compute Double Exponential Moving Average (DEMA).
        
        DEMA = 2 * EMA(close) - EMA(EMA(close))
        Reduces lag compared to simple EMA.
        
        Args:
            close: Close price series
            
        Returns:
            DEMA values
        """
        ema1 = EMAIndicator(close=close, window=self.dema_window).ema_indicator()
        ema2 = EMAIndicator(close=ema1, window=self.dema_window).ema_indicator()
        dema = 2 * ema1 - ema2
        return dema
    
    def compute_vwap(
        self, 
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series, 
        volume: pd.Series
    ) -> pd.Series:
        """
        Compute Volume Weighted Average Price (VWAP).
        
        VWAP = cumsum(typical_price * volume) / cumsum(volume)
        Using rolling window instead of cumulative for stationarity.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            volume: Volume series
            
        Returns:
            VWAP values
        """
        typical_price = (high + low + close) / 3
        vwap = (
            (typical_price * volume).rolling(window=self.vwap_window).sum() /
            volume.rolling(window=self.vwap_window).sum()
        )
        return vwap
    
    def compute_macd_signal_diff(self, close: pd.Series) -> pd.Series:
        """
        Compute MACD Signal Line Difference.
        
        MACD = EMA(12) - EMA(26)
        Signal = EMA(MACD, 9)
        Diff = MACD - Signal (histogram)
        
        Args:
            close: Close price series
            
        Returns:
            MACD histogram (MACD - Signal)
        """
        macd = MACD(
            close=close,
            window_slow=self.macd_long,
            window_fast=self.macd_short,
            window_sign=self.macd_signal
        )
        return macd.macd_diff()
    
    def compute_rsi(self, close: pd.Series) -> pd.Series:
        """
        Compute Relative Strength Index (RSI).
        
        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss
        
        Args:
            close: Close price series
            
        Returns:
            RSI values (0-100)
        """
        rsi = RSIIndicator(close=close, window=self.rsi_window)
        return rsi.rsi()
    
    def compute_stoch(
        self, 
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series
    ) -> pd.Series:
        """
        Compute Stochastic Oscillator (%K).
        
        %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            
        Returns:
            Stochastic %K values (0-100)
        """
        stoch = StochasticOscillator(
            high=high,
            low=low,
            close=close,
            window=self.stoch_window,
            smooth_window=3
        )
        return stoch.stoch()
    
    def compute_cci(
        self, 
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series
    ) -> pd.Series:
        """
        Compute Commodity Channel Index (CCI).
        
        CCI = (Typical Price - SMA) / (0.015 * Mean Deviation)
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            
        Returns:
            CCI values
        """
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(window=self.cci_window).mean()
        mean_deviation = typical_price.rolling(window=self.cci_window).apply(
            lambda x: np.abs(x - x.mean()).mean()
        )
        cci = (typical_price - sma) / (0.015 * mean_deviation)
        return cci
    
    def compute_adx(
        self, 
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series
    ) -> pd.Series:
        """
        Compute Average Directional Index (ADX).
        
        Measures trend strength (0-100), not direction.
        ADX > 25 indicates strong trend.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            
        Returns:
            ADX values (0-100)
        """
        adx = ADXIndicator(
            high=high,
            low=low,
            close=close,
            window=self.adx_window
        )
        return adx.adx()
    
    def compute_aroon(
        self, 
        high: pd.Series, 
        low: pd.Series
    ) -> tuple[pd.Series, pd.Series]:
        """
        Compute Aroon Indicator (Up and Down).
        
        Aroon Up = ((window - periods since highest high) / window) * 100
        Aroon Down = ((window - periods since lowest low) / window) * 100
        
        Args:
            high: High price series
            low: Low price series
            
        Returns:
            Tuple of (aroon_up, aroon_down) series
        """
        aroon = AroonIndicator(high=high, low=low, window=self.aroon_window)
        return aroon.aroon_up(), aroon.aroon_down()
    
    def compute_bbands(self, close: pd.Series) -> tuple[pd.Series, pd.Series]:
        """
        Compute Bollinger Bands (Upper and Lower).
        
        Upper = SMA + 2 * std
        Lower = SMA - 2 * std
        
        Args:
            close: Close price series
            
        Returns:
            Tuple of (upper_band, lower_band) series
        """
        bbands = BollingerBands(close=close, window=self.bbands_window, window_dev=2)
        return bbands.bollinger_hband(), bbands.bollinger_lband()
    
    def compute_adl(
        self, 
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series, 
        volume: pd.Series
    ) -> pd.Series:
        """
        Compute Accumulation/Distribution Line (ADL).
        
        ADL = cumsum(((close - low) - (high - close)) / (high - low) * volume)
        Measures money flow based on price position within range.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            volume: Volume series
            
        Returns:
            ADL values
        """
        adl = AccDistIndexIndicator(high=high, low=low, close=close, volume=volume)
        return adl.acc_dist_index()
    
    def compute_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Compute On-Balance Volume (OBV).
        
        OBV adds volume on up days, subtracts on down days.
        Measures buying/selling pressure.
        
        Args:
            close: Close price series
            volume: Volume series
            
        Returns:
            OBV values
        """
        obv = OnBalanceVolumeIndicator(close=close, volume=volume)
        return obv.on_balance_volume()
    
    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all technical indicators and add to DataFrame.
        
        Args:
            df: DataFrame with OHLCV columns (open, high, low, close, volume)
            
        Returns:
            DataFrame with all technical indicators added
        """
        df = df.copy()
        
        # Extract OHLCV columns
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['volume']
        
        # Trend indicators
        df['dema'] = self.compute_dema(close)
        df['macd_signal_diffs'] = self.compute_macd_signal_diff(close)
        df['adx'] = self.compute_adx(high, low, close)
        df['aroon_up'], df['aroon_down'] = self.compute_aroon(high, low)
        
        # Momentum indicators
        df['rsi'] = self.compute_rsi(close)
        df['stoch'] = self.compute_stoch(high, low, close)
        df['cci'] = self.compute_cci(high, low, close)
        
        # Volatility indicators
        df['bband_up'], df['bband_down'] = self.compute_bbands(close)
        
        # Volume indicators
        df['vwap'] = self.compute_vwap(high, low, close, volume)
        df['adl'] = self.compute_adl(high, low, close, volume)
        df['obv'] = self.compute_obv(close, volume)
        
        return df


if __name__ == '__main__':
    # Example usage
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from data.preprocessing.ohlcv import OHLCVPreprocessor
    
    # Load and preprocess OHLCV data
    preprocessor = OHLCVPreprocessor()
    df = preprocessor.load('data/storage/BTC.csv')
    
    # Compute technical indicators
    ta_calc = TechnicalIndicators()
    df = ta_calc.compute_all(df)
    
    print("Columns:", df.columns.tolist())
    print("\nShape:", df.shape)
    print("\nSample indicators:")
    print(df[['timestamp', 'close', 'rsi', 'macd_signal_diffs', 'adx']].tail(10))
