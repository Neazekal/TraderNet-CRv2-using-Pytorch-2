"""
Binance OHLCV Data Downloader using CCXT.

Downloads hourly candlestick data from Binance exchange.
Handles pagination for large date ranges and supports resuming downloads.
"""

import ccxt
import pandas as pd
import time
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import (
    RATE_LIMIT_MS, MARKET_TYPE, CANDLES_PER_REQUEST, 
    DOWNLOAD_SLEEP, SUPPORTED_CRYPTOS, TIMEFRAME
)


class BinanceDownloader:
    """
    CCXT-based Binance downloader for OHLCV data.
    
    Features:
    - Download hourly candles for any supported crypto pair
    - Handle pagination for large date ranges
    - Save to CSV format
    - Resume from last downloaded timestamp
    
    Example:
        downloader = BinanceDownloader('BTC/USDT')
        df = downloader.download(start_date='2020-01-01')
        downloader.save('data/storage/BTC.csv')
    """
    
    OHLCV_COLUMNS = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    def __init__(self, symbol: str, timeframe: str = '1h'):
        """
        Initialize downloader with trading pair.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe (default: '1h')
        """
        self.symbol = symbol
        self.timeframe = timeframe
        # Binance Futures rate limit: 2400 weight/min
        # fetch_ohlcv uses ~5 weight per call
        # Safe: 2400/5 = 480 calls/min = 8 calls/sec = 125ms between calls
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'rateLimit': RATE_LIMIT_MS,
            'options': {'defaultType': MARKET_TYPE}
        })
        self.data: pd.DataFrame = pd.DataFrame()
        
    def _parse_date(self, date_str: str) -> int:
        """Convert date string to milliseconds timestamp."""
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        return int(dt.timestamp() * 1000)
    
    def _get_timeframe_ms(self) -> int:
        """Get timeframe duration in milliseconds."""
        timeframe_map = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
        }
        return timeframe_map.get(self.timeframe, 60 * 60 * 1000)
    
    def download(
        self, 
        start_date: str, 
        end_date: str = None,
        limit_per_request: int = CANDLES_PER_REQUEST,
        sleep_seconds: float = DOWNLOAD_SLEEP
    ) -> pd.DataFrame:
        """
        Download OHLCV data between dates.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format (default: now)
            limit_per_request: Max candles per API request (default: 1000)
            sleep_seconds: Sleep between requests to avoid rate limits
            
        Returns:
            DataFrame with OHLCV data
        """
        since = self._parse_date(start_date)
        until = self._parse_date(end_date) if end_date else int(time.time() * 1000)
        
        timeframe_ms = self._get_timeframe_ms()
        total_candles = (until - since) // timeframe_ms
        
        all_ohlcv = []
        current = since
        
        with tqdm(total=total_candles, desc=f"Downloading {self.symbol}") as pbar:
            while current < until:
                try:
                    ohlcv = self.exchange.fetch_ohlcv(
                        symbol=self.symbol,
                        timeframe=self.timeframe,
                        since=current,
                        limit=limit_per_request
                    )
                    
                    if not ohlcv:
                        break
                    
                    all_ohlcv.extend(ohlcv)
                    
                    # Update progress
                    pbar.update(len(ohlcv))
                    
                    # Move to next batch
                    current = ohlcv[-1][0] + timeframe_ms
                    
                    # Rate limiting
                    time.sleep(sleep_seconds)
                    
                except ccxt.NetworkError as e:
                    print(f"Network error: {e}. Retrying in 5 seconds...")
                    time.sleep(5)
                except ccxt.ExchangeError as e:
                    print(f"Exchange error: {e}")
                    break
        
        # Convert to DataFrame
        self.data = pd.DataFrame(all_ohlcv, columns=self.OHLCV_COLUMNS)
        
        # Convert timestamp to datetime
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'], unit='ms')
        
        # Remove duplicates and sort
        self.data = self.data.drop_duplicates(subset=['timestamp'])
        self.data = self.data.sort_values('timestamp').reset_index(drop=True)
        
        # Filter to date range
        if end_date:
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            self.data = self.data[self.data['timestamp'] < end_dt]
        
        print(f"Downloaded {len(self.data)} candles for {self.symbol}")
        return self.data
    
    def save(self, filepath: str) -> None:
        """
        Save downloaded data to CSV.
        
        Args:
            filepath: Path to save CSV file
        """
        if self.data.empty:
            raise ValueError("No data to save. Call download() first.")
        
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        self.data.to_csv(filepath, index=False)
        print(f"Saved {len(self.data)} rows to {filepath}")
    
    def load(self, filepath: str) -> pd.DataFrame:
        """
        Load existing data from CSV.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame with OHLCV data
        """
        self.data = pd.read_csv(filepath)
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        print(f"Loaded {len(self.data)} rows from {filepath}")
        return self.data
    
    def update(self, filepath: str) -> pd.DataFrame:
        """
        Update existing data file with new candles.
        
        Args:
            filepath: Path to existing CSV file
            
        Returns:
            Updated DataFrame
        """
        # Load existing data
        existing = self.load(filepath)
        
        # Get last timestamp
        last_timestamp = existing['timestamp'].max()
        start_date = (last_timestamp + pd.Timedelta(hours=1)).strftime('%Y-%m-%d')
        
        # Download new data
        new_data = self.download(start_date=start_date)
        
        if not new_data.empty:
            # Combine and deduplicate
            self.data = pd.concat([existing, new_data], ignore_index=True)
            self.data = self.data.drop_duplicates(subset=['timestamp'])
            self.data = self.data.sort_values('timestamp').reset_index(drop=True)
            
            # Save updated data
            self.save(filepath)
        
        return self.data


def download_all_cryptos(
    cryptos: dict,
    output_dir: str = 'data/storage/',
    timeframe: str = '1h'
) -> None:
    """
    Download OHLCV data for all supported cryptocurrencies.
    
    Args:
        cryptos: Dictionary of crypto configs (from config.py)
        output_dir: Directory to save CSV files
        timeframe: Candle timeframe
    """
    for name, config in cryptos.items():
        print(f"\n{'='*50}")
        print(f"Downloading {name}...")
        print(f"{'='*50}")
        
        downloader = BinanceDownloader(
            symbol=config['symbol'],
            timeframe=timeframe
        )
        
        start_date = f"{config['start_year']}-01-01"
        downloader.download(start_date=start_date)
        downloader.save(f"{output_dir}/{name}.csv")


if __name__ == '__main__':
    # Example usage
    # Download single crypto
    # downloader = BinanceDownloader('BTC/USDT')
    # downloader.download(start_date='2023-01-01')
    # downloader.save('data/storage/BTC.csv')
    
    # Download all supported cryptos
    download_all_cryptos(SUPPORTED_CRYPTOS)
