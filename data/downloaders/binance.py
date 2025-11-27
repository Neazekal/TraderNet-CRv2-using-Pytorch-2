"""
Binance OHLCV and Funding Rate Data Downloader using CCXT.

Downloads hourly candlestick data and funding rate history from Binance Futures.
Handles pagination for large date ranges and supports resuming downloads.
Uses API keys from .env file for authenticated endpoints.
"""

import ccxt
import pandas as pd
import time
import os
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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
    FUNDING_COLUMNS = ['timestamp', 'symbol', 'funding_rate', 'funding_timestamp']
    
    def __init__(self, symbol: str, timeframe: str = '1h', use_api_keys: bool = False):
        """
        Initialize downloader with trading pair.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe (default: '1h')
            use_api_keys: Whether to use API keys from .env (required for some endpoints)
        """
        self.symbol = symbol
        self.timeframe = timeframe
        
        # Exchange configuration
        exchange_config = {
            'enableRateLimit': True,
            'rateLimit': 100,  # ms between requests (CCXT handles this)
            'options': {'defaultType': 'future'}  # Use USDT-M Futures
        }
        
        # Add API keys if requested (needed for funding rate history)
        if use_api_keys:
            api_key = os.getenv('BINANCE_API_KEY')
            api_secret = os.getenv('BINANCE_API_SECRET')
            if api_key and api_secret:
                exchange_config['apiKey'] = api_key
                exchange_config['secret'] = api_secret
            else:
                print("Warning: API keys not found in .env file. Some features may not work.")
        
        # Binance Futures rate limit: 2400 weight/min
        # fetch_ohlcv uses ~5 weight per call
        # Safe: 2400/5 = 480 calls/min = 8 calls/sec = 125ms between calls
        # Using 100ms (rateLimit) for CCXT's built-in throttling
        self.exchange = ccxt.binance(exchange_config)
        self.data: pd.DataFrame = pd.DataFrame()
        self.funding_data: pd.DataFrame = pd.DataFrame()
        
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
    
    def download_funding_rate(
        self, 
        start_date: str, 
        end_date: str = None,
        limit_per_request: int = 1000,
        sleep_seconds: float = 0.15
    ) -> pd.DataFrame:
        """
        Download funding rate history for the symbol.
        
        Funding rates are collected every 8 hours on Binance Futures.
        Requires API keys to be set in .env file.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format (default: now)
            limit_per_request: Max records per API request (default: 1000)
            sleep_seconds: Sleep between requests to avoid rate limits
            
        Returns:
            DataFrame with funding rate data
        """
        since = self._parse_date(start_date)
        until = self._parse_date(end_date) if end_date else int(time.time() * 1000)
        
        # Funding rate interval is 8 hours (28800000 ms)
        funding_interval_ms = 8 * 60 * 60 * 1000
        total_records = (until - since) // funding_interval_ms
        
        all_funding = []
        current = since
        
        # Convert symbol format for API (BTC/USDT -> BTCUSDT)
        api_symbol = self.symbol.replace('/', '')
        
        with tqdm(total=total_records, desc=f"Downloading {self.symbol} funding rates") as pbar:
            while current < until:
                try:
                    # Use Binance's fundingRate endpoint via CCXT
                    params = {
                        'symbol': api_symbol,
                        'startTime': current,
                        'endTime': until,
                        'limit': limit_per_request
                    }
                    
                    # Fetch funding rate history using fapiPublic endpoint
                    response = self.exchange.fapiPublicGetFundingRate(params)
                    
                    if not response:
                        break
                    
                    all_funding.extend(response)
                    
                    # Update progress
                    pbar.update(len(response))
                    
                    if len(response) < limit_per_request:
                        # No more data
                        break
                    
                    # Move to next batch (use last fundingTime + 1ms)
                    current = int(response[-1]['fundingTime']) + 1
                    
                    # Rate limiting
                    time.sleep(sleep_seconds)
                    
                except ccxt.NetworkError as e:
                    print(f"Network error: {e}. Retrying in 5 seconds...")
                    time.sleep(5)
                except ccxt.ExchangeError as e:
                    print(f"Exchange error: {e}")
                    break
                except Exception as e:
                    print(f"Error fetching funding rate: {e}")
                    break
        
        if not all_funding:
            print(f"No funding rate data found for {self.symbol}")
            self.funding_data = pd.DataFrame(columns=['timestamp', 'funding_rate'])
            return self.funding_data
        
        # Convert to DataFrame
        self.funding_data = pd.DataFrame(all_funding)
        
        # Rename and format columns
        self.funding_data = self.funding_data.rename(columns={
            'fundingTime': 'timestamp',
            'fundingRate': 'funding_rate'
        })
        
        # Keep only relevant columns
        self.funding_data = self.funding_data[['timestamp', 'funding_rate']]
        
        # Convert timestamp to datetime
        self.funding_data['timestamp'] = pd.to_datetime(
            self.funding_data['timestamp'].astype(int), unit='ms'
        )
        
        # Convert funding_rate to float
        self.funding_data['funding_rate'] = self.funding_data['funding_rate'].astype(float)
        
        # Remove duplicates and sort
        self.funding_data = self.funding_data.drop_duplicates(subset=['timestamp'])
        self.funding_data = self.funding_data.sort_values('timestamp').reset_index(drop=True)
        
        print(f"Downloaded {len(self.funding_data)} funding rate records for {self.symbol}")
        return self.funding_data
    
    def save_funding(self, filepath: str) -> None:
        """
        Save funding rate data to CSV.
        
        Args:
            filepath: Path to save CSV file
        """
        if self.funding_data.empty:
            raise ValueError("No funding data to save. Call download_funding_rate() first.")
        
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        self.funding_data.to_csv(filepath, index=False)
        print(f"Saved {len(self.funding_data)} funding rate rows to {filepath}")
    
    def download_all(
        self, 
        start_date: str, 
        end_date: str = None,
        include_funding: bool = True
    ) -> tuple:
        """
        Download both OHLCV and funding rate data.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format (default: now)
            include_funding: Whether to download funding rate data
            
        Returns:
            Tuple of (ohlcv_df, funding_df) DataFrames
        """
        # Download OHLCV data
        ohlcv_df = self.download(start_date=start_date, end_date=end_date)
        
        # Download funding rate data if requested
        funding_df = pd.DataFrame()
        if include_funding:
            funding_df = self.download_funding_rate(start_date=start_date, end_date=end_date)
        
        return ohlcv_df, funding_df
    
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
    timeframe: str = '1h',
    include_funding: bool = True
) -> None:
    """
    Download OHLCV and funding rate data for all supported cryptocurrencies.
    
    Args:
        cryptos: Dictionary of crypto configs (from config.py)
        output_dir: Directory to save CSV files
        timeframe: Candle timeframe (default: 1h)
        include_funding: Whether to download funding rate data
    """
    for name, config in cryptos.items():
        print(f"\n{'='*50}")
        print(f"Downloading {name}...")
        print(f"{'='*50}")
        
        downloader = BinanceDownloader(
            symbol=config['symbol'],
            timeframe=timeframe,
            use_api_keys=include_funding  # Use API keys if downloading funding rates
        )
        
        start_date = f"{config['start_year']}-01-01"
        
        # Download OHLCV data
        downloader.download(start_date=start_date)
        downloader.save(f"{output_dir}/{name}.csv")
        
        # Download funding rate data if requested
        if include_funding:
            try:
                downloader.download_funding_rate(start_date=start_date)
                downloader.save_funding(f"{output_dir}/{name}_funding.csv")
            except Exception as e:
                print(f"Warning: Could not download funding rate for {name}: {e}")


if __name__ == '__main__':
    # Example usage
    # Download single crypto with funding rate
    # downloader = BinanceDownloader('BTC/USDT', use_api_keys=True)
    # downloader.download_all(start_date='2023-01-01')
    # downloader.save('data/storage/BTC.csv')
    # downloader.save_funding('data/storage/BTC_funding.csv')
    
    # Download all supported cryptos (1h timeframe only)
    download_all_cryptos(SUPPORTED_CRYPTOS, include_funding=True)
