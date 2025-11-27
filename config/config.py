"""
TraderNet-CRv2 PyTorch Configuration

Contains all hyperparameters, supported cryptocurrencies, and settings.
"""

# Supported cryptocurrencies (Binance USDT-M Futures)
# Start years based on Binance Futures listing dates (Futures launched Sep 2019)
SUPPORTED_CRYPTOS = {
    'BTC': {'symbol': 'BTC/USDT', 'start_year': 2019},   # Futures launched Sep 2019
    'ETH': {'symbol': 'ETH/USDT', 'start_year': 2019},   # Listed since Futures launch
    'XRP': {'symbol': 'XRP/USDT', 'start_year': 2020},   # Futures pair added 2020
    'SOL': {'symbol': 'SOL/USDT', 'start_year': 2021},   # Futures listed 2021
    'BNB': {'symbol': 'BNB/USDT', 'start_year': 2020},   # Futures pair added 2020
    'TRX': {'symbol': 'TRX/USDT', 'start_year': 2020},   # Futures listed 2020
    'DOGE': {'symbol': 'DOGE/USDT', 'start_year': 2021}, # Futures listed Apr 2021
}

# Data settings
TIMEFRAME = '1h'
SEQUENCE_LENGTH = 12      # N previous hours for state
HORIZON = 20              # K hours lookahead for reward
FEES = 0.01               # 1% transaction fee

# Feature list (19 features)
FEATURES = [
    # Log returns (5)
    'open_log_returns', 'high_log_returns', 'low_log_returns',
    'close_log_returns', 'volume_log_returns',
    # Time (1)
    'hour',
    # Technical indicators - Trend (4)
    'macd_signal_diffs', 'aroon_up', 'aroon_down', 'adx',
    # Technical indicators - Momentum (3)
    'stoch', 'rsi', 'cci',
    # Technical indicators - Price relative (4)
    'close_dema', 'close_vwap', 'bband_up_close', 'close_bband_down',
    # Technical indicators - Volume (2)
    'adl_diffs2', 'obv_diffs2'
]

# Technical indicator window parameters
TA_PARAMS = {
    'dema_window': 15,
    'vwap_window': 10,
    'macd_short': 12,
    'macd_long': 26,
    'macd_signal': 9,
    'rsi_window': 14,
    'stoch_window': 14,
    'cci_window': 20,
    'adx_window': 14,
    'aroon_window': 25,
    'bbands_window': 20,
}

# PPO hyperparameters
PPO_PARAMS = {
    'learning_rate': 0.0005,
    'epsilon_clip': 0.3,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'num_epochs': 40,
    'batch_size': 128,
}

# Network architecture
NETWORK_PARAMS = {
    'conv_filters': 32,
    'conv_kernel': 3,
    'fc_layers': [256, 256],
    'activation': 'gelu',
}

# Smurf parameters
SMURF_HOLD_REWARD = 0.0055

# N-Consecutive parameters
N_CONSECUTIVE_WINDOW = 2

# Paths
DATA_DIR = 'data/storage/'
CHECKPOINT_DIR = 'checkpoints/'

# Evaluation settings
EVAL_HOURS = 2250  # ~3 months for evaluation
