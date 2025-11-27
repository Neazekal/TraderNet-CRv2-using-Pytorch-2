"""
TraderNet-CRv2 PyTorch Configuration

Contains all hyperparameters, supported cryptocurrencies, and settings.
Centralized configuration for the entire project.
"""

# =============================================================================
# Supported Cryptocurrencies
# =============================================================================
# Binance USDT-M Futures pairs
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

# =============================================================================
# Data Download Settings
# =============================================================================
EXCHANGE = 'binance'
MARKET_TYPE = 'future'          # 'spot' or 'future'
TIMEFRAME = '1h'                # Candle timeframe
RATE_LIMIT_MS = 100             # Milliseconds between API requests
DOWNLOAD_SLEEP = 0.1            # Additional sleep between batches (seconds)
CANDLES_PER_REQUEST = 1000      # Max candles per API request

# =============================================================================
# Environment Settings
# =============================================================================
SEQUENCE_LENGTH = 12            # N previous hours for state observation
HORIZON = 20                    # K hours lookahead for reward calculation
FEES = 0.001                    # Transaction fee (0.1% = 0.001 for Binance Futures)

# Action space
NUM_ACTIONS = 3
ACTION_BUY = 0
ACTION_SELL = 1
ACTION_HOLD = 2
ACTION_NAMES = {0: 'BUY', 1: 'SELL', 2: 'HOLD'}

# =============================================================================
# Capital Management (Position Trading)
# =============================================================================
INITIAL_CAPITAL = 10000.0       # Starting capital in USDT
RISK_PER_TRADE = 0.02           # Risk 2% of capital per trade
LEVERAGE = 10                   # Leverage multiplier (1 = no leverage, 10 = 10x)
MAX_POSITION_SIZE = 0.5         # Max 50% of capital in single position

# =============================================================================
# Feature Engineering
# =============================================================================
NUM_FEATURES = 19

# Feature list (19 features total)
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

# =============================================================================
# Reward Function Settings
# =============================================================================
# Smurf agent parameters
SMURF_HOLD_REWARD = 0.0055      # Fixed positive reward for HOLD in Smurf

# Reward types
REWARD_MARKET_LIMIT = 'market_limit'
REWARD_SMURF = 'smurf'

# =============================================================================
# Safety Mechanisms
# =============================================================================
# N-Consecutive rule: require N consecutive same actions before executing
N_CONSECUTIVE_WINDOW = 2

# =============================================================================
# PPO Hyperparameters
# =============================================================================
PPO_PARAMS = {
    'learning_rate': 0.0005,
    'epsilon_clip': 0.3,
    'gamma': 0.99,              # Discount factor
    'gae_lambda': 0.95,         # GAE lambda for advantage estimation
    'num_epochs': 40,           # PPO epochs per update
    'batch_size': 128,
    'value_loss_coef': 0.5,     # Value loss coefficient
    'entropy_coef': 0.01,       # Entropy bonus coefficient
    'max_grad_norm': 0.5,       # Gradient clipping
}

# =============================================================================
# Network Architecture
# =============================================================================
NETWORK_PARAMS = {
    'conv_filters': 32,
    'conv_kernel': 3,
    'fc_layers': [256, 256],
    'activation': 'gelu',
    'dropout': 0.0,             # Dropout rate (0 = no dropout)
}

# =============================================================================
# Training Settings
# =============================================================================
TRAINING_PARAMS = {
    'total_timesteps': 1_000_000,   # Total training steps
    'eval_freq': 10_000,            # Evaluate every N steps
    'save_freq': 50_000,            # Save checkpoint every N steps
    'log_freq': 1000,               # Log metrics every N steps
    'seed': 42,                     # Random seed for reproducibility
}

# =============================================================================
# Evaluation Settings
# =============================================================================
EVAL_HOURS = 2250               # ~3 months for evaluation (last N hours)
EVAL_EPISODES = 1               # Number of evaluation episodes

# =============================================================================
# Paths
# =============================================================================
DATA_DIR = 'data/storage/'
DATASET_DIR = 'data/datasets/'
CHECKPOINT_DIR = 'checkpoints/'
LOG_DIR = 'logs/'

# =============================================================================
# Derived Constants (computed from above)
# =============================================================================
# Observation shape for neural network input
OBS_SHAPE = (SEQUENCE_LENGTH, NUM_FEATURES)  # (12, 19)
