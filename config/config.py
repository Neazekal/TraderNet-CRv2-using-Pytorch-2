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
TIMEFRAME = '1h'                # Candle timeframe (primary for trading)
RATE_LIMIT_MS = 100             # Milliseconds between API requests
DOWNLOAD_SLEEP = 0.1            # Additional sleep between batches (seconds)
CANDLES_PER_REQUEST = 1000      # Max candles per API request

# =============================================================================
# Environment Settings
# =============================================================================
SEQUENCE_LENGTH = 12            # N previous hours for state observation
HORIZON = 20                    # K hours lookahead for reward calculation
FEES = 0.001                    # Transaction fee (0.1% = 0.001 for Binance Futures)

# Action space (Position-based actions)
# Gymnasium uses Discrete(3) with {0, 1, 2}, we map to intuitive position values
# LONG: Open or keep LONG position (flip from SHORT if needed)
# SHORT: Open or keep SHORT position (flip from LONG if needed)
# FLAT: Close any position and go flat
NUM_ACTIONS = 3

# Gymnasium action indices (what the agent outputs)
ACTION_LONG = 0   # Agent outputs 0 for LONG
ACTION_SHORT = 1  # Agent outputs 1 for SHORT
ACTION_FLAT = 2   # Agent outputs 2 for FLAT
ACTION_NAMES = {0: 'LONG', 1: 'SHORT', 2: 'FLAT'}

# Position values (intuitive representation for calculations)
# Sign indicates market direction: +1 bullish, -1 bearish, 0 neutral
POSITION_LONG = 1     # +1 = bullish/long
POSITION_SHORT = -1   # -1 = bearish/short
POSITION_FLAT = 0     # 0 = neutral/no position
POSITION_NAMES = {1: 'LONG', -1: 'SHORT', 0: 'FLAT'}

# Mapping: Gymnasium action -> Position value
ACTION_TO_POSITION = {
    ACTION_LONG: POSITION_LONG,    # 0 -> +1
    ACTION_SHORT: POSITION_SHORT,  # 1 -> -1
    ACTION_FLAT: POSITION_FLAT,    # 2 -> 0
}

# Reverse mapping: Position value -> Gymnasium action
POSITION_TO_ACTION = {
    POSITION_LONG: ACTION_LONG,    # +1 -> 0
    POSITION_SHORT: ACTION_SHORT,  # -1 -> 1
    POSITION_FLAT: ACTION_FLAT,    # 0 -> 2
}

# Legacy aliases for backward compatibility
ACTION_BUY = ACTION_LONG
ACTION_SELL = ACTION_SHORT
ACTION_HOLD = ACTION_FLAT

# =============================================================================
# Capital Management (Position Trading)
# =============================================================================
INITIAL_CAPITAL = 10000.0       # Starting capital in USDT
RISK_PER_TRADE = 0.02           # Risk 2% of capital per trade
LEVERAGE = 10                   # Leverage multiplier (1 = no leverage, 10 = 10x)
MAX_POSITION_SIZE = 0.5         # Max 50% of capital in single position

# Stop-Loss / Take-Profit
STOP_LOSS = 0.02                # 2% stop-loss (auto-close on loss)
TAKE_PROFIT = 0.04              # 4% take-profit (auto-close on profit)
USE_ISOLATED_MARGIN = True      # Isolated margin (max loss = position margin only)

# Slippage Settings (simulates market impact and execution delays)
SLIPPAGE_ENABLED = True         # Enable random slippage simulation
SLIPPAGE_MEAN = 0.0001          # Mean slippage (0.01% = 1 bps)
SLIPPAGE_STD = 0.0002           # Slippage std deviation (0.02% = 2 bps)
SLIPPAGE_MAX = 0.001            # Maximum slippage cap (0.1% = 10 bps)

# =============================================================================
# Feature Engineering
# =============================================================================
NUM_FEATURES = 28  # 19 base features + regime + 8 funding features

# Feature list (19 features + regime + 8 funding)
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
    'adl_diffs2', 'obv_diffs2',
    # Market regime (1) - encoded as 0-3
    'regime_encoded',
    # Funding rate (1) - raw funding rate is already meaningful
    'funding_rate'
]

# Funding feature columns
FUNDING_FEATURES = ['funding_rate']

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
# Market Regime Detection Parameters
# =============================================================================
# Regimes: TRENDING_UP, TRENDING_DOWN, HIGH_VOLATILITY, RANGING
REGIME_PARAMS = {
    'volatility_window': 24,           # Hours for rolling volatility
    'volatility_high_percentile': 75,  # Above this = HIGH_VOLATILITY
    'volatility_low_percentile': 25,   # Below this = low vol (for RANGING)
    'trend_window': 50,                # SMA period for trend direction
    'adx_threshold': 25,               # ADX above this = strong trend
    'lookback_for_percentiles': 720,   # 30 days for adaptive thresholds
}

# =============================================================================
# Funding Rate Parameters
# =============================================================================
FUNDING_PARAMS = {
    'lookback_periods': 21,             # Rolling window for funding stats (7 days of 8h data)
    'high_funding_threshold': 0.001,    # 0.1% per 8h = extreme funding
}

# Funding Fee Settings (for position trading environment)
FUNDING_FEE_ENABLED = True              # Apply funding fees when holding positions

# =============================================================================
# Reward Function Settings
# =============================================================================
# Smurf agent parameters
SMURF_HOLD_REWARD = 0.0055      # Fixed positive reward for HOLD in Smurf

# Reward types
REWARD_MARKET_LIMIT = 'market_limit'
REWARD_SMURF = 'smurf'

# Drawdown Penalty Settings
# Penalizes agent when equity falls below peak (encourages capital preservation)
DRAWDOWN_PENALTY_ENABLED = True
DRAWDOWN_PENALTY_THRESHOLD = 0.05   # Start penalizing at 5% drawdown
DRAWDOWN_PENALTY_SCALE = 0.5        # Penalty multiplier (higher = more penalty)
DRAWDOWN_PENALTY_MAX = 0.1          # Maximum penalty per step

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
OBS_SHAPE = (SEQUENCE_LENGTH, NUM_FEATURES)  # (12, 28)
