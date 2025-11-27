# TraderNet-CRv2 PyTorch

A PyTorch implementation of **TraderNet-CRv2** - a Deep Reinforcement Learning system for cryptocurrency trading that combines PPO with technical analysis and safety mechanisms.

Based on the paper: *"Combining deep reinforcement learning with technical analysis and trend monitoring on cryptocurrency markets"* (Neural Computing and Applications, 2023)

Original TensorFlow implementation: [kochlisGit/TraderNet-CRv2](https://github.com/kochlisGit/TraderNet-CRv2)

---

## Features

- **PPO Agent**: Proximal Policy Optimization for trading decisions (BUY/SELL/HOLD)
- **Technical Analysis**: 11 indicators (MACD, RSI, Bollinger Bands, ADX, etc.)
- **N-Consecutive Rule**: Safety mechanism requiring N consecutive same actions
- **Smurf Integration**: Conservative secondary agent for risk management
- **Multi-Crypto Support**: BTC, ETH, XRP, SOL, BNB, TRX, DOGE

---

## Project Structure

```
tradernet-pytorch/
├── config/
│   └── config.py              # Centralized hyperparameters & settings
├── data/
│   ├── downloaders/
│   │   └── binance.py         # CCXT Binance Futures downloader
│   ├── preprocessing/
│   │   ├── ohlcv.py           # Log returns & hour extraction
│   │   └── technical.py       # Derived features
│   ├── datasets/
│   │   ├── builder.py         # Dataset building pipeline
│   │   └── utils.py           # Train/eval split utilities
│   └── storage/               # Raw OHLCV data (gitignored)
├── analysis/
│   └── technical/
│       └── indicators_calc.py # Technical indicator calculations
├── environments/
│   ├── trading_env.py         # Gymnasium trading environment
│   └── rewards/
│       ├── base.py            # Base reward class
│       ├── market_limit.py    # MarketLimitOrder reward
│       └── smurf.py           # Smurf conservative reward
├── agents/                    # (Phase 4+) PPO agent & networks
├── rules/                     # (Phase 4+) N-Consecutive & Smurf
├── metrics/                   # (Phase 4+) Trading metrics
├── checkpoints/               # Model checkpoints (gitignored)
├── IMPLEMENTATION_PLAN.md     # Detailed implementation plan
└── requirements.txt           # Python dependencies
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Neazekal/TraderNet-CRv2-using-Pytorch-2.git
cd TraderNet-CRv2-using-Pytorch-2
```

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Phase 1: Data Download

Download historical OHLCV data from Binance Futures.

### Download all supported cryptocurrencies

```bash
python -m data.downloaders.binance
```

This downloads hourly candles for all 7 cryptos from their Futures listing date to today.

### Download a single cryptocurrency

```python
from data.downloaders.binance import BinanceDownloader

# Download BTC from 2023
downloader = BinanceDownloader('BTC/USDT')
downloader.download(start_date='2023-01-01')
downloader.save('data/storage/BTC.csv')
```

### Download with custom date range

```python
downloader = BinanceDownloader('ETH/USDT')
downloader.download(start_date='2022-01-01', end_date='2023-12-31')
downloader.save('data/storage/ETH_2022_2023.csv')
```

### Update existing data

```python
downloader = BinanceDownloader('BTC/USDT')
downloader.update('data/storage/BTC.csv')  # Fetches new candles since last timestamp
```

### Supported Cryptocurrencies

| Symbol | Pair | Data Start |
|--------|------|------------|
| BTC | BTC/USDT | 2019 |
| ETH | ETH/USDT | 2019 |
| XRP | XRP/USDT | 2020 |
| SOL | SOL/USDT | 2021 |
| BNB | BNB/USDT | 2020 |
| TRX | TRX/USDT | 2020 |
| DOGE | DOGE/USDT | 2021 |

---

## Phase 2: Data Preprocessing

Process raw OHLCV data into training-ready features.

### Build all datasets

```bash
python -m data.datasets.builder
```

This processes all cryptocurrencies and saves:
- `data/datasets/{CRYPTO}_processed.csv` - Scaled features
- `data/datasets/{CRYPTO}_processed.scaler.pkl` - Fitted scaler

### Build a single dataset

```python
from data.datasets.builder import DatasetBuilder

builder = DatasetBuilder()
df, scaler = builder.build('data/storage/BTC.csv')
builder.save('data/datasets/BTC_processed.csv')
```

### Preprocessing Pipeline

The pipeline performs these steps:

1. **Load raw OHLCV** - timestamp, open, high, low, close, volume
2. **Calculate log returns** - Stationary price features
3. **Extract hour** - Temporal pattern (0-23)
4. **Compute technical indicators** - 11 indicators
5. **Compute derived features** - Price relative positions, volume acceleration
6. **Min-Max scaling** - Normalize to [0, 1]
7. **Drop NaN rows** - Remove indicator warm-up period

### Features (19 total)

| Category | Features |
|----------|----------|
| Log Returns (5) | open, high, low, close, volume |
| Time (1) | hour |
| Trend (4) | macd_signal_diffs, aroon_up, aroon_down, adx |
| Momentum (3) | stoch, rsi, cci |
| Price Relative (4) | close_dema, close_vwap, bband_up_close, close_bband_down |
| Volume (2) | adl_diffs2, obv_diffs2 |

### Prepare training data

```python
from data.datasets.utils import prepare_training_data

# Load processed dataset and split into train/eval
data = prepare_training_data('data/datasets/BTC_processed.csv')

# Access training sequences
train_sequences = data['train']['sequences']  # Shape: (N, 12, 19)
train_closes = data['train']['closes']        # For reward calculation

# Access evaluation sequences
eval_sequences = data['eval']['sequences']
eval_closes = data['eval']['closes']
```

### Train/Eval Split

- **Training**: All data except last 2250 hours
- **Evaluation**: Last 2250 hours (~3 months)

### Sequence Format

Each state is a sliding window of 12 hourly timesteps with 19 features:
- Shape: `(num_samples, 12, 19)`
- Used as input to the Conv1D neural network

---

## Phase 3: Trading Environment

Two Gymnasium-compatible trading environments for reinforcement learning.

### 1. TradingEnv (Single-Step Rewards)

Original paper implementation - each step is independent, rewards based on potential profit.

```python
from environments.trading_env import create_trading_env

env = create_trading_env('data/datasets/BTC_processed.csv', reward_type='market_limit')
```

### 2. PositionTradingEnv (Realistic Trading)

Position-based environment that simulates real trading:
- Must open position before closing
- HOLD keeps position open
- Actual P&L calculated on close with fees

```python
from environments.position_trading_env import create_position_trading_env

env = create_position_trading_env('data/datasets/BTC_processed.csv')

obs, info = env.reset()

# Open LONG position
obs, reward, _, _, info = env.step(0)  # BUY -> LONG
print(info['position'])  # 'LONG'

# Hold position
obs, reward, _, _, info = env.step(2)  # HOLD
print(info['unrealized_pnl'])  # Current P&L

# Close position
obs, reward, _, _, info = env.step(1)  # SELL -> FLAT
print(reward)  # Realized P&L with fees
```

### Position States

```
FLAT  + BUY  -> LONG   (open long)
FLAT  + SELL -> SHORT  (open short)
LONG  + HOLD -> LONG   (keep holding)
LONG  + SELL -> FLAT   (close long, get P&L)
SHORT + HOLD -> SHORT  (keep holding)
SHORT + BUY  -> FLAT   (close short, get P&L)
```

### Environment Comparison

| Feature | TradingEnv | PositionTradingEnv |
|---------|------------|-------------------|
| Position tracking | No | Yes (FLAT/LONG/SHORT) |
| HOLD behavior | Independent step | Maintains position |
| Rewards | Potential profit | Actual P&L on close |
| Fees | Once per step | On open + close |
| Use case | Paper replication | Realistic trading |

### Reward Functions (TradingEnv only)

| Type | HOLD Reward | Use Case |
|------|-------------|----------|
| `market_limit` | Negative (penalize inaction) | Main TraderNet agent |
| `smurf` | +0.0055 (encourage holding) | Conservative Smurf agent |

---

## Configuration

All hyperparameters are centralized in `config/config.py`:

```python
# Data Download Settings
EXCHANGE = 'binance'
MARKET_TYPE = 'future'        # 'spot' or 'future'
TIMEFRAME = '1h'

# Environment Settings
SEQUENCE_LENGTH = 12          # State window size (N hours)
HORIZON = 20                  # Reward lookahead (K hours)
FEES = 0.01                   # Transaction fee (1%)
NUM_ACTIONS = 3               # BUY, SELL, HOLD

# Feature Engineering
NUM_FEATURES = 19
OBS_SHAPE = (12, 19)          # Observation shape for neural network

# Reward Settings
SMURF_HOLD_REWARD = 0.0055    # Fixed HOLD reward for Smurf agent

# PPO Hyperparameters
PPO_PARAMS = {
    'learning_rate': 0.0005,
    'epsilon_clip': 0.3,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'num_epochs': 40,
    'batch_size': 128,
    'value_loss_coef': 0.5,
    'entropy_coef': 0.01,
    'max_grad_norm': 0.5,
}

# Network Architecture
NETWORK_PARAMS = {
    'conv_filters': 32,
    'conv_kernel': 3,
    'fc_layers': [256, 256],
    'activation': 'gelu',
}

# Training Settings
TRAINING_PARAMS = {
    'total_timesteps': 1_000_000,
    'eval_freq': 10_000,
    'save_freq': 50_000,
    'seed': 42,
}

# Paths
DATA_DIR = 'data/storage/'
DATASET_DIR = 'data/datasets/'
CHECKPOINT_DIR = 'checkpoints/'
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download data (takes ~10-15 minutes for all cryptos)
python -m data.downloaders.binance

# 3. Process data (takes ~2-3 minutes)
python -m data.datasets.builder

# 4. Test the trading environment
python -c "
from environments.trading_env import create_trading_env

env = create_trading_env('data/datasets/BTC_processed.csv')
print(f'Observation space: {env.observation_space}')
print(f'Action space: {env.action_space}')
print(f'Episode length: {env.episode_length}')

# Run a few steps
obs, info = env.reset()
for _ in range(10):
    action = env.action_space.sample()
    obs, reward, done, _, info = env.step(action)
    print(f'Action: {info[\"action\"]}, Reward: {reward:.4f}')
"
```

Expected output:
```
Observation space: Box(0.0, 1.0, (12, 19), float32)
Action space: Discrete(3)
Episode length: 54451
Action: HOLD, Reward: -0.0155
Action: BUY, Reward: 0.0123
...
```

---

## Roadmap

- [x] **Phase 1**: Project setup & Binance data downloader
- [x] **Phase 2**: Technical analysis & preprocessing pipeline
- [x] **Phase 3**: Trading environment & reward functions
- [ ] **Phase 4**: Neural networks (Actor/Critic)
- [ ] **Phase 5**: PPO agent implementation
- [ ] **Phase 6**: Safety mechanisms (N-Consecutive, Smurf)
- [ ] **Phase 7**: Training & evaluation scripts
- [ ] **Phase 8**: Metrics & visualization

---

## References

1. Kochliaridis et al. (2023) - *"Combining deep reinforcement learning with technical analysis and trend monitoring on cryptocurrency markets"*
2. Schulman et al. (2017) - *"Proximal Policy Optimization Algorithms"*
3. Original implementation: [kochlisGit/TraderNet-CRv2](https://github.com/kochlisGit/TraderNet-CRv2)

---

## License

This project is for educational and research purposes.

---

## Contributing

Contributions are welcome! Please read the [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for details on the architecture and planned features.
