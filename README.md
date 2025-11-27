# TraderNet-CRv2 PyTorch

A PyTorch implementation of **TraderNet-CRv2** - a Deep Reinforcement Learning system for cryptocurrency trading that combines PPO with technical analysis and safety mechanisms.

Based on the paper: *"Combining deep reinforcement learning with technical analysis and trend monitoring on cryptocurrency markets"* (Neural Computing and Applications, 2023)

Original TensorFlow implementation: [kochlisGit/TraderNet-CRv2](https://github.com/kochlisGit/TraderNet-CRv2)

---

## Features

- **PPO Agent**: Proximal Policy Optimization for trading decisions (LONG/SHORT/FLAT)
- **Technical Analysis**: 11 indicators (MACD, RSI, Bollinger Bands, ADX, etc.)
- **Realistic Trading**: Position-based environment with capital management
- **Risk Management**: Stop-Loss, Take-Profit, and leverage support
- **Instant Position Flip**: Switch from LONG to SHORT (or vice versa) in one step
- **N-Consecutive Rule**: Safety mechanism requiring N consecutive same actions
- **Smurf Integration**: Conservative secondary agent for risk management
- **Multi-Crypto Support**: BTC, ETH, XRP, SOL, BNB, TRX, DOGE

---

## Project Structure

```
tradernet-pytorch/
├── config/
│   └── config.py                  # Centralized hyperparameters & settings
├── data/
│   ├── downloaders/
│   │   └── binance.py             # CCXT Binance Futures downloader
│   ├── preprocessing/
│   │   ├── ohlcv.py               # Log returns & hour extraction
│   │   ├── technical.py           # Derived features
│   │   └── regime.py              # Market regime detection
│   ├── datasets/
│   │   ├── builder.py             # Dataset building pipeline
│   │   └── utils.py               # Train/eval split utilities
│   └── storage/                   # Raw OHLCV data (gitignored)
├── analysis/
│   └── technical/
│       └── indicators_calc.py     # Technical indicator calculations
├── environments/
│   ├── trading_env.py             # Paper replication environment
│   ├── position_trading_env.py    # Realistic trading environment
│   └── rewards/
│       ├── base.py                # Base reward class
│       ├── market_limit.py        # MarketLimitOrder reward
│       └── smurf.py               # Smurf conservative reward
├── agents/                        # (Phase 4+) PPO agent & networks
├── rules/                         # (Phase 4+) N-Consecutive & Smurf
├── metrics/                       # (Phase 4+) Trading metrics
├── checkpoints/                   # Model checkpoints (gitignored)
├── IMPLEMENTATION_PLAN.md         # Detailed implementation plan
└── requirements.txt               # Python dependencies
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

Download historical OHLCV and funding rate data from Binance Futures.

### Download all supported cryptocurrencies

```bash
python -m data.downloaders.binance
```

This downloads:
- **OHLCV data**: Hourly candles (open, high, low, close, volume)
- **Funding rate data**: 8-hour funding rates (collected every 8 hours)

Data is fetched for all 7 cryptos from their Futures listing date to today.

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

### Data Files

After download, the following files are saved:
- `data/storage/{CRYPTO}.csv` - OHLCV data (hourly candles)
- `data/storage/{CRYPTO}_funding.csv` - Funding rate data (8-hour intervals)

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
6. **Detect market regime** - Classify market conditions (4 regimes)
7. **Merge funding rate** - 8-hour funding data forward-filled to hourly
8. **Min-Max scaling** - Normalize to [0, 1]
9. **Drop NaN rows** - Remove indicator warm-up period

### Market Regime Detection

The system classifies market conditions into 4 regimes using rolling volatility and trend strength:

| Regime | Description | Detection Criteria |
|--------|-------------|-------------------|
| TRENDING_UP | Strong uptrend | ADX > 25 and price above 50-period SMA |
| TRENDING_DOWN | Strong downtrend | ADX > 25 and price below 50-period SMA |
| HIGH_VOLATILITY | Choppy/uncertain | Volatility in top 25% of recent history |
| RANGING | Sideways market | Low volatility with weak trend |

This lightweight approach (no HMM or complex models) provides regime awareness for the agent to:
- Adapt strategy based on market conditions
- Avoid overtrading in ranging markets
- Be more aggressive in trending markets
- Be cautious in high volatility periods

### Features (21 total)

| Category | Features |
|----------|----------|
| Log Returns (5) | open, high, low, close, volume |
| Time (1) | hour |
| Trend (4) | macd_signal_diffs, aroon_up, aroon_down, adx |
| Momentum (3) | stoch, rsi, cci |
| Price Relative (4) | close_dema, close_vwap, bband_up_close, close_bband_down |
| Volume (2) | adl_diffs2, obv_diffs2 |
| Regime (1) | regime_encoded (0=TRENDING_UP, 1=TRENDING_DOWN, 2=HIGH_VOLATILITY, 3=RANGING) |
| Funding (1) | funding_rate (raw 8-hour funding rate, forward-filled) |

### Funding Rate Feature

Funding rate is a key sentiment indicator for futures trading:
- **Positive funding**: Longs pay shorts - market is over-leveraged long (bullish sentiment)
- **Negative funding**: Shorts pay longs - market is over-leveraged short (bearish sentiment)
- **Near zero**: Balanced market

The raw funding rate is used directly as it already provides meaningful signal.

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

Each state is a sliding window of 12 hourly timesteps with 21 features:
- Shape: `(num_samples, 12, 21)`
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

Position-based environment that simulates real futures trading with:
- Capital management (balance, position sizing)
- Leverage support (isolated margin)
- Stop-Loss and Take-Profit triggers
- Instant position flipping (LONG to SHORT in one step)
- Transaction fees on entry and exit

```python
from environments.position_trading_env import create_position_trading_env

env = create_position_trading_env('data/datasets/BTC_processed.csv')

obs, info = env.reset()
print(f"Balance: ${info['balance']:,.2f}")

# Open LONG position
obs, reward, _, _, info = env.step(0)  # LONG action
print(f"Position: {info['position']}, Entry: ${info['entry_price']:,.2f}")
print(f"SL: ${info['sl_price']:,.2f}, TP: ${info['tp_price']:,.2f}")

# Flip to SHORT (closes LONG, opens SHORT in one step)
obs, reward, _, _, info = env.step(1)  # SHORT action
print(f"Flipped: {info['flipped']}, P&L: ${info['pnl_dollars']:,.2f}")

# Close position (go FLAT)
obs, reward, _, _, info = env.step(2)  # FLAT action
print(f"Position: {info['position']}, Trade closed: {info['trade_closed']}")
```

### Action Space

**Gymnasium Actions** (what the agent outputs): `Discrete(3)` with values `{0, 1, 2}`

| Gymnasium Action | Name | Description |
|------------------|------|-------------|
| 0 | LONG | Go/stay long |
| 1 | SHORT | Go/stay short |
| 2 | FLAT | Close position |

**Position Values** (internal representation): `{+1, -1, 0}`

| Position Value | Name | Meaning |
|----------------|------|---------|
| +1 | LONG | Bullish - profit when price goes up |
| -1 | SHORT | Bearish - profit when price goes down |
| 0 | FLAT | Neutral - no position |

The sign indicates market direction, enabling math-friendly calculations:
```python
pnl = position * log(exit_price / entry_price)
# LONG (+1):  +1 * log(exit/entry) = profit when price increases
# SHORT (-1): -1 * log(exit/entry) = profit when price decreases
```

**Action Matrix**:

| Current Position | LONG Action | SHORT Action | FLAT Action |
|------------------|-------------|--------------|-------------|
| FLAT | Open LONG | Open SHORT | Do nothing |
| LONG | Keep LONG | Flip to SHORT | Close LONG |
| SHORT | Flip to LONG | Keep SHORT | Close SHORT |

**Instant Flip**: When in LONG and taking SHORT action (or vice versa), the environment:
1. Closes the current position (realizes P&L)
2. Immediately opens a new position in the opposite direction

This allows the agent to react instantly to market reversals without waiting.

### Capital Management

```python
env = create_position_trading_env(
    'data/datasets/BTC_processed.csv',
    initial_capital=10000,   # Starting balance
    risk_per_trade=0.02,     # Risk 2% of capital per trade
    leverage=10              # 10x leverage (isolated margin)
)
```

**Position Sizing**:
- Position Size = Capital x Risk per Trade x Leverage
- Example: $10,000 x 2% x 10 = $2,000 position size

### Stop-Loss and Take-Profit

Automatic SL/TP based on percentage from entry price:

```python
env = create_position_trading_env(
    'data/datasets/BTC_processed.csv',
    stop_loss=0.02,    # 2% stop-loss
    take_profit=0.04   # 4% take-profit (2:1 reward-risk ratio)
)
```

**SL/TP Behavior**:
- Checked at each step using high/low prices (intra-candle triggers)
- SL/TP triggers are processed BEFORE the agent's action
- Maximum loss per trade is limited to risk_per_trade (isolated margin)

### Drawdown Penalty

The environment applies a drawdown penalty to discourage excessive losses:

```python
env = create_position_trading_env(
    'data/datasets/BTC_processed.csv',
    drawdown_penalty_threshold=0.1,  # Start penalizing at 10% drawdown
    drawdown_penalty_factor=0.5      # Penalty multiplier
)
```

**How it works**:
- Tracks peak equity throughout the episode
- Calculates current drawdown: `(peak_equity - current_equity) / peak_equity`
- When drawdown exceeds threshold, penalty is applied:
  - `penalty = (drawdown - threshold) * penalty_factor`
  - Subtracted from the reward

**Example**:
- Threshold: 10%, Factor: 0.5
- Current drawdown: 15%
- Penalty: `(0.15 - 0.10) * 0.5 = 0.025` subtracted from reward

This teaches the agent to:
- Preserve capital during losing streaks
- Cut losses early rather than hoping for recovery
- Maintain consistent equity curve

### Exit Types

| Exit Reason | Description |
|-------------|-------------|
| `stop_loss` | Price hit stop-loss level |
| `take_profit` | Price hit take-profit level |
| `flip` | Position flipped to opposite direction |
| `manual` | Agent chose FLAT action |
| `end_episode` | Episode ended with open position |

### Info Dictionary

The environment returns detailed information:

```python
info = {
    # Position
    'position': 'LONG',           # FLAT, LONG, or SHORT
    'position_value': 1,          # +1 (LONG), -1 (SHORT), 0 (FLAT)
    'entry_price': 50000.0,
    'sl_price': 49000.0,
    'tp_price': 52000.0,
    
    # Capital
    'balance': 10000.0,
    'equity': 10150.0,            # Balance + unrealized P&L
    'position_size': 2000.0,
    
    # P&L
    'unrealized_pnl': 0.015,      # Log return
    'unrealized_pnl_dollars': 30.0,
    'realized_pnl_dollars': 0.0,
    
    # Statistics
    'num_trades': 5,
    'win_rate': 0.6,
    'roi': 0.015,
    'max_drawdown': 0.02,
    
    # Exit info (when trade closes)
    'trade_closed': True,
    'exit_reason': 'take_profit',
    'flipped': False,
    'pnl_dollars': 80.0,
}
```

### Environment Comparison

| Feature | TradingEnv | PositionTradingEnv |
|---------|------------|-------------------|
| Position tracking | No | Yes (FLAT/LONG/SHORT) |
| Capital management | No | Yes (balance, equity) |
| Leverage | No | Yes (configurable) |
| Stop-Loss/Take-Profit | No | Yes (auto-trigger) |
| Instant flip | N/A | Yes (LONG to SHORT) |
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
NUM_ACTIONS = 3               # LONG, SHORT, FLAT

# Gymnasium action indices
ACTION_LONG = 0               # Agent outputs 0 for LONG
ACTION_SHORT = 1              # Agent outputs 1 for SHORT
ACTION_FLAT = 2               # Agent outputs 2 for FLAT

# Position values (for calculations)
POSITION_LONG = 1             # +1 = bullish
POSITION_SHORT = -1           # -1 = bearish
POSITION_FLAT = 0             # 0 = neutral

# Feature Engineering
NUM_FEATURES = 21
OBS_SHAPE = (12, 21)          # Observation shape for neural network

# Reward Settings
SMURF_HOLD_REWARD = 0.0055    # Fixed HOLD reward for Smurf agent

# Trading Settings (PositionTradingEnv)
TRADING_PARAMS = {
    'initial_capital': 10000,   # Starting balance in USD
    'risk_per_trade': 0.02,     # Risk 2% of capital per trade
    'leverage': 10,             # 10x leverage (isolated margin)
    'stop_loss': 0.02,          # 2% stop-loss from entry
    'take_profit': 0.04,        # 4% take-profit from entry (2:1 RR)
}

# Drawdown Penalty Settings
DRAWDOWN_PARAMS = {
    'drawdown_penalty_threshold': 0.1,   # Start penalizing at 10% drawdown
    'drawdown_penalty_factor': 0.5,      # Penalty multiplier
}

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
from environments.position_trading_env import create_position_trading_env

env = create_position_trading_env('data/datasets/BTC_processed.csv')
print(f'Observation space: {env.observation_space}')
print(f'Action space: {env.action_space}')
print(f'Actions: LONG(0), SHORT(1), FLAT(2)')

obs, info = env.reset()
print(f'Initial balance: \${info[\"balance\"]:,.2f}')

# Open LONG
obs, reward, _, _, info = env.step(0)
print(f'Opened LONG at \${info[\"entry_price\"]:,.2f}')

# Flip to SHORT
obs, reward, _, _, info = env.step(1)
print(f'Flipped to SHORT, P&L: \${info[\"pnl_dollars\"]:,.2f}')

# Close position
obs, reward, _, _, info = env.step(2)
print(f'Closed position, Final balance: \${info[\"balance\"]:,.2f}')
"
```

Expected output:
```
Observation space: Box(0.0, 1.0, (12, 21), float32)
Action space: Discrete(3)
Actions: LONG(0), SHORT(1), FLAT(2)
Initial balance: $10,000.00
Opened LONG at $50,123.45
Flipped to SHORT, P&L: $-12.34
Closed position, Final balance: $9,975.21
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
