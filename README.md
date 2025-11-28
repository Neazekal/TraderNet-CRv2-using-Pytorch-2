# TraderNet-CRv2 PyTorch

A PyTorch implementation of **TraderNet-CRv2** - a Deep Reinforcement Learning system for cryptocurrency trading that combines QR-DQN and Categorical SAC with technical analysis and safety mechanisms.

Based on the paper: *"Combining deep reinforcement learning with technical analysis and trend monitoring on cryptocurrency markets"* (Neural Computing and Applications, 2023)

Original TensorFlow implementation: [kochlisGit/TraderNet-CRv2](https://github.com/kochlisGit/TraderNet-CRv2)

---

## Features

- **RL Agents**: QR-DQN (distributional) and Categorical SAC for trading decisions (LONG/SHORT/FLAT)
- **Technical Analysis**: 11 indicators (MACD, RSI, Bollinger Bands, ADX, etc.)
- **Realistic Trading**: Position-based environment with capital management
- **Risk Management**: Stop-Loss, Take-Profit, and leverage support
- **Instant Position Flip**: Switch from LONG to SHORT (or vice versa) in one step
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
├── agents/
│   └── networks/
│       ├── actor.py               # Actor network (policy)
│       └── critic.py              # Critic network (value function)
├── metrics/                       # (Phase 5+) Trading metrics
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

### Features (28 total: 21 active + 7 reserved)

**Active Features (21)**:

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

**Reserved Features (7)**: Additional funding-derived features reserved for future experimentation (e.g., funding rate moving averages, differences, volatility).

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

Each state is a sliding window of 12 hourly timesteps with 28 features:
- Shape: `(num_samples, 12, 28)`
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

### Funding Fee

The environment simulates real Binance Futures funding fees that occur every 8 hours (at 00:00, 08:00, 16:00 UTC):

```python
env = create_position_trading_env(
    'data/datasets/BTC_processed.csv',
    enable_funding_fee=True  # Enable funding fee simulation (default: True)
)
```

**How it works**:
- Funding fee is charged/paid when holding a position at funding times
- Fee = Position Size x Funding Rate
- **Positive funding rate**: LONG pays, SHORT receives
- **Negative funding rate**: SHORT pays, LONG receives
- Funding rate is read from the `funding_rate` column in the dataset

**Example**:
- Position: LONG, Size: $2,000
- Funding rate: 0.01% (positive)
- Fee: $2,000 x 0.0001 = $0.20 deducted from balance

This teaches the agent to:
- Consider funding costs when holding positions long-term
- Potentially close positions before unfavorable funding times
- Factor in funding direction when choosing LONG vs SHORT

### Random Slippage

The environment simulates realistic order execution with random slippage:

```python
env = create_position_trading_env(
    'data/datasets/BTC_processed.csv',
    enable_slippage=True,           # Enable slippage simulation (default: True)
    slippage_mean=0.0001,           # Mean slippage: 0.01% (1 bps)
    slippage_std=0.00005            # Standard deviation: 0.005%
)
```

**How it works**:
- Slippage is sampled from a normal distribution: `N(mean, std)`
- Applied when opening or closing positions (not during hold)
- Always works against the trader (worse execution price)
  - LONG entry: price slightly higher
  - LONG exit: price slightly lower
  - SHORT entry: price slightly lower
  - SHORT exit: price slightly higher

**Example**:
- Entry price (market): $50,000
- Slippage: 0.015% (sampled)
- LONG actual entry: $50,000 x 1.00015 = $50,007.50

**Why it matters**:
- Real exchanges have order book depth and market impact
- Large orders move the price against you
- High-frequency strategies are more affected by slippage
- Teaches agent that frequent trading has hidden costs

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

### Funding Fee in Info

When funding fee is applied, the info dictionary includes:

```python
info = {
    # ... other fields ...
    'funding_fee_applied': True,
    'funding_fee': -0.20,  # Negative = paid, Positive = received
    'funding_rate': 0.0001,
}

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
| Funding fee | No | Yes (8-hour intervals) |
| Random slippage | No | Yes (configurable) |
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

## Phase 4: Neural Networks

Deep neural networks for the discrete RL agents (QR-DQN and Categorical SAC) using PyTorch. A shared Conv1D + FC backbone feeds lightweight heads (policy logits, Q-values, quantile values).

### Actor Network

The Actor network uses the shared backbone and a categorical policy head to output action probabilities for the action space.

```python
from agents.networks.actor import ActorNetwork

actor = ActorNetwork()
print(f"Parameters: {sum(p.numel() for p in actor.parameters()):,}")
# Parameters: 151,459
```

**Architecture**:
- **Conv1D**: 28 input channels → 32 filters, kernel size 3
- **Flatten**: 320 features
- **FC Layer 1**: 320 → 256 (GELU activation)
- **FC Layer 2**: 256 → 256 (GELU activation)
- **Output**: 256 → 3 (Softmax for action probabilities)

**Input**: `(batch, 12, 28)` - 12 timesteps × 28 features
**Output**: `(batch, 3)` - Probabilities for LONG/SHORT/FLAT

**Methods**:
- `forward(state)` - Get action probabilities
- `get_action(state, deterministic=False)` - Sample action and log probability
- `evaluate_actions(states, actions)` - Compute log probs & entropy (used by categorical SAC)

### Critic Network

The Critic network uses the same backbone and a value head; the backbone can also be paired with Q/quantile heads (see `agents/networks/heads.py`) for QR-DQN or categorical SAC critics.

```python
from agents.networks.critic import CriticNetwork

critic = CriticNetwork()
print(f"Parameters: {sum(p.numel() for p in critic.parameters()):,}")
# Parameters: 150,945
```

**Architecture**:
- **Conv1D**: 28 input channels → 32 filters, kernel size 3
- **Flatten**: 320 features
- **FC Layer 1**: 320 → 256 (GELU activation)
- **FC Layer 2**: 256 → 256 (GELU activation)
- **Output**: 256 → 1 (Linear for value estimate)

**Input**: `(batch, 12, 28)` - 12 timesteps × 28 features
**Output**: `(batch, 1)` - State value estimate

**Methods**:
- `forward(state)` - Get state value
- `get_value(state)` - Single state value estimate
- `evaluate_states(states)` - Batch value estimates for training loops

### Key Features

**Shared Architecture**:
- Both networks use the same Conv1D + FC backbone
- Only the output layer differs (3 actions vs 1 value)
- GELU activation throughout (Gaussian Error Linear Unit)

**Weight Initialization**:
- Conv/Hidden layers: Xavier and Kaiming initialization
- Critic output layer: Uniform[-0.03, 0.03] for stable value learning

**How to use with QR-DQN & Categorical SAC**:
- For **QR-DQN**, keep the Conv1D + FC backbone and replace the output layer with a quantile head of shape `(num_actions, num_quantiles)`.
- For **Categorical SAC**, use the actor as the categorical policy head and pair it with twin Q-networks that share the same backbone but output Q-values per action.
- Both networks support batched forward passes for efficient training.

### Testing Networks

```python
import torch
from agents.networks.actor import ActorNetwork
from agents.networks.critic import CriticNetwork

# Create networks
actor = ActorNetwork()
critic = CriticNetwork()

# Random state input (12 timesteps, 28 features)
state = torch.randn(12, 28)

# Actor: Get action probabilities
action_probs = actor(state.unsqueeze(0))
print(f"Action probs: {action_probs}")  # [0.33, 0.33, 0.34] (sums to 1.0)

# Actor: Sample an action
action, log_prob = actor.get_action(state, deterministic=False)
print(f"Action: {action}, Log prob: {log_prob:.4f}")  # Action: 2, Log prob: -1.0986

# Critic: Get state value
value = critic.get_value(state)
print(f"State value: {value:.4f}")  # State value: 0.0234
```

### Network Configuration

All network parameters are centralized in `config/config.py`:

```python
NETWORK_PARAMS = {
    'conv_filters': 32,
    'conv_kernel': 3,
    'fc_layers': [256, 256],
    'activation': 'gelu',
    'dropout': 0.0,
}

NETWORK_INIT_PARAMS = {
    'value_head_init_range': 0.03,      # Critic output layer init range
    'log_prob_epsilon': 1e-8,           # Numerical stability for log prob
}
```

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
NUM_FEATURES = 28             # Total features (21 active + 7 reserved)
OBS_SHAPE = (12, 28)          # Observation shape for neural network

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

# Slippage Settings
SLIPPAGE_PARAMS = {
    'enable_slippage': True,             # Enable random slippage
    'slippage_mean': 0.0001,             # Mean slippage: 0.01% (1 bps)
    'slippage_std': 0.00005,             # Standard deviation: 0.005%
}

# QR-DQN Hyperparameters
QR_DQN_PARAMS = {
    'learning_rate': 0.0005,
    'gamma': 0.99,
    'num_quantiles': 51,
    'batch_size': 128,
    'target_update_interval': 2000,
    'huber_kappa': 1.0,
    'replay_buffer_size': 500_000,
    'priority_alpha': 0.6,
    'priority_beta_start': 0.4,
    'priority_beta_frames': 500_000,
}

# Categorical SAC Hyperparameters
CATEGORICAL_SAC_PARAMS = {
    'learning_rate': 0.0005,
    'gamma': 0.99,
    'tau': 0.005,              # Target smoothing
    'batch_size': 256,
    'entropy_target': -1.0,    # For auto temperature
    'alpha_init': 0.2,         # Initial entropy temperature
    'replay_buffer_size': 500_000,
    'target_update_interval': 1,
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
Observation space: Box(0.0, 1.0, (12, 28), float32)
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
- [x] **Phase 4**: Neural networks (Actor/Critic)
- [ ] **Phase 5**: RL agents (QR-DQN + Categorical SAC)
- [ ] **Phase 6**: Training & evaluation scripts
- [ ] **Phase 7**: Metrics & visualization

---

## References

1. Kochliaridis et al. (2023) - *"Combining deep reinforcement learning with technical analysis and trend monitoring on cryptocurrency markets"*
2. Haarnoja et al. (2018) - *"Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL"*
3. Dabney et al. (2018) - *"Distributional RL with Quantile Regression"*
4. Original implementation: [kochlisGit/TraderNet-CRv2](https://github.com/kochlisGit/TraderNet-CRv2)

---

## License

This project is for educational and research purposes.

---

## Contributing

Contributions are welcome! Please read the [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for details on the architecture and planned features.
