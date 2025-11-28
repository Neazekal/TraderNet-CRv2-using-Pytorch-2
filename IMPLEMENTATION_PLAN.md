# Implementation Plan

**Project:** TraderNet-CRv2 PyTorch Implementation  
**Status:** Phase 6 Complete

---

## Phase Overview

| Phase | Description | Status | LOC |
|-------|-------------|--------|-----|
| 1 | Data Download | ✅ Complete | ~500 |
| 2 | Preprocessing | ✅ Complete | ~800 |
| 3 | Trading Environment | ✅ Complete | ~1,200 |
| 4 | Neural Networks | ✅ Complete | ~600 |
| 5 | RL Agents | ✅ Complete | ~1,100 |
| 6 | Training & Evaluation | ✅ Complete | ~800 |
| 7 | Visualization | 🔲 Planned | ~400 |

**Total Code:** ~5,400 lines

---

## Phase 1: Data Download ✅

### Files
- `data/downloaders/binance.py` - CCXT-based downloader

### Features
- Downloads OHLCV from Binance Futures
- Downloads funding rates (8-hour intervals)
- Supports: BTC, ETH, XRP, SOL, BNB, TRX, DOGE
- Saves to `data/storage/{crypto}_raw.csv`

### Usage
```bash
python -m data.downloaders.binance --crypto BTC
python -m data.downloaders.binance --all
```

---

## Phase 2: Preprocessing ✅

### Files
- `data/preprocessing/ohlcv.py` - Log returns, hour extraction
- `data/preprocessing/technical.py` - Technical indicators
- `data/preprocessing/regime.py` - Market regime detection
- `data/preprocessing/funding.py` - Funding rate processing

### Features (21 total)
| Index | Feature | Category |
|-------|---------|----------|
| 0-4 | log_return_open/high/low/close/volume | Returns |
| 5 | hour | Time |
| 6-9 | macd_diff, aroon_up, aroon_down, adx | Trend |
| 10-12 | stoch_d, rsi, cci | Momentum |
| 13-16 | close_dema_diff, close_vwap_diff, bb_upper_dist, bb_lower_dist | Price |
| 17-18 | adl_diff, obv_diff | Volume |
| 19 | regime | Market State |
| 20 | funding_rate | Funding |

### Usage
```bash
python -m data.datasets.builder --crypto BTC
python -m data.datasets.builder --all
```

---

## Phase 3: Trading Environment ✅

### Files
- `environments/position_trading_env.py` - Main environment
- `environments/rewards/market_limit.py` - Reward function

### Environment Specs
```python
# Observation
observation_space = Box(0, 1, (12, 21))  # 12 timesteps × 21 features

# Actions
action_space = Discrete(3)  # LONG=0, SHORT=1, FLAT=2

# Position values
POSITION_LONG = +1   # Bullish
POSITION_SHORT = -1  # Bearish
POSITION_FLAT = 0    # No position
```

### Trading Parameters
```python
INITIAL_CAPITAL = 10000.0
RISK_PER_TRADE = 0.02    # 2%
LEVERAGE = 10
STOP_LOSS = 0.02         # 2%
TAKE_PROFIT = 0.04       # 4%
FEES = 0.001             # 0.1%
```

---

## Phase 4: Neural Networks ✅

### Files
- `agents/networks/backbone.py` - Shared Conv1D backbone
- `agents/networks/heads.py` - Policy and value heads
- `agents/networks/actor.py` - Actor network
- `agents/networks/critic.py` - Critic network

### Architecture
```
Input: (batch, 12, 21)
    ↓
Permute: (batch, 21, 12)
    ↓
Conv1D(21→32, kernel=3)
    ↓
GELU
    ↓
Flatten: (batch, 320)
    ↓
FC(320→256) → GELU
    ↓
FC(256→256) → GELU
    ↓
Output head (varies by agent)
```

---

## Phase 5: RL Agents ✅

### QR-DQN Agent
**File:** `agents/qrdqn_agent.py`

```python
# Architecture
Q-Network: Backbone → FC(256, 51*3) → reshape(batch, 3, 51)

# Hyperparameters
learning_rate = 0.0005
gamma = 0.99
num_quantiles = 51
batch_size = 128
target_update_interval = 2000
epsilon: 1.0 → 0.01 over 500K steps
```

### Categorical SAC Agent
**File:** `agents/categorical_sac_agent.py`

```python
# Architecture
Actor: Backbone → FC(256, 3) → Softmax
Q1/Q2: Backbone → FC(256, 3)

# Hyperparameters
learning_rate = 0.0005
gamma = 0.99
tau = 0.005
batch_size = 256
entropy_target = -1.0
alpha_init = 0.2
```

### Replay Buffer
**File:** `agents/buffers/replay_buffer.py`

- Prioritized Experience Replay
- Capacity: 500,000 transitions
- Alpha (priority): 0.6
- Beta (importance sampling): 0.4 → 1.0

---

## Phase 6: Training & Evaluation ✅

### Training Script
**File:** `train.py`

```bash
# QR-DQN
python train.py --agent qrdqn --crypto BTC --timesteps 1000000

# SAC
python train.py --agent sac --crypto ETH --timesteps 500000
```

**Features:**
- Progress bar with real-time metrics
- Automatic best model saving
- Checkpoint naming: `{agent}_{crypto}_best.pt`, `{agent}_{crypto}_last.pt`
- Resume from checkpoint

### Evaluation Script
**File:** `evaluate.py`

```bash
python evaluate.py --checkpoint checkpoints/qrdqn_BTC_best.pt --crypto BTC
```

**Output:**
- Return statistics (mean, std, min, max)
- Risk metrics (Sharpe, Sortino, Max Drawdown)
- Trading stats (win rate, profit factor, trades)

---

## Phase 7: Visualization (Planned)

### Planned Features
1. **Equity Curves**
   - Plot balance over time
   - Compare with buy-and-hold

2. **Drawdown Analysis**
   - Drawdown over time
   - Underwater plot

3. **Trade Analysis**
   - Entry/exit visualization
   - Win/loss distribution

4. **Training Curves**
   - Loss over time
   - Mean return over training

### Proposed Files
- `analysis/visualization/equity.py`
- `analysis/visualization/drawdown.py`
- `analysis/visualization/trades.py`
- `analysis/visualization/training.py`

---

## Architecture Decisions

### Why No Smurf Agent?
The original paper used a Smurf agent as a conservative gatekeeper. In this implementation:
- Position-based environment already tracks positions
- Stop-Loss automatically closes losing trades
- Take-Profit locks in gains
- Drawdown penalty discourages risky behavior
- FLAT action allows agent to choose not to trade

### Position Values: +1, -1, 0
Makes P&L calculation intuitive:
```python
pnl = position * log(exit_price / entry_price)
# LONG (+1): profit when price increases
# SHORT (-1): profit when price decreases
```

### 21 Features (Not 28)
Original plan had 28 features (21 active + 7 reserved). Simplified to just use the 21 active features.

---

## Dependencies

```
torch>=2.0.0
gymnasium>=0.29.0
numpy>=1.24.0
pandas>=2.0.0
ccxt>=4.0.0
ta>=0.10.0
tqdm>=4.65.0
```

---

## References

- Paper: Kochliaridis et al. (2023), Neural Computing and Applications
- Original TensorFlow: https://github.com/kochlisGit/TraderNet-CRv2
