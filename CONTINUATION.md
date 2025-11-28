# Continuation Guide for AI Assistants

**Purpose:** This document helps any AI assistant quickly understand and continue working on this project.

**Last Updated:** 2025-11-28

---

## Project Overview

**TraderNet-CRv2 PyTorch** is a Deep Reinforcement Learning system for cryptocurrency futures trading.

### What This Project Does
1. Downloads hourly OHLCV data from Binance Futures
2. Creates 21 technical features (log returns, indicators, regime, funding rate)
3. Trains RL agents (QR-DQN or SAC) to trade LONG/SHORT/FLAT
4. Evaluates on held-out data (~3 months)

### Key Design Decisions
- **No Smurf agent**: Removed - SL/TP provides sufficient risk management
- **21 features**: Active features only (CSV has 27 columns but only 21 are used as features)
- **Position values**: LONG(+1), SHORT(-1), FLAT(0) for intuitive P&L math
- **Checkpoint naming**: `{agent}_{crypto}_best.pt` and `{agent}_{crypto}_last.pt`
- **Best model metric**: `mean_return` (higher is better, positive = profitable)

---

## Quick Commands

```bash
# Download data
python -m data.downloaders.binance --crypto BTC

# Build dataset (creates data/datasets/BTC_processed.csv)
python -m data.datasets.builder --crypto BTC

# Train QR-DQN
python train.py --agent qrdqn --crypto BTC --timesteps 100000

# Train SAC
python train.py --agent sac --crypto BTC --timesteps 100000

# Evaluate
python evaluate.py --checkpoint checkpoints/qrdqn_BTC_best.pt --crypto BTC

# Quick test (for debugging)
python train.py --agent qrdqn --crypto BTC --timesteps 500 --eval-freq 200
```

---

## Critical Data Flow

### CSV Structure (27 columns, but only 21 used as features)
```
Column 0-5:   timestamp, open, high, low, close, volume  (NOT features - used for rewards/SL/TP)
Column 6-26:  21 actual features used by neural network
```

### Data Loading (IMPORTANT - Common Bug Source)
```python
# CORRECT way to load data:
from data.datasets.utils import prepare_training_data
data = prepare_training_data('data/datasets/BTC_processed.csv')

# data['train'] and data['eval'] contain:
#   'sequences': np.ndarray (num_samples, 12, 21)  # NN input
#   'highs': np.ndarray      # For reward calculation
#   'lows': np.ndarray       # For reward calculation  
#   'closes': np.ndarray     # For reward calculation
#   'df': pd.DataFrame       # Original data with all columns

# WRONG - Don't pass DataFrame directly to environment:
# env = PositionTradingEnv(data=df)  # ERROR!

# CORRECT:
env = PositionTradingEnv(
    sequences=data['train']['sequences'],
    highs=data['train']['highs'],
    lows=data['train']['lows'],
    closes=data['train']['closes'],
)
```

### Data Split
```python
EVAL_HOURS = 2250  # ~3 months held out for evaluation

# Training: All data EXCEPT last 2250 hours
# Evaluation: Last 2250 hours (e.g., Aug-Nov 2025 for BTC)
```

---

## File Structure

```
├── train.py                           # Main training script
├── evaluate.py                        # Evaluation script
├── config/config.py                   # ALL hyperparameters (check here first!)
├── environments/
│   └── position_trading_env.py        # Main trading environment
├── agents/
│   ├── qrdqn_agent.py                 # QR-DQN (distributional Q-learning)
│   └── categorical_sac_agent.py       # SAC (entropy-regularized)
│   ├── networks/                      # Neural network components
│   └── buffers/replay_buffer.py       # Prioritized Experience Replay
├── data/
│   ├── downloaders/binance.py         # Download OHLCV + funding
│   └── datasets/
│       ├── builder.py                 # Build processed CSV
│       └── utils.py                   # prepare_training_data() - USE THIS!
└── checkpoints/                       # Saved models
```

---

## Environment Details

### PositionTradingEnv
```python
# State space
observation_space = Box(0, 1, shape=(12, 21))  # 12 timesteps × 21 features

# Action space  
action_space = Discrete(3)
# 0 = LONG (go/stay long, flip if short)
# 1 = SHORT (go/stay short, flip if long)
# 2 = FLAT (close position)

# Position values (internal tracking)
POSITION_LONG = +1
POSITION_SHORT = -1
POSITION_FLAT = 0
```

### Reward Calculation
```python
# When a trade closes:
reward = position * log(exit_price / entry_price) - 2 * fees - drawdown_penalty

# position: +1 (LONG) or -1 (SHORT)
# fees: 0.001 (0.1% per side, x2 for entry+exit = 0.2% round trip)

# When holding (no trade close):
reward = 0  # Default
```

### Risk Management (Automatic)
```python
STOP_LOSS = 0.02      # 2% - auto-closes losing position
TAKE_PROFIT = 0.04    # 4% - auto-closes winning position
LEVERAGE = 10         # 10x isolated margin
RISK_PER_TRADE = 0.02 # Risk 2% of balance per trade
```

---

## Agent Details

### QR-DQN
```python
# Exploration: Epsilon-greedy
epsilon: 1.0 → 0.01 over 500,000 steps  # Starts 100% random!

# Network: Conv1D(21→32) → FC(256) → FC(256) → Quantile head (51 × 3)
# Target update: Hard copy every 2000 steps
# Batch size: 128
```

### Categorical SAC
```python
# Exploration: Entropy-based (no epsilon)
# Networks: Actor + Twin Q-networks
# Target update: Soft update (τ=0.005) every step
# Batch size: 256
# Entropy temperature: Auto-tuned
```

---

## Training Metrics

### mean_return (Primary Metric)
```python
mean_return = average(sum of rewards per episode)

# Interpretation:
# Negative (e.g., -150): Losing money (bad)
# Around 0: Breaking even
# Positive (e.g., +50): Making profit (good!)

# Early training will show negative values because epsilon=1.0 (random actions)
# Should improve as epsilon decreases and agent learns
```

### Checkpoint Saving
- `{agent}_{crypto}_best.pt`: Saved when mean_return improves
- `{agent}_{crypto}_last.pt`: Saved at end of training

---

## Common Issues & Solutions

### 1. "expected 28 channels, got 21"
**Cause**: Old code had hardcoded 28 features
**Fix**: Networks now use `NUM_FEATURES` from config (= 21)

### 2. "PositionTradingEnv got unexpected argument 'data'"
**Cause**: Passing DataFrame instead of arrays
**Fix**: Use `prepare_training_data()` and pass sequences/highs/lows/closes separately

### 3. "train_eval_split() got unexpected argument 'train_ratio'"
**Cause**: Wrong function signature
**Fix**: Use `prepare_training_data()` which handles everything correctly

### 4. Training shows negative mean_return
**Cause**: Normal! Early training has epsilon≈1.0 (random actions)
**Fix**: Train longer (500K+ steps), or use SAC (no epsilon, faster learning)

---

## Configuration Reference (config/config.py)

```python
# Features
NUM_FEATURES = 21
SEQUENCE_LENGTH = 12
EVAL_HOURS = 2250  # ~3 months for evaluation

# Actions
NUM_ACTIONS = 3  # LONG=0, SHORT=1, FLAT=2

# Trading
INITIAL_CAPITAL = 10000.0
FEES = 0.001           # 0.1% per trade
STOP_LOSS = 0.02       # 2%
TAKE_PROFIT = 0.04     # 4%
LEVERAGE = 10

# QR-DQN
QR_DQN_PARAMS = {
    'learning_rate': 0.0005,
    'gamma': 0.99,
    'num_quantiles': 51,
    'batch_size': 128,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay_frames': 500_000,
    'target_update_interval': 2000,
}

# SAC
CATEGORICAL_SAC_PARAMS = {
    'learning_rate': 0.0005,
    'gamma': 0.99,
    'tau': 0.005,
    'batch_size': 256,
    'alpha_init': 0.2,
}
```

---

## What's NOT Implemented (Phase 7 TODO)

1. **Visualization**: No equity curves, drawdown plots, or trade charts yet
2. **Hyperparameter tuning**: No Optuna integration
3. **Walk-forward testing**: No rolling window backtesting

---

## Tips for AI Assistants

1. **Always check `config/config.py`** for current parameter values
2. **Use `prepare_training_data()`** - never load CSV directly for training
3. **Environment needs arrays**, not DataFrame: sequences, highs, lows, closes
4. **Test changes with short runs**: `--timesteps 500 --eval-freq 200`
5. **mean_return higher = better** (positive = profitable)
6. **Early training will look bad** - epsilon starts at 1.0 (100% random)
