# TraderNet-CRv2 PyTorch

A PyTorch implementation of **TraderNet-CRv2** - a Deep Reinforcement Learning system for cryptocurrency futures trading using QR-DQN and Categorical SAC agents.

Based on: *"Combining deep reinforcement learning with technical analysis and trend monitoring on cryptocurrency markets"* (Neural Computing and Applications, 2023)

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download data (BTC example)
python -m data.downloaders.binance --crypto BTC

# 3. Build dataset
python -m data.datasets.builder --crypto BTC

# 4. Train agent
python train.py --agent qrdqn --crypto BTC --timesteps 100000

# 5. Evaluate
python evaluate.py --checkpoint checkpoints/qrdqn_BTC_best.pt --crypto BTC
```

---

## Features

| Feature | Description |
|---------|-------------|
| **RL Agents** | QR-DQN (distributional) and Categorical SAC |
| **Actions** | LONG (0), SHORT (1), FLAT (2) with instant position flipping |
| **Risk Management** | Stop-Loss (2%), Take-Profit (4%), leverage (10x) |
| **Technical Indicators** | MACD, RSI, Bollinger Bands, ADX, Stochastic, CCI, etc. |
| **Features** | 21 features (log returns, technicals, regime, funding rate) |
| **Supported Cryptos** | BTC, ETH, XRP, SOL, BNB, TRX, DOGE |

---

## Project Structure

```
TraderNet-CRv2-using-Pytorch-2/
├── config/
│   └── config.py                  # All hyperparameters & settings
├── data/
│   ├── downloaders/binance.py     # Download OHLCV + funding rates
│   ├── preprocessing/             # Feature engineering
│   └── datasets/                  # Dataset building & splitting
├── environments/
│   ├── position_trading_env.py    # Main trading environment (Gymnasium)
│   └── rewards/market_limit.py    # Reward function
├── agents/
│   ├── networks/                  # Actor, Critic, Backbone networks
│   ├── buffers/replay_buffer.py   # Prioritized Experience Replay
│   ├── qrdqn_agent.py             # QR-DQN agent
│   └── categorical_sac_agent.py   # Categorical SAC agent
├── train.py                       # Training script
├── evaluate.py                    # Evaluation script
├── utils/                         # Logging, checkpoints, metrics
├── checkpoints/                   # Saved models
└── logs/                          # Training logs
```

---

## Training

### Train QR-DQN
```bash
python train.py --agent qrdqn --crypto BTC --timesteps 1000000
```

### Train Categorical SAC
```bash
python train.py --agent sac --crypto ETH --timesteps 500000
```

### Training Options
| Option | Description | Default |
|--------|-------------|---------|
| `--agent` | Agent type: `qrdqn` or `sac` | Required |
| `--crypto` | Cryptocurrency: BTC, ETH, XRP, SOL, BNB, TRX, DOGE | Required |
| `--timesteps` | Total training steps | 1,000,000 |
| `--eval-freq` | Evaluate every N steps | 10,000 |
| `--device` | `cpu` or `cuda` | auto-detect |
| `--no-multi-gpu` | Disable multi-GPU training | False |
| `--resume` | Resume from checkpoint path | None |

### GPU Configuration

#### Recommended Settings by GPU

| GPU | VRAM | QR-DQN Batch | SAC Batch | Multi-GPU? |
|-----|------|--------------|-----------|------------|
| GTX 1650 | 4GB | 256 | 512 | No |
| T4 | 16GB | 1024 | 2048 | No |
| P100 | 16GB | 2048 | 4096 | No |
| V100 | 32GB | 4096 | 8192 | No |
| A100 | 40GB | 8192 | 16384 | No |

#### Why Single GPU is Faster for This Model

This model is **memory-bandwidth bound**, not compute-bound:
- Small model size (~1-2MB parameters)
- Frequent replay buffer sampling
- DataParallel overhead > parallelization benefit

**Memory bandwidth comparison:**
| GPU | Memory Bandwidth |
|-----|-----------------|
| T4 | 320 GB/s |
| P100 | 732 GB/s |
| V100 | 900 GB/s |

**Example**: 1x P100 (732 GB/s) is faster than 2x T4 (640 GB/s combined, minus sync overhead)

#### GPU Training Commands

```bash
# Single GPU (recommended for this model)
python train.py --agent qrdqn --crypto BTC --no-multi-gpu

# Multi-GPU (only if you need more VRAM, not speed)
python train.py --agent qrdqn --crypto BTC
```

#### Changing Batch Size

Edit `config/config.py`:
```python
QR_DQN_PARAMS = {
    'batch_size': 2048,  # Increase for P100/V100
    ...
}
```

### Checkpoints
Training saves two files:
- `{agent}_{crypto}_best.pt` - Best model (highest mean_return)
- `{agent}_{crypto}_last.pt` - Final model at end of training

Example: `qrdqn_BTC_best.pt`, `sac_ETH_last.pt`

### Progress Bar
Training shows real-time progress:
```
Training QR-DQN: 45%|████████████| 450000/1000000 [2:30:00<3:00:00, 50.0step/s]
                    eps=0.102, loss=0.0234, ep_ret=-12.5, episodes=8
```

### Best Model Selection
The best model is saved when `mean_return` improves (higher is better).
- Positive mean_return = profitable trading
- Negative mean_return = losing trades

---

## Evaluation

### Basic Evaluation
```bash
python evaluate.py --checkpoint checkpoints/qrdqn_BTC_best.pt --crypto BTC
```

### Options
| Option | Description | Default |
|--------|-------------|---------|
| `--checkpoint` | Path to checkpoint file | Required |
| `--crypto` | Cryptocurrency to evaluate on | Required |
| `--num-episodes` | Number of evaluation episodes | 5 |
| `--save-trades` | Save trade log to CSV | None |
| `--verbose` | Print detailed output | False |

### Output Metrics
```
EVALUATION RESULTS - QRDQN on BTC
======================================================================
Return Statistics:
  Mean Return:           12.5400
  Std Return:             2.3100
  
Risk-Adjusted Metrics:
  Sharpe Ratio:            1.85
  Sortino Ratio:           2.41
  Max Drawdown:           8.50%
  
Trading Statistics:
  Total Trades:             156
  Win Rate:              58.33%
  Profit Factor:           1.42
======================================================================
```

---

## Environment Details

### Position-Based Trading
The agent manages positions in a realistic futures trading simulation:

| Position | Value | Description |
|----------|-------|-------------|
| LONG | +1 | Bullish - profit when price goes up |
| SHORT | -1 | Bearish - profit when price goes down |
| FLAT | 0 | No position - waiting for opportunity |

### Action Space
| Action | Index | Effect |
|--------|-------|--------|
| LONG | 0 | Open/keep LONG, flip if SHORT |
| SHORT | 1 | Open/keep SHORT, flip if LONG |
| FLAT | 2 | Close any position |

### Capital Management
```python
INITIAL_CAPITAL = 10000.0    # Starting balance (USDT)
RISK_PER_TRADE = 0.02        # Risk 2% per trade
LEVERAGE = 10                # 10x leverage
STOP_LOSS = 0.02             # 2% stop-loss
TAKE_PROFIT = 0.04           # 4% take-profit
FEES = 0.001                 # 0.1% transaction fee
```

### State Observation
- Shape: `(12, 21)` - 12 timesteps × 21 features
- Normalized to [0, 1] range
- Includes: log returns, technical indicators, market regime, funding rate

---

## Data Pipeline

### 1. Download Data
```bash
# Download single crypto
python -m data.downloaders.binance --crypto BTC

# Download all supported cryptos
python -m data.downloaders.binance --all
```

### 2. Build Dataset
```bash
# Build single crypto dataset
python -m data.datasets.builder --crypto BTC

# Build all datasets
python -m data.datasets.builder --all
```

### Features (21 total)
| Category | Features |
|----------|----------|
| Log Returns (5) | open, high, low, close, volume |
| Time (1) | hour of day |
| Trend (4) | MACD signal diff, Aroon Up/Down, ADX |
| Momentum (3) | Stochastic, RSI, CCI |
| Price Relative (4) | Close vs DEMA/VWAP, Bollinger Band distances |
| Volume (2) | ADL diff, OBV diff |
| Regime (1) | Market regime (0-3) |
| Funding (1) | Funding rate |

---

## Agent Architecture

### QR-DQN (Quantile Regression DQN)
- **Purpose**: Distributional Q-learning - learns full return distribution
- **Network**: Conv1D backbone → 51 quantiles × 3 actions
- **Exploration**: Epsilon-greedy (1.0 → 0.01)
- **Target Update**: Hard update every 2000 steps

### Categorical SAC (Soft Actor-Critic)
- **Purpose**: Entropy-regularized policy learning
- **Networks**: Actor (policy) + Twin Q-networks (critics)
- **Exploration**: Entropy temperature auto-tuning
- **Target Update**: Soft update (τ=0.005) every step

### Shared Components
- **Backbone**: Conv1D(21→32) → FC(256) → FC(256) → GELU activation
- **Replay Buffer**: Prioritized Experience Replay (500K capacity)
- **Batch Size**: 128 (QR-DQN), 256 (SAC)

---

## Configuration

All hyperparameters are in `config/config.py`:

```python
# QR-DQN
QR_DQN_PARAMS = {
    'learning_rate': 0.0005,
    'gamma': 0.99,
    'num_quantiles': 51,
    'batch_size': 128,
    'target_update_interval': 2000,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay_frames': 500_000,
}

# Categorical SAC
CATEGORICAL_SAC_PARAMS = {
    'learning_rate': 0.0005,
    'gamma': 0.99,
    'tau': 0.005,
    'batch_size': 256,
    'entropy_target': -1.0,
    'alpha_init': 0.2,
}

# Training
TRAINING_PARAMS = {
    'total_timesteps': 1_000_000,
    'eval_freq': 10_000,
    'warmup_steps': 10_000,  # SAC only
}
```

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA (optional, for GPU training)

Key dependencies:
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

## Training Tips

1. **Start with shorter runs** to verify everything works:
   ```bash
   python train.py --agent qrdqn --crypto BTC --timesteps 10000
   ```

2. **Monitor mean_return** - should trend upward (less negative → positive)

3. **Training time estimates** (with GPU):
   - 100K steps: ~30 minutes
   - 500K steps: ~2.5 hours
   - 1M steps: ~5 hours

4. **Memory usage**: ~4GB GPU memory for default settings

5. **Resume training** if interrupted:
   ```bash
   python train.py --agent qrdqn --crypto BTC --resume checkpoints/qrdqn_BTC_last.pt
   ```

---

## Differences from Original Paper

| Aspect | Original Paper | This Implementation |
|--------|---------------|---------------------|
| Framework | TensorFlow | PyTorch |
| Actions | BUY/SELL/HOLD | LONG/SHORT/FLAT (position-based) |
| Environment | Single-step rewards | Full position tracking with SL/TP |
| Smurf Agent | Yes (conservative gatekeeper) | No (SL/TP provides risk management) |
| Capital | Not tracked | Full capital management with leverage |

---

## License

This project is for educational and research purposes. See the original paper for academic citations.

## References

- Original Paper: [DOI: 10.1007/s00521-023-08516-x](https://doi.org/10.1007/s00521-023-08516-x)
- Original TensorFlow Code: [kochlisGit/TraderNet-CRv2](https://github.com/kochlisGit/TraderNet-CRv2)
