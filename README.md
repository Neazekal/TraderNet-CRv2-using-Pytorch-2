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
│   └── config.py              # Hyperparameters & crypto configs
├── data/
│   ├── downloaders/
│   │   └── binance.py         # CCXT Binance Futures downloader
│   ├── preprocessing/
│   │   ├── ohlcv.py           # Log returns & hour extraction
│   │   ├── technical.py       # Derived features
│   │   └── regime.py          # Market regime detection
│   ├── datasets/
│   │   ├── builder.py         # Dataset building pipeline
│   │   └── utils.py           # Train/eval split utilities
│   └── storage/               # Raw OHLCV data (gitignored)
├── analysis/
│   └── technical/
│       └── indicators_calc.py # Technical indicator calculations
├── environments/              # (Phase 3) Trading environment
├── agents/                    # (Phase 3+) PPO agent & networks
├── rules/                     # (Phase 3+) N-Consecutive & Smurf
├── metrics/                   # (Phase 3+) Trading metrics
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
6. **Detect market regime** - Classify market conditions
7. **Min-Max scaling** - Normalize to [0, 1]
8. **Drop NaN rows** - Remove indicator warm-up period

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

### Features (20 total)

| Category | Features |
|----------|----------|
| Log Returns (5) | open, high, low, close, volume |
| Time (1) | hour |
| Trend (4) | macd_signal_diffs, aroon_up, aroon_down, adx |
| Momentum (3) | stoch, rsi, cci |
| Price Relative (4) | close_dema, close_vwap, bband_up_close, close_bband_down |
| Volume (2) | adl_diffs2, obv_diffs2 |
| Regime (1) | regime_encoded (0=TRENDING_UP, 1=TRENDING_DOWN, 2=HIGH_VOLATILITY, 3=RANGING) |

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

Each state is a sliding window of 12 hourly timesteps with 20 features:
- Shape: `(num_samples, 12, 20)`
- Used as input to the Conv1D neural network

---

## Configuration

All hyperparameters are in `config/config.py`:

```python
# Data settings
TIMEFRAME = '1h'
SEQUENCE_LENGTH = 12      # State window size
HORIZON = 20              # Reward lookahead
FEES = 0.01               # 1% transaction fee

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

# Technical indicator windows
TA_PARAMS = {
    'dema_window': 15,
    'rsi_window': 14,
    'macd_short': 12,
    'macd_long': 26,
    # ... etc
}
```

---

## 📈 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download data (takes ~10-15 minutes for all cryptos)
python -m data.downloaders.binance

# 3. Process data (takes ~2-3 minutes)
python -m data.datasets.builder

# 4. Verify data is ready
python -c "
from data.datasets.utils import prepare_training_data
data = prepare_training_data('data/datasets/BTC_processed.csv')
print(f'Train sequences: {data[\"train\"][\"sequences\"].shape}')
print(f'Eval sequences: {data[\"eval\"][\"sequences\"].shape}')
"
```

Expected output:
```
Train sequences: (52221, 12, 19)
Eval sequences: (2239, 12, 19)
```

---

## 🗺️ Roadmap

- [x] **Phase 1**: Project setup & Binance data downloader
- [x] **Phase 2**: Technical analysis & preprocessing pipeline
- [ ] **Phase 3**: Trading environment & reward functions
- [ ] **Phase 4**: Neural networks (Actor/Critic)
- [ ] **Phase 5**: PPO agent implementation
- [ ] **Phase 6**: Safety mechanisms (N-Consecutive, Smurf)
- [ ] **Phase 7**: Training & evaluation scripts
- [ ] **Phase 8**: Metrics & visualization

---

## 📚 References

1. Kochliaridis et al. (2023) - *"Combining deep reinforcement learning with technical analysis and trend monitoring on cryptocurrency markets"*
2. Schulman et al. (2017) - *"Proximal Policy Optimization Algorithms"*
3. Original implementation: [kochlisGit/TraderNet-CRv2](https://github.com/kochlisGit/TraderNet-CRv2)

---

## 📄 License

This project is for educational and research purposes.

---

## 🤝 Contributing

Contributions are welcome! Please read the [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for details on the architecture and planned features.
