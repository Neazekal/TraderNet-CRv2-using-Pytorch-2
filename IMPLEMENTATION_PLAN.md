# TraderNet-CRv2 PyTorch Migration Plan

**Created:** 2025-11-27  
**Source:** [kochlisGit/TraderNet-CRv2](https://github.com/kochlisGit/TraderNet-CRv2)  
**Paper:** "Combining deep reinforcement learning with technical analysis and trend monitoring on cryptocurrency markets" (Neural Computing and Applications, 2023)

---

## Phase 1: Project Setup & Data Pipeline

### 1.1 Project Structure
```
tradernet_pytorch/
├── config/
│   └── config.py              # Hyperparameters, crypto configs
├── data/
│   ├── downloaders/
│   │   └── binance.py         # CCXT Binance downloader
│   ├── preprocessing/
│   │   ├── ohlcv.py           # OHLCV preprocessing, log returns
│   │   └── technical.py       # Technical indicator preprocessing
│   └── datasets/
│       ├── builder.py         # Dataset builder
│       └── utils.py           # Train/eval split utilities
├── analysis/
│   └── technical/
│       ├── indicators/        # Individual indicator implementations
│       └── config.py          # Standard TA config with window params
├── environments/
│   ├── trading_env.py         # Gymnasium environment
│   └── rewards/
│       ├── base.py            # Base reward function
│       ├── market_limit.py    # MarketLimitOrder reward
│       └── smurf.py           # Smurf reward
├── agents/
│   ├── networks/
│   │   ├── actor.py           # Actor network
│   │   └── critic.py          # Critic network
│   ├── qrdqn_agent.py         # QR-DQN agent
│   ├── categorical_sac_agent.py # Categorical SAC agent
│   └── buffers.py             # Replay/rollout buffer
├── rules/
│   └── smurfing.py            # Smurf integration wrapper
├── metrics/
│   └── trading/
│       ├── base.py            # Base metric class
│       ├── pnl.py             # Cumulative returns
│       ├── sharpe.py          # Sharpe ratio
│       ├── sortino.py         # Sortino ratio
│       └── drawdown.py        # Maximum drawdown
├── train.py
├── evaluate.py
└── requirements.txt
```

### 1.2 Dependencies
```
torch>=2.0
gymnasium
ccxt                  # Binance data via CCXT
ta                    # Technical indicators
pandas
numpy
scikit-learn
matplotlib            # Plotting results
tqdm                  # Progress bars
```

### 1.3 Phase 1 Tasks

| Task | Description | Files to Create |
|------|-------------|-----------------|
| 1.3.1 | Create directory structure | All folders + `__init__.py` |
| 1.3.2 | Create requirements.txt | `requirements.txt` |
| 1.3.3 | Create config | `config/config.py` |
| 1.3.4 | Create Binance downloader | `data/downloaders/binance.py` |

### 1.4 Config Contents (`config/config.py`)

```python
# Supported cryptocurrencies
supported_cryptos = {
    'BTC': {'symbol': 'BTC/USDT', 'start_year': 2017},
    'ETH': {'symbol': 'ETH/USDT', 'start_year': 2017},
    'ADA': {'symbol': 'ADA/USDT', 'start_year': 2017},
    # ... etc
}

# Data settings
TIMEFRAME = '1h'
SEQUENCE_LENGTH = 12      # N previous hours for state
HORIZON = 20              # K hours lookahead for reward
FEES = 0.01               # 1% transaction fee

# Feature list (20 features - no Google Trends)
FEATURES = [
    'open_log_returns', 'high_log_returns', 'low_log_returns',
    'close_log_returns', 'volume_log_returns', 'hour',
    'macd_signal_diffs', 'stoch', 'aroon_up', 'aroon_down',
    'rsi', 'adx', 'cci', 'close_dema', 'close_vwap',
    'bband_up_close', 'close_bband_down', 'adl_diffs2', 'obv_diffs2'
]

# Technical indicator windows
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

# QR-DQN hyperparameters
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

# Categorical SAC hyperparameters
CATEGORICAL_SAC_PARAMS = {
    'learning_rate': 0.0005,
    'gamma': 0.99,
    'tau': 0.005,
    'batch_size': 256,
    'entropy_target': -1.0,
    'alpha_init': 0.2,
    'replay_buffer_size': 500_000,
    'target_update_interval': 1,
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

# Paths
DATA_DIR = 'data/storage/'
CHECKPOINT_DIR = 'checkpoints/'
```

### 1.5 Binance Downloader (`data/downloaders/binance.py`)

```python
"""
CCXT-based Binance downloader for OHLCV data.

Features:
- Download hourly candles for any supported crypto
- Handle pagination for large date ranges
- Save to CSV format
- Resume from last downloaded timestamp
"""

class BinanceDownloader:
    def __init__(self, symbol: str, timeframe: str = '1h'):
        """Initialize with trading pair (e.g., 'BTC/USDT')"""
        pass
    
    def download(self, start_date: str, end_date: str = None) -> pd.DataFrame:
        """Download OHLCV data between dates"""
        pass
    
    def save(self, filepath: str):
        """Save downloaded data to CSV"""
        pass
    
    def load(self, filepath: str) -> pd.DataFrame:
        """Load existing data from CSV"""
        pass
```

### 1.6 Data Flow

```
Binance API (via CCXT)
        │
        ▼
┌─────────────────┐
│ BinanceDownloader│  → Raw OHLCV CSV
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ OHLCVPreprocessing│  → Add log returns, hour
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ TechnicalAnalysis │  → Add 12 indicators
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ DatasetBuilder   │  → Final dataset (20 features)
└─────────────────┘
        │
        ▼
   Train/Eval Split
```

---

## Phase 2: Technical Analysis & Preprocessing

### 2.1 Technical Indicators
- Implement indicator classes using `ta` library (DEMA, VWAP, MACD, RSI, STOCH, CCI, ADX, AROON, BBANDS, ADL, OBV)
- Keep same window parameters from original
- Implement derived features: `close_dema`, `close_vwap`, `adl_diffs2`, `obv_diffs2`

### 2.2 Technical Indicator Parameters

| Indicator | Parameter | Default Value |
|-----------|-----------|---------------|
| DEMA | window | 15 |
| VWAP | window | 10 |
| MACD | short_window | 12 |
| MACD | long_window | 26 |
| MACD | signal_period | 9 |
| RSI | window | 14 |
| STOCH | window | 14 |
| CCI | window | 20 |
| ADX | window | 14 |
| AROON | window | 25 |
| BBANDS | window | 20 |

### 2.3 Preprocessing Pipeline
- OHLCV log returns calculation
- Min-Max scaling to [0, 1]
- Drop NaN rows (from indicator warm-up periods)

### 2.4 Feature Engineering (20 features)
```python
features = [
    # Log returns (5)
    'open_log_returns', 'high_log_returns', 'low_log_returns',
    'close_log_returns', 'volume_log_returns',
    # Time (1)
    'hour',
    # Technical indicators (12)
    'macd_signal_diffs', 'stoch', 'aroon_up', 'aroon_down', 
    'rsi', 'adx', 'cci', 'close_dema', 'close_vwap', 
    'bband_up_close', 'close_bband_down',
    # Volume-based (2) - Second-order differences!
    'adl_diffs2', 'obv_diffs2'
]
```

### 2.5 Derived Feature Calculations
```python
close_dema = close - dema
close_vwap = close - vwap
bband_up_close = bband_up - close
close_bband_down = close - bband_down
adl_diffs = adl.diff()       # First-order difference
adl_diffs2 = adl_diffs.diff() # Second-order difference (USED IN MODEL)
obv_diffs = obv.diff()
obv_diffs2 = obv_diffs.diff() # Second-order difference (USED IN MODEL)
```

---

## Phase 3: Environment (Gymnasium)

### 3.1 TradingEnvironment
```python
class TradingEnv(gymnasium.Env):
    observation_space: Box(shape=(12, 20))  # N=12 timesteps, 20 features
    action_space: Discrete(3)  # BUY=0, SELL=1, HOLD=2
```

### 3.2 Environment Parameters
| Parameter | Value |
|-----------|-------|
| Sequence length (N) | 12 |
| Horizon (K) | 20 |
| Fees | 1% (0.01) |

### 3.3 Reward Functions

#### MarketLimitOrder Reward (TraderNet)
```python
def compute_rewards(highs, lows, closes, horizon, fees):
    rewards = np.zeros((len(closes) - horizon, 3))
    for i in range(timeframe_size, len(closes) - horizon + 1):
        # BUY: log(max_high / close)
        rewards[i, 0] = np.log(highs[i:i+horizon].max() / closes[i-1])
        # SELL: log(close / min_low)
        rewards[i, 1] = np.log(closes[i-1] / lows[i:i+horizon].min())
    
    # Apply fees
    fee_adjustment = np.log((1 - fees) / (1 + fees))
    rewards[:, 0:2] += fee_adjustment
    
    # HOLD: negative of max possible reward, capped at 0
    hold_rewards = -rewards.max(axis=1)
    hold_rewards[hold_rewards > 0] = 0
    rewards[:, 2] = hold_rewards
    
    return rewards
```

#### Smurf Reward
```python
# Same as MarketLimitOrder but:
# - hold_reward = 0.0055 (positive constant)
smurf_rf[:, 2] = 0.0055
```

---

## Phase 4: Neural Networks (PyTorch)

### 4.1 Actor Network
```python
class ActorNetwork(nn.Module):
    """
    Architecture:
    - Conv1D(in_channels=20, out_channels=32, kernel_size=3)
    - Flatten
    - Linear(flatten_size, 256)
    - Linear(256, 256)
    - Linear(256, 3)  # 3 actions
    - Softmax
    
    Activation: GELU
    """
```

### 4.2 Critic Network
```python
class CriticNetwork(nn.Module):
    """
    Architecture:
    - Conv1D(in_channels=20, out_channels=32, kernel_size=3)
    - Flatten
    - Linear(flatten_size, 256)
    - Linear(256, 256)
    - Linear(256, 1)  # Value output
    
    Activation: GELU
    """
```

### 4.3 Network Parameters
| Parameter | Value |
|-----------|-------|
| Conv layers | [(32, 3)] (filters, kernel) |
| FC layers | [256, 256] |
| Activation | GELU |
| Output init | Uniform[-0.03, 0.03] for value head |

---

## Phase 5: RL Agents (QR-DQN & Categorical SAC)

### 5.1 Key Hyperparameters
| Agent | Highlight Params |
|-------|------------------|
| QR-DQN | lr=5e-4, gamma=0.99, num_quantiles=51, batch_size=128, target_update=2000, prioritized replay (alpha=0.6, beta_start=0.4) |
| Categorical SAC | lr=5e-4, gamma=0.99, tau=0.005, batch_size=256, entropy_target=-1.0, alpha_init=0.2, target_update=1 |

### 5.2 Implementation Details
- Shared Conv1D + FC backbone for all heads.
- **QR-DQN:** quantile regression Huber loss; target network updates; prioritized replay buffer.
- **Categorical SAC:** twin Q-networks, categorical policy logits, entropy temperature auto-tuning; soft target updates.
- Replay buffer supports priorities and optional n-step returns.
- Evaluation policy: greedy (QR-DQN) / max-prob or deterministic (Cat-SAC).

### 5.3 Core Losses (sketch)
```python
# QR-DQN quantile Huber loss
td_errors = target_quantiles - predicted_quantiles
huber = huber_loss(td_errors, kappa)
quantile_weights = torch.abs(tau_hat - (td_errors < 0).float())
loss = (quantile_weights * huber).mean()

# Categorical SAC losses
q1, q2 = twin_q(states, actions)
target_v = torch.min(q1_target, q2_target) - alpha * log_pi
critic_loss = F.mse_loss(q1, target_v) + F.mse_loss(q2, target_v)
policy_loss = (alpha * log_pi - torch.min(q1_policy, q2_policy)).mean()
alpha_loss = -(log_alpha * (log_pi + entropy_target).detach()).mean()
```

---

## Phase 6: Metrics & Evaluation

(NOTE: Actions are executed immediately with position-based policy:
LONG->LONG holds position, LONG->SHORT flips immediately, etc.)

### 6.1 Trading Metrics

| Metric | Formula |
|--------|---------|
| Cumulative Returns (CR) | `sum(log_returns)` |
| Cumulative PNL (CP) | `exp(CR)` |
| Investment Risk (IR) | `std(log_returns)` |
| Sharpe Ratio | `exp(mean(returns) / std(returns))` |
| Sortino Ratio | `exp(mean(returns) / std(negative_returns))` |
| Maximum Drawdown (MDD) | `1 - min(cumsum / peak)` |

### 6.2 Metric Implementations
```python
class CumulativeLogReturn(Metric):
    def update(self, log_pnl: float):
        self._log_pnl_sum += log_pnl
    
    def result(self) -> float:
        return self._log_pnl_sum

class SharpeRatio(Metric):
    def result(self) -> float:
        returns = np.array(self._episode_log_pnls)
        return np.exp(returns.mean() / returns.std())

class SortinoRatio(Metric):
    def result(self) -> float:
        returns = np.array(self._episode_log_pnls)
        downside_std = returns[returns < 0].std()
        return np.exp(returns.mean() / downside_std)

class MaximumDrawdown(Metric):
    def result(self) -> float:
        return 1 - min(self._hourly_mdds)
```

---

## Phase 7: Training Pipeline

### 7.1 Data Split
- Training: All data except last 2250 hours
- Evaluation: Last 2250 hours (~3 months)
- Different evaluation timelines per crypto to avoid overlapping market conditions

### 7.2 Training Flow
1. Load preprocessed dataset
2. Create train/eval environments
3. Train TraderNet (QR-DQN with MarketLimitOrder reward)
4. Train Smurf (Categorical SAC with Smurf reward)
5. Save best checkpoints by average return

### 7.3 Integrated Evaluation
1. Load TraderNet and Smurf checkpoints
2. Run integrated policy on evaluation set
3. Compute all trading metrics

### 7.4 Supported Cryptocurrencies
```python
supported_cryptos = {
    'BTC': {'symbol': 'BTC/USDT', 'start_year': 2017},
    'ETH': {'symbol': 'ETH/USDT', 'start_year': 2017},
    'ADA': {'symbol': 'ADA/USDT', 'start_year': 2017},
    'XRP': {'symbol': 'XRP/USDT', 'start_year': 2019},
    'LTC': {'symbol': 'LTC/USDT', 'start_year': 2018},
    'SOL': {'symbol': 'SOL/USDT', 'start_year': 2020},
    'BNB': {'symbol': 'BNB/USDT', 'start_year': 2019},
    'DOGE': {'symbol': 'DOGE/USDT', 'start_year': 2020},
    'MATIC': {'symbol': 'MATIC/USDT', 'start_year': 2020},
    'DOT': {'symbol': 'DOT/USDT', 'start_year': 2021},
    'AVAX': {'symbol': 'AVAX/USDT', 'start_year': 2021},
}
```

---

## Implementation Order

| Step | Task | Priority | Estimated Effort |
|------|------|----------|------------------|
| 1 | Project structure + config | High | Low |
| 2 | Data preprocessing (port from TF) | High | Medium |
| 3 | Technical indicators | High | Medium |
| 4 | Gymnasium environment | High | Medium |
| 5 | Reward functions | High | Low |
| 6 | Actor/Critic networks | High | Medium |
| 7 | RL agents (QR-DQN & Categorical SAC) | High | High |
| 8 | Smurf mechanism | Medium | Low |
| 9 | Metrics | Medium | Low |
| 10 | Training script | High | Medium |
| 11 | Evaluation script | Medium | Medium |

---

## Key Differences from Original

| Aspect | Original (TensorFlow) | PyTorch Version |
|--------|----------------------|-----------------|
| Framework | TensorFlow 2.x + TF-Agents | PyTorch 2.x |
| Environment | gym + TFPyEnvironment wrappers | Gymnasium |
| Agents | TF-Agents clipped policy-gradient agent | QR-DQN + Categorical SAC (custom) |
| Networks | tf.keras.layers | torch.nn.Module |
| Replay Buffer | TFUniformReplayBuffer | Custom RolloutBuffer |
| Checkpointing | TF Checkpointer | torch.save/load_state_dict |
| Data types | tf.float32 | torch.float32 |

---

## References

1. Paper: Kochliaridis et al. (2023) "Combining deep reinforcement learning with technical analysis and trend monitoring on cryptocurrency markets"
2. Original repo: https://github.com/kochlisGit/TraderNet-CRv2
3. SAC paper: Haarnoja et al. (2018) "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL"
4. QR-DQN paper: Dabney et al. (2018) "Distributional RL with Quantile Regression"
