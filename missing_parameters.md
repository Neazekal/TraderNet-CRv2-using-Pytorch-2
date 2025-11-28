# Missing Parameters from Paper (Found in Official Repo)

**Source:** https://github.com/kochlisGit/TraderNet-CRv2  
**Date:** 2025-11-26

---

## 0. Technical Indicator Default Parameters

| Indicator | Parameter | Default Value |
|-----------|-----------|---------------|
| **DEMA** | window | `15` |
| **VWAP** | window | `10` |
| **MACD** | short_window | `12` |
| **MACD** | long_window | `26` |
| **MACD** | signal_period | `9` |
| **RSI** | window | `14` |
| **STOCH** | window | `14` |
| **CCI** | window | `20` |
| **ADX** | window | `14` |
| **AROON** | window | `25` |
| **BBANDS** | window | `20` |
| **ADL** | (no window) | Computed from OHLCV |
| **OBV** | (no window) | Computed from close + volume |

### Derived Feature Calculations:
```python
# From database/preprocessing/ta/ta.py
close_dema = close - dema
close_vwap = close - vwap
bband_up_close = bband_up - close
close_bband_down = close - bband_down
adl_diffs = adl.diff()      # First-order difference
obv_diffs = obv.diff()      # First-order difference
```

### ADL & OBV Processing Pipeline:
```
ADL (raw) → adl_diffs (1st diff) → adl_diffs2 (2nd diff) → USED IN MODEL
OBV (raw) → obv_diffs (1st diff) → obv_diffs2 (2nd diff) → USED IN MODEL
```
**Note:** Raw ADL/OBV values are NOT used directly. Only second-order differences 
(`adl_diffs2`, `obv_diffs2`) are fed to the model. This captures acceleration/deceleration 
of volume flow rather than absolute values.

---

## 1. Feature Engineering Details

### Full Feature List (21 features):
```python
regression_features = [
    'open_log_returns', 'high_log_returns', 'low_log_returns',
    'close_log_returns', 'volume_log_returns', 'trades_log_returns', 'hour',
    'macd_signal_diffs', 'stoch', 'aroon_up', 'aroon_down', 'rsi', 'adx', 'cci',
    'close_dema', 'close_vwap', 'bband_up_close', 'close_bband_down', 
    'adl_diffs2', 'obv_diffs2', 'trends'
]
```

### Derived Features (not raw indicators):
- `close_dema` - Close price relative to DEMA
- `close_vwap` - Close price relative to VWAP
- `bband_up_close` - Upper Bollinger Band relative to close
- `close_bband_down` - Close relative to lower Bollinger Band
- `adl_diffs2` - ADL second-order differences
- `obv_diffs2` - OBV second-order differences
- `macd_signal_diffs` - MACD minus signal line

---

## 3. Data Configuration

| Parameter | Value |
|-----------|-------|
| **Timeframe** | 1 hour candles |
| **Data source** | CoinAPI |
| **Social indicator** | Google Trends (PyTrends library) |

### Supported Cryptocurrencies with Start Years:
```python
supported_cryptos = {
    'BTC': start_year=2017,
    'ETH': start_year=2017,
    'ADA': start_year=2017,
    'XRP': start_year=2019,
    'LTC': start_year=2018,
    'SOL': start_year=2020,
    'BNB': start_year=2019,
    'DOGE': start_year=2020,
    'MATIC': start_year=2020,
    'TRON': start_year=2018,
    'DOT': start_year=2021,
    'AVAX': start_year=2021,
    'XMR': start_year=2018,
    'BAT': start_year=2018,
    'LRC': start_year=2018
}
```

---

## 4. Implementation Framework

| Component | Library |
|-----------|---------|
| **DRL Framework** | TensorFlow + TF-Agents (NOT PyTorch) |
| **Environment** | Custom Gym environment |
| **Technical Indicators** | `ta` library (TA-Lib alternative) |
| **Google Trends** | `pytrends` library |

---

## 5. N-Consecutive Implementation

```python
class NConsecutive:
    def __init__(self, window_size: int):
        self._window_size = window_size
        self._actions_queue = []

    def filter(self, action: int) -> int:
        if len(self._actions_queue) < self._window_size:
            self._actions_queue.insert(0, action)
            return HOLD
        
        self._actions_queue.pop(-1)
        self._actions_queue.insert(0, action)
        return action if len(set(self._actions_queue)) == 1 else HOLD
```

---

## 6. Reward Function Implementation

```python
# MarketLimitOrder reward (BUY action example):
reward = np.log(highs[i: i + horizon].max() / closes[i - 1])

# SELL action:
reward = np.log(closes[i - 1] / lows[i: i + horizon].min())

# Fee calculation (applied to BUY and SELL rewards):
fees = np.log((1 - fees_percentage) / (1 + fees_percentage))
rewards[:, 0:2] += fees  # Add fees to BUY (index 0) and SELL (index 1)

# HOLD reward calculation:
hold_rewards = -rewards.max(axis=1)  # Negative of max possible reward
hold_rewards[hold_rewards > 0] = 0   # Cap at 0 (no positive hold reward for TraderNet)
```

---

## 7. RL Implementation Details (QR-DQN & Categorical SAC)

| Parameter | Value | Notes |
|-----------|-------|-------|
| **QR-DQN learning rate** | `0.0005` | Distributional value updates |
| **Num quantiles** | `51` | Standard QR-DQN setting |
| **Target update interval** | `2000` | Soft copy frequency |
| **Prioritized replay (alpha, beta_start)** | `0.6`, `0.4` | Matches README defaults |
| **Categorical SAC learning rate** | `0.0005` | Shared across policy and Q nets |
| **Tau** | `0.005` | Target smoothing |
| **Entropy target** | `-1.0` | For temperature auto-tuning |
| **Alpha init** | `0.2` | Initial temperature |
| **Batch sizes** | `128` (QR-DQN), `256` (Cat-SAC) | Off-policy updates |
| **Activation function** | `gelu` | Backbone activation |
| **Gamma (discount)** | `0.99` | Matches paper |

---

## 8. File Paths (for reference)

```python
checkpoint_dir = 'database/storage/checkpoints/'
ohlcv_history_filepath = 'database/storage/downloads/ohlcv/{}.csv'
gtrends_history_filepath = 'database/storage/downloads/gtrends/{}.csv'
dataset_save_filepath = 'database/storage/datasets/{}.csv'
```

---

## 9. Dependencies

```
Python >= 3.6
numpy
pandas
matplotlib
tensorflow
tf-agents
ta (technical analysis library)
pytrends
scikit-learn
```
