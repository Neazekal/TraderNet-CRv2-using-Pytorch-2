# Project Continuation Guide

**Last Updated:** 2025-11-27  
**Current Phase:** Phase 3 Complete, Ready for Phase 4

---

## Current Status

### Completed Phases

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | COMPLETE | Data download (Binance Futures OHLCV + Funding Rate) |
| Phase 2 | COMPLETE | Preprocessing (Technical indicators + Regime detection) |
| Phase 3 | COMPLETE | Trading environment (Position-based with realistic features) |

### Next Phase: Phase 4 - Neural Networks

Implement the Actor-Critic networks for PPO:

1. **ActorNetwork** (`agents/networks/actor.py`)
   - Input: (batch, 12, 21) - sequence of 12 timesteps with 21 features
   - Conv1D(21, 32, kernel=3) -> Flatten -> FC(256) -> FC(256) -> FC(3) -> Softmax
   - Output: Action probabilities for LONG/SHORT/FLAT

2. **CriticNetwork** (`agents/networks/critic.py`)
   - Same architecture but outputs single value (state value)
   - Conv1D(21, 32, kernel=3) -> Flatten -> FC(256) -> FC(256) -> FC(1)

3. **Key parameters** (from config.py):
   ```python
   NETWORK_PARAMS = {
       'conv_filters': 32,
       'conv_kernel': 3,
       'fc_layers': [256, 256],
       'activation': 'gelu',
   }
   ```

---

## Key Design Decisions Made

### 1. Action Space
- **Decision:** 3 actions (LONG=0, SHORT=1, FLAT=2)
- **Position values:** +1 (LONG), -1 (SHORT), 0 (FLAT)
- **Reason:** Sign indicates market direction for easy P&L calculation

### 2. Instant Position Flip
- **Decision:** LONG->SHORT or SHORT->LONG happens in one step
- **How:** Close current position, immediately open opposite
- **Reason:** Faster reaction to market reversals vs waiting one step

### 3. No HMM for Regime Detection
- **Decision:** Use simple rolling volatility + ADX instead
- **Reason:** HMM requires more compute, rolling indicators are lightweight for live trading

### 4. Single Timeframe
- **Decision:** Only 1h timeframe (removed 4h, 1D)
- **Reason:** Simplicity for initial implementation

### 5. Futures Only
- **Decision:** Use Binance Futures data, not Spot
- **Reason:** Can SHORT, has funding rate data, matches real trading

### 6. Funding Fee Simulation
- **Decision:** Apply real funding fees every 8 hours (00:00, 08:00, 16:00 UTC)
- **Reason:** Teaches agent cost of holding positions long-term

---

## File Structure Summary

```
Key files to understand:
├── config/config.py           # ALL hyperparameters centralized here
├── data/
│   ├── downloaders/binance.py # Downloads OHLCV + funding from Binance Futures
│   ├── preprocessing/
│   │   ├── technical.py       # Computes 11 technical indicators
│   │   └── regime.py          # Market regime detection (4 states)
│   └── datasets/builder.py    # Builds final processed dataset
├── environments/
│   ├── trading_env.py         # Paper replication (single-step rewards)
│   └── position_trading_env.py # MAIN ENV: realistic trading
└── analysis/technical/
    └── indicators_calc.py     # Individual indicator calculations
```

---

## Environment Features Summary

The `PositionTradingEnv` includes:
- Capital management (balance, equity tracking)
- Position sizing (risk_per_trade * leverage)
- Stop-Loss / Take-Profit (auto-trigger)
- Funding fee (8-hour intervals)
- Random slippage (configurable mean/std)
- Drawdown penalty (configurable threshold)
- Instant position flip (LONG<->SHORT)
- Transaction fees on open/close

---

## How to Continue

### For Phase 4 (Neural Networks):

```python
# Example actor network structure
import torch
import torch.nn as nn
from config.config import NETWORK_PARAMS, OBS_SHAPE, NUM_ACTIONS

class ActorNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv1D: input shape (batch, seq_len=12, features=21)
        # PyTorch Conv1d expects (batch, channels, seq_len)
        # So need to transpose: (batch, 21, 12)
        self.conv = nn.Conv1d(
            in_channels=OBS_SHAPE[1],  # 21 features
            out_channels=NETWORK_PARAMS['conv_filters'],  # 32
            kernel_size=NETWORK_PARAMS['conv_kernel']  # 3
        )
        self.activation = nn.GELU()
        
        # After conv: (batch, 32, 12-3+1) = (batch, 32, 10)
        flatten_size = 32 * 10
        
        self.fc1 = nn.Linear(flatten_size, NETWORK_PARAMS['fc_layers'][0])
        self.fc2 = nn.Linear(NETWORK_PARAMS['fc_layers'][0], NETWORK_PARAMS['fc_layers'][1])
        self.output = nn.Linear(NETWORK_PARAMS['fc_layers'][1], NUM_ACTIONS)
        
    def forward(self, x):
        # x: (batch, 12, 21)
        x = x.transpose(1, 2)  # (batch, 21, 12)
        x = self.activation(self.conv(x))
        x = x.flatten(1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = torch.softmax(self.output(x), dim=-1)
        return x
```

### For Phase 5 (PPO):

Key hyperparameters in config.py:
```python
PPO_PARAMS = {
    'learning_rate': 0.0005,
    'epsilon_clip': 0.3,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'num_epochs': 40,
    'batch_size': 128,
}
```

---

## Testing Commands

```bash
# Test data download
python -m data.downloaders.binance

# Test preprocessing
python -m data.datasets.builder

# Test environment
python -c "
from environments.position_trading_env import create_position_trading_env
env = create_position_trading_env('data/datasets/BTC_processed.csv')
obs, info = env.reset()
print(f'Obs shape: {obs.shape}')
print(f'Balance: \${info[\"balance\"]:,.2f}')
"
```

---

## Git Branches

- `main` - Stable, merged phases
- `phase-1-data-pipeline` - Data download (merged)
- `phase-2-preprocessing` - Technical indicators (merged)  
- `phase-3-environment` - Trading environment (merged)

For Phase 4, create: `phase-4-neural-networks`
