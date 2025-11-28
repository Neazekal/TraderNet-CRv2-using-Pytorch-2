# Project Status

**Last Updated:** 2025-11-28  
**Project:** TraderNet-CRv2 PyTorch Implementation  
**Status:** Phase 6 Complete (Training & Evaluation Working)

---

## Quick Summary

| Component | Status | Notes |
|-----------|--------|-------|
| Data Download | ✅ Complete | Binance Futures OHLCV + funding rates |
| Preprocessing | ✅ Complete | 21 features with Min-Max scaling |
| Trading Environment | ✅ Complete | Position-based with SL/TP |
| Neural Networks | ✅ Complete | Actor, Critic, Conv1D backbone |
| QR-DQN Agent | ✅ Complete | Distributional Q-learning |
| Categorical SAC | ✅ Complete | Entropy-regularized policy |
| Training Script | ✅ Complete | Progress bar, checkpoints |
| Evaluation Script | ✅ Complete | Full metrics report |
| Visualization | 🔲 Planned | Phase 7 |

---

## What Works Now

### Training
```bash
# Train QR-DQN on BTC
python train.py --agent qrdqn --crypto BTC --timesteps 100000

# Train SAC on ETH  
python train.py --agent sac --crypto ETH --timesteps 100000
```

**Features:**
- Real-time progress bar with metrics (loss, epsilon/alpha, episode return)
- Automatic best model saving when mean_return improves
- Saves only 2 files: `{agent}_{crypto}_best.pt` and `{agent}_{crypto}_last.pt`
- Resume training from checkpoint

### Evaluation
```bash
python evaluate.py --checkpoint checkpoints/qrdqn_BTC_best.pt --crypto BTC
```

**Output includes:**
- Return statistics (mean, std, min, max)
- Risk metrics (Sharpe, Sortino, Max Drawdown, Calmar)
- Trading stats (win rate, profit factor, total trades)

---

## Key Files

| File | Purpose |
|------|---------|
| `train.py` | Main training script |
| `evaluate.py` | Evaluation script |
| `config/config.py` | All hyperparameters |
| `environments/position_trading_env.py` | Trading environment |
| `agents/qrdqn_agent.py` | QR-DQN implementation |
| `agents/categorical_sac_agent.py` | SAC implementation |

---

## Architecture

### Trading Environment (PositionTradingEnv)
- **State**: (12 timesteps, 21 features) normalized [0,1]
- **Actions**: LONG(0), SHORT(1), FLAT(2)
- **Positions**: LONG(+1), SHORT(-1), FLAT(0)
- **Risk**: 2% per trade, 10x leverage, 2% SL, 4% TP

### QR-DQN Agent
- Conv1D(21→32) → FC(256) → FC(256) → Quantile head (51 × 3)
- Epsilon-greedy exploration: 1.0 → 0.01 over 500K steps
- Target network hard update every 2000 steps
- Prioritized replay buffer (500K capacity)

### Categorical SAC Agent
- Actor: Conv1D → FC → Softmax policy
- Twin Q-networks for critic
- Entropy temperature auto-tuning
- Soft target updates (τ=0.005)

---

## Recent Changes

### 2025-11-28
- Removed Smurf mechanism (not needed with SL/TP)
- Added progress bars (tqdm) to training
- Changed checkpoint naming: `{agent}_{crypto}_best.pt`, `{agent}_{crypto}_last.pt`
- Fixed feature channel mismatch (28 → 21)
- Fixed `evaluate.py` data loading

---

## Next Steps (Phase 7)

1. **Visualization**
   - Equity curve plotting
   - Drawdown visualization
   - Trade analysis charts

2. **Backtesting**
   - Walk-forward testing
   - Out-of-sample evaluation

3. **Hyperparameter Tuning**
   - Optuna integration
   - Learning rate scheduling

---

## Supported Cryptocurrencies

| Crypto | Symbol | Data Available From |
|--------|--------|---------------------|
| BTC | BTC/USDT | 2019 |
| ETH | ETH/USDT | 2019 |
| XRP | XRP/USDT | 2020 |
| SOL | SOL/USDT | 2021 |
| BNB | BNB/USDT | 2020 |
| TRX | TRX/USDT | 2021 |
| DOGE | DOGE/USDT | 2021 |

---

## Checkpoints

Training creates checkpoints in `checkpoints/`:
```
checkpoints/
├── qrdqn_BTC_best.pt    # Best QR-DQN model for BTC
├── qrdqn_BTC_last.pt    # Last QR-DQN model for BTC
├── sac_ETH_best.pt      # Best SAC model for ETH
└── sac_ETH_last.pt      # Last SAC model for ETH
```

Best model is saved when `mean_return` increases (higher = better).
