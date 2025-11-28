# Project Continuation Guide

**Last Updated:** 2025-11-28
**Current Phase:** Phase 5 Complete, Ready for Phase 6
**Project:** TraderNet-CRv2 PyTorch - Deep RL Cryptocurrency Trading System

---

## Project Overview

A PyTorch implementation of TraderNet-CRv2 that combines QR-DQN (distributional DQN) and Categorical SAC with technical analysis for cryptocurrency futures trading. The system uses deep reinforcement learning to learn profitable trading strategies while managing risk through stop-loss, take-profit, and position sizing.

**Paper:** "Combining deep reinforcement learning with technical analysis and trend monitoring on cryptocurrency markets" (Neural Computing and Applications, 2023)

**Original Repo:** [kochlisGit/TraderNet-CRv2](https://github.com/kochlisGit/TraderNet-CRv2) (TensorFlow)

---

## Current Status

### Completed Phases (5 of 7)

| Phase | Status | Lines of Code | Description |
|-------|--------|---------------|-------------|
| **Phase 1** | COMPLETE | ~1,500 | Data download (Binance Futures OHLCV + Funding Rate) |
| **Phase 2** | COMPLETE | ~1,500 | Preprocessing (21 features: technical indicators + regime detection) |
| **Phase 3** | COMPLETE | ~1,300 | Trading environment (Position-based with realistic costs) |
| **Phase 4** | COMPLETE | ~529 | Neural Networks (Actor-Critic backbones) |
| **Phase 5** | COMPLETE | ~1,944 | RL Agents (QR-DQN + Categorical SAC + Replay Buffer) |
| **Phase 6** | NEXT | TBD | Training & Evaluation scripts |
| **Phase 7** | PLANNED | TBD | Metrics & Visualization |

**Total Implementation:** ~6,773 lines of production code
**Test Code:** ~453 lines (Phase 5 unit tests)
**Documentation:** ~1,800+ lines

---

## Phase 5 Completion Summary

### What Was Built

**1. Prioritized Experience Replay Buffer** ✅
- `agents/buffers/replay_buffer.py` (390 LOC)
- Store experiences with TD-error priorities
- Importance sampling for off-policy learning
- Beta annealing (0.4 → 1.0 over time)
- O(log N) efficient operations

**2. QR-DQN Agent** ✅
- `agents/qrdqn_agent.py` (505 LOC)
- Distributional Q-learning with 51 quantiles
- Quantile Huber loss with tau-weighted importance
- Target network (hard updates every 2000 steps)
- Epsilon-greedy exploration
- 206K parameters

**3. Categorical SAC Agent** ✅
- `agents/categorical_sac_agent.py` (596 LOC)
- Entropy-regularized policy learning
- Twin Q-networks (319K parameters)
- Entropy temperature auto-tuning
- Soft target updates (tau=0.005, every step)

**4. Configuration Centralization** ✅
- All settings moved to `config/config.py`
- `AGENT_TRAINING_PARAMS` for shared settings
- Enhanced `QR_DQN_PARAMS` with epsilon decay

**5. Comprehensive Testing** ✅
- `test_phase5_agents.py` (318 LOC)
- Unit tests for all components
- 100% test pass rate (CPU + GPU)

---

## Next Phase: Phase 6 - Training & Evaluation Scripts

### What Needs to Be Built

**1. Training Script** (`train.py`)
- Load preprocessed datasets
- Create train/eval environments
- Training loop with periodic evaluation
- Checkpoint management (save best models)
- Logging and metrics collection

**2. Agent-Specific Training** (`train_qrdqn.py` & `train_categorical_sac.py`)
- QR-DQN: Epsilon decay schedule
- SAC: Alpha (entropy) tracking
- Both: Replay buffer warmup, gradient accumulation

**3. Evaluation Script** (`evaluate.py`)
- Load trained checkpoints
- Run on test set (validation data)
- Compute trading metrics:
  - Cumulative returns
  - Sharpe ratio
  - Sortino ratio
  - Maximum drawdown
  - Win rate
  - Trade statistics

**4. Key Components to Implement:**

```python
def train_qrdqn(env, agent, config):
    """Training loop for QR-DQN."""
    # Initialize epsilon
    # Collect experiences
    # Train agent with replay buffer
    # Evaluate periodically
    # Save best checkpoints

def train_categorical_sac(env, agent, config):
    """Training loop for Categorical SAC."""
    # Warm up replay buffer
    # Collect experiences with stochastic policy
    # Train actor, critics, and alpha
    # Evaluate periodically
    # Save best checkpoints

def evaluate(env, agent, num_episodes):
    """Evaluate trained agent on test set."""
    # Run episodes
    # Collect metrics
    # Return statistics
```

**5. Hyperparameters** (already in `config/config.py`):

```python
QR_DQN_PARAMS = {
    'learning_rate': 5e-4,
    'gamma': 0.99,
    'num_quantiles': 51,
    'target_update_interval': 2000,
    'batch_size': 128,
    'huber_kappa': 1.0,
    'replay_buffer_size': 500_000,
    'priority_alpha': 0.6,
    'priority_beta_start': 0.4,
    'priority_beta_frames': 500_000,
}

CATEGORICAL_SAC_PARAMS = {
    'learning_rate': 5e-4,
    'gamma': 0.99,
    'tau': 0.005,
    'batch_size': 256,
    'entropy_target': -1.0,
    'alpha_init': 0.2,
    'replay_buffer_size': 500_000,
    'target_update_interval': 1,
}
```

**5. Training Settings** (already in `config/config.py`):

```python
TRAINING_PARAMS = {
    'total_timesteps': 1_000_000,  # Total training steps
    'eval_freq': 10_000,           # Evaluate every N steps
    'save_freq': 50_000,           # Save checkpoint every N steps
    'log_freq': 1000,              # Log metrics every N steps
    'seed': 42,                    # Random seed
}
```

---

## 📁 Current File Structure

```
TraderNet-CRv2-using-Pytorch-2/
├── config/
│   └── config.py                    # All hyperparameters centralized
│
├── data/
│   ├── downloaders/
│   │   └── binance.py               # CCXT Binance Futures downloader
│   ├── preprocessing/
│   │   ├── ohlcv.py                 # Log returns & hour extraction
│   │   ├── technical.py             # Derived technical features
│   │   ├── regime.py                # Market regime detection (4 states)
│   │   └── funding.py               # Funding rate processing
│   └── datasets/
│       ├── builder.py               # Complete preprocessing pipeline
│       └── utils.py                 # Train/eval split utilities
│
├── analysis/
│   └── technical/
│       └── indicators_calc.py       # 11 technical indicators
│
├── environments/
│   ├── trading_env.py               # Paper replication environment
│   ├── position_trading_env.py      # MAIN ENV: Realistic trading
│   └── rewards/
│       ├── base.py                  # Base reward class
│       ├── market_limit.py          # MarketLimitOrder reward
│       └── smurf.py                 # Smurf conservative reward
│
├── agents/
│   ├── __init__.py                  # Module exports
│   ├── networks/
│   │   ├── __init__.py              # Network exports
│   │   ├── actor.py                 # ActorNetwork (151,459 params)
│   │   └── critic.py                # CriticNetwork (150,945 params)
│   ├── buffers/                     # NEXT: Replay/quantile buffer
│   │   └── rollout_buffer.py        # TODO: Experience storage + prioritized replay
│   ├── qrdqn_agent.py               # TODO: QR-DQN training algorithm
│   └── categorical_sac_agent.py     # TODO: Categorical SAC training algorithm
│
├── metrics/                         # PLANNED: Trading metrics
│   └── trading/
│       ├── base.py                  # TODO: Base metric class
│       ├── pnl.py                   # TODO: Cumulative returns
│       ├── sharpe.py                # TODO: Sharpe ratio
│       ├── sortino.py               # TODO: Sortino ratio
│       └── drawdown.py              # TODO: Maximum drawdown
│
├── checkpoints/                     # Model checkpoints (gitignored)
├── logs/                            # Training logs (gitignored)
├── data/storage/                    # Raw CSV data (gitignored)
├── data/datasets/                   # Processed datasets (gitignored)
│
├── README.md                        # Complete user guide (810 lines)
├── CONTINUATION.md                  # This file - continuation guide
├── IMPLEMENTATION_PLAN.md           # Detailed 7-phase plan (510 lines)
├── papers.md                        # Paper summary and references
├── missing_parameters.md            # Parameters from original repo
└── requirements.txt                 # Python dependencies
```

---

## Key Design Decisions

### 1. Action Space (Position-Based)
- **Actions:** 3 discrete - LONG(0), SHORT(1), FLAT(2)
- **Position Values:** +1 (LONG), -1 (SHORT), 0 (FLAT)
- **Rationale:** Sign indicates market direction for easy P&L calculation
  ```python
  pnl = position * log(exit_price / entry_price)
  # LONG (+1):  profit when price goes up
  # SHORT (-1): profit when price goes down
  ```

### 2. Instant Position Flip
- **Behavior:** LONG→SHORT or SHORT→LONG happens in ONE step
- **Implementation:** Close current position, immediately open opposite
- **Rationale:** Faster reaction to market reversals vs waiting one step
- **Trade-off:** More complex logic, but more realistic trader behavior

### 3. Lightweight Regime Detection
- **Method:** Rolling volatility + ADX instead of Hidden Markov Model
- **Regimes:** TRENDING_UP, TRENDING_DOWN, HIGH_VOLATILITY, RANGING
- **Rationale:** HMM requires training/state estimation, rolling indicators are stateless and real-time friendly
- **Benefit:** Easier to interpret and debug

### 4. Single Timeframe (1h only)
- **Decision:** Use only 1h candles (original paper used 1h, 4h, 1d)
- **Rationale:** Simplicity for initial implementation
- **Trade-off:** Less information, but cleaner architecture

### 5. Futures-Only with Funding Fees
- **Data Source:** Binance Futures (not Spot)
- **Benefits:**
  - Can SHORT natively
  - Funding rate is valuable sentiment indicator
  - Matches real trading conditions
- **Implementation:** Funding fee applied every 8 hours (00:00, 08:00, 16:00 UTC)

### 6. Position-Based Rewards (PositionTradingEnv)
- **Reward Timing:** Only when closing positions (not unrealized P&L)
- **Rationale:** Encourages completing trades, prevents "holding forever"
- **Trade-off:** Sparse rewards during holding periods

### 7. 28 Features (21 Active + 7 Reserved)
- **Active:** 21 features currently used for training
- **Reserved:** 7 slots for future funding-derived features (moving averages, volatility, etc.)
- **Rationale:** Fixed network input size, flexibility for experimentation

---

## Neural Network Architecture (Phase 4 - Complete)

### ActorNetwork (151,459 parameters)
```
Input: (batch, 12, 28)  →  12 timesteps × 28 features

Conv1D: 28 channels → 32 filters, kernel=3
  ↓ GELU activation
Flatten: 320 features (32 filters × 10 timesteps)
  ↓
FC Layer 1: 320 → 256
  ↓ GELU activation
FC Layer 2: 256 → 256
  ↓ GELU activation
Output: 256 → 3 (categorical policy logits)
  ↓ Softmax
Action Probabilities: [P(LONG), P(SHORT), P(FLAT)]
```

**Methods:**
- `forward(state)` → action probabilities
- `get_action(state, deterministic)` → sampled action + log prob
- `evaluate_actions(states, actions)` → log probs + entropy (used by categorical SAC)

### CriticNetwork (150,945 parameters)
```
Input: (batch, 12, 28)  →  12 timesteps × 28 features

Conv1D: 28 channels → 32 filters, kernel=3
  ↓ GELU activation
Flatten: 320 features
  ↓
FC Layer 1: 320 → 256
  ↓ GELU activation
FC Layer 2: 256 → 256
  ↓ GELU activation
Output: 256 → 1
  ↓ Linear (no activation)
State Value: V(s)
```

**Methods:**
- `forward(state)` → state value
- `get_value(state)` → single value estimate
- `evaluate_states(states)` → batch values for training

**Heads:** Additional heads live in `agents/networks/heads.py` for quantile Q-values, standard Q-values, and categorical policy logits, all fed by the shared Conv1D + FC backbone.

**Weight Initialization:**
- Conv/Hidden layers: Xavier and Kaiming initialization
- Critic output layer: Uniform[-0.03, 0.03] for stable value learning

---

## Dataset & Features

### Supported Cryptocurrencies (7)
| Symbol | Pair | Data Start | Status |
|--------|------|------------|--------|
| BTC | BTC/USDT | 2019 | Available |
| ETH | ETH/USDT | 2019 | Available |
| XRP | XRP/USDT | 2020 | Available |
| SOL | SOL/USDT | 2021 | Available |
| BNB | BNB/USDT | 2020 | Available |
| TRX | TRX/USDT | 2020 | Available |
| DOGE | DOGE/USDT | 2021 | Available |

### Features (28 total)

**Active Features (21):**

| Category | Count | Features |
|----------|-------|----------|
| Log Returns | 5 | open, high, low, close, volume |
| Time | 1 | hour (0-23) |
| Trend Indicators | 4 | macd_signal_diffs, aroon_up, aroon_down, adx |
| Momentum Indicators | 3 | stoch, rsi, cci |
| Price Relative | 4 | close_dema, close_vwap, bband_up_close, close_bband_down |
| Volume | 2 | adl_diffs2, obv_diffs2 (second-order differences) |
| Market Regime | 1 | regime_encoded (0-3) |
| Funding Rate | 1 | funding_rate (raw 8-hour rate) |

**Reserved Features (7):** Placeholder slots for future funding-derived features

### Data Pipeline
```
Binance Futures API
  ↓
Raw OHLCV (hourly) + Funding Rate (8-hour)
  ↓
Log Returns + Hour Extraction
  ↓
Technical Indicators (11 indicators)
  ↓
Derived Features (price relatives, volume acceleration)
  ↓
Market Regime Detection (4 regimes)
  ↓
Merge Funding Rate (forward-fill to hourly)
  ↓
Min-Max Scaling [0, 1]
  ↓
Processed Dataset (28 features) + Scaler
  ↓
Sequences (12 timesteps, 28 features) + Train/Eval Split
```

### Train/Eval Split
- **Training:** All data except last 2250 hours
- **Evaluation:** Last 2250 hours (~3 months)

---

## Trading Environment Features

### PositionTradingEnv (Main Environment)

**Capital Management:**
- Initial capital: $10,000 (configurable)
- Position sizing: `Capital × Risk × Leverage`
- Risk per trade: 2% (configurable)
- Leverage: 10x (configurable)
- Isolated margin (max loss = risk amount)

**Risk Controls:**
- **Stop-Loss:** 2% from entry (auto-close on loss)
- **Take-Profit:** 4% from entry (2:1 reward-risk ratio)
- SL/TP checked using intra-candle high/low prices

**Realistic Costs:**
- **Transaction fees:** 0.1% per trade (Binance Futures maker/taker)
- **Funding fees:** Every 8 hours when holding position
  - Positive funding: LONG pays, SHORT receives
  - Negative funding: SHORT pays, LONG receives
- **Slippage:** Random normal distribution (mean=0.01%, std=0.02%, max=0.1%)

**Drawdown Penalty:**
- Threshold: 5% drawdown (starts penalizing)
- Penalty: 0.5 × (drawdown - threshold), capped at 0.1
- Encourages capital preservation

**Observation Space:** `Box(0, 1, (12, 28), float32)`
**Action Space:** `Discrete(3)` - LONG(0), SHORT(1), FLAT(2)

---

## Testing Commands

### Test Each Component

```bash
# 1. Test data download
python -m data.downloaders.binance

# 2. Test preprocessing
python -m data.datasets.builder

# 3. Test environment
python -c "
from environments.position_trading_env import create_position_trading_env

env = create_position_trading_env('data/datasets/BTC_processed.csv')
obs, info = env.reset()
print(f'Observation shape: {obs.shape}')
print(f'Initial balance: \${info[\"balance\"]:,.2f}')

# Open LONG
obs, reward, _, _, info = env.step(0)
print(f'Position: {info[\"position\"]}, Entry: \${info[\"entry_price\"]:,.2f}')
"

# 4. Test Actor network
python -c "
import sys
sys.path.insert(0, '.')
from agents.networks.actor import ActorNetwork
import torch

actor = ActorNetwork()
print(f'Actor parameters: {sum(p.numel() for p in actor.parameters()):,}')

state = torch.randn(12, 28)
action_probs = actor(state.unsqueeze(0))
print(f'Action probs: {action_probs}')
print(f'Sum: {action_probs.sum():.6f}')
"

# 5. Test Critic network
python -c "
import sys
sys.path.insert(0, '.')
from agents.networks.critic import CriticNetwork
import torch

critic = CriticNetwork()
print(f'Critic parameters: {sum(p.numel() for p in critic.parameters()):,}')

state = torch.randn(12, 28)
value = critic.get_value(state)
print(f'State value: {value:.4f}')
"

# 6. Run both network tests
PYTHONPATH=/home/ngkhoa/TraderNet-CRv2-using-Pytorch-2:$PYTHONPATH python agents/networks/actor.py
PYTHONPATH=/home/ngkhoa/TraderNet-CRv2-using-Pytorch-2:$PYTHONPATH python agents/networks/critic.py
```

---

## Git Workflow

### Branches

```
main                       ← Stable, all merged phases
├── phase1-project-setup   ← Phase 1 (merged)
├── phase2-preprocessing   ← Phase 2 (merged)
├── phase3-environment     ← Phase 3 (merged)
└── phase4-neural-networks ← Phase 4 (merged)

Next: phase5-rl-agents     ← Create for Phase 5
```

### Workflow for Phase 5

```bash
# 1. Create Phase 5 branch
git checkout main
git checkout -b phase5-rl-agents

# 2. Implement RL components
# ... buffer + QR-DQN + Categorical SAC ...

# 3. Test thoroughly
# ... run tests ...

# 4. Commit with clear messages
git add agents/buffers/rollout_buffer.py
git commit -m "feat(buffers): add prioritized replay with quantile targets"

git add agents/qrdqn_agent.py
git commit -m "feat(agents): add QR-DQN agent"

git add agents/categorical_sac_agent.py
git commit -m "feat(agents): add categorical SAC agent"

# 5. Push to remote
git push -u origin phase5-rl-agents

# 6. Merge to main when complete
git checkout main
git merge phase5-rl-agents --no-ff
git push origin main
```

---

## Important References

### Implementation References
1. **Original Paper:** Kochliaridis et al. (2023) - "Combining deep reinforcement learning with technical analysis and trend monitoring on cryptocurrency markets"
2. **SAC Paper:** Haarnoja et al. (2018) - "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL"
3. **QR-DQN Paper:** Dabney et al. (2018) - "Distributional RL with Quantile Regression"
4. **Original TensorFlow Repo:** [kochlisGit/TraderNet-CRv2](https://github.com/kochlisGit/TraderNet-CRv2)

### Code References
- `config/config.py` - Single source of truth for all hyperparameters
- `IMPLEMENTATION_PLAN.md` - Detailed 7-phase roadmap
- `papers.md` - Paper summary and key concepts
- `missing_parameters.md` - Parameters extracted from original repo

---

## Tips for Next Developer/Bot

### Understanding the Codebase
1. **Start with:** `README.md` for high-level overview (now includes Phase 5)
2. **Then read:** `CONTINUATION.md` (this file) for current status
3. **Check:** `config/config.py` to understand all centralized settings
4. **Review:** `PROJECT_STATUS.md` for quick status reference
5. **Deep dive:** Phase-specific sections below

### Before Starting Phase 6
1. Verify all Phase 1-5 components work:
   ```bash
   python test_phase5_agents.py  # Should pass all tests
   ```
2. Ensure data is downloaded (`data/storage/*.csv`)
3. Ensure datasets are processed (`data/datasets/*_processed.csv`)
4. Test Phase 5 agents work on environment:
   ```bash
   python -c "from agents import QRDQNAgent, CategoricalSACAgent; print('Agents OK')"
   ```
5. Create `phase6-training` branch from `main`

### During Phase 6 Implementation
1. **Read Phase 6 section** in this file (above)
2. **Follow existing patterns:**
   - Use centralized config from `config/config.py`
   - Import agents/buffer from Phase 5: `from agents import QRDQNAgent, CategoricalSACAgent`
   - Add comprehensive docstrings
   - Include test code in `if __name__ == '__main__'` blocks
   - Use type hints throughout
3. **Training Loop Architecture:**
   ```
   Load environment (PositionTradingEnv)
     ↓
   Load or create agents (QR-DQN or SAC)
     ↓
   For each timestep:
     1. Select action (epsilon-greedy or stochastic)
     2. Execute action in environment
     3. Add experience to replay buffer
     4. Train agent (if buffer has enough samples)
     5. Periodically evaluate on test set
     6. Save best checkpoint
   ```
4. **Key Integration Points:**
   - Agents use `config.QR_DQN_PARAMS` / `config.CATEGORICAL_SAC_PARAMS`
   - Environment uses `config.PositionTradingEnv` parameters
   - All training settings in `config.TRAINING_PARAMS`

### Common Pitfalls to Avoid
- Don't hardcode hyperparameters (use `config/config.py`)
- Don't forget Conv1D expects (batch, channels, seq_length)
- Don't train without warmup on empty replay buffer
- Don't mix stochastic (SAC) and epsilon-greedy (QR-DQN) settings
- Don't forget to detach tensors when converting to numpy for priorities
- Don't commit without clear, descriptive commit messages
- Don't merge to main without comprehensive testing

### What Makes This Implementation Special
1. **Realistic Trading Simulation:**
   - Funding fees (unique to futures)
   - Random slippage
   - Instant position flips
   - Intra-candle SL/TP triggers

2. **Clean Architecture:**
   - Centralized configuration
   - Modular components
   - Type hints throughout
   - Comprehensive documentation

3. **Production-Ready:**
   - Proper git workflow
   - Phase-based development
   - Extensive testing
   - Clear continuation path

---

## Summary for Next Session

**What's Done (Phase 1-5):**
- ✅ Data pipeline (Binance Futures OHLCV + Funding)
- ✅ Preprocessing (21 active features + regime detection)
- ✅ Trading environment (Realistic position-based)
- ✅ Neural networks (Actor-Critic, 302K total params)
- ✅ Prioritized Experience Replay (500K capacity, PER)
- ✅ QR-DQN Agent (206K params, distributional learning)
- ✅ Categorical SAC Agent (319K params, entropy-regularized)
- ✅ Configuration centralization (all settings in config.py)
- ✅ Comprehensive testing (Phase 5 unit tests, 100% pass rate)

**What's Next (Phase 6):**
- Training script with full pipeline
- Agent-specific training loops
- Evaluation metrics computation
- Checkpoint management
- Logging and visualization

**Ready to Start Phase 6:**
```bash
git checkout main
git checkout -b phase6-training
# Start implementing train.py and evaluate.py
```

**Expected Timeline:**
- Phase 5: 1,944 lines (COMPLETE)
- Phase 6: ~400-600 lines (Training/Eval scripts)
- Phase 7: ~300-400 lines (Metrics & Visualization)

**Total Project:** ~7,500-7,900 lines when complete

---

**Last Updated:** 2025-11-28
**Next Update:** After Phase 6 completion
**Questions?** Check README.md (Phase 5 section), PROJECT_STATUS.md, or config/config.py
