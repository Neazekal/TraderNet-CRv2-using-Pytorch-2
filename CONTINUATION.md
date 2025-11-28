# Continuation Guide for AI Assistants

**Purpose:** This document helps any AI assistant quickly understand and continue working on this project.

**Last Updated:** 2025-11-28

---

## Project Overview

**TraderNet-CRv2 PyTorch** is a Deep Reinforcement Learning system for cryptocurrency futures trading.

### Core Components
1. **Data Pipeline**: Downloads from Binance, creates 21 features
2. **Environment**: Gymnasium-compatible, position-based trading with SL/TP
3. **Agents**: QR-DQN and Categorical SAC
4. **Training**: Unified script with progress bar, saves best/last checkpoints

### Key Decisions Made
- **No Smurf agent**: Position-based environment with SL/TP provides sufficient risk management
- **21 features** (not 28): Active features only, no reserved slots
- **Position values**: LONG(+1), SHORT(-1), FLAT(0) for easy P&L math
- **Checkpoint naming**: `{agent}_{crypto}_best.pt` and `{agent}_{crypto}_last.pt`

---

## Quick Commands

```bash
# Download data
python -m data.downloaders.binance --crypto BTC

# Build dataset
python -m data.datasets.builder --crypto BTC

# Train
python train.py --agent qrdqn --crypto BTC --timesteps 100000

# Evaluate
python evaluate.py --checkpoint checkpoints/qrdqn_BTC_best.pt --crypto BTC

# Quick test (500 steps)
python train.py --agent qrdqn --crypto BTC --timesteps 500 --eval-freq 200
```

---

## File Structure (Important Files Only)

```
├── train.py                           # Main training script
├── evaluate.py                        # Evaluation script
├── config/config.py                   # ALL hyperparameters
├── environments/
│   └── position_trading_env.py        # Main trading environment
├── agents/
│   ├── qrdqn_agent.py                 # QR-DQN agent
│   └── categorical_sac_agent.py       # SAC agent
├── data/
│   ├── downloaders/binance.py         # Data download
│   └── datasets/
│       ├── builder.py                 # Dataset building
│       └── utils.py                   # prepare_training_data()
└── checkpoints/                       # Saved models
```

---

## How Training Works

### train.py Flow
1. `load_dataset(crypto)` → calls `prepare_training_data()` → returns dict with sequences/prices
2. `create_environments(data)` → creates PositionTradingEnv for train/eval
3. `create_agent(agent_type, device)` → creates QRDQNAgent or CategoricalSACAgent
4. `train_qrdqn()` or `train_categorical_sac()` → main training loop

### Training Loop (QR-DQN)
```python
for step in tqdm(range(total_timesteps)):
    # Epsilon decay
    epsilon = epsilon_start - progress * (epsilon_start - epsilon_end)
    
    # Select action (epsilon-greedy)
    action = agent.select_action(obs, epsilon=epsilon)
    
    # Environment step
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    # Store in replay buffer
    agent.add_experience(obs, action, reward, next_obs, done)
    
    # Train
    if buffer_ready:
        metrics = agent.train_step()
    
    # Evaluation & checkpointing
    if step % eval_freq == 0:
        eval_metrics = evaluate_agent(eval_env, agent)
        if mean_return improved:
            save best checkpoint
```

### Checkpoint Format
```python
# QR-DQN checkpoint
{
    'step': int,
    'agent_state': q_network.state_dict(),
    'target_state': target_q_network.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'metrics': {'mean_return': float, ...},
}

# SAC checkpoint  
{
    'step': int,
    'actor_state': actor.state_dict(),
    'q1_state', 'q2_state', 'target_q1_state', 'target_q2_state': ...,
    'actor_optimizer', 'q1_optimizer', 'q2_optimizer', 'alpha_optimizer': ...,
    'log_alpha': tensor,
    'metrics': {...},
}
```

---

## Environment Details

### PositionTradingEnv
```python
# Constructor requires:
PositionTradingEnv(
    sequences: np.ndarray,      # (num_samples, 12, 21)
    highs: np.ndarray,          # aligned with sequences
    lows: np.ndarray,
    closes: np.ndarray,
    funding_rates: np.ndarray,  # optional
)

# Action space: Discrete(3)
# - 0: LONG
# - 1: SHORT  
# - 2: FLAT

# Observation space: Box(0, 1, (12, 21))
```

### State-Action-Position Mapping
```
Current Position | Action LONG | Action SHORT | Action FLAT
-----------------|-------------|--------------|------------
FLAT (0)         | Open LONG   | Open SHORT   | Do nothing
LONG (+1)        | Keep LONG   | Flip→SHORT   | Close LONG
SHORT (-1)       | Flip→LONG   | Keep SHORT   | Close SHORT
```

---

## Agent Details

### QRDQNAgent
```python
# Key attributes
agent.q_network           # Main Q-network
agent.target_q_network    # Target network (hard update)
agent.replay_buffer       # PrioritizedReplayBuffer
agent.batch_size          # 128
agent.num_quantiles       # 51

# Key methods
agent.select_action(obs, epsilon=0.1)  # Returns action (0, 1, or 2)
agent.add_experience(obs, action, reward, next_obs, done)
agent.train_step()  # Returns {'loss': float, 'mean_td_error': float, ...}
```

### CategoricalSACAgent
```python
# Key attributes
agent.actor               # Actor network (policy)
agent.q1_network          # Q-network 1
agent.q2_network          # Q-network 2
agent.target_q1_network   # Target Q1
agent.target_q2_network   # Target Q2
agent.log_alpha           # Entropy temperature (learnable)

# Key methods
agent.select_action(obs, deterministic=False)
agent.add_experience(obs, action, reward, next_obs, done)
agent.train_step()  # Returns {'actor_loss', 'q1_loss', 'q2_loss', 'alpha', ...}
```

---

## Common Issues & Solutions

### 1. Feature dimension mismatch
**Error**: `expected input to have 28 channels, but got 21`
**Fix**: Networks use `NUM_FEATURES` from config (should be 21)

### 2. Wrong environment constructor
**Error**: `PositionTradingEnv() got unexpected argument 'data'`
**Fix**: Pass sequences, highs, lows, closes separately (not DataFrame)

### 3. train_eval_split wrong signature
**Error**: `train_eval_split() got unexpected argument 'train_ratio'`
**Fix**: Use `prepare_training_data()` instead which handles everything

---

## Configuration Reference

### config/config.py Key Settings
```python
# Features
NUM_FEATURES = 21
SEQUENCE_LENGTH = 12
FEATURES = ['log_return_open', 'log_return_high', ...]  # 21 items

# Actions
NUM_ACTIONS = 3
ACTION_LONG, ACTION_SHORT, ACTION_FLAT = 0, 1, 2

# Positions  
POSITION_LONG, POSITION_SHORT, POSITION_FLAT = 1, -1, 0

# Trading
INITIAL_CAPITAL = 10000.0
RISK_PER_TRADE = 0.02
LEVERAGE = 10
STOP_LOSS = 0.02
TAKE_PROFIT = 0.04
FEES = 0.001

# Training
TRAINING_PARAMS = {
    'total_timesteps': 1_000_000,
    'eval_freq': 10_000,
    'log_freq': 1000,
    'warmup_steps': 10_000,
}
```

---

## What Needs Work (Phase 7)

1. **Visualization**: Equity curves, drawdown plots, trade analysis
2. **Better Metrics**: Information ratio, payoff ratio, recovery factor
3. **Hyperparameter Tuning**: Optuna integration
4. **Backtesting**: Walk-forward validation

---

## Tips for AI Assistants

1. **Always check `config/config.py`** for parameter values
2. **Use `prepare_training_data()`** not raw DataFrame loading
3. **Environment needs arrays**, not DataFrame: sequences, highs, lows, closes
4. **Test changes with short runs**: `--timesteps 500 --eval-freq 200`
5. **Check checkpoint format** matches agent type when loading
