# Phase 6 Implementation Plan: Training & Evaluation

**Phase:** 6 of 7
**Status:** Planning
**Dependencies:** Phase 5 Complete (QR-DQN + Categorical SAC Agents)
**Estimated LOC:** 400-600 lines

---

## Overview

Phase 6 implements the training and evaluation pipeline that brings together all previous phases into a complete RL trading system. This includes training loops, checkpoint management, metrics collection, and evaluation scripts.

---

## Goals

1. **Training Pipeline**: Complete training loop for both QR-DQN and Categorical SAC
2. **Evaluation System**: Test trained agents on validation data
3. **Metrics Collection**: Trading performance metrics (PnL, Sharpe, Sortino, etc.)
4. **Checkpoint Management**: Save/load best models automatically
5. **Logging System**: Track training progress and performance

---

## File Structure

```
TraderNet-CRv2-using-Pytorch-2/
├── train.py                        # Main unified training script (NEW)
├── evaluate.py                     # Evaluation script (NEW)
├── utils/
│   ├── __init__.py                 # Utils module exports (NEW)
│   ├── logger.py                   # Training logger (NEW)
│   ├── checkpoint.py               # Checkpoint manager (NEW)
│   └── metrics.py                  # Trading metrics calculator (NEW)
├── metrics/
│   └── trading/
│       ├── __init__.py             # Already exists
│       ├── sharpe.py               # Sharpe ratio (NEW)
│       ├── sortino.py              # Sortino ratio (NEW)
│       └── drawdown.py             # Max drawdown (NEW)
└── test_phase6_training.py         # Phase 6 tests (NEW)
```

---

## Component Breakdown

### 1. Main Training Script (`train.py`)

**Purpose:** Unified training interface for both agents

**Key Features:**
- Command-line argument parsing
- Agent selection (QR-DQN or Categorical SAC)
- Cryptocurrency selection
- Load preprocessed dataset
- Create train/eval environments
- Training loop with periodic evaluation
- Automatic checkpoint saving
- Progress logging

**Usage:**
```bash
# Train QR-DQN on BTC
python train.py --agent qrdqn --crypto BTC --timesteps 1000000

# Train Categorical SAC on ETH with custom settings
python train.py --agent sac --crypto ETH --timesteps 500000 --eval-freq 5000

# Resume from checkpoint
python train.py --agent qrdqn --crypto BTC --resume checkpoints/qrdqn_BTC_best.pt
```

**Implementation:**
```python
def train(
    agent_type: str,
    crypto: str,
    total_timesteps: int,
    eval_freq: int,
    save_freq: int,
    resume_path: Optional[str] = None,
):
    """Main training function."""
    # 1. Load environment and dataset
    # 2. Create or load agent
    # 3. Training loop:
    #    - Collect experience
    #    - Train agent
    #    - Periodic evaluation
    #    - Save best checkpoint
    # 4. Final evaluation and summary
```

### 2. Evaluation Script (`evaluate.py`)

**Purpose:** Evaluate trained agents on test set

**Key Features:**
- Load trained checkpoint
- Run on evaluation environment
- Compute all trading metrics
- Generate performance report
- Optionally save trade log

**Usage:**
```bash
# Evaluate QR-DQN checkpoint
python evaluate.py --checkpoint checkpoints/qrdqn_BTC_best.pt --crypto BTC

# Evaluate with detailed output
python evaluate.py --checkpoint checkpoints/sac_ETH_best.pt --crypto ETH --verbose

# Save trade log
python evaluate.py --checkpoint checkpoints/qrdqn_BTC_best.pt --crypto BTC --save-trades trades.csv
```

**Implementation:**
```python
def evaluate(
    checkpoint_path: str,
    crypto: str,
    num_episodes: int = 1,
    save_trades: Optional[str] = None,
):
    """Evaluate trained agent."""
    # 1. Load checkpoint and create agent
    # 2. Create evaluation environment
    # 3. Run episodes with deterministic policy
    # 4. Collect metrics
    # 5. Generate report
    # 6. Optionally save trade log
```

### 3. Training Logger (`utils/logger.py`)

**Purpose:** Track and log training progress

**Features:**
- Console logging with progress bars
- File logging (CSV format for metrics)
- Tensorboard support (optional)
- Metric aggregation

**Interface:**
```python
class TrainingLogger:
    def __init__(self, log_dir: str, experiment_name: str):
        """Initialize logger."""

    def log_step(self, step: int, metrics: Dict[str, float]):
        """Log single training step."""

    def log_episode(self, episode: int, metrics: Dict[str, float]):
        """Log episode metrics."""

    def log_eval(self, step: int, eval_metrics: Dict[str, float]):
        """Log evaluation metrics."""

    def save(self):
        """Save logs to disk."""
```

### 4. Checkpoint Manager (`utils/checkpoint.py`)

**Purpose:** Save and load model checkpoints

**Features:**
- Save best model based on metric
- Save periodic checkpoints
- Load checkpoint with state restoration
- Checkpoint cleanup (keep only N best)

**Interface:**
```python
class CheckpointManager:
    def __init__(
        self,
        checkpoint_dir: str,
        metric_name: str = 'eval_return',
        mode: str = 'max',
        keep_n_best: int = 3,
    ):
        """Initialize checkpoint manager."""

    def save_checkpoint(
        self,
        agent,
        step: int,
        metrics: Dict[str, float],
        is_best: bool = False,
    ):
        """Save checkpoint."""

    def load_checkpoint(self, checkpoint_path: str, agent):
        """Load checkpoint into agent."""

    def get_best_checkpoint(self) -> Optional[str]:
        """Get path to best checkpoint."""
```

### 5. Trading Metrics (`utils/metrics.py`)

**Purpose:** Calculate trading performance metrics

**Metrics to Implement:**
- **Cumulative Return**: Total log return
- **Sharpe Ratio**: Risk-adjusted return
- **Sortino Ratio**: Downside risk-adjusted return
- **Maximum Drawdown**: Peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Average Trade Duration**: Mean holding time
- **Profit Factor**: Gross profit / gross loss
- **Calmar Ratio**: Return / max drawdown

**Interface:**
```python
class TradingMetrics:
    def __init__(self):
        """Initialize metrics tracker."""

    def update(self, reward: float, pnl: float, trade_info: Dict):
        """Update metrics with step info."""

    def reset(self):
        """Reset all metrics."""

    def compute(self) -> Dict[str, float]:
        """Compute and return all metrics."""

    def summary(self) -> str:
        """Return formatted summary."""
```

---

## Training Pipeline Architecture

### QR-DQN Training Loop

```python
def train_qrdqn(env, eval_env, agent, config):
    """Training loop for QR-DQN."""

    # Initialize
    epsilon = config['epsilon_start']
    epsilon_end = config['epsilon_end']
    epsilon_decay = config['epsilon_decay_frames']

    obs, _ = env.reset()
    episode_reward = 0
    episode_steps = 0

    logger = TrainingLogger(...)
    checkpoint_mgr = CheckpointManager(...)

    for step in range(config['total_timesteps']):
        # Epsilon decay
        epsilon = max(
            epsilon_end,
            epsilon_start - (step / epsilon_decay) * (epsilon_start - epsilon_end)
        )

        # Select action
        action = agent.select_action(obs, epsilon=epsilon)

        # Environment step
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Store experience
        agent.add_experience(obs, action, reward, next_obs, done)

        # Train agent
        if agent.replay_buffer.is_ready(agent.batch_size):
            metrics = agent.train_step()

            if step % config['log_freq'] == 0 and metrics:
                logger.log_step(step, {
                    'loss': metrics['loss'],
                    'td_error': metrics['mean_td_error'],
                    'epsilon': epsilon,
                    'buffer_size': len(agent.replay_buffer),
                })

        # Episode tracking
        episode_reward += reward
        episode_steps += 1

        if done:
            logger.log_episode(step, {
                'episode_reward': episode_reward,
                'episode_length': episode_steps,
            })
            obs, _ = env.reset()
            episode_reward = 0
            episode_steps = 0
        else:
            obs = next_obs

        # Periodic evaluation
        if step % config['eval_freq'] == 0:
            eval_metrics = evaluate_agent(eval_env, agent, num_episodes=3)
            logger.log_eval(step, eval_metrics)

            # Save checkpoint
            is_best = checkpoint_mgr.is_better(eval_metrics['mean_return'])
            checkpoint_mgr.save_checkpoint(agent, step, eval_metrics, is_best)

        # Periodic save
        if step % config['save_freq'] == 0:
            checkpoint_mgr.save_checkpoint(agent, step, {}, is_best=False)

    logger.save()
    return agent
```

### Categorical SAC Training Loop

```python
def train_categorical_sac(env, eval_env, agent, config):
    """Training loop for Categorical SAC."""

    obs, _ = env.reset()
    episode_reward = 0
    episode_steps = 0

    logger = TrainingLogger(...)
    checkpoint_mgr = CheckpointManager(...)

    # Warmup: fill replay buffer with random actions
    warmup_steps = config.get('warmup_steps', 10000)
    for step in range(warmup_steps):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.add_experience(obs, action, reward, next_obs, done)
        obs = env.reset()[0] if done else next_obs

    obs, _ = env.reset()

    for step in range(config['total_timesteps']):
        # Select action (stochastic policy for exploration)
        action = agent.select_action(obs, deterministic=False)

        # Environment step
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Store experience
        agent.add_experience(obs, action, reward, next_obs, done)

        # Train agent
        if agent.replay_buffer.is_ready(agent.batch_size):
            metrics = agent.train_step()

            if step % config['log_freq'] == 0 and metrics:
                logger.log_step(step, {
                    'actor_loss': metrics['actor_loss'],
                    'q1_loss': metrics['q1_loss'],
                    'q2_loss': metrics['q2_loss'],
                    'alpha': metrics['alpha'],
                    'entropy': metrics['entropy'],
                    'buffer_size': len(agent.replay_buffer),
                })

        # Episode tracking
        episode_reward += reward
        episode_steps += 1

        if done:
            logger.log_episode(step, {
                'episode_reward': episode_reward,
                'episode_length': episode_steps,
            })
            obs, _ = env.reset()
            episode_reward = 0
            episode_steps = 0
        else:
            obs = next_obs

        # Periodic evaluation
        if step % config['eval_freq'] == 0:
            eval_metrics = evaluate_agent(eval_env, agent, num_episodes=3)
            logger.log_eval(step, eval_metrics)

            # Save checkpoint
            is_best = checkpoint_mgr.is_better(eval_metrics['mean_return'])
            checkpoint_mgr.save_checkpoint(agent, step, eval_metrics, is_best)

        # Periodic save
        if step % config['save_freq'] == 0:
            checkpoint_mgr.save_checkpoint(agent, step, {}, is_best=False)

    logger.save()
    return agent
```

### Evaluation Function

```python
def evaluate_agent(env, agent, num_episodes: int = 5) -> Dict[str, float]:
    """Evaluate agent on environment."""

    episode_returns = []
    episode_lengths = []
    metrics_tracker = TradingMetrics()

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            # Deterministic action for evaluation
            if hasattr(agent, 'select_action'):
                # QR-DQN: epsilon=0 for greedy
                # SAC: deterministic=True for max probability
                if 'epsilon' in agent.__init__.__code__.co_varnames:
                    action = agent.select_action(obs, epsilon=0.0)
                else:
                    action = agent.select_action(obs, deterministic=True)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

            # Update metrics
            if 'pnl_dollars' in info and info.get('trade_closed', False):
                metrics_tracker.update(reward, info['pnl_dollars'], info)

            obs = next_obs

        episode_returns.append(episode_reward)
        episode_lengths.append(episode_length)

    # Compute metrics
    trading_metrics = metrics_tracker.compute()

    return {
        'mean_return': np.mean(episode_returns),
        'std_return': np.std(episode_returns),
        'mean_length': np.mean(episode_lengths),
        **trading_metrics,
    }
```

---

## Configuration Updates

Add to `config/config.py`:

```python
# =============================================================================
# Training Settings (Phase 6)
# =============================================================================
TRAINING_PARAMS = {
    'total_timesteps': 1_000_000,   # Total training steps
    'eval_freq': 10_000,            # Evaluate every N steps
    'save_freq': 50_000,            # Save checkpoint every N steps
    'log_freq': 1000,               # Log metrics every N steps
    'seed': 42,                     # Random seed for reproducibility
    'warmup_steps': 10_000,         # SAC warmup (random actions)
    'eval_episodes': 5,             # Episodes per evaluation
}

# Checkpoint settings
CHECKPOINT_PARAMS = {
    'keep_n_best': 3,               # Keep top N checkpoints
    'metric_name': 'mean_return',   # Metric to optimize
    'mode': 'max',                  # 'max' or 'min'
}

# Logging settings
LOGGING_PARAMS = {
    'log_dir': 'logs/',
    'use_tensorboard': False,       # Enable Tensorboard logging
    'console_log_freq': 100,        # Console output frequency
}
```

---

## Implementation Order

### Phase 6.1: Utilities (Estimated: 150 LOC)
1. `utils/__init__.py` - Module setup
2. `utils/metrics.py` - Trading metrics calculator (60 LOC)
3. `utils/logger.py` - Training logger (50 LOC)
4. `utils/checkpoint.py` - Checkpoint manager (40 LOC)

### Phase 6.2: Training Scripts (Estimated: 200 LOC)
1. `train.py` - Main training script with:
   - Argument parsing
   - Environment setup
   - Agent creation
   - Training loop delegation
   - QR-DQN training function
   - SAC training function

### Phase 6.3: Evaluation Scripts (Estimated: 100 LOC)
1. `evaluate.py` - Evaluation script with:
   - Checkpoint loading
   - Evaluation loop
   - Metrics computation
   - Report generation

### Phase 6.4: Metrics Implementation (Estimated: 100 LOC)
1. `metrics/trading/sharpe.py` - Sharpe ratio (30 LOC)
2. `metrics/trading/sortino.py` - Sortino ratio (30 LOC)
3. `metrics/trading/drawdown.py` - Max drawdown (40 LOC)

### Phase 6.5: Testing (Estimated: 100 LOC)
1. `test_phase6_training.py` - Unit tests:
   - Test metrics calculation
   - Test checkpoint save/load
   - Test logger functionality
   - Test training loop (small scale)

---

## Testing Strategy

### Unit Tests

```python
# test_phase6_training.py

def test_trading_metrics():
    """Test metrics calculation."""
    metrics = TradingMetrics()

    # Simulate trades
    metrics.update(reward=0.01, pnl=100, trade_info={'win': True})
    metrics.update(reward=-0.005, pnl=-50, trade_info={'win': False})

    result = metrics.compute()
    assert 'sharpe_ratio' in result
    assert 'sortino_ratio' in result
    assert 'max_drawdown' in result
    assert 'win_rate' in result

def test_checkpoint_manager():
    """Test checkpoint save/load."""
    import tempfile
    from agents import QRDQNAgent

    with tempfile.TemporaryDirectory() as tmpdir:
        agent = QRDQNAgent()
        mgr = CheckpointManager(tmpdir, metric_name='return', mode='max')

        # Save checkpoint
        mgr.save_checkpoint(agent, step=100, metrics={'return': 0.5}, is_best=True)

        # Load checkpoint
        agent2 = QRDQNAgent()
        mgr.load_checkpoint(mgr.get_best_checkpoint(), agent2)

        # Verify
        assert agent.total_steps == agent2.total_steps

def test_training_integration():
    """Test small-scale training."""
    # Create minimal environment and agent
    # Run training for 1000 steps
    # Verify metrics are logged
    # Verify checkpoints are saved
```

### Integration Test

```bash
# Test QR-DQN training (short run)
python train.py --agent qrdqn --crypto BTC --timesteps 1000 --eval-freq 500

# Test SAC training (short run)
python train.py --agent sac --crypto BTC --timesteps 1000 --eval-freq 500

# Test evaluation
python evaluate.py --checkpoint checkpoints/qrdqn_BTC_latest.pt --crypto BTC
```

---

## Success Criteria

### Phase 6 Complete When:

- ✅ Training script runs for both QR-DQN and SAC
- ✅ Evaluation script loads checkpoints and computes metrics
- ✅ Metrics are calculated correctly (Sharpe, Sortino, Drawdown)
- ✅ Checkpoints are saved and loaded without errors
- ✅ Logger tracks training progress to console and file
- ✅ All unit tests pass (100% coverage for new code)
- ✅ Full training run completes on BTC dataset
- ✅ Documentation updated with usage examples

---

## Key Challenges & Solutions

### Challenge 1: Epsilon Decay Scheduling
**Solution:** Linear decay from `epsilon_start` to `epsilon_end` over `epsilon_decay_frames`

### Challenge 2: SAC Replay Buffer Warmup
**Solution:** Pre-fill buffer with random actions before training starts

### Challenge 3: Metric Computation Timing
**Solution:** Only update metrics when trades are closed (check `trade_closed` flag)

### Challenge 4: Checkpoint Storage
**Solution:** Keep only N best checkpoints to save disk space

### Challenge 5: Environment Episode Length
**Solution:** Set max episode length based on dataset size, reset on done

---

## Example Usage

### Training QR-DQN on BTC

```bash
python train.py \
    --agent qrdqn \
    --crypto BTC \
    --timesteps 1000000 \
    --eval-freq 10000 \
    --save-freq 50000 \
    --log-dir logs/qrdqn_btc \
    --checkpoint-dir checkpoints/qrdqn_btc
```

### Training Categorical SAC on ETH

```bash
python train.py \
    --agent sac \
    --crypto ETH \
    --timesteps 500000 \
    --eval-freq 5000 \
    --warmup-steps 10000 \
    --log-dir logs/sac_eth \
    --checkpoint-dir checkpoints/sac_eth
```

### Evaluating Trained Agent

```bash
python evaluate.py \
    --checkpoint checkpoints/qrdqn_btc/qrdqn_BTC_best.pt \
    --crypto BTC \
    --num-episodes 10 \
    --save-trades results/qrdqn_btc_trades.csv \
    --verbose
```

---

## Expected Output

### Training Output (Console)

```
=== Training QR-DQN on BTC ===
Dataset: data/datasets/BTC_processed.csv
Training samples: 45000
Evaluation samples: 2250

Step    Loss    TD-Error  Epsilon  Buffer   Eval-Return  Best
-------------------------------------------------------------
0       -       -         1.000    0        -            -
1000    0.234   2.145     0.998    1000     -            -
10000   0.189   1.876     0.980    10000    0.045        ✓
20000   0.156   1.654     0.960    20000    0.052        ✓
...
```

### Evaluation Output

```
=== Evaluation Results ===
Checkpoint: checkpoints/qrdqn_BTC_best.pt
Cryptocurrency: BTC
Episodes: 10

Performance Metrics:
  Mean Return: 0.124 ± 0.032
  Sharpe Ratio: 1.82
  Sortino Ratio: 2.15
  Max Drawdown: 8.3%

Trading Statistics:
  Total Trades: 145
  Win Rate: 58.6%
  Avg Trade Duration: 12.3 hours
  Profit Factor: 1.45
  Calmar Ratio: 1.49

Trade log saved to: results/qrdqn_btc_trades.csv
```

---

## Next Steps (Phase 7)

After Phase 6 completion:
- Advanced visualizations (equity curve, drawdown plot)
- Backtesting framework
- Multi-crypto portfolio management
- Hyperparameter tuning scripts
- Performance comparison tools

---

**Estimated Timeline:** 3-5 days for full Phase 6 implementation and testing
**LOC Estimate:** 400-600 lines (utilities + scripts + tests)
**Dependencies:** Phase 5 complete ✅
