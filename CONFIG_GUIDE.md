# Centralized Configuration Guide (Phase 6)

All Phase 6 settings are now centralized in `config/config.py`. This provides a single source of truth for all training, evaluation, logging, metrics, and data loading parameters.

## Configuration Parameters

### 1. TRAINING_PARAMS

Controls the main training loop behavior.

```python
TRAINING_PARAMS = {
    'total_timesteps': 1_000_000,   # Total training steps
    'eval_freq': 10_000,            # Evaluate every N steps
    'save_freq': 50_000,            # Save checkpoint every N steps
    'log_freq': 1000,               # Log metrics every N steps
    'seed': 42,                     # Random seed for reproducibility
    'warmup_steps': 10_000,         # SAC warmup with random actions
    'eval_episodes': 5,             # Episodes per evaluation
}
```

**Usage:**
- Modify `total_timesteps` to train for longer/shorter
- Adjust `eval_freq` to evaluate more/less often
- Change `seed` for different random initializations

### 2. CHECKPOINT_PARAMS

Controls model checkpoint saving behavior.

```python
CHECKPOINT_PARAMS = {
    'keep_n_best': 3,               # Keep top N checkpoints
    'metric_name': 'mean_return',   # Metric to optimize ('mean_return', 'sharpe_ratio', etc.)
    'mode': 'max',                  # 'max' to maximize, 'min' to minimize
    'save_interval': 50_000,        # Save checkpoint every N steps (auto-saved)
}
```

**Usage:**
- Change `metric_name` to optimize for different metrics (e.g., 'sharpe_ratio', 'max_drawdown')
- Adjust `keep_n_best` to save more/fewer checkpoints (saves disk space)
- Use `mode: 'min'` for metrics where lower is better (e.g., 'max_drawdown')

### 3. LOGGING_PARAMS

Controls training logging and file output.

```python
LOGGING_PARAMS = {
    'log_dir': 'logs/',              # Directory for experiment logs
    'checkpoint_dir': 'checkpoints/', # Directory for model checkpoints
    'use_tensorboard': False,        # Enable TensorBoard logging (future)
    'console_log_freq': 100,         # Print console output frequency
    'save_interval': 100,            # Save logs to disk frequency
}
```

**Usage:**
- Change `log_dir` to organize logs differently
- Set `use_tensorboard: True` for TensorBoard integration (when implemented)
- Adjust `console_log_freq` to see more/fewer console updates

### 4. DATA_LOADING_PARAMS

Controls data preprocessing and splitting.

```python
DATA_LOADING_PARAMS = {
    'train_ratio': 0.95,            # Train/eval split (95% train, 5% eval)
    'shuffle': False,               # Shuffle data before splitting
    'normalize': True,              # Normalize features (uses existing scalers)
}
```

**Usage:**
- Adjust `train_ratio` for different train/eval splits (0.9 = 90% train, 10% eval)
- Set `shuffle: True` to randomly mix data (not recommended for time series)
- Set `normalize: False` if using pre-normalized data

### 5. METRICS_PARAMS

Controls trading metrics calculation.

```python
METRICS_PARAMS = {
    'risk_free_rate': 0.0,          # Annual risk-free rate (0% = default)
    'periods_per_year': 252,        # Annualization factor (252 for daily, 8760 for hourly)
    'initial_capital': 10000.0,     # Starting capital for P&L tracking
}
```

**Usage:**
- Change `risk_free_rate` if considering safe investment returns (e.g., 0.02 for 2%)
- Use `periods_per_year: 8760` if working with hourly returns (1h timeframe)
- Adjust `initial_capital` to match your simulation setup

### 6. EVALUATION_PARAMS

Controls agent evaluation behavior.

```python
EVALUATION_PARAMS = {
    'num_episodes': 5,              # Episodes per evaluation run
    'deterministic': True,          # Use greedy policy (no exploration)
    'seed': 42,                     # Evaluation seed
}
```

**Usage:**
- Increase `num_episodes` for more stable evaluation metrics
- Set `deterministic: False` to evaluate with exploration (stochastic)
- Use same seed as training for reproducibility

## Using the Configuration

### In Your Code

All Phase 6 modules automatically use centralized config:

```python
# In train.py
from config.config import TRAINING_PARAMS, CHECKPOINT_PARAMS
train_data, eval_data = train_eval_split(df, train_ratio=DATA_LOADING_PARAMS['train_ratio'])

# In utils/metrics.py
metrics = TradingMetrics()  # Automatically uses METRICS_PARAMS defaults

# In utils/logger.py
logger = TrainingLogger()  # Automatically uses LOGGING_PARAMS defaults

# In utils/checkpoint.py
checkpoint_mgr = CheckpointManager()  # Automatically uses CHECKPOINT_PARAMS defaults
```

### Command Line Overrides

You can still override config via command line:

```bash
# Use config defaults
python train.py --agent qrdqn --crypto BTC

# Override specific parameters
python train.py --agent qrdqn --crypto BTC --timesteps 500000 --eval-freq 5000

# Override directories
python train.py --agent qrdqn --crypto BTC --log-dir my_logs --checkpoint-dir my_checkpoints
```

## Common Configuration Scenarios

### Scenario 1: Quick Testing (Small Scale)

```python
TRAINING_PARAMS = {
    'total_timesteps': 10_000,      # Short training
    'eval_freq': 1_000,             # Frequent evaluation
    'save_freq': 5_000,             # Frequent saves
    'log_freq': 100,                # Verbose logging
    ...
}

CHECKPOINT_PARAMS = {
    'keep_n_best': 1,               # Save space
    ...
}
```

### Scenario 2: Long Training Run

```python
TRAINING_PARAMS = {
    'total_timesteps': 10_000_000,  # Very long training
    'eval_freq': 50_000,            # Less frequent eval
    'save_freq': 100_000,           # Less frequent saves
    'log_freq': 5000,               # Less verbose
    ...
}

CHECKPOINT_PARAMS = {
    'keep_n_best': 5,               # Keep more good models
    ...
}
```

### Scenario 3: Risk-Adjusted Optimization

```python
CHECKPOINT_PARAMS = {
    'metric_name': 'sharpe_ratio',  # Optimize for risk-adjusted returns
    'mode': 'max',
    ...
}

METRICS_PARAMS = {
    'risk_free_rate': 0.02,         # 2% risk-free rate
    'periods_per_year': 8760,       # Hourly data
    ...
}
```

### Scenario 4: Minimize Drawdown Risk

```python
CHECKPOINT_PARAMS = {
    'metric_name': 'max_drawdown',  # Optimize to minimize drawdown
    'mode': 'min',                  # min because lower is better
    ...
}
```

### Scenario 5: Different Data Split

```python
DATA_LOADING_PARAMS = {
    'train_ratio': 0.80,            # 80% train, 20% eval
    'shuffle': False,               # Keep chronological order
    ...
}
```

## Configuration Flow

```
config/config.py (source of truth)
    ↓
train.py → uses TRAINING_PARAMS, DATA_LOADING_PARAMS, etc.
    ↓
utils/metrics.py → uses METRICS_PARAMS
utils/logger.py → uses LOGGING_PARAMS
utils/checkpoint.py → uses CHECKPOINT_PARAMS
evaluate.py → uses EVALUATION_PARAMS, DATA_LOADING_PARAMS
```

## Important Notes

1. **Single Source of Truth**: All settings in `config/config.py` - no scattered magic numbers
2. **Backward Compatible**: Command line arguments still work and override config
3. **Type Hints**: All parameters are documented with types and descriptions
4. **Modular Design**: Each component (metrics, logging, checkpoint) reads its own params
5. **Easy Defaults**: Run without changes uses sensible defaults
6. **Easy Customization**: Change one place to affect entire system

## Accessing Configuration Programmatically

```python
from config.config import (
    TRAINING_PARAMS,
    CHECKPOINT_PARAMS,
    LOGGING_PARAMS,
    DATA_LOADING_PARAMS,
    METRICS_PARAMS,
    EVALUATION_PARAMS,
)

# Access any parameter
print(f"Training for {TRAINING_PARAMS['total_timesteps']} steps")
print(f"Evaluating every {TRAINING_PARAMS['eval_freq']} steps")
print(f"Risk-free rate: {METRICS_PARAMS['risk_free_rate']}")
print(f"Train ratio: {DATA_LOADING_PARAMS['train_ratio']}")
```

## Future Extensibility

To add new Phase 6 settings:

1. Add new dict to `config/config.py`:
   ```python
   NEW_COMPONENT_PARAMS = {
       'setting1': value1,
       'setting2': value2,
       ...
   }
   ```

2. Import in your module:
   ```python
   from config.config import NEW_COMPONENT_PARAMS
   ```

3. Use with defaults:
   ```python
   def __init__(self, param1=None):
       self.param1 = param1 if param1 is not None else NEW_COMPONENT_PARAMS['setting1']
   ```

## Summary

All Phase 6 components now read from a centralized configuration system in `config/config.py`:

| Component | Config Dict | Key Parameters |
|-----------|------------|-----------------|
| train.py | TRAINING_PARAMS | total_timesteps, eval_freq, save_freq, log_freq, warmup_steps |
| utils/metrics.py | METRICS_PARAMS | risk_free_rate, periods_per_year, initial_capital |
| utils/logger.py | LOGGING_PARAMS | log_dir, checkpoint_dir, console_log_freq |
| utils/checkpoint.py | CHECKPOINT_PARAMS | keep_n_best, metric_name, mode, save_interval |
| Data loading | DATA_LOADING_PARAMS | train_ratio, shuffle, normalize |
| evaluate.py | EVALUATION_PARAMS | num_episodes, deterministic, seed |

This provides complete transparency and makes it easy to experiment with different configurations without modifying code.
