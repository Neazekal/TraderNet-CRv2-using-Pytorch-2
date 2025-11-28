#!/usr/bin/env python3
"""
Main training script for QR-DQN and Categorical SAC agents.

Unified interface for training both agents on cryptocurrency futures data.

Usage:
    python train.py --agent qrdqn --crypto BTC --timesteps 1000000
    python train.py --agent sac --crypto ETH --timesteps 500000
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Optional, Dict

from config.config import (
    SUPPORTED_CRYPTOS, TRAINING_PARAMS, CHECKPOINT_PARAMS, LOGGING_PARAMS,
    QR_DQN_PARAMS, CATEGORICAL_SAC_PARAMS, NUM_ACTIONS, OBS_SHAPE,
)
from data.datasets.builder import DatasetBuilder
from data.datasets.utils import train_eval_split
from environments.position_trading_env import PositionTradingEnv
from agents.qrdqn_agent import QRDQNAgent
from agents.categorical_sac_agent import CategoricalSACAgent
from utils.logger import TrainingLogger
from utils.checkpoint import CheckpointManager
from utils.metrics import TradingMetrics


def load_dataset(crypto: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and split dataset for training and evaluation.

    Args:
        crypto: Cryptocurrency symbol (e.g., 'BTC', 'ETH')

    Returns:
        Tuple of (train_data, eval_data)
    """
    dataset_path = Path(f"data/datasets/{crypto}_processed.csv")

    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        print("Please download and preprocess data first using data/downloaders/binance.py")
        raise FileNotFoundError(f"Dataset: {dataset_path}")

    # Load full dataset
    print(f"Loading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"Loaded {len(df)} samples")

    # Split into train and eval
    train_data, eval_data = train_eval_split(df, train_ratio=0.95)
    print(f"Train: {len(train_data)} samples, Eval: {len(eval_data)} samples")

    return train_data, eval_data


def create_environments(
    train_data: pd.DataFrame,
    eval_data: pd.DataFrame,
) -> tuple[PositionTradingEnv, PositionTradingEnv]:
    """
    Create training and evaluation environments.

    Args:
        train_data: Training dataset
        eval_data: Evaluation dataset

    Returns:
        Tuple of (train_env, eval_env)
    """
    train_env = PositionTradingEnv(data=train_data)
    eval_env = PositionTradingEnv(data=eval_data)

    print(f"Train environment: {len(train_data)} samples")
    print(f"Eval environment: {len(eval_data)} samples")

    return train_env, eval_env


def create_agent(
    agent_type: str,
    device: torch.device,
) -> torch.nn.Module:
    """
    Create RL agent.

    Args:
        agent_type: 'qrdqn' or 'sac'
        device: torch device

    Returns:
        Agent instance
    """
    if agent_type.lower() == "qrdqn":
        print("Creating QR-DQN agent...")
        agent = QRDQNAgent(
            num_actions=NUM_ACTIONS,
            learning_rate=QR_DQN_PARAMS['learning_rate'],
            gamma=QR_DQN_PARAMS['gamma'],
            num_quantiles=QR_DQN_PARAMS['num_quantiles'],
            device=device,
        )
    elif agent_type.lower() == "sac":
        print("Creating Categorical SAC agent...")
        agent = CategoricalSACAgent(
            num_actions=NUM_ACTIONS,
            learning_rate=CATEGORICAL_SAC_PARAMS['learning_rate'],
            gamma=CATEGORICAL_SAC_PARAMS['gamma'],
            tau=CATEGORICAL_SAC_PARAMS['tau'],
            device=device,
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    return agent


def train_qrdqn(
    env: PositionTradingEnv,
    eval_env: PositionTradingEnv,
    agent: QRDQNAgent,
    total_timesteps: int,
    eval_freq: int,
    save_freq: int,
    log_dir: str,
    checkpoint_dir: str,
    resume_checkpoint: Optional[str] = None,
):
    """
    Training loop for QR-DQN agent.

    Args:
        env: Training environment
        eval_env: Evaluation environment
        agent: QR-DQN agent
        total_timesteps: Total training steps
        eval_freq: Evaluation frequency
        save_freq: Checkpoint save frequency
        log_dir: Logging directory
        checkpoint_dir: Checkpoint directory
        resume_checkpoint: Path to checkpoint to resume from
    """
    # Initialize logger and checkpoint manager
    logger = TrainingLogger(log_dir=log_dir, experiment_name="qrdqn_training")
    checkpoint_mgr = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        metric_name=CHECKPOINT_PARAMS['metric_name'],
        mode=CHECKPOINT_PARAMS['mode'],
        keep_n_best=CHECKPOINT_PARAMS['keep_n_best'],
    )

    # Load checkpoint if provided
    if resume_checkpoint:
        checkpoint_mgr.load_checkpoint(resume_checkpoint, agent)
        start_step = agent.total_steps
    else:
        start_step = 0

    # Initialize training state
    epsilon_start = QR_DQN_PARAMS['epsilon_start']
    epsilon_end = QR_DQN_PARAMS['epsilon_end']
    epsilon_decay_frames = QR_DQN_PARAMS['epsilon_decay_frames']

    obs, _ = env.reset()
    episode_reward = 0
    episode_length = 0

    print(f"Starting QR-DQN training: {total_timesteps} steps")
    print(f"Epsilon decay: {epsilon_start:.3f} -> {epsilon_end:.3f} over {epsilon_decay_frames} steps")

    for step in range(start_step, start_step + total_timesteps):
        # Epsilon decay (linear)
        progress = min(step / epsilon_decay_frames, 1.0)
        epsilon = epsilon_start - progress * (epsilon_start - epsilon_end)

        # Select action
        action = agent.select_action(obs, epsilon=epsilon)

        # Environment step
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Add experience to replay buffer
        agent.add_experience(obs, action, reward, next_obs, done)

        # Training step
        if agent.replay_buffer.is_ready(agent.batch_size):
            metrics = agent.train_step()

            if step % TRAINING_PARAMS['log_freq'] == 0 and metrics:
                logger.log_step(step, {
                    'loss': metrics['loss'],
                    'td_error': metrics['mean_td_error'],
                    'epsilon': epsilon,
                    'buffer_size': metrics['buffer_size'],
                })

        # Episode tracking
        episode_reward += reward
        episode_length += 1

        if done:
            logger.log_episode(step, {
                'reward': episode_reward,
                'length': episode_length,
            })
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
        else:
            obs = next_obs

        # Periodic evaluation
        if (step + 1) % eval_freq == 0:
            eval_metrics = evaluate_agent(eval_env, agent, agent_type='qrdqn')
            logger.log_eval(step, eval_metrics)

            # Save checkpoint
            is_best = checkpoint_mgr._is_better(eval_metrics['mean_return'])
            checkpoint_mgr.save_checkpoint(agent, step, eval_metrics, is_best=is_best)

        # Periodic save
        if (step + 1) % save_freq == 0:
            checkpoint_mgr.save_checkpoint(agent, step, {}, is_best=False)

    logger.save()
    print("QR-DQN training complete!")


def train_categorical_sac(
    env: PositionTradingEnv,
    eval_env: PositionTradingEnv,
    agent: CategoricalSACAgent,
    total_timesteps: int,
    eval_freq: int,
    save_freq: int,
    log_dir: str,
    checkpoint_dir: str,
    resume_checkpoint: Optional[str] = None,
):
    """
    Training loop for Categorical SAC agent.

    Args:
        env: Training environment
        eval_env: Evaluation environment
        agent: Categorical SAC agent
        total_timesteps: Total training steps
        eval_freq: Evaluation frequency
        save_freq: Checkpoint save frequency
        log_dir: Logging directory
        checkpoint_dir: Checkpoint directory
        resume_checkpoint: Path to checkpoint to resume from
    """
    # Initialize logger and checkpoint manager
    logger = TrainingLogger(log_dir=log_dir, experiment_name="sac_training")
    checkpoint_mgr = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        metric_name=CHECKPOINT_PARAMS['metric_name'],
        mode=CHECKPOINT_PARAMS['mode'],
        keep_n_best=CHECKPOINT_PARAMS['keep_n_best'],
    )

    # Load checkpoint if provided
    if resume_checkpoint:
        checkpoint_mgr.load_checkpoint(resume_checkpoint, agent)
        start_step = agent.total_steps
    else:
        start_step = 0

    # Warmup: fill replay buffer with random actions
    warmup_steps = TRAINING_PARAMS.get('warmup_steps', 10000)
    print(f"Warmup: Filling replay buffer with {warmup_steps} random actions...")

    obs, _ = env.reset()
    for _ in range(warmup_steps):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.add_experience(obs, action, reward, next_obs, done)
        obs = env.reset()[0] if done else next_obs

    obs, _ = env.reset()
    episode_reward = 0
    episode_length = 0

    print(f"Starting Categorical SAC training: {total_timesteps} steps")

    for step in range(start_step, start_step + total_timesteps):
        # Select action (stochastic)
        action = agent.select_action(obs, deterministic=False)

        # Environment step
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Add experience to replay buffer
        agent.add_experience(obs, action, reward, next_obs, done)

        # Training step
        if agent.replay_buffer.is_ready(agent.batch_size):
            metrics = agent.train_step()

            if step % TRAINING_PARAMS['log_freq'] == 0 and metrics:
                logger.log_step(step, {
                    'actor_loss': metrics['actor_loss'],
                    'q1_loss': metrics['q1_loss'],
                    'q2_loss': metrics['q2_loss'],
                    'alpha': metrics['alpha'],
                    'buffer_size': metrics['buffer_size'],
                })

        # Episode tracking
        episode_reward += reward
        episode_length += 1

        if done:
            logger.log_episode(step, {
                'reward': episode_reward,
                'length': episode_length,
            })
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
        else:
            obs = next_obs

        # Periodic evaluation
        if (step + 1) % eval_freq == 0:
            eval_metrics = evaluate_agent(eval_env, agent, agent_type='sac')
            logger.log_eval(step, eval_metrics)

            # Save checkpoint
            is_best = checkpoint_mgr._is_better(eval_metrics['mean_return'])
            checkpoint_mgr.save_checkpoint(agent, step, eval_metrics, is_best=is_best)

        # Periodic save
        if (step + 1) % save_freq == 0:
            checkpoint_mgr.save_checkpoint(agent, step, {}, is_best=False)

    logger.save()
    print("Categorical SAC training complete!")


def evaluate_agent(
    env: PositionTradingEnv,
    agent,
    agent_type: str = 'qrdqn',
    num_episodes: int = 3,
) -> Dict[str, float]:
    """
    Evaluate agent on environment.

    Args:
        env: Evaluation environment
        agent: Agent to evaluate
        agent_type: 'qrdqn' or 'sac'
        num_episodes: Number of episodes to run

    Returns:
        Dictionary of evaluation metrics
    """
    episode_returns = []
    episode_lengths = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_return = 0
        episode_length = 0
        done = False

        while not done:
            # Deterministic action
            if agent_type.lower() == 'qrdqn':
                action = agent.select_action(obs, epsilon=0.0)
            else:  # sac
                action = agent.select_action(obs, deterministic=True)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_return += reward
            episode_length += 1

            obs = next_obs

        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)

    return {
        'mean_return': float(np.mean(episode_returns)),
        'std_return': float(np.std(episode_returns)),
        'mean_length': float(np.mean(episode_lengths)),
    }


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train RL agents for crypto trading")

    parser.add_argument(
        "--agent",
        type=str,
        choices=["qrdqn", "sac"],
        required=True,
        help="Agent type to train",
    )
    parser.add_argument(
        "--crypto",
        type=str,
        choices=list(SUPPORTED_CRYPTOS.keys()),
        required=True,
        help="Cryptocurrency to trade",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=TRAINING_PARAMS['total_timesteps'],
        help="Total training timesteps",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=TRAINING_PARAMS['eval_freq'],
        help="Evaluation frequency (steps)",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=TRAINING_PARAMS['save_freq'],
        help="Checkpoint save frequency (steps)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=LOGGING_PARAMS['log_dir'],
        help="Logging directory",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=TRAINING_PARAMS['seed'],
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for training",
    )

    args = parser.parse_args()

    # Set seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load dataset
    print(f"\nLoading {args.crypto} dataset...")
    train_data, eval_data = load_dataset(args.crypto)

    # Create environments
    print(f"\nCreating environments...")
    train_env, eval_env = create_environments(train_data, eval_data)

    # Create agent
    print(f"\nCreating {args.agent.upper()} agent...")
    agent = create_agent(args.agent.lower(), device)

    # Training
    print(f"\n{'='*60}")
    print(f"Training {args.agent.upper()} on {args.crypto}")
    print(f"{'='*60}\n")

    if args.agent.lower() == "qrdqn":
        train_qrdqn(
            env=train_env,
            eval_env=eval_env,
            agent=agent,
            total_timesteps=args.timesteps,
            eval_freq=args.eval_freq,
            save_freq=args.save_freq,
            log_dir=args.log_dir,
            checkpoint_dir=args.checkpoint_dir,
            resume_checkpoint=args.resume,
        )
    else:  # sac
        train_categorical_sac(
            env=train_env,
            eval_env=eval_env,
            agent=agent,
            total_timesteps=args.timesteps,
            eval_freq=args.eval_freq,
            save_freq=args.save_freq,
            log_dir=args.log_dir,
            checkpoint_dir=args.checkpoint_dir,
            resume_checkpoint=args.resume,
        )


if __name__ == "__main__":
    main()
