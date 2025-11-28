#!/usr/bin/env python3
"""
Main training script for QR-DQN and Categorical SAC agents.

Unified interface for training both agents on cryptocurrency futures data.
Supports multi-GPU training when multiple GPUs are available.

Usage:
    python train.py --agent qrdqn --crypto BTC --timesteps 1000000
    python train.py --agent sac --crypto ETH --timesteps 500000
    python train.py --agent qrdqn --crypto BTC --timesteps 1000000 --no-multi-gpu
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Optional, Dict
from tqdm import tqdm

from config.config import (
    SUPPORTED_CRYPTOS, TRAINING_PARAMS, CHECKPOINT_PARAMS, LOGGING_PARAMS,
    DATA_LOADING_PARAMS, METRICS_PARAMS, EVALUATION_PARAMS,
    QR_DQN_PARAMS, CATEGORICAL_SAC_PARAMS, NUM_ACTIONS, OBS_SHAPE,
    EVAL_HOURS, SEQUENCE_LENGTH, FEATURES,
)
from data.datasets.utils import prepare_training_data
from environments.position_trading_env import PositionTradingEnv
from agents.qrdqn_agent import QRDQNAgent
from agents.categorical_sac_agent import CategoricalSACAgent
from utils.logger import TrainingLogger
from utils.checkpoint import CheckpointManager
from utils.metrics import TradingMetrics
from utils.distributed import print_gpu_info, get_device


def load_dataset(crypto: str) -> dict:
    """
    Load and prepare dataset for training and evaluation.

    Args:
        crypto: Cryptocurrency symbol (e.g., 'BTC', 'ETH')

    Returns:
        Dictionary with train/eval data including sequences and prices
    """
    dataset_path = Path(f"data/datasets/{crypto}_processed.csv")

    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        print("Please download and preprocess data first:")
        print("  python -m data.downloaders.binance")
        print("  python -m data.datasets.builder")
        raise FileNotFoundError(f"Dataset: {dataset_path}")

    # Load and prepare all data
    print(f"Loading dataset: {dataset_path}")
    data = prepare_training_data(str(dataset_path))
    
    print(f"Train: {len(data['train']['sequences'])} samples")
    print(f"Eval: {len(data['eval']['sequences'])} samples")

    return data


def create_environments(data: dict) -> tuple[PositionTradingEnv, PositionTradingEnv]:
    """
    Create training and evaluation environments.

    Args:
        data: Dictionary from prepare_training_data with sequences and prices

    Returns:
        Tuple of (train_env, eval_env)
    """
    # Get funding rates if available
    train_funding = None
    eval_funding = None
    
    if 'funding_rate' in data['train']['df'].columns:
        train_funding = data['train']['df']['funding_rate'].values[SEQUENCE_LENGTH-1:].astype(np.float32)
    if 'funding_rate' in data['eval']['df'].columns:
        eval_funding = data['eval']['df']['funding_rate'].values[SEQUENCE_LENGTH-1:].astype(np.float32)
    
    train_env = PositionTradingEnv(
        sequences=data['train']['sequences'],
        highs=data['train']['highs'],
        lows=data['train']['lows'],
        closes=data['train']['closes'],
        funding_rates=train_funding,
    )
    
    eval_env = PositionTradingEnv(
        sequences=data['eval']['sequences'],
        highs=data['eval']['highs'],
        lows=data['eval']['lows'],
        closes=data['eval']['closes'],
        funding_rates=eval_funding,
    )

    print(f"Train environment: {len(data['train']['sequences'])} steps")
    print(f"Eval environment: {len(data['eval']['sequences'])} steps")

    return train_env, eval_env


def create_agent(
    agent_type: str,
    device: torch.device,
    use_multi_gpu: bool = True,
) -> torch.nn.Module:
    """
    Create RL agent.

    Args:
        agent_type: 'qrdqn' or 'sac'
        device: torch device
        use_multi_gpu: Whether to enable multi-GPU training

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
            use_multi_gpu=use_multi_gpu,
        )
    elif agent_type.lower() == "sac":
        print("Creating Categorical SAC agent...")
        agent = CategoricalSACAgent(
            num_actions=NUM_ACTIONS,
            learning_rate=CATEGORICAL_SAC_PARAMS['learning_rate'],
            gamma=CATEGORICAL_SAC_PARAMS['gamma'],
            tau=CATEGORICAL_SAC_PARAMS['tau'],
            device=device,
            use_multi_gpu=use_multi_gpu,
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
    log_dir: str,
    checkpoint_dir: str,
    crypto: str,
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
        log_dir: Logging directory
        checkpoint_dir: Checkpoint directory
        crypto: Cryptocurrency symbol (for checkpoint naming)
        resume_checkpoint: Path to checkpoint to resume from
    """
    # Initialize logger and checkpoint manager
    logger = TrainingLogger(log_dir=log_dir, experiment_name=f"qrdqn_{crypto}")
    checkpoint_mgr = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        metric_name=CHECKPOINT_PARAMS['metric_name'],
        mode=CHECKPOINT_PARAMS['mode'],
        keep_n_best=CHECKPOINT_PARAMS['keep_n_best'],
    )
    
    # Checkpoint naming with crypto
    best_checkpoint_name = f"qrdqn_{crypto}_best.pt"
    last_checkpoint_name = f"qrdqn_{crypto}_last.pt"

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
    num_episodes = 0
    last_loss = 0.0

    print(f"Starting QR-DQN training: {total_timesteps} steps")
    print(f"Epsilon decay: {epsilon_start:.3f} -> {epsilon_end:.3f} over {epsilon_decay_frames} steps")

    # Progress bar
    pbar = tqdm(range(start_step, start_step + total_timesteps), desc="Training QR-DQN", unit="step")
    
    for step in pbar:
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

            if metrics:
                last_loss = metrics['loss']
                
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
            num_episodes += 1
            logger.log_episode(step, {
                'reward': episode_reward,
                'length': episode_length,
            })
            # Update progress bar with episode info
            pbar.set_postfix({
                'eps': f"{epsilon:.3f}",
                'loss': f"{last_loss:.4f}",
                'ep_ret': f"{episode_reward:.2f}",
                'episodes': num_episodes,
            })
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
        else:
            obs = next_obs

        # Periodic evaluation
        if (step + 1) % eval_freq == 0:
            pbar.set_description("Evaluating...")
            eval_metrics = evaluate_agent(eval_env, agent, agent_type='qrdqn')
            logger.log_eval(step, eval_metrics)
            pbar.set_description("Training QR-DQN")

            # Save best checkpoint if improved
            if checkpoint_mgr._is_better(eval_metrics['mean_return']):
                checkpoint_mgr.best_value = eval_metrics['mean_return']
                best_path = Path(checkpoint_dir) / best_checkpoint_name
                torch.save({
                    'step': step,
                    'agent_state': agent.q_network.state_dict(),
                    'target_state': agent.target_q_network.state_dict(),
                    'optimizer_state': agent.optimizer.state_dict(),
                    'metrics': eval_metrics,
                }, best_path)
                print(f"\n✓ New best checkpoint saved: {best_path} (mean_return: {eval_metrics['mean_return']:.4f})")
    
    pbar.close()
    
    # Save last checkpoint
    last_path = Path(checkpoint_dir) / last_checkpoint_name
    torch.save({
        'step': step,
        'agent_state': agent.q_network.state_dict(),
        'target_state': agent.target_q_network.state_dict(),
        'optimizer_state': agent.optimizer.state_dict(),
        'metrics': {},
    }, last_path)
    print(f"Last checkpoint saved: {last_path}")

    logger.save()
    print("QR-DQN training complete!")


def train_categorical_sac(
    env: PositionTradingEnv,
    eval_env: PositionTradingEnv,
    agent: CategoricalSACAgent,
    total_timesteps: int,
    eval_freq: int,
    log_dir: str,
    checkpoint_dir: str,
    crypto: str,
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
        log_dir: Logging directory
        checkpoint_dir: Checkpoint directory
        crypto: Cryptocurrency symbol (for checkpoint naming)
        resume_checkpoint: Path to checkpoint to resume from
    """
    # Initialize logger and checkpoint manager
    logger = TrainingLogger(log_dir=log_dir, experiment_name=f"sac_{crypto}")
    checkpoint_mgr = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        metric_name=CHECKPOINT_PARAMS['metric_name'],
        mode=CHECKPOINT_PARAMS['mode'],
        keep_n_best=CHECKPOINT_PARAMS['keep_n_best'],
    )
    
    # Checkpoint naming with crypto
    best_checkpoint_name = f"sac_{crypto}_best.pt"
    last_checkpoint_name = f"sac_{crypto}_last.pt"

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
    for _ in tqdm(range(warmup_steps), desc="Warmup", unit="step"):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.add_experience(obs, action, reward, next_obs, done)
        obs = env.reset()[0] if done else next_obs

    obs, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    num_episodes = 0
    last_actor_loss = 0.0

    print(f"Starting Categorical SAC training: {total_timesteps} steps")

    # Progress bar
    pbar = tqdm(range(start_step, start_step + total_timesteps), desc="Training SAC", unit="step")
    
    for step in pbar:
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

            if metrics:
                last_actor_loss = metrics['actor_loss']
                
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
            num_episodes += 1
            logger.log_episode(step, {
                'reward': episode_reward,
                'length': episode_length,
            })
            # Update progress bar with episode info
            pbar.set_postfix({
                'alpha': f"{agent.log_alpha.exp().item():.3f}",
                'loss': f"{last_actor_loss:.4f}",
                'ep_ret': f"{episode_reward:.2f}",
                'episodes': num_episodes,
            })
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
        else:
            obs = next_obs

        # Periodic evaluation
        if (step + 1) % eval_freq == 0:
            pbar.set_description("Evaluating...")
            eval_metrics = evaluate_agent(eval_env, agent, agent_type='sac')
            logger.log_eval(step, eval_metrics)
            pbar.set_description("Training SAC")

            # Save best checkpoint if improved
            if checkpoint_mgr._is_better(eval_metrics['mean_return']):
                checkpoint_mgr.best_value = eval_metrics['mean_return']
                best_path = Path(checkpoint_dir) / best_checkpoint_name
                torch.save({
                    'step': step,
                    'actor_state': agent.actor.state_dict(),
                    'q1_state': agent.q1_network.state_dict(),
                    'q2_state': agent.q2_network.state_dict(),
                    'target_q1_state': agent.target_q1_network.state_dict(),
                    'target_q2_state': agent.target_q2_network.state_dict(),
                    'actor_optimizer': agent.actor_optimizer.state_dict(),
                    'q1_optimizer': agent.q1_optimizer.state_dict(),
                    'q2_optimizer': agent.q2_optimizer.state_dict(),
                    'alpha_optimizer': agent.alpha_optimizer.state_dict(),
                    'log_alpha': agent.log_alpha.data,
                    'metrics': eval_metrics,
                }, best_path)
                print(f"\n✓ New best checkpoint saved: {best_path} (mean_return: {eval_metrics['mean_return']:.4f})")

    pbar.close()
    
    # Save last checkpoint
    last_path = Path(checkpoint_dir) / last_checkpoint_name
    torch.save({
        'step': step,
        'actor_state': agent.actor.state_dict(),
        'q1_state': agent.q1_network.state_dict(),
        'q2_state': agent.q2_network.state_dict(),
        'target_q1_state': agent.target_q1_network.state_dict(),
        'target_q2_state': agent.target_q2_network.state_dict(),
        'actor_optimizer': agent.actor_optimizer.state_dict(),
        'q1_optimizer': agent.q1_optimizer.state_dict(),
        'q2_optimizer': agent.q2_optimizer.state_dict(),
        'alpha_optimizer': agent.alpha_optimizer.state_dict(),
        'log_alpha': agent.log_alpha.data,
        'metrics': {},
    }, last_path)
    print(f"Last checkpoint saved: {last_path}")
    
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
    parser.add_argument(
        "--no-multi-gpu",
        action="store_true",
        help="Disable multi-GPU training even if multiple GPUs are available",
    )

    args = parser.parse_args()
    
    # Create checkpoint directory
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Set seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device and multi-GPU setup
    device = torch.device(args.device)
    use_multi_gpu = not args.no_multi_gpu
    
    # Print GPU information
    print_gpu_info()
    print(f"\nUsing device: {device}")
    if use_multi_gpu:
        print(f"Multi-GPU training: ENABLED")
    else:
        print(f"Multi-GPU training: DISABLED")

    # Load dataset
    print(f"\nLoading {args.crypto} dataset...")
    data = load_dataset(args.crypto)

    # Create environments
    print(f"\nCreating environments...")
    train_env, eval_env = create_environments(data)

    # Create agent
    print(f"\nCreating {args.agent.upper()} agent...")
    agent = create_agent(args.agent.lower(), device, use_multi_gpu)

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
            log_dir=args.log_dir,
            checkpoint_dir=args.checkpoint_dir,
            crypto=args.crypto,
            resume_checkpoint=args.resume,
        )
    else:  # sac
        train_categorical_sac(
            env=train_env,
            eval_env=eval_env,
            agent=agent,
            total_timesteps=args.timesteps,
            eval_freq=args.eval_freq,
            log_dir=args.log_dir,
            checkpoint_dir=args.checkpoint_dir,
            crypto=args.crypto,
            resume_checkpoint=args.resume,
        )


if __name__ == "__main__":
    main()
