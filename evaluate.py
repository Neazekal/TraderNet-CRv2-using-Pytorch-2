#!/usr/bin/env python3
"""
Evaluation script for trained RL agents.

Evaluates checkpoint on test data and generates performance report.

Usage:
    python evaluate.py --checkpoint checkpoints/best.pt --crypto BTC
    python evaluate.py --checkpoint checkpoints/qrdqn_ETH_best.pt --crypto ETH --num-episodes 10
"""

import argparse
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict

from config.config import SUPPORTED_CRYPTOS, DATA_LOADING_PARAMS, EVALUATION_PARAMS, METRICS_PARAMS
from data.datasets.utils import train_eval_split
from environments.position_trading_env import PositionTradingEnv
from agents.qrdqn_agent import QRDQNAgent
from agents.categorical_sac_agent import CategoricalSACAgent
from utils.metrics import TradingMetrics
from utils.checkpoint import CheckpointManager


def detect_agent_type(checkpoint_path: str) -> str:
    """
    Detect agent type from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        'qrdqn' or 'sac'
    """
    checkpoint_path = Path(checkpoint_path)
    name = checkpoint_path.stem.lower()

    if 'sac' in name:
        return 'sac'
    elif 'qrdqn' in name or 'dqn' in name:
        return 'qrdqn'
    else:
        # Default to QR-DQN
        return 'qrdqn'


def load_agent(checkpoint_path: str, agent_type: str, device: torch.device):
    """
    Load agent from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        agent_type: 'qrdqn' or 'sac'
        device: torch device

    Returns:
        Loaded agent
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")

    if agent_type.lower() == 'qrdqn':
        agent = QRDQNAgent(device=device)
    elif agent_type.lower() == 'sac':
        agent = CategoricalSACAgent(device=device)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    checkpoint = torch.load(checkpoint_path, weights_only=False)

    if checkpoint['agent_state'] is not None:
        agent.load_state_dict(checkpoint['agent_state'])

    print(f"Agent loaded successfully (step: {checkpoint['step']})")

    return agent


def load_eval_data(crypto: str) -> pd.DataFrame:
    """
    Load evaluation dataset.

    Args:
        crypto: Cryptocurrency symbol

    Returns:
        Evaluation dataset
    """
    dataset_path = Path(f"data/datasets/{crypto}_processed.csv")

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print(f"Loading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)

    # Use last (1 - train_ratio) for evaluation
    _, eval_data = train_eval_split(
        df,
        train_ratio=DATA_LOADING_PARAMS['train_ratio'],
        shuffle=DATA_LOADING_PARAMS['shuffle']
    )
    print(f"Evaluation dataset: {len(eval_data)} samples")

    return eval_data


def evaluate_agent(
    env: PositionTradingEnv,
    agent,
    agent_type: str,
    num_episodes: int = 5,
    save_trades: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Evaluate agent and collect metrics.

    Args:
        env: Evaluation environment
        agent: Agent to evaluate
        agent_type: 'qrdqn' or 'sac'
        num_episodes: Number of episodes to run
        save_trades: Optional path to save trade log
        verbose: Print detailed output

    Returns:
        Dictionary of metrics
    """
    metrics_tracker = TradingMetrics()
    episode_returns = []
    episode_lengths = []
    all_trades = []

    print(f"\nRunning {num_episodes} evaluation episodes...")

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

            # Update metrics
            if info.get('trade_closed', False):
                metrics_tracker.update(
                    reward=reward,
                    pnl=info.get('pnl_dollars', 0),
                    trade_info=info,
                )

                # Record trade
                if save_trades:
                    all_trades.append({
                        'episode': episode,
                        'step': episode_length,
                        'action': info.get('action', 0),
                        'position': info.get('position', 0),
                        'entry_price': info.get('entry_price', 0),
                        'exit_price': info.get('exit_price', 0),
                        'pnl_percent': info.get('pnl_percent', 0),
                        'pnl_dollars': info.get('pnl_dollars', 0),
                        'duration': info.get('duration', 0),
                    })

            obs = next_obs

        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)

        if verbose:
            print(f"  Episode {episode + 1}: Return={episode_return:.4f}, Length={episode_length}")

    # Compute metrics
    computed_metrics = metrics_tracker.compute()

    result = {
        'mean_return': float(np.mean(episode_returns)),
        'std_return': float(np.std(episode_returns)),
        'median_return': float(np.median(episode_returns)),
        'min_return': float(np.min(episode_returns)),
        'max_return': float(np.max(episode_returns)),
        'mean_length': float(np.mean(episode_lengths)),
        'num_episodes': num_episodes,
    }

    # Add computed trading metrics
    result.update(computed_metrics)

    # Save trades if requested
    if save_trades and all_trades:
        trades_df = pd.DataFrame(all_trades)
        trades_df.to_csv(save_trades, index=False)
        print(f"\nTrade log saved to: {save_trades}")

    return result


def print_results(results: Dict[str, float], crypto: str, agent_type: str):
    """
    Print evaluation results in formatted manner.

    Args:
        results: Dictionary of metrics
        crypto: Cryptocurrency symbol
        agent_type: Agent type
    """
    print("\n" + "=" * 70)
    print(f"EVALUATION RESULTS - {agent_type.upper()} on {crypto}")
    print("=" * 70)

    print("\nReturn Statistics:")
    print(f"  Mean Return:        {results['mean_return']:>10.4f}")
    print(f"  Std Return:         {results['std_return']:>10.4f}")
    print(f"  Median Return:      {results['median_return']:>10.4f}")
    print(f"  Min Return:         {results['min_return']:>10.4f}")
    print(f"  Max Return:         {results['max_return']:>10.4f}")

    print("\nRisk-Adjusted Metrics:")
    print(f"  Sharpe Ratio:       {results.get('sharpe_ratio', 0):>10.2f}")
    print(f"  Sortino Ratio:      {results.get('sortino_ratio', 0):>10.2f}")
    print(f"  Max Drawdown:       {results.get('max_drawdown', 0):>10.2%}")
    print(f"  Calmar Ratio:       {results.get('calmar_ratio', 0):>10.2f}")

    print("\nTrading Statistics:")
    print(f"  Total Trades:       {int(results.get('total_trades', 0)):>10}")
    print(f"  Win Rate:           {results.get('win_rate', 0):>10.2%}")
    print(f"  Profit Factor:      {results.get('profit_factor', 0):>10.2f}")
    print(f"  Avg Trade Duration: {results.get('avg_trade_duration', 0):>10.1f}h")

    print("\nExecution Statistics:")
    print(f"  Mean Episode Length: {results['mean_length']:>9.0f}")
    print(f"  Num Episodes:       {int(results['num_episodes']):>10}")

    print("=" * 70 + "\n")


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate trained RL agent")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--crypto",
        type=str,
        choices=list(SUPPORTED_CRYPTOS.keys()),
        required=True,
        help="Cryptocurrency to evaluate on",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=5,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--save-trades",
        type=str,
        default=None,
        help="Path to save trade log CSV",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output",
    )
    parser.add_argument(
        "--agent-type",
        type=str,
        choices=["qrdqn", "sac"],
        default=None,
        help="Agent type (auto-detected if not specified)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for evaluation",
    )

    args = parser.parse_args()

    # Device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Detect agent type if not specified
    agent_type = args.agent_type
    if agent_type is None:
        agent_type = detect_agent_type(args.checkpoint)
        print(f"Detected agent type: {agent_type.upper()}")

    # Load agent
    print(f"\nLoading {agent_type.upper()} agent...")
    agent = load_agent(args.checkpoint, agent_type, device)

    # Load evaluation data
    print(f"\nLoading {args.crypto} evaluation data...")
    eval_data = load_eval_data(args.crypto)

    # Create evaluation environment
    eval_env = PositionTradingEnv(data=eval_data)

    # Evaluate
    print(f"\n{'='*70}")
    print(f"Evaluating {agent_type.upper()} on {args.crypto}")
    print(f"{'='*70}")

    results = evaluate_agent(
        env=eval_env,
        agent=agent,
        agent_type=agent_type,
        num_episodes=args.num_episodes,
        save_trades=args.save_trades,
        verbose=args.verbose,
    )

    # Print results
    print_results(results, args.crypto, agent_type)


if __name__ == "__main__":
    main()
