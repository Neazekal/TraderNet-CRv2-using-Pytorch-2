"""
Phase 5 Integration Test: QR-DQN & Categorical SAC with Trading Environment

Tests the complete training pipeline:
1. Environment interaction
2. Experience collection
3. Agent training with both QR-DQN and Categorical SAC
4. Checkpoint save/load
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import torch
from pathlib import Path

from environments.position_trading_env import create_position_trading_env
from agents.qrdqn_agent import QRDQNAgent
from agents.categorical_sac_agent import CategoricalSACAgent
from config.config import ACTION_NAMES


def test_qrdqn_training():
    """Test QR-DQN training loop."""
    print("\n" + "="*60)
    print("Testing QR-DQN Agent Training")
    print("="*60)

    # Create environment
    env = create_position_trading_env('data/datasets/BTC_processed.csv')
    agent = QRDQNAgent(device=torch.device('cpu'))

    print(f"Environment: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.n}")
    print(f"Agent Q-network: {sum(p.numel() for p in agent.q_network.parameters()):,} params")

    # Training loop
    total_reward = 0
    episode_steps = 0
    max_steps_per_episode = 50

    obs, info = env.reset()
    print(f"\nInitial balance: ${info['balance']:,.2f}")

    # Collect experiences and train
    for step in range(500):
        # Epsilon-greedy exploration (decay epsilon over time)
        epsilon = max(0.1, 1.0 - step / 250)

        # Select and execute action
        action = agent.select_action(obs, epsilon=epsilon)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated or episode_steps >= max_steps_per_episode

        # Add to replay buffer
        agent.add_experience(obs, action, reward, next_obs, done)

        total_reward += reward
        episode_steps += 1

        # Train
        if step % 10 == 0:
            metrics = agent.train_step()
            if metrics:
                print(f"Step {step:3d} | "
                      f"Loss: {metrics['loss']:.4f} | "
                      f"TD-error: {metrics['mean_td_error']:.4f} | "
                      f"Buffer: {metrics['buffer_size']}")

        if done:
            obs, info = env.reset()
            episode_steps = 0
            print(f"  Episode end | Balance: ${info['balance']:,.2f}, "
                  f"Episode return: {total_reward:.4f}")
            total_reward = 0
        else:
            obs = next_obs

    # Save checkpoint
    checkpoint_path = Path('checkpoints/test_qrdqn.pt')
    checkpoint_path.parent.mkdir(exist_ok=True)
    agent.save_checkpoint(checkpoint_path)
    print(f"\n✓ Checkpoint saved to {checkpoint_path}")

    # Load checkpoint
    agent2 = QRDQNAgent(device=torch.device('cpu'))
    agent2.load_checkpoint(checkpoint_path)
    print(f"✓ Checkpoint loaded successfully")
    print(f"✓ QR-DQN training test passed!")


def test_categorical_sac_training():
    """Test Categorical SAC training loop."""
    print("\n" + "="*60)
    print("Testing Categorical SAC Agent Training")
    print("="*60)

    # Create environment
    env = create_position_trading_env('data/datasets/BTC_processed.csv')
    agent = CategoricalSACAgent(device=torch.device('cpu'))

    print(f"Environment: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.n}")
    print(f"Actor: {sum(p.numel() for p in agent.actor.parameters()):,} params")
    print(f"Q-networks: {sum(p.numel() for p in agent.q1_network.parameters()):,} each")

    # Training loop
    total_reward = 0
    episode_steps = 0
    max_steps_per_episode = 50

    obs, info = env.reset()
    print(f"\nInitial balance: ${info['balance']:,.2f}")

    # Collect experiences and train
    for step in range(500):
        # Stochastic action selection (exploration)
        action = agent.select_action(obs, deterministic=False)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated or episode_steps >= max_steps_per_episode

        # Add to replay buffer
        agent.add_experience(obs, action, reward, next_obs, done)

        total_reward += reward
        episode_steps += 1

        # Train
        if step % 10 == 0:
            metrics = agent.train_step()
            if metrics:
                print(f"Step {step:3d} | "
                      f"Actor loss: {metrics['actor_loss']:.4f} | "
                      f"Q-loss: {metrics['q1_loss']:.4f} | "
                      f"Alpha: {metrics['alpha']:.4f} | "
                      f"Entropy: {metrics['entropy']:.4f}")

        if done:
            obs, info = env.reset()
            episode_steps = 0
            print(f"  Episode end | Balance: ${info['balance']:,.2f}, "
                  f"Episode return: {total_reward:.4f}")
            total_reward = 0
        else:
            obs = next_obs

    # Save checkpoint
    checkpoint_path = Path('checkpoints/test_categorical_sac.pt')
    checkpoint_path.parent.mkdir(exist_ok=True)
    agent.save_checkpoint(checkpoint_path)
    print(f"\n✓ Checkpoint saved to {checkpoint_path}")

    # Load checkpoint
    agent2 = CategoricalSACAgent(device=torch.device('cpu'))
    agent2.load_checkpoint(checkpoint_path)
    print(f"✓ Checkpoint loaded successfully")
    print(f"✓ Categorical SAC training test passed!")


def test_agent_comparison():
    """Compare action selection between agents on same environment."""
    print("\n" + "="*60)
    print("Comparing Agent Action Selection")
    print("="*60)

    env = create_position_trading_env('data/datasets/BTC_processed.csv')
    qrdqn = QRDQNAgent(device=torch.device('cpu'))
    sac = CategoricalSACAgent(device=torch.device('cpu'))

    obs, info = env.reset()

    # Get actions from both agents
    qrdqn_action = qrdqn.select_action(obs, epsilon=0.0)  # Greedy
    sac_stochastic = sac.select_action(obs, deterministic=False)
    sac_deterministic = sac.select_action(obs, deterministic=True)

    print(f"Initial state shape: {obs.shape}")
    print(f"QR-DQN (greedy): {qrdqn_action} ({ACTION_NAMES[qrdqn_action]})")
    print(f"SAC (stochastic): {sac_stochastic} ({ACTION_NAMES[sac_stochastic]})")
    print(f"SAC (deterministic): {sac_deterministic} ({ACTION_NAMES[sac_deterministic]})")
    print(f"\n✓ Agent comparison test passed!")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("PHASE 5 INTEGRATION TEST: QR-DQN & CATEGORICAL SAC")
    print("="*70)

    try:
        test_qrdqn_training()
        test_categorical_sac_training()
        test_agent_comparison()

        print("\n" + "="*70)
        print("ALL PHASE 5 TESTS PASSED!")
        print("="*70)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
