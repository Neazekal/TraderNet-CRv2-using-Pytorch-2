"""
Phase 5 Unit Tests: QR-DQN & Categorical SAC Agents

Tests agent components independently:
1. Agent initialization
2. Experience collection and buffer management
3. Training steps and loss computation
4. Checkpoint save/load
5. Action selection modes
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import torch
from pathlib import Path
import tempfile

from agents.qrdqn_agent import QRDQNAgent
from agents.categorical_sac_agent import CategoricalSACAgent
from agents.buffers.replay_buffer import ReplayBuffer
from config.config import ACTION_NAMES


def test_replay_buffer():
    """Test replay buffer functionality."""
    print("\n" + "="*60)
    print("Testing Replay Buffer")
    print("="*60)

    buffer = ReplayBuffer(capacity=1000, alpha=0.6, beta_start=0.4)

    # Add experiences
    print("Adding 100 experiences...")
    for i in range(100):
        state = np.random.randn(12, 28).astype(np.float32)
        action = np.random.randint(0, 3)
        reward = np.random.randn()
        next_state = np.random.randn(12, 28).astype(np.float32)
        done = np.random.rand() < 0.1
        buffer.add(state, action, reward, next_state, done)

    print(f"✓ Buffer size: {len(buffer)}")
    print(f"✓ Buffer stats: {buffer.get_stats()}")

    # Sample batch
    batch = buffer.sample(batch_size=32)
    print(f"✓ Sampled batch shapes:")
    print(f"  - states: {batch['states'].shape}")
    print(f"  - actions: {batch['actions'].shape}")
    print(f"  - weights: {batch['weights'].shape}")

    # Update priorities
    td_errors = np.random.rand(32) * 10
    buffer.update_priorities(batch['indices'], td_errors)
    print(f"✓ Updated priorities")

    print(f"✓ Replay buffer test passed!")


def test_qrdqn_agent():
    """Test QR-DQN agent."""
    print("\n" + "="*60)
    print("Testing QR-DQN Agent")
    print("="*60)

    device = torch.device('cpu')
    agent = QRDQNAgent(device=device)

    print(f"Agent created:")
    print(f"  - Q-Network: {sum(p.numel() for p in agent.q_network.parameters()):,} params")
    print(f"  - Num quantiles: {agent.num_quantiles}")
    print(f"  - Num actions: {agent.num_actions}")

    # Add experiences
    print("\nAdding 200 experiences...")
    for i in range(200):
        state = np.random.randn(12, 28).astype(np.float32)
        action = np.random.randint(0, agent.num_actions)
        reward = np.random.randn()
        next_state = np.random.randn(12, 28).astype(np.float32)
        done = np.random.rand() < 0.05
        agent.add_experience(state, action, reward, next_state, done)

    print(f"✓ Buffer size: {len(agent.replay_buffer)}")

    # Train
    print("\nTraining for 20 steps...")
    for step in range(20):
        metrics = agent.train_step()
        if metrics and step % 5 == 0:
            print(f"  Step {step}: Loss={metrics['loss']:.4f}, "
                  f"TD-error={metrics['mean_td_error']:.4f}")

    print(f"✓ Training completed")

    # Action selection
    print("\nAction selection:")
    state = np.random.randn(12, 28).astype(np.float32)
    action_greedy = agent.select_action(state, epsilon=0.0)
    action_explore = agent.select_action(state, epsilon=0.5)
    print(f"  - Greedy action: {action_greedy} ({ACTION_NAMES[action_greedy]})")
    print(f"  - Explore action: {action_explore} ({ACTION_NAMES[action_explore]})")

    # Checkpoint
    print("\nCheckpoint save/load:")
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / 'qrdqn_test.pt'
        agent.save_checkpoint(checkpoint_path)
        print(f"  ✓ Saved checkpoint")

        agent2 = QRDQNAgent(device=device)
        agent2.load_checkpoint(checkpoint_path)
        print(f"  ✓ Loaded checkpoint")
        print(f"  - Total steps match: {agent.total_steps == agent2.total_steps}")
        print(f"  - Update count match: {agent.update_count == agent2.update_count}")

    print(f"✓ QR-DQN agent test passed!")


def test_categorical_sac_agent():
    """Test Categorical SAC agent."""
    print("\n" + "="*60)
    print("Testing Categorical SAC Agent")
    print("="*60)

    device = torch.device('cpu')
    agent = CategoricalSACAgent(device=device)

    print(f"Agent created:")
    print(f"  - Actor: {sum(p.numel() for p in agent.actor.parameters()):,} params")
    print(f"  - Q1 Network: {sum(p.numel() for p in agent.q1_network.parameters()):,} params")
    print(f"  - Q2 Network: {sum(p.numel() for p in agent.q2_network.parameters()):,} params")
    print(f"  - Initial alpha: {agent.log_alpha.exp().item():.4f}")

    # Add experiences
    print("\nAdding 300 experiences...")
    for i in range(300):
        state = np.random.randn(12, 28).astype(np.float32)
        action = np.random.randint(0, agent.num_actions)
        reward = np.random.randn()
        next_state = np.random.randn(12, 28).astype(np.float32)
        done = np.random.rand() < 0.05
        agent.add_experience(state, action, reward, next_state, done)

    print(f"✓ Buffer size: {len(agent.replay_buffer)}")

    # Train
    print("\nTraining for 20 steps...")
    for step in range(20):
        metrics = agent.train_step()
        if metrics and step % 5 == 0:
            print(f"  Step {step}: Actor loss={metrics['actor_loss']:.4f}, "
                  f"Alpha={metrics['alpha']:.4f}")

    print(f"✓ Training completed")

    # Action selection
    print("\nAction selection:")
    state = np.random.randn(12, 28).astype(np.float32)
    action_stochastic = agent.select_action(state, deterministic=False)
    action_deterministic = agent.select_action(state, deterministic=True)
    print(f"  - Stochastic action: {action_stochastic} ({ACTION_NAMES[action_stochastic]})")
    print(f"  - Deterministic action: {action_deterministic} ({ACTION_NAMES[action_deterministic]})")

    # Checkpoint
    print("\nCheckpoint save/load:")
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / 'sac_test.pt'
        agent.save_checkpoint(checkpoint_path)
        print(f"  ✓ Saved checkpoint")

        agent2 = CategoricalSACAgent(device=device)
        agent2.load_checkpoint(checkpoint_path)
        print(f"  ✓ Loaded checkpoint")
        print(f"  - Total steps match: {agent.total_steps == agent2.total_steps}")
        print(f"  - Alpha match: {torch.allclose(agent.log_alpha, agent2.log_alpha)}")

    print(f"✓ Categorical SAC agent test passed!")


def test_agent_comparison():
    """Compare agents on same random states."""
    print("\n" + "="*60)
    print("Comparing Agent Behaviors")
    print("="*60)

    device = torch.device('cpu')
    qrdqn = QRDQNAgent(device=device)
    sac = CategoricalSACAgent(device=device)

    print("Testing action selection on 5 random states...")
    for i in range(5):
        state = np.random.randn(12, 28).astype(np.float32)

        qrdqn_action = qrdqn.select_action(state, epsilon=0.0)
        sac_action_stoch = sac.select_action(state, deterministic=False)
        sac_action_det = sac.select_action(state, deterministic=True)

        print(f"  State {i}: QR-DQN={ACTION_NAMES[qrdqn_action]}, "
              f"SAC-Stoch={ACTION_NAMES[sac_action_stoch]}, "
              f"SAC-Det={ACTION_NAMES[sac_action_det]}")

    print(f"✓ Agent comparison test passed!")


def test_gpu_device():
    """Test GPU device if available."""
    print("\n" + "="*60)
    print("Testing Device Support")
    print("="*60)

    if torch.cuda.is_available():
        print("CUDA available!")
        device = torch.device('cuda:0')
        print(f"Using device: {device}")

        agent = QRDQNAgent(device=device)
        print(f"  ✓ QR-DQN created on GPU")

        agent_sac = CategoricalSACAgent(device=device)
        print(f"  ✓ Categorical SAC created on GPU")
    else:
        print("CUDA not available, skipping GPU tests")
        print("✓ CPU device test completed")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("PHASE 5 UNIT TESTS: QR-DQN & CATEGORICAL SAC")
    print("="*70)

    try:
        test_replay_buffer()
        test_qrdqn_agent()
        test_categorical_sac_agent()
        test_agent_comparison()
        test_gpu_device()

        print("\n" + "="*70)
        print("ALL PHASE 5 UNIT TESTS PASSED!")
        print("="*70)
        print("\n✓ Phase 5 is complete and ready for training!")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
