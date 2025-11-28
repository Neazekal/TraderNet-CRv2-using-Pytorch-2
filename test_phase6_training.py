"""
Unit tests for Phase 6: Training & Evaluation

Tests for:
- Trading metrics calculation
- Logger functionality
- Checkpoint manager
- Training loops (small scale)
- Evaluation
"""

import pytest
import tempfile
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from utils.metrics import TradingMetrics
from utils.logger import TrainingLogger
from utils.checkpoint import CheckpointManager
from metrics.trading.sharpe import calculate_sharpe_ratio, SharpeRatioCalculator
from metrics.trading.sortino import calculate_sortino_ratio, SortinoRatioCalculator
from metrics.trading.drawdown import (
    calculate_max_drawdown,
    calculate_max_drawdown_detailed,
    DrawdownCalculator,
)


class TestTradingMetrics:
    """Test TradingMetrics class."""

    def test_initialization(self):
        """Test metric initialization."""
        metrics = TradingMetrics()
        assert metrics.risk_free_rate == 0.0
        assert len(metrics.rewards) == 0
        assert metrics.equity_curve == [10000.0]

    def test_reset(self):
        """Test reset functionality."""
        metrics = TradingMetrics()
        metrics.all_returns.append(0.01)
        metrics.reset()
        assert len(metrics.all_returns) == 0
        assert len(metrics.pnls) == 0

    def test_update_with_trade(self):
        """Test update with trade info."""
        metrics = TradingMetrics()
        metrics.update(0.01, 100.0, {'trade_closed': True})
        assert len(metrics.pnls) == 1
        assert metrics.pnls[0] == 100.0

    def test_cumulative_return(self):
        """Test cumulative return calculation."""
        metrics = TradingMetrics()
        # Simulate 10% gain
        metrics.update(0.01, 1000.0, {})
        result = metrics.compute()
        assert 'cumulative_return' in result
        assert result['cumulative_return'] == pytest.approx(0.1, abs=0.01)

    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        metrics = TradingMetrics()
        # Add returns
        for _ in range(100):
            metrics.all_returns.append(0.001)

        result = metrics.compute()
        assert 'sharpe_ratio' in result
        assert isinstance(result['sharpe_ratio'], float)

    def test_win_rate(self):
        """Test win rate calculation."""
        metrics = TradingMetrics()
        # 3 wins, 1 loss
        metrics.update(0.01, 100.0, {'trade_closed': True})
        metrics.update(0.01, 100.0, {'trade_closed': True})
        metrics.update(0.01, 100.0, {'trade_closed': True})
        metrics.update(0.01, -50.0, {'trade_closed': True})

        result = metrics.compute()
        assert result['win_rate'] == pytest.approx(0.75, abs=0.01)

    def test_max_drawdown(self):
        """Test maximum drawdown calculation."""
        metrics = TradingMetrics()
        # Create a V-shaped equity curve
        metrics.update(0.01, 0.0, {})
        metrics.update(-0.05, -500.0, {})
        metrics.update(-0.05, -500.0, {})
        metrics.update(0.01, 1000.0, {})

        result = metrics.compute()
        assert 'max_drawdown' in result
        assert result['max_drawdown'] >= 0

    def test_summary(self):
        """Test summary string generation."""
        metrics = TradingMetrics()
        metrics.update(0.01, 100.0, {'trade_closed': True})
        summary = metrics.summary()
        assert "Trading Metrics" in summary
        assert "Sharpe Ratio" in summary
        assert "Win Rate" in summary


class TestTrainingLogger:
    """Test TrainingLogger class."""

    def test_initialization(self):
        """Test logger initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TrainingLogger(log_dir=tmpdir, experiment_name="test")
            assert logger.log_dir == Path(tmpdir)
            assert logger.experiment_name == "test"
            assert (logger.exp_dir / "steps.csv").parent.exists()

    def test_log_step(self):
        """Test logging a training step."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TrainingLogger(log_dir=tmpdir)
            logger.log_step(0, {'loss': 0.5, 'reward': 0.01})
            assert logger.step_count == 1

    def test_log_episode(self):
        """Test logging an episode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TrainingLogger(log_dir=tmpdir)
            logger.log_episode(0, {'reward': 100.0, 'length': 50})
            assert logger.episode_count == 1

    def test_log_eval(self):
        """Test logging evaluation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TrainingLogger(log_dir=tmpdir)
            logger.log_eval(1000, {'mean_return': 0.05, 'std_return': 0.02})
            assert logger.eval_count == 1

    def test_csv_creation(self):
        """Test CSV file creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TrainingLogger(log_dir=tmpdir, experiment_name="csv_test")
            logger.log_step(0, {'loss': 0.5})
            logger.log_episode(0, {'reward': 100})
            logger.log_eval(0, {'mean_return': 0.05})

            assert logger.step_log_file.exists()
            assert logger.episode_log_file.exists()
            assert logger.eval_log_file.exists()

    def test_multiple_logs(self):
        """Test multiple log entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TrainingLogger(log_dir=tmpdir)
            for i in range(5):
                logger.log_step(i, {'loss': 0.5 - i * 0.01})
            assert logger.step_count == 5


class TestCheckpointManager:
    """Test CheckpointManager class."""

    def test_initialization(self):
        """Test checkpoint manager initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(checkpoint_dir=tmpdir, mode='max')
            assert mgr.checkpoint_dir == Path(tmpdir)
            assert mgr.mode == 'max'
            assert mgr.best_value == float('-inf')

    def test_is_better_max(self):
        """Test metric comparison for maximization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(checkpoint_dir=tmpdir, mode='max')
            assert mgr._is_better(0.5) is True
            assert mgr._is_better(0.3) is False

    def test_is_better_min(self):
        """Test metric comparison for minimization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(checkpoint_dir=tmpdir, mode='min')
            assert mgr._is_better(0.5) is True
            assert mgr._is_better(0.8) is False

    def test_save_checkpoint(self):
        """Test checkpoint saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(checkpoint_dir=tmpdir)

            # Create a simple model
            model = torch.nn.Linear(10, 3)

            # Mock agent with state_dict
            class MockAgent:
                def state_dict(self):
                    return model.state_dict()

            agent = MockAgent()
            path = mgr.save_checkpoint(agent, step=100, metrics={'return': 0.5})
            assert path.exists()

    def test_get_best_checkpoint(self):
        """Test getting best checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(checkpoint_dir=tmpdir)

            # No checkpoint yet
            assert mgr.get_best_checkpoint() is None

            # Create and save checkpoint
            class MockAgent:
                def state_dict(self):
                    return {}

            agent = MockAgent()
            mgr.save_checkpoint(agent, step=100, metrics={'return': 0.5}, is_best=True)

            # Now should have best checkpoint
            best = mgr.get_best_checkpoint()
            assert best is not None
            assert best.name == "best.pt"

    def test_keep_n_best(self):
        """Test keeping only N best checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(checkpoint_dir=tmpdir, mode='max', keep_n_best=2)

            class MockAgent:
                def state_dict(self):
                    return {}

            agent = MockAgent()

            # Save 4 checkpoints with increasing metric
            for i in range(4):
                mgr.save_checkpoint(
                    agent,
                    step=i * 100,
                    metrics={'return': float(i) * 0.1},
                    is_best=(i >= 2),  # Last 2 are best
                )

            # Should only have 2 in best list
            assert len(mgr.best_checkpoints) <= 2


class TestSharpeRatio:
    """Test Sharpe ratio calculation."""

    def test_constant_returns(self):
        """Test Sharpe ratio with constant returns."""
        returns = np.array([0.01, 0.01, 0.01, 0.01])
        sharpe = calculate_sharpe_ratio(returns)
        # Constant returns should have 0 volatility
        assert sharpe == pytest.approx(0.0, abs=1e-6)

    def test_normal_returns(self):
        """Test Sharpe ratio with normal returns."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.01, 252)
        sharpe = calculate_sharpe_ratio(returns)
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)

    def test_calculator(self):
        """Test incremental Sharpe ratio calculator."""
        calc = SharpeRatioCalculator()
        for _ in range(100):
            calc.add_return(0.001)

        sharpe = calc.calculate()
        assert isinstance(sharpe, float)

    def test_with_risk_free_rate(self):
        """Test Sharpe ratio with non-zero risk-free rate."""
        returns = np.array([0.02, 0.02, 0.02, 0.02])
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.01)
        assert isinstance(sharpe, float)


class TestSortinoRatio:
    """Test Sortino ratio calculation."""

    def test_no_losses(self):
        """Test Sortino with no losses."""
        returns = np.array([0.01, 0.01, 0.01, 0.01])
        sortino = calculate_sortino_ratio(returns)
        assert sortino >= 0

    def test_with_losses(self):
        """Test Sortino with both gains and losses."""
        returns = np.array([0.02, 0.02, -0.02, 0.01, -0.01, 0.02])
        sortino = calculate_sortino_ratio(returns)
        assert isinstance(sortino, float)
        assert not np.isnan(sortino)

    def test_calculator(self):
        """Test incremental Sortino calculator."""
        calc = SortinoRatioCalculator()
        returns = [0.02, -0.01, 0.03, -0.02, 0.01]
        for ret in returns:
            calc.add_return(ret)

        sortino = calc.calculate()
        assert isinstance(sortino, float)


class TestDrawdown:
    """Test drawdown calculation."""

    def test_no_drawdown(self):
        """Test with monotonically increasing equity."""
        equity = np.array([1000, 1100, 1200, 1300])
        dd = calculate_max_drawdown(equity)
        assert dd == pytest.approx(0.0, abs=1e-6)

    def test_simple_drawdown(self):
        """Test simple V-shaped drawdown."""
        equity = np.array([1000, 1100, 900, 1050])  # 100/1100 = 9.1% drawdown
        dd = calculate_max_drawdown(equity)
        assert dd > 0
        assert dd < 0.15

    def test_detailed_drawdown(self):
        """Test detailed drawdown calculation."""
        equity = np.array([1000, 1200, 900, 1100])
        dd, peak_idx, trough_idx = calculate_max_drawdown_detailed(equity)
        assert dd > 0
        assert peak_idx < trough_idx
        assert equity[peak_idx] > equity[trough_idx]

    def test_calculator(self):
        """Test incremental drawdown calculator."""
        calc = DrawdownCalculator()
        equity_values = [1000, 1100, 1000, 900, 1050, 1200]
        for val in equity_values:
            calc.add_value(val)

        dd = calc.calculate_max_drawdown()
        assert dd > 0


# Integration tests


class TestSmallScaleTraining:
    """Small-scale training integration tests."""

    @pytest.mark.skip(reason="Requires full environment setup")
    def test_qrdqn_training_step(self):
        """Test QR-DQN training loop (small scale)."""
        # This test requires full environment and agent setup
        # Skipped for now as it requires data files
        pass

    @pytest.mark.skip(reason="Requires full environment setup")
    def test_sac_training_step(self):
        """Test SAC training loop (small scale)."""
        # This test requires full environment and agent setup
        # Skipped for now as it requires data files
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
