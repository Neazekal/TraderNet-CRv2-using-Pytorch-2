"""
Checkpoint manager for saving and loading model states.

Features:
- Save best model based on metric
- Save periodic checkpoints
- Load checkpoint with state restoration
- Automatic cleanup of old checkpoints
"""

import os
import torch
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass

from config.config import CHECKPOINT_PARAMS, LOGGING_PARAMS


@dataclass
class CheckpointInfo:
    """Information about a checkpoint."""
    path: Path
    step: int
    metric_value: float
    is_best: bool


class CheckpointManager:
    """Manage model checkpoints with automatic best model tracking."""

    def __init__(
        self,
        checkpoint_dir: str = None,
        metric_name: str = None,
        mode: str = None,
        keep_n_best: int = None,
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints (default: from CHECKPOINT_PARAMS)
            metric_name: Name of metric to optimize (default: from CHECKPOINT_PARAMS)
            mode: 'max' to maximize metric, 'min' to minimize (default: from CHECKPOINT_PARAMS)
            keep_n_best: Number of best checkpoints to keep (default: from CHECKPOINT_PARAMS)
        """
        if checkpoint_dir is None:
            checkpoint_dir = LOGGING_PARAMS.get('checkpoint_dir', 'checkpoints')
        if metric_name is None:
            metric_name = CHECKPOINT_PARAMS.get('metric_name', 'mean_return')
        if mode is None:
            mode = CHECKPOINT_PARAMS.get('mode', 'max')
        if keep_n_best is None:
            keep_n_best = CHECKPOINT_PARAMS.get('keep_n_best', 3)

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.metric_name = metric_name
        self.mode = mode
        self.keep_n_best = keep_n_best

        # Track best checkpoints
        self.best_checkpoints: list[CheckpointInfo] = []
        self.best_value = float('-inf') if mode == 'max' else float('inf')

    def save_checkpoint(
        self,
        agent,
        step: int,
        metrics: Dict[str, float],
        is_best: bool = False,
    ) -> Path:
        """
        Save checkpoint.

        Args:
            agent: Agent with state_dict() method
            step: Current training step
            metrics: Dictionary of metrics
            is_best: Whether this is the best checkpoint

        Returns:
            Path to saved checkpoint
        """
        # Generate checkpoint name
        checkpoint_name = f"checkpoint_step_{step}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        # Prepare checkpoint dictionary
        checkpoint = {
            'step': step,
            'agent_state': agent.state_dict() if hasattr(agent, 'state_dict') else None,
            'metrics': metrics,
            'is_best': is_best,
        }

        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

        # Track if this is a best checkpoint
        if self.metric_name in metrics:
            metric_value = metrics[self.metric_name]
            if self._is_better(metric_value):
                self.best_value = metric_value
                info = CheckpointInfo(
                    path=checkpoint_path,
                    step=step,
                    metric_value=metric_value,
                    is_best=True,
                )
                self.best_checkpoints.append(info)
                self.best_checkpoints.sort(
                    key=lambda x: x.metric_value,
                    reverse=(self.mode == 'max')
                )

                # Keep only top K best checkpoints
                if len(self.best_checkpoints) > self.keep_n_best:
                    old_checkpoint = self.best_checkpoints.pop(-1)
                    if old_checkpoint.path.exists():
                        old_checkpoint.path.unlink()
                        print(f"Removed old checkpoint: {old_checkpoint.path}")

                # Save best checkpoint separately
                best_path = self.checkpoint_dir / "best.pt"
                torch.save(checkpoint, best_path)
                print(f"Best checkpoint updated: {best_path}")

        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str, agent) -> Dict:
        """
        Load checkpoint into agent.

        Args:
            checkpoint_path: Path to checkpoint file
            agent: Agent to load state into

        Returns:
            Checkpoint dictionary
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, weights_only=False)

        # Load agent state
        if checkpoint['agent_state'] is not None and hasattr(agent, 'load_state_dict'):
            agent.load_state_dict(checkpoint['agent_state'])
            print(f"Agent state loaded from: {checkpoint_path}")

        return checkpoint

    def get_best_checkpoint(self) -> Optional[Path]:
        """Get path to best checkpoint."""
        best_path = self.checkpoint_dir / "best.pt"
        if best_path.exists():
            return best_path
        return None

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to latest checkpoint."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_step_*.pt"),
            key=lambda x: int(x.stem.split('_')[-1]),
            reverse=True
        )
        return checkpoints[0] if checkpoints else None

    def _is_better(self, value: float) -> bool:
        """Check if metric value is better than current best."""
        if self.mode == 'max':
            return value > self.best_value
        else:
            return value < self.best_value

    def list_checkpoints(self) -> list[CheckpointInfo]:
        """List all saved best checkpoints."""
        return sorted(
            self.best_checkpoints,
            key=lambda x: x.metric_value,
            reverse=(self.mode == 'max')
        )
