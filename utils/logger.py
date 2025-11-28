"""
Training progress logger with file and console output.

Features:
- Console logging with formatted output
- CSV file logging for metrics
- Metric aggregation and summary statistics
"""

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List


class TrainingLogger:
    """Logger for tracking training progress and metrics."""

    def __init__(self, log_dir: str = "logs", experiment_name: Optional[str] = None):
        """
        Initialize training logger.

        Args:
            log_dir: Directory to save logs
            experiment_name: Name of the experiment (for file naming)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Generate experiment name if not provided
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name

        # Create experiment directory
        self.exp_dir = self.log_dir / experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # Initialize CSV files
        self.step_log_file = self.exp_dir / "steps.csv"
        self.episode_log_file = self.exp_dir / "episodes.csv"
        self.eval_log_file = self.exp_dir / "evaluations.csv"

        # Create CSV files with headers
        self.step_headers = None
        self.episode_headers = None
        self.eval_headers = None

        # For buffering writes
        self.step_buffer: List[Dict] = []
        self.episode_buffer: List[Dict] = []
        self.eval_buffer: List[Dict] = []

        # Statistics
        self.step_count = 0
        self.episode_count = 0
        self.eval_count = 0

    def log_step(self, step: int, metrics: Dict[str, float]):
        """
        Log single training step.

        Args:
            step: Step number
            metrics: Dictionary of metrics to log
        """
        self.step_count += 1

        # Add timestamp and step number
        log_entry = {'step': step, 'timestamp': datetime.now().isoformat()}
        log_entry.update(metrics)

        # Initialize headers if needed
        if self.step_headers is None:
            self.step_headers = list(log_entry.keys())
            self._write_csv_header(self.step_log_file, self.step_headers)

        # Write to CSV
        self._append_csv(self.step_log_file, self.step_headers, log_entry)

        # Console output every 10 steps
        if self.step_count % 10 == 0:
            self._print_step(step, metrics)

    def log_episode(self, episode: int, metrics: Dict[str, float]):
        """
        Log episode metrics.

        Args:
            episode: Episode number
            metrics: Dictionary of episode metrics
        """
        self.episode_count += 1

        # Add timestamp
        log_entry = {'episode': episode, 'timestamp': datetime.now().isoformat()}
        log_entry.update(metrics)

        # Initialize headers if needed
        if self.episode_headers is None:
            self.episode_headers = list(log_entry.keys())
            self._write_csv_header(self.episode_log_file, self.episode_headers)

        # Write to CSV
        self._append_csv(self.episode_log_file, self.episode_headers, log_entry)

    def log_eval(self, step: int, eval_metrics: Dict[str, float]):
        """
        Log evaluation metrics.

        Args:
            step: Training step at which evaluation occurred
            eval_metrics: Dictionary of evaluation metrics
        """
        self.eval_count += 1

        # Add timestamp and step
        log_entry = {'step': step, 'eval_num': self.eval_count, 'timestamp': datetime.now().isoformat()}
        log_entry.update(eval_metrics)

        # Initialize headers if needed
        if self.eval_headers is None:
            self.eval_headers = list(log_entry.keys())
            self._write_csv_header(self.eval_log_file, self.eval_headers)

        # Write to CSV
        self._append_csv(self.eval_log_file, self.eval_headers, log_entry)

        # Console output for evaluations
        self._print_eval(step, eval_metrics)

    def save(self):
        """Save all buffered logs to disk and print summary."""
        # Ensure all files exist with proper headers
        if self.step_headers is None and self.step_buffer:
            self.step_headers = list(self.step_buffer[0].keys())
            self._write_csv_header(self.step_log_file, self.step_headers)

        if self.episode_headers is None and self.episode_buffer:
            self.episode_headers = list(self.episode_buffer[0].keys())
            self._write_csv_header(self.episode_log_file, self.episode_headers)

        if self.eval_headers is None and self.eval_buffer:
            self.eval_headers = list(self.eval_buffer[0].keys())
            self._write_csv_header(self.eval_log_file, self.eval_headers)

        # Print summary
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE - Summary")
        print("=" * 60)
        print(f"Experiment: {self.experiment_name}")
        print(f"Log directory: {self.exp_dir}")
        print(f"Training steps: {self.step_count}")
        print(f"Episodes: {self.episode_count}")
        print(f"Evaluations: {self.eval_count}")
        print(f"Logs saved to:")
        print(f"  - Steps: {self.step_log_file}")
        print(f"  - Episodes: {self.episode_log_file}")
        print(f"  - Evaluations: {self.eval_log_file}")
        print("=" * 60)

    def _write_csv_header(self, filepath: Path, headers: List[str]):
        """Write header row to CSV file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

    def _append_csv(self, filepath: Path, headers: List[str], row: Dict):
        """Append row to CSV file."""
        with open(filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writerow(row)

    def _print_step(self, step: int, metrics: Dict[str, float]):
        """Print training step summary."""
        metric_str = "  ".join([f"{k}: {v:.4f}" for k, v in list(metrics.items())[:4]])
        print(f"Step {step:8d}  {metric_str}")

    def _print_eval(self, step: int, eval_metrics: Dict[str, float]):
        """Print evaluation summary."""
        print("\n" + "-" * 60)
        print(f"EVALUATION at Step {step}")
        print("-" * 60)
        for k, v in eval_metrics.items():
            if isinstance(v, float):
                print(f"  {k:25s}: {v:10.4f}")
            else:
                print(f"  {k:25s}: {v}")
        print("-" * 60 + "\n")
