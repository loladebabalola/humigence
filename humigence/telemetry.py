"""Telemetry and metrics tracking for Humigence training."""

import json
import logging
import time
from collections import deque
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)
console = Console()


class TrainingTelemetry:
    """Track training metrics, throughput, and VRAM usage."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.step_times = deque(maxlen=window_size)
        self.step_losses = deque(maxlen=window_size)
        self.step_tokens = deque(maxlen=window_size)
        self.start_time = time.time()
        self.last_eval_time = time.time()
        self.peak_vram = 0.0

        # Reset VRAM tracking
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def record_step(
        self, step: int, loss: float, tokens_processed: int, step_time: float
    ) -> None:
        """Record metrics for a training step."""
        self.step_times.append(step_time)
        self.step_losses.append(loss)
        self.step_tokens.append(tokens_processed)

        # Update peak VRAM
        if torch.cuda.is_available():
            current_vram = torch.cuda.max_memory_allocated() / (1024**3)  # GB
            self.peak_vram = max(self.peak_vram, current_vram)

    def get_current_metrics(self) -> dict:
        """Get current training metrics."""
        if not self.step_times:
            return {}

        # Calculate throughput metrics
        avg_step_time = sum(self.step_times) / len(self.step_times)
        tokens_per_sec = (
            sum(self.step_tokens) / sum(self.step_times)
            if sum(self.step_times) > 0
            else 0
        )

        # Calculate loss metrics
        current_loss = self.step_losses[-1] if self.step_losses else 0
        avg_loss = (
            sum(self.step_losses) / len(self.step_losses) if self.step_losses else 0
        )

        # Calculate tokens per step
        avg_tokens_per_step = (
            sum(self.step_tokens) / len(self.step_tokens) if self.step_tokens else 0
        )

        # Calculate throughput stability (jitter)
        if len(self.step_times) >= 3:
            recent_times = list(self.step_times)[-3:]
            time_mean = sum(recent_times) / len(recent_times)
            time_variance = sum((t - time_mean) ** 2 for t in recent_times) / len(
                recent_times
            )
            throughput_jitter = (
                (time_variance**0.5) / time_mean * 100 if time_mean > 0 else 0
            )
        else:
            throughput_jitter = 0.0

        return {
            "step": len(self.step_times),
            "current_loss": current_loss,
            "avg_loss": avg_loss,
            "tokens_per_step": avg_tokens_per_step,
            "tokens_per_sec": tokens_per_sec,
            "throughput_jitter_pct": throughput_jitter,
            "peak_vram_gb": self.peak_vram,
            "avg_step_time": avg_step_time,
            "total_training_time": time.time() - self.start_time,
        }

    def print_telemetry_table(self, step: int, total_steps: int) -> None:
        """Print a formatted telemetry table."""
        metrics = self.get_current_metrics()
        if not metrics:
            return

        table = Table(title=f"📊 Training Telemetry (Step {step}/{total_steps})")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        table.add_column("Unit", style="green")

        # Progress
        progress_pct = (step / total_steps * 100) if total_steps > 0 else 0
        table.add_row("Progress", f"{progress_pct:.1f}", "%")

        # Loss
        table.add_row("Current Loss", f"{metrics['current_loss']:.4f}", "")
        table.add_row("Avg Loss", f"{metrics['avg_loss']:.4f}", "")

        # Throughput
        table.add_row("Tokens/Step", f"{metrics['tokens_per_step']:.0f}", "tokens")
        table.add_row("Tokens/sec", f"{metrics['tokens_per_sec']:.0f}", "tokens/s")
        table.add_row(
            "Throughput Jitter", f"{metrics['throughput_jitter_pct']:.1f}", "%"
        )

        # Memory
        table.add_row("Peak VRAM", f"{metrics['peak_vram_gb']:.2f}", "GB")

        # Timing
        table.add_row("Avg Step Time", f"{metrics['avg_step_time']:.3f}", "seconds")
        table.add_row("Total Time", f"{metrics['total_training_time']:.0f}", "seconds")

        console.print(table)

    def save_metrics(self, run_dir: Path, step: int) -> None:
        """Save metrics to JSONL file."""
        metrics = self.get_current_metrics()
        if not metrics:
            return

        metrics_file = run_dir / "metrics.jsonl"
        metrics["timestamp"] = time.time()
        metrics["step"] = step

        with open(metrics_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")

    def reset_eval_timer(self) -> None:
        """Reset the evaluation timer."""
        self.last_eval_time = time.time()

    def get_eval_interval_metrics(self) -> dict:
        """Get metrics since last evaluation."""
        interval_time = time.time() - self.last_eval_time
        return {
            "eval_interval_time": interval_time,
            "steps_since_eval": len(self.step_times),
            "tokens_since_eval": sum(self.step_tokens),
        }


def log_memory_usage(logger: logging.Logger, device: int = 0) -> None:
    """Log current GPU memory usage."""
    if not torch.cuda.is_available():
        return

    allocated = torch.cuda.memory_allocated(device) / (1024**3)
    reserved = torch.cuda.memory_reserved(device) / (1024**3)
    total = torch.cuda.get_device_properties(device).total_memory / (1024**3)

    logger.info(
        f"GPU {device} Memory: {allocated:.2f} GB allocated, "
        f"{reserved:.2f} GB reserved, {total:.2f} GB total"
    )


def estimate_tokens_per_second(
    batch_size: int, seq_length: int, model_params: int, gpu_memory_gb: float
) -> float:
    """Estimate tokens per second based on hardware specs."""
    # Rough estimation based on model size and GPU memory
    # This is a simplified heuristic
    base_tokens_per_sec = 1000  # Base assumption

    # Adjust for model size (smaller models are faster)
    if model_params < 1e9:  # < 1B params
        size_factor = 1.5
    elif model_params < 3e9:  # 1-3B params
        size_factor = 1.0
    else:  # > 3B params
        size_factor = 0.7

    # Adjust for GPU memory (more memory = potentially faster)
    memory_factor = min(gpu_memory_gb / 16.0, 2.0)  # Normalize to 16GB

    return base_tokens_per_sec * size_factor * memory_factor
