"""
Logging utilities for Humigence.
Provides structured logging with Rich formatting.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import rich.console
import rich.logging
import rich.traceback
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table


def setup_logging(
    level: str = "INFO", log_file: Path | None = None, rich_console: bool = True
) -> logging.Logger:
    """
    Set up logging configuration for Humigence.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging
        rich_console: Whether to use Rich console formatting

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("humigence")
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatter
    if rich_console:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )

    # Console handler
    if rich_console:
        console_handler = RichHandler(
            console=rich.console.Console(), show_time=True, show_path=False, markup=True
        )
        console_handler.setLevel(getattr(logging, level.upper()))
        logger.addHandler(console_handler)
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def log_system_info(logger: logging.Logger) -> None:
    """Log system information for reproducibility."""
    import platform

    import torch

    logger.info("=" * 60)
    logger.info("SYSTEM INFORMATION")
    logger.info("=" * 60)

    # System info
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"PyTorch: {torch.__version__}")

    # CUDA info
    if torch.cuda.is_available():
        logger.info(f"CUDA: {torch.version.cuda}")
        logger.info(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        logger.warning("CUDA not available")

    # Working directory
    logger.info(f"Working Directory: {Path.cwd()}")
    logger.info("=" * 60)


def log_config_summary(logger: logging.Logger, config: dict) -> None:
    """Log a summary of the configuration."""
    logger.info("CONFIGURATION SUMMARY")
    logger.info("=" * 60)

    # Create a table for better readability
    table = Table(title="Configuration", show_header=True, header_style="bold magenta")
    table.add_column("Section", style="cyan")
    table.add_column("Key", style="white")
    table.add_column("Value", style="green")

    def add_config_items(section_name: str, section_data: dict, prefix: str = ""):
        for key, value in section_data.items():
            if isinstance(value, dict):
                add_config_items(section_name, value, f"{prefix}{key}.")
            else:
                full_key = f"{prefix}{key}" if prefix else key
                table.add_row(section_name, full_key, str(value))

    for section, section_data in config.items():
        if isinstance(section_data, dict):
            add_config_items(section, section_data)
        else:
            table.add_row("General", section, str(section_data))

    # Print the table
    console = Console()
    console.print(table)
    logger.info("=" * 60)


def log_training_progress(
    logger: logging.Logger,
    step: int,
    total_steps: int,
    loss: float,
    learning_rate: float,
    tokens_per_sec: float,
    memory_used: float,
) -> None:
    """Log training progress information."""
    progress = (step / total_steps) * 100 if total_steps > 0 else 0

    logger.info(
        f"Step {step}/{total_steps} ({progress:.1f}%) | "
        f"Loss: {loss:.4f} | "
        f"LR: {learning_rate:.2e} | "
        f"Tokens/sec: {tokens_per_sec:.1f} | "
        f"Memory: {memory_used:.1f} GB"
    )


def log_evaluation_results(
    logger: logging.Logger, metrics: dict, step: int | None = None
) -> None:
    """Log evaluation results."""
    step_info = f" (Step {step})" if step is not None else ""
    logger.info(f"EVALUATION RESULTS{step_info}")
    logger.info("=" * 60)

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")

    for metric, value in metrics.items():
        if isinstance(value, float):
            table.add_row(metric, f"{value:.4f}")
        else:
            table.add_row(metric, str(value))

    console = Console()
    console.print(table)
    logger.info("=" * 60)


def create_run_logger(
    run_name: str, log_dir: Path, level: str = "INFO"
) -> logging.Logger:
    """
    Create a logger for a specific training run.

    Args:
        run_name: Name of the training run
        log_dir: Directory to store log files
        level: Logging level

    Returns:
        Configured logger for the run
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{run_name}_{timestamp}.log"

    return setup_logging(level=level, log_file=log_file, rich_console=True)


def log_memory_usage(logger: logging.Logger) -> None:
    """Log current memory usage."""
    try:
        import psutil

        # System memory
        memory = psutil.virtual_memory()
        logger.info(
            f"System Memory: {memory.used / (1024**3):.1f} GB / "
            f"{memory.total / (1024**3):.1f} GB "
            f"({memory.percent:.1f}%)"
        )

        # GPU memory (if available)
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                total = torch.cuda.get_device_properties(i).total_memory / (1024**3)

                logger.info(
                    f"GPU {i} Memory: {allocated:.1f} GB allocated, "
                    f"{reserved:.1f} GB reserved, {total:.1f} GB total"
                )

    except ImportError:
        logger.warning("psutil not available, cannot log memory usage")
    except Exception as e:
        logger.warning(f"Failed to log memory usage: {e}")
