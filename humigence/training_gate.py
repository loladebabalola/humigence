"""
Training Readiness Gate for Humigence.
Validates that all prerequisites are met before starting training.
"""

from pathlib import Path
from typing import Any

from datasets import Dataset
from rich.console import Console

from .config import Config

console = Console()


class TrainingReadinessError(Exception):
    """Raised when training readiness checks fail."""
    pass


def validate_training_readiness(
    config: Config,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    runs_dir: Path
) -> None:
    """
    Validate that all prerequisites are met for training.

    Args:
        config: Configuration object
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        runs_dir: Runs directory path

    Raises:
        TrainingReadinessError: If any prerequisite is not met
    """
    # Ensure directories exist
    runs_dir.mkdir(parents=True, exist_ok=True)
    Path(config.export.artifacts_dir).mkdir(parents=True, exist_ok=True)

    # Dataset checks
    if train_dataset is None or len(train_dataset) == 0:
        raise TrainingReadinessError(
            "No training samples found after preprocessing. "
            "Choose the Bundled demo or supply a valid JSONL file."
        )

    if eval_dataset is None or len(eval_dataset) == 0:
        raise TrainingReadinessError(
            "No validation samples found after preprocessing. "
            "Check your data split configuration."
        )

    console.print("[green]✓ Training readiness validation passed[/green]")


def validate_fsdp_config(config: Config) -> dict[str, Any]:
    """
    Validate and resolve FSDP configuration conflicts.

    Args:
        config: Configuration object

    Returns:
        dict: Resolved FSDP configuration
    """
    fsdp_config = {}

    # Check for FSDP conflicts and resolve them
    if hasattr(config.train, 'fsdp') and hasattr(config.train, 'fsdp_full_shard'):
        if config.train.fsdp and config.train.fsdp_full_shard:
            console.print(
                "[yellow]⚠️  Both fsdp and fsdp_full_shard are set. "
                "Using fsdp_full_shard and disabling fsdp.[/yellow]"
            )
            fsdp_config['fsdp'] = None
            fsdp_config['fsdp_full_shard'] = True
        elif config.train.fsdp and not config.train.fsdp_full_shard:
            fsdp_config['fsdp'] = True
            fsdp_config['fsdp_full_shard'] = None
        elif not config.train.fsdp and config.train.fsdp_full_shard:
            fsdp_config['fsdp'] = None
            fsdp_config['fsdp_full_shard'] = True
        else:
            # Neither set, use defaults
            fsdp_config['fsdp'] = None
            fsdp_config['fsdp_full_shard'] = None

    return fsdp_config


def validate_training_arguments_compatibility() -> dict[str, Any]:
    """
    Detect installed transformers version and return compatible TrainingArguments.

    Returns:
        dict: Version-compatible TrainingArguments configuration
    """
    try:
        import transformers
        version = transformers.__version__
        console.print(f"[cyan]Detected transformers version: {version}[/cyan]")

        # Feature detection for different versions
        compatible_args = {
            "do_train": True,
            "do_eval": True,
            "report_to": ["none"],
        }

        # Version-specific compatibility
        if version >= "4.30.0":
            compatible_args.update({
                "evaluation_strategy": "steps",
                "save_strategy": "steps",
                "logging_strategy": "steps",
                "lr_scheduler_type": "cosine",
            })
        elif version >= "4.20.0":
            compatible_args.update({
                "eval_strategy": "steps",
                "save_strategy": "steps",
                "logging_strategy": "steps",
                "lr_scheduler": "cosine",
            })
        else:
            # Older versions - use basic args
            compatible_args.update({
                "eval_strategy": "steps",
                "save_strategy": "steps",
            })

        return compatible_args

    except ImportError:
        console.print("[yellow]Warning: transformers not available, using basic args[/yellow]")
        return {
            "do_train": True,
            "do_eval": True,
            "report_to": ["none"],
        }
