#!/usr/bin/env python3
"""Humigence CLI - Production-grade QLoRA fine-tuning for local GPUs."""

import json
import os
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

try:
    from . import __version__
except ImportError:
    # Fallback for when running outside package context
    __version__ = "0.2.0"
from .acceptance import AcceptanceGates
from .config import Config
from .data_utils import create_demo_dataset
from .eval import ModelEvaluator
from .infer import ModelInferencer
from .model_utils import ensure_model_available
from .pack import ModelPacker
from .plan import TrainingPlanner
from .preprocess import DataPreprocessor, PreprocessingEmptyTrainError
from .train import QLoRATrainer
from .wizard import run_wizard

# Default config path (project-root relative)
DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "humigence.basic.json"
)

# Initialize Typer app and Rich console
app = typer.Typer(
    name="humigence",
    help="Production-grade QLoRA fine-tuning for local GPUs",
    add_completion=True,
    rich_markup_mode="rich",
    no_args_is_help=False,
)


@app.callback()
def _root(
    ctx: typer.Context,
    config: str = typer.Option(
        str(DEFAULT_CONFIG),
        "--config",
        "-c",
        help="Config path used when the wizard autostarts.",
    ),
    run: str = typer.Option(
        None,
        "--run",
        help="If wizard autostarts, preferred action after config: plan|validate|pipeline (default: plan).",
    ),
    train: bool = typer.Option(
        False,
        "--train",
        help="Allow training when autostarting (also honors TRAIN=1).",
    ),
    no_wizard: bool = typer.Option(
        False,
        "--no-wizard",
        help="Do not autostart the wizard; just show help.",
    ),
):
    """Default entrypoint when no subcommand is provided."""
    if ctx.invoked_subcommand:
        return

    import os
    import sys

    from rich.console import Console

    console = Console()

    if no_wizard or not sys.stdin.isatty() or not sys.stdout.isatty():
        typer.echo(ctx.get_help())
        raise typer.Exit(0)

    default_cmd = (os.getenv("HUMIGENCE_DEFAULT_CMD", "wizard")).lower()
    default_run = (run or os.getenv("HUMIGENCE_WIZARD_RUN", "plan")).lower()
    allow_train = train or (os.getenv("TRAIN") == "1")

    if default_cmd in ("wizard", "init"):
        try:
            from .wizard import run_wizard

            wizard_result = run_wizard(
                Path(config), run=default_run, allow_train=allow_train
            )

            # Check if wizard was cancelled or failed
            if wizard_result["exit_code"] != 0:
                raise typer.Exit(wizard_result["exit_code"])

            if wizard_result["next_action"] is None:
                console.print(
                    "[yellow]Wizard completed without selecting an action[/yellow]"
                )
                raise typer.Exit(0)

            # Load the config for execution
            config_obj = Config.from_file(wizard_result["config_path"])

            # Execute the chosen action
            console.print(
                f"\n[bold blue]🎯 Executing: {wizard_result['next_action']}[/bold blue]"
            )

            if wizard_result["next_action"] == "plan":
                # Run training plan
                planner = TrainingPlanner(config_obj)
                plan_result = planner.plan_training()

                # Save training plan
                runs_dir = Path("runs") / config_obj.project
                runs_dir.mkdir(parents=True, exist_ok=True)
                plan_file = runs_dir / "training_plan.json"

                with open(plan_file, "w") as f:
                    json.dump(plan_result, f, indent=2, default=str)

                console.print(f"[green]✓ Training plan saved to: {plan_file}[/green]")
                console.print("\n[green]💡 Next: humigence validate[/green]")
                console.print(
                    "[yellow]💡 To run full training pipeline: humigence init --run pipeline --train[/yellow]"
                )

            elif wizard_result["next_action"] == "validate":
                # Run validation
                console.print(
                    "[yellow]⚠️  Validation runner not yet implemented[/yellow]"
                )
                console.print("\n[green]💡 Next: humigence pipeline[/green]")
                console.print(
                    "[yellow]💡 To run full training pipeline: humigence init --run pipeline --train[/yellow]"
                )

            elif wizard_result["next_action"] == "pipeline":
                # Run pipeline
                if wizard_result["train"]:
                    console.print(
                        "[green]🚀 Starting full training pipeline with training enabled![/green]"
                    )
                    console.print(
                        "[blue]This will execute: Plan → Preprocess → Train → Eval → Pack → Acceptance[/blue]"
                    )
                else:
                    console.print(
                        "[yellow]⚠️  Pipeline will run without training (training is disabled)[/yellow]"
                    )
                    console.print(
                        "[yellow]💡 To enable training, run: humigence init --run pipeline --train[/yellow]"
                    )

                exit_code = run_pipeline(config_obj, wizard_result["train"])
                if exit_code != 0:
                    raise typer.Exit(exit_code)

                console.print("\n[green]🎉 Pipeline completed successfully![/green]")

            console.print(
                f"\n[green]✅ Action '{wizard_result['next_action']}' completed successfully![/green]"
            )

            # Provide next steps guidance
            if wizard_result["next_action"] == "pipeline" and wizard_result["train"]:
                console.print("\n[bold green]🎯 Next Steps:[/bold green]")
                console.print(
                    "[green]✓ Training completed! Your model is ready.[/green]"
                )
                console.print(
                    f"[green]📁 Check results in: runs/{config_obj.project}/[/green]"
                )
                console.print(
                    "[green]💡 Run inference: humigence infer --prompt 'Your prompt here'[/green]"
                )
            elif (
                wizard_result["next_action"] == "pipeline"
                and not wizard_result["train"]
            ):
                console.print("\n[bold yellow]⚠️  Training was skipped![/bold yellow]")
                console.print(
                    "[yellow]💡 To run training: humigence init --run pipeline --train[/yellow]"
                )
                console.print(
                    "[yellow]💡 Or use existing config: humigence pipeline --config configs/humigence.basic.json --train[/yellow]"
                )

        except Exception as e:
            console.print(f"[red]Wizard failed:[/red] {e}")
            typer.echo(ctx.get_help())
            raise typer.Exit(2) from None
    else:
        typer.echo(ctx.get_help())
        raise typer.Exit(0)


console = Console()


# Global options
def get_config_path(
    config: str = typer.Option(
        str(DEFAULT_CONFIG),
        "--config",
        "-c",
        help="Path to configuration file",
    )
) -> Path:
    """Get and validate config path."""
    config_path = Path(config)
    if not config_path.exists():
        console.print(f"[red]Error: Config file not found: {config_path}[/red]")
        raise typer.Exit(2)
    return config_path


def get_project_name(
    project: str
    | None = typer.Option(
        None, "--project", "-p", help="Override project name from config"
    )
) -> str | None:
    """Get project name override."""
    return project


def get_verbose_quiet(
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress output"),
) -> tuple[bool, bool]:
    """Get verbosity settings."""
    return verbose, quiet


def load_config(config_path: Path, project_override: str | None = None) -> Config:
    """Load and validate configuration."""
    try:
        with open(config_path) as f:
            config_data = json.load(f)

        if project_override:
            config_data["project"] = project_override

        return Config(**config_data)
    except Exception as e:
        console.print(f"[red]Error loading config: {e}[/red]")
        raise typer.Exit(2) from None


def print_next_command(suggestion: str) -> None:
    """Print suggested next command."""
    console.print(f"\n[green]💡 Next suggested command:[/green] {suggestion}")


def check_training_allowed() -> bool:
    """Check if training is explicitly allowed."""
    return os.getenv("TRAIN") == "1"


def print_training_disabled_warning() -> None:
    """Print warning about training being disabled by default."""
    console.print(
        Panel(
            "[yellow]⚠️  Training is disabled by default for safety.[/yellow]\n"
            "Use [bold]--train[/bold] flag or set [bold]TRAIN=1[/bold] environment variable to enable training.",
            title="Training Disabled",
            border_style="yellow",
        )
    )


@app.command()
def plan(
    config: Path = typer.Option(
        str(DEFAULT_CONFIG),
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    project: str
    | None = typer.Option(
        None, "--project", "-p", help="Override project name from config"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress output"),
) -> None:
    """Create training plan without executing (dry-run)."""
    try:
        # Load configuration
        config_obj = load_config(config, project)

        # Create training planner
        planner = TrainingPlanner(config_obj)

        # Generate plan
        plan_result = planner.plan_training()

        # Create runs directory
        runs_dir = Path("runs") / config_obj.project
        runs_dir.mkdir(parents=True, exist_ok=True)

        # Write training plan to JSON
        plan_file = runs_dir / "training_plan.json"
        with open(plan_file, "w") as f:
            json.dump(plan_result, f, indent=2, default=str)

        # Display plan summary
        if not quiet:
            console.print(
                f"\n[bold green]✅ Training plan generated: {plan_file}[/bold green]"
            )

            # Create summary table
            table = Table(title="Training Plan Summary")
            table.add_column("Component", style="cyan")
            table.add_column("Details", style="white")

            table.add_row("Project", config_obj.project)
            table.add_row("Model", config_obj.model.repo)
            table.add_row("Precision Mode", config_obj.train.precision_mode)
            table.add_row("Dataset", config_obj.data.raw_path)
            table.add_row("Max Sequence Length", str(config_obj.data.max_seq_len))
            table.add_row("LoRA Rank", str(config_obj.train.lora.r))
            table.add_row("Learning Rate", str(config_obj.train.lr))
            table.add_row(
                "Target Tokens/Step", str(config_obj.train.tokens_per_step_target)
            )

            console.print(table)

            # Print precision banner
            console.print(
                f"\n[bold]PRECISION MODE={config_obj.train.precision_mode}[/bold]"
            )

            # Print precision config if available
            if "precision_config" in plan_result:
                precision = plan_result["precision_config"]
                if "mode" in precision:
                    console.print(f"[bold]DTYPE={precision['mode']}[/bold]")
                if "lora_targets" in precision:
                    console.print(
                        f"[bold]LORA TARGETS={precision['lora_targets']}[/bold]"
                    )

            # Print VRAM plan if available
            if "vram_projection" in plan_result:
                vram = plan_result["vram_projection"]
                if "recommended_config" in vram:
                    recommended = vram["recommended_config"]
                    console.print("\n[bold]Auto-VRAM Plan:[/bold]")
                    console.print(
                        f"micro_batch_size={recommended.get('micro_batch_size', 'N/A')}"
                    )
                    console.print(f"grad_accum={recommended.get('grad_accum', 'N/A')}")
                    console.print(
                        f"projected_vram_gb={recommended.get('projected_vram_gb', 'N/A')} GB"
                    )

        print_next_command("humigence validate")

    except Exception as e:
        console.print(f"[red]Error creating training plan: {e}[/red]")
        raise typer.Exit(1) from None


@app.command()
def validate(
    config: Path = typer.Option(
        str(DEFAULT_CONFIG),
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    project: str
    | None = typer.Option(
        None, "--project", "-p", help="Override project name from config"
    ),
    download_missing: bool = typer.Option(
        False, "--download-missing", help="Download missing models automatically"
    ),
    train: bool = typer.Option(
        False, "--train", help="Enable training (overrides TRAIN env var)"
    ),
    strict: bool = typer.Option(
        True, "--strict", help="Exit non-zero on acceptance gate failures"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress output"),
) -> None:
    """Produce evidence pack without training (unless --train specified)."""
    try:
        # Check if training is allowed
        training_allowed = train or check_training_allowed()

        if not training_allowed:
            print_training_disabled_warning()

        # Load configuration
        config_obj = load_config(config, project)

        # Create validation directory
        validation_dir = Path("validation")
        validation_dir.mkdir(exist_ok=True)

        # Environment info
        if not quiet:
            console.print("[bold]🔍 Environment Validation[/bold]")

        # Write environment info
        env_info = {
            "cuda_available": "torch.cuda.is_available()",
            "gpu_count": "torch.cuda.device_count() if torch.cuda.is_available() else 0",
            "python_version": sys.version,
            "humigence_version": __version__,
        }

        with open(validation_dir / "env.txt", "w") as f:
            for key, value in env_info.items():
                f.write(f"{key}: {value}\n")

        # Git commit info
        try:
            import subprocess

            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=Path.cwd(),
            )
            if result.returncode == 0:
                commit_sha = result.stdout.strip()
                with open(validation_dir / "commit.txt", "w") as f:
                    f.write(f"commit: {commit_sha}\n")
                if not quiet:
                    console.print(f"✅ Git commit: {commit_sha[:8]}")
        except Exception:
            if not quiet:
                console.print("⚠️  Could not determine git commit")

        # Model presence check
        if not quiet:
            console.print("[bold]📥 Model Validation[/bold]")

        model_path = config_obj.model.local_path or config_obj.model.repo
        if Path(model_path).exists():
            if not quiet:
                console.print(f"✅ Model found at: {model_path}")
        elif download_missing:
            if not quiet:
                console.print(f"📥 Downloading model: {config_obj.model.repo}")
            # TODO: Implement model download logic
        else:
            if not quiet:
                console.print(f"⚠️  Model not found: {model_path}")
                console.print("Use --download-missing to download automatically")

        # Data preprocessing
        if not quiet:
            console.print("[bold]🔄 Data Preprocessing[/bold]")

        try:
            preprocessor = DataPreprocessor(config_obj)
            data_report = preprocessor.preprocess_data()

            # Write data report
            with open(validation_dir / "data_report.json", "w") as f:
                json.dump(data_report, f, indent=2, default=str)

            # Write sample rows
            with open(validation_dir / "sample_rows.jsonl", "w") as f:
                # Extract samples from the report structure
                if isinstance(data_report, dict) and "train" in data_report:
                    samples = data_report["train"][:10]  # First 10 samples
                    for sample in samples:
                        f.write(json.dumps(sample) + "\n")

            if not quiet:
                console.print("✅ Data preprocessing complete")

        except Exception as e:
            if not quiet:
                console.print(f"⚠️  Data preprocessing failed: {e}")

        # Check for existing checkpoint and run eval if available
        runs_dir = Path("runs") / config_obj.project
        if runs_dir.exists() and any(runs_dir.glob("checkpoint-*")):
            if not quiet:
                console.print("[bold]📊 Running Evaluation[/bold]")

            try:
                evaluator = ModelEvaluator(config_obj)
                eval_result = evaluator.evaluate_model()

                # Write eval report
                eval_report_file = Path("runs/humigence/eval_report.json")
                eval_report_file.parent.mkdir(parents=True, exist_ok=True)

                # Handle both Pydantic models and regular dicts
                if hasattr(eval_result, 'dict'):
                    eval_data = eval_result.dict()
                else:
                    eval_data = eval_result

                with open(eval_report_file, "w") as f:
                    json.dump(eval_data, f, indent=2, default=str)

                if not quiet:
                    console.print("✅ Evaluation complete")

            except Exception as e:
                if not quiet:
                    console.print(f"⚠️  Evaluation failed: {e}")
        else:
            if not quiet:
                console.print("ℹ️  No checkpoint found, skipping evaluation")

        # Run acceptance gates
        if not quiet:
            console.print("[bold]🎯 Acceptance Gates[/bold]")

        try:
            gates = AcceptanceGates(config_obj, runs_dir)
            acceptance_result = gates.evaluate_training_run()

            # Write acceptance report
            with open(validation_dir / "acceptance_report.json", "w") as f:
                json.dump(acceptance_result.dict(), f, indent=2, default=str)

            if acceptance_result.passed:
                if not quiet:
                    console.print("✅ Acceptance gates passed")
                exit_code = 0
            else:
                if not quiet:
                    console.print("❌ Acceptance gates failed")
                exit_code = 3 if strict else 0

        except Exception as e:
            if not quiet:
                console.print(f"⚠️  Acceptance gates failed: {e}")
            exit_code = 3 if strict else 0

        # Run tests and lint
        if not quiet:
            console.print("[bold]🧪 Code Quality[/bold]")

        try:
            # Run tests
            import subprocess

            result = subprocess.run(
                ["python", "-m", "pytest", "tests/", "-q"],
                capture_output=True,
                text=True,
                cwd=Path.cwd(),
            )
            with open(validation_dir / "tests.txt", "w") as f:
                f.write(f"exit_code: {result.returncode}\n")
                f.write(f"stdout: {result.stdout}\n")
                f.write(f"stderr: {result.stderr}\n")

            # Run lint
            result = subprocess.run(
                ["ruff", "check", "."], capture_output=True, text=True, cwd=Path.cwd()
            )
            with open(validation_dir / "lint.txt", "w") as f:
                f.write(f"exit_code: {result.returncode}\n")
                f.write(f"stdout: {result.stdout}\n")
                f.write(f"stderr: {result.stderr}\n")

            if not quiet:
                console.print("✅ Code quality checks complete")

        except Exception as e:
            if not quiet:
                console.print(f"⚠️  Code quality checks failed: {e}")

        if not quiet:
            console.print("\n[bold green]✅ Validation complete![/bold green]")
            console.print(f"📁 Evidence pack written to: {validation_dir}")

        print_next_command("humigence pipeline")

        # Exit with appropriate code
        if "exit_code" in locals():
            raise typer.Exit(exit_code)

    except typer.Exit:
        # Re-raise typer.Exit to allow normal program termination
        raise
    except Exception as e:
        console.print(f"[red]Error during validation: {e}[/red]")
        raise typer.Exit(1) from None


# Pipeline command removed - using pipeline_direct instead


@app.command()
def preprocess(
    config: Path = typer.Option(
        str(DEFAULT_CONFIG),
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    project: str
    | None = typer.Option(
        None, "--project", "-p", help="Override project name from config"
    ),
    max_seq_len: int
    | None = typer.Option(None, "--max-seq-len", help="Override max sequence length"),
    split: str = typer.Option(
        "0.8,0.1,0.1", "--split", help="Train,val,test split ratios"
    ),
    packing: bool
    | None = typer.Option(None, "--packing", help="Enable/disable packing"),
    seed: int | None = typer.Option(None, "--seed", help="Random seed for splitting"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress output"),
) -> None:
    """Run data preprocessing, splitting, and packing."""
    try:
        # Load configuration
        config_obj = load_config(config, project)

        # Override config with CLI options
        if max_seq_len is not None:
            config_obj.data.max_seq_len = max_seq_len
        if packing is not None:
            config_obj.data.packing = packing
        if seed is not None:
            config_obj.data.seed = seed

        # Parse split ratios
        try:
            split_ratios = [float(x) for x in split.split(",")]
            if len(split_ratios) == 3:
                config_obj.data.split = {
                    "train": split_ratios[0],
                    "val": split_ratios[1],
                    "test": split_ratios[2],
                }
        except ValueError:
            console.print("[red]Error: Invalid split format. Use '0.8,0.1,0.1'[/red]")
            raise typer.Exit(2) from None

        if not quiet:
            console.print("[bold]🔄 Starting Data Preprocessing[/bold]")
            console.print(f"Dataset: {config_obj.data.raw_path}")
            console.print(f"Max sequence length: {config_obj.data.max_seq_len}")
            console.print(f"Packing: {config_obj.data.packing}")
            console.print(f"Split: {config_obj.data.split}")

        # Run preprocessing
        preprocessor = DataPreprocessor(config_obj)
        data_report = preprocessor.preprocess_data()

        # Write reports
        validation_dir = Path("validation")
        validation_dir.mkdir(exist_ok=True)

        with open(validation_dir / "data_report.json", "w") as f:
            json.dump(data_report, f, indent=2, default=str)

        with open(validation_dir / "sample_rows.jsonl", "w") as f:
            # Extract samples from the report structure
            if isinstance(data_report, dict) and "train" in data_report:
                samples = data_report["train"][:10]
                for sample in samples:
                    f.write(json.dumps(sample) + "\n")

        if not quiet:
            console.print("✅ Preprocessing complete")
            console.print(f"📊 Data report: {validation_dir / 'data_report.json'}")
            console.print(f"📝 Sample rows: {validation_dir / 'sample_rows.jsonl'}")

        print_next_command("humigence train --train")

    except Exception as e:
        console.print(f"[red]Error during preprocessing: {e}[/red]")
        raise typer.Exit(1) from None


@app.command()
def train(
    config: Path = typer.Option(
        str(DEFAULT_CONFIG),
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    project: str
    | None = typer.Option(
        None, "--project", "-p", help="Override project name from config"
    ),
    train: bool = typer.Option(
        False, "--train", help="Enable training (overrides TRAIN env var)"
    ),
    precision_mode: str
    | None = typer.Option(None, "--precision-mode", help="Override precision mode"),
    epochs: int
    | None = typer.Option(None, "--epochs", help="Override number of epochs"),
    lr: float | None = typer.Option(None, "--lr", help="Override learning rate"),
    lora_r: int | None = typer.Option(None, "--lora-r", help="Override LoRA rank"),
    tokens_per_step_target: int
    | None = typer.Option(
        None, "--tokens-per-step-target", help="Override tokens per step target"
    ),
    eval_every_steps: int
    | None = typer.Option(None, "--eval-every-steps", help="Override eval frequency"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress output"),
) -> None:
    """Run QLoRA training (only when explicitly allowed)."""
    try:
        # Check if training is allowed
        training_allowed = train or check_training_allowed()

        if not training_allowed:
            print_training_disabled_warning()
            raise typer.Exit(0)

        # Load configuration
        config_obj = load_config(config, project)

        # Override config with CLI options
        if precision_mode is not None:
            config_obj.train.precision_mode = precision_mode
        if epochs is not None:
            config_obj.train.epochs = epochs
        if lr is not None:
            config_obj.train.lr = lr
        if lora_r is not None:
            config_obj.lora.r = lora_r
        if tokens_per_step_target is not None:
            config_obj.train.tokens_per_step_target = tokens_per_step_target
        if eval_every_steps is not None:
            config_obj.train.eval_every_steps = eval_every_steps

        if not quiet:
            console.print("[bold]🚀 Starting QLoRA Training[/bold]")
            console.print(f"Project: {config_obj.project}")
            console.print(f"Precision mode: {config_obj.train.precision_mode}")
            console.print(f"Learning rate: {config_obj.train.lr}")
            console.print(f"LoRA rank: {config_obj.lora.r}")

        # Print precision banner
        console.print(f"[bold]PRECISION MODE={config_obj.train.precision_mode}[/bold]")

        # Create trainer and start training
        trainer = QLoRATrainer(config_obj)

        # Get VRAM fit info if available
        try:
            # This would typically come from the trainer's setup
            if hasattr(trainer, "get_vram_fit_info"):
                vram_info = trainer.get_vram_fit_info()
                if vram_info:
                    console.print(
                        f"micro_batch_size={vram_info.get('micro_batch_size', 'N/A')}"
                    )
                    console.print(f"grad_accum={vram_info.get('grad_accum', 'N/A')}")
                    console.print(
                        f"effective tokens/step={vram_info.get('effective_tokens_per_step', 'N/A')}"
                    )
        except Exception:
            pass

        # Start training
        trainer.train()

        if not quiet:
            console.print("✅ Training complete")
            console.print(f"📊 Metrics: runs/{config_obj.project}/metrics.jsonl")
            console.print(f"📝 Logs: runs/{config_obj.project}/train.log")

        print_next_command("humigence eval")

    except typer.Exit:
        # Re-raise typer.Exit to allow normal program termination
        raise
    except Exception as e:
        console.print(f"[red]Error during training: {e}[/red]")
        raise typer.Exit(1) from None


@app.command()
def eval(
    config: Path = typer.Option(
        str(DEFAULT_CONFIG),
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    project: str
    | None = typer.Option(
        None, "--project", "-p", help="Override project name from config"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress output"),
) -> None:
    """Run quantitative and qualitative evaluation."""
    try:
        # Load configuration
        config_obj = load_config(config, project)

        if not quiet:
            console.print("[bold]📊 Starting Model Evaluation[/bold]")
            console.print(f"Project: {config_obj.project}")

        # Create evaluator and run evaluation
        evaluator = ModelEvaluator(config_obj)
        eval_result = evaluator.evaluate_model()

        # Write eval report
        runs_dir = Path("runs") / config_obj.project
        runs_dir.mkdir(parents=True, exist_ok=True)

        eval_file = runs_dir / "eval_report.json"

        # Handle both Pydantic models and regular dicts
        if hasattr(eval_result, 'dict'):
            eval_data = eval_result.dict()
        else:
            eval_data = eval_result

        with open(eval_file, "w") as f:
            json.dump(eval_data, f, indent=2, default=str)

        if not quiet:
            console.print("✅ Evaluation complete")
            console.print(f"📊 Report: {eval_file}")

            # Display key metrics if available
            if hasattr(eval_result, "loss"):
                console.print(f"Loss: {eval_result.loss:.4f}")
            if hasattr(eval_result, "perplexity"):
                console.print(f"Perplexity: {eval_result.perplexity:.4f}")

        print_next_command("humigence pack")

    except Exception as e:
        console.print(f"[red]Error during evaluation: {e}[/red]")
        raise typer.Exit(1) from None


@app.command()
def pack(
    config: Path = typer.Option(
        str(DEFAULT_CONFIG),
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    project: str
    | None = typer.Option(
        None, "--project", "-p", help="Override project name from config"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress output"),
) -> None:
    """Pack model artifacts for deployment."""
    try:
        # Load configuration
        config_obj = load_config(config, project)

        if not quiet:
            console.print("[bold]📦 Starting Model Packing[/bold]")
            console.print(f"Project: {config_obj.project}")

        # Create packer and pack artifacts
        packer = ModelPacker(config_obj)
        packer.pack_model()

        if not quiet:
            console.print("✅ Packing complete")
            console.print("📁 Artifacts: artifacts/humigence/")

        print_next_command("humigence infer --prompt 'Your prompt here'")

    except Exception as e:
        console.print(f"[red]Error during packing: {e}[/red]")
        raise typer.Exit(1) from None


@app.command()
def infer(
    prompt: str = typer.Argument(..., help="Input prompt for inference"),
    config: Path = typer.Option(
        str(DEFAULT_CONFIG),
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    project: str
    | None = typer.Option(
        None, "--project", "-p", help="Override project name from config"
    ),
    temperature: float = typer.Option(
        0.2, "--temperature", "-t", help="Sampling temperature"
    ),
    max_new_tokens: int = typer.Option(
        256, "--max-new-tokens", "-m", help="Maximum new tokens to generate"
    ),
    save_proof: bool = typer.Option(
        False, "--save-proof", help="Save inference to validation/infer.txt"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress output"),
) -> None:
    """Run inference with the trained model."""
    try:
        # Load configuration
        config_obj = load_config(config, project)

        if not quiet:
            console.print("[bold]🤖 Starting Inference[/bold]")
            console.print(f"Prompt: {prompt}")
            console.print(f"Temperature: {temperature}")
            console.print(f"Max new tokens: {max_new_tokens}")

        # Check if artifacts exist
        artifacts_dir = Path("artifacts/humigence")
        if not artifacts_dir.exists():
            console.print(
                "[red]Error: Model artifacts not found. Run 'humigence pack' first.[/red]"
            )
            raise typer.Exit(5)

        # Create inferencer and run inference
        inferencer = ModelInferencer(config_obj)
        generation = inferencer.generate_response(
            prompt=prompt,
            max_length=max_new_tokens,
            temperature=temperature
        )

        # Display generation
        if not quiet:
            console.print("\n[bold]Generated:[/bold]")
            console.print(generation)

        # Save proof if requested
        if save_proof:
            validation_dir = Path("validation")
            validation_dir.mkdir(exist_ok=True)

            with open(validation_dir / "infer.txt", "a") as f:
                f.write(f"--- {typer.get_current_time()} ---\n")
                f.write(f"Prompt: {prompt}\n")
                f.write(f"Generation: {generation}\n")
                f.write(f"Temperature: {temperature}\n")
                f.write(f"Max new tokens: {max_new_tokens}\n\n")

            if not quiet:
                console.print(f"💾 Proof saved to: {validation_dir / 'infer.txt'}")

        print_next_command("humigence tokens")

    except typer.Exit:
        # Re-raise typer.Exit to allow normal program termination
        raise
    except Exception as e:
        console.print(f"[red]Error during inference: {e}[/red]")
        raise typer.Exit(1) from None


@app.command()
def model(
    action: str = typer.Argument(..., help="Action: download or check"),
    config: Path = typer.Option(
        str(DEFAULT_CONFIG),
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    project: str
    | None = typer.Option(
        None, "--project", "-p", help="Override project name from config"
    ),
    force: bool = typer.Option(
        False, "--force", help="Force download even if model exists"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress output"),
) -> None:
    """Manage base model (download or check status)."""
    try:
        # Load configuration
        config_obj = load_config(config, project)

        if action == "download":
            if not quiet:
                console.print("[bold]📥 Downloading Base Model[/bold]")
                console.print(f"Model: {config_obj.model.repo}")

            # TODO: Implement model download logic
            if not quiet:
                console.print("⚠️  Model download not yet implemented")
                console.print("Please download manually or use Hugging Face CLI")

        elif action == "check":
            if not quiet:
                console.print("[bold]🔍 Checking Model Status[/bold]")

            model_path = config_obj.model.local_path or config_obj.model.repo

            if Path(model_path).exists():
                # Get model size
                size_bytes = sum(
                    f.stat().st_size for f in Path(model_path).rglob("*") if f.is_file()
                )
                size_gb = size_bytes / (1024**3)

                if not quiet:
                    console.print(f"✅ Model found at: {model_path}")
                    console.print(f"📊 Size on disk: {size_gb:.2f} GB")
            else:
                if not quiet:
                    console.print(f"❌ Model not found: {model_path}")
                    console.print("Use 'humigence model download' to download")
                    raise typer.Exit(5)
        else:
            console.print(
                f"[red]Error: Unknown action '{action}'. Use 'download' or 'check'.[/red]"
            )
            raise typer.Exit(2)

        print_next_command("humigence plan")

    except typer.Exit:
        # Re-raise typer.Exit to allow normal program termination
        raise
    except Exception as e:
        console.print(f"[red]Error during model operation: {e}[/red]")
        raise typer.Exit(1) from None


@app.command()
def tokens(
    config: Path = typer.Option(
        str(DEFAULT_CONFIG),
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    project: str
    | None = typer.Option(
        None, "--project", "-p", help="Override project name from config"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress output"),
) -> None:
    """Show last known training metrics."""
    try:
        # Load configuration
        config_obj = load_config(config, project)

        if not quiet:
            console.print("[bold]📊 Training Metrics[/bold]")
            console.print(f"Project: {config_obj.project}")

        # Check for metrics file
        metrics_file = Path("runs") / config_obj.project / "metrics.jsonl"

        if metrics_file.exists():
            # Read last line (most recent metrics)
            with open(metrics_file) as f:
                lines = f.readlines()
                if lines:
                    last_metrics = json.loads(lines[-1])

                    # Create metrics table
                    table = Table(title="Last Training Metrics")
                    table.add_column("Metric", style="cyan")
                    table.add_column("Value", style="white")

                    if "tokens_per_step" in last_metrics:
                        table.add_row(
                            "Tokens per Step", str(last_metrics["tokens_per_step"])
                        )
                    if "tokens_per_sec" in last_metrics:
                        table.add_row(
                            "Tokens per Second", str(last_metrics["tokens_per_sec"])
                        )
                    if "peak_vram_gb" in last_metrics:
                        table.add_row(
                            "Peak VRAM (GB)", str(last_metrics["peak_vram_gb"])
                        )
                    if "loss" in last_metrics:
                        table.add_row("Loss", f"{last_metrics['loss']:.4f}")

                    console.print(table)
                else:
                    if not quiet:
                        console.print("ℹ️  No metrics found in file")
        else:
            if not quiet:
                console.print("ℹ️  No metrics file found")
                console.print("Run training first: humigence train --train")

        print_next_command("humigence eval")

    except Exception as e:
        console.print(f"[red]Error reading metrics: {e}[/red]")
        raise typer.Exit(1) from None


@app.command()
def config(
    action: str = typer.Argument(..., help="Action: view or set"),
    key: str | None = typer.Argument(None, help="Config key (for set action)"),
    value: str | None = typer.Argument(None, help="Config value (for set action)"),
    config: Path = typer.Option(
        str(DEFAULT_CONFIG),
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    project: str
    | None = typer.Option(
        None, "--project", "-p", help="Override project name from config"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress output"),
) -> None:
    """View or modify configuration."""
    try:
        if action == "view":
            # Load and display config
            config_obj = load_config(config, project)

            if not quiet:
                console.print("[bold]📋 Configuration[/bold]")
                console.print(f"File: {config}")
                if project:
                    console.print(f"Project override: {project}")

            # Pretty print config
            config_json = json.dumps(config_obj.model_dump(), indent=2, default=str)
            console.print(config_json)

        elif action == "set":
            if not key or not value:
                console.print(
                    "[red]Error: Both key and value required for 'set' action[/red]"
                )
                raise typer.Exit(2)

            # Load config
            with open(config) as f:
                config_data = json.load(f)

            # Parse dotted key path
            keys = key.split(".")
            current = config_data

            # Navigate to parent of target key
            for k in keys[:-1]:
                if k not in current:
                    console.print(f"[red]Error: Key '{k}' not found in config[/red]")
                    raise typer.Exit(2)
                current = current[k]

            target_key = keys[-1]

            # Try to convert value to appropriate type
            try:
                # Check if it's a number
                if value.lower() in ("true", "false"):
                    converted_value = value.lower() == "true"
                elif "." in value:
                    converted_value = float(value)
                else:
                    converted_value = int(value)
            except ValueError:
                converted_value = value

            # Set the value
            current[target_key] = converted_value

            # Write back to file
            with open(config, "w") as f:
                json.dump(config_data, f, indent=2)

            if not quiet:
                console.print(f"✅ Set {key} = {converted_value}")
                console.print(f"💾 Updated: {config}")
        else:
            console.print(
                f"[red]Error: Unknown action '{action}'. Use 'view' or 'set'.[/red]"
            )
            raise typer.Exit(2)

        if action == "view":
            print_next_command("humigence config set <key> <value>")
        else:
            print_next_command("humigence config view")

    except Exception as e:
        console.print(f"[red]Error during config operation: {e}[/red]")
        raise typer.Exit(1) from None


@app.command()
def doctor(
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress output"),
) -> None:
    """Run environment diagnostics."""
    try:
        if not quiet:
            console.print("[bold]🔍 Environment Diagnostics[/bold]")

        # Check CUDA availability
        try:
            import torch

            cuda_available = torch.cuda.is_available()
            gpu_count = torch.cuda.device_count() if cuda_available else 0

            if not quiet:
                console.print(f"CUDA Available: {'✅' if cuda_available else '❌'}")
                console.print(f"GPU Count: {gpu_count}")

            if cuda_available and gpu_count > 0:
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / (
                        1024**3
                    )
                    if not quiet:
                        console.print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        except ImportError:
            if not quiet:
                console.print("❌ PyTorch not available")

        # Check bitsandbytes
        try:
            import bitsandbytes

            if not quiet:
                console.print(f"✅ bitsandbytes: {bitsandbytes.__version__}")
        except ImportError:
            if not quiet:
                console.print("❌ bitsandbytes not available")

        # Check HF cache path
        hf_home = os.getenv("HF_HOME", "~/.cache/huggingface")
        hf_path = Path(hf_home).expanduser()
        if not quiet:
            console.print(f"HF Cache: {hf_path}")
            console.print(f"HF Cache exists: {'✅' if hf_path.exists() else '❌'}")

        # Check permissions and directories
        dirs_to_check = ["data/", "runs/", "artifacts/", "validation/"]

        for dir_path in dirs_to_check:
            path = Path(dir_path)
            if not quiet:
                console.print(f"\n📁 {dir_path}:")

            # Check if directory exists
            if path.exists():
                if not quiet:
                    console.print("  Exists: ✅")

                # Check read/write permissions
                try:
                    test_file = path / ".test_write"
                    test_file.write_text("test")
                    test_file.unlink()
                    if not quiet:
                        console.print("  Write: ✅")
                except Exception:
                    if not quiet:
                        console.print("  Write: ❌")

                try:
                    list(path.iterdir())
                    if not quiet:
                        console.print("  Read: ✅")
                except Exception:
                    if not quiet:
                        console.print("  Read: ❌")
            else:
                if not quiet:
                    console.print("  Exists: ❌")

        if not quiet:
            console.print("\n[bold green]✅ Diagnostics complete[/bold green]")

        print_next_command("humigence plan")

    except Exception as e:
        console.print(f"[red]Error during diagnostics: {e}[/red]")
        raise typer.Exit(1) from None


@app.command()
def version(
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed version info"
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress output"),
) -> None:
    """Show version information."""
    try:
        if not quiet:
            console.print(f"[bold]Humigence v{__version__}[/bold]")

        if verbose:
            # Get git SHA
            try:
                import subprocess

                result = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    cwd=Path.cwd(),
                )
                if result.returncode == 0:
                    commit_sha = result.stdout.strip()[:8]
                    if not quiet:
                        console.print(f"Git SHA: {commit_sha}")
            except Exception:
                pass

            # Show dependency versions
            dependencies = ["torch", "transformers", "peft", "bitsandbytes"]

            for dep in dependencies:
                try:
                    module = __import__(dep)
                    version = getattr(module, "__version__", "unknown")
                    if not quiet:
                        console.print(f"{dep}: {version}")
                except ImportError:
                    if not quiet:
                        console.print(f"{dep}: not installed")

        print_next_command("humigence --help")

    except Exception as e:
        console.print(f"[red]Error getting version: {e}[/red]")
        raise typer.Exit(1) from None


def validate_config_for_pipeline(config: Config) -> tuple[bool, list[str]]:
    """Validate configuration for pipeline execution.

    Args:
        config: Configuration object to validate

    Returns:
        tuple: (is_valid, list_of_errors)
    """
    errors = []

    # Check required paths exist or can be created
    try:
        # Check data file
        raw_path = Path(config.data.raw_path)
        if not raw_path.exists():
            errors.append(f"Raw data file not found: {raw_path}")

        # Check if output directories can be created
        runs_dir = Path("runs") / config.project
        runs_dir.mkdir(parents=True, exist_ok=True)

        artifacts_dir = Path(config.export.artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

    except Exception as e:
        errors.append(f"Cannot create output directories: {e}")

    # Check model configuration
    if not config.model.repo:
        errors.append("Model repository not specified")

    # Check training configuration
    if config.train.precision_mode not in [
        "qlora_nf4",
        "lora_fp16",
        "lora_bf16",
        "lora_int8",
    ]:
        errors.append(f"Invalid precision mode: {config.train.precision_mode}")

    if config.train.lora.r <= 0:
        errors.append(f"Invalid LoRA rank: {config.train.lora.r}")

    if config.train.lora.alpha <= 0:
        errors.append(f"Invalid LoRA alpha: {config.train.lora.alpha}")

    # Check data configuration
    if config.data.max_seq_len <= 0 or config.data.max_seq_len > 8192:
        errors.append(f"Invalid max sequence length: {config.data.max_seq_len}")

    if config.data.data_schema not in ["chat_messages", "instruction_output"]:
        errors.append(f"Invalid data schema: {config.data.data_schema}")

    return len(errors) == 0, errors


def _load_config_with_source(path: Path) -> Config:
    cfg = Config.from_file(path)
    # Remember where to persist automatic updates (model local_path, etc.)
    cfg._source_path = Path(path).expanduser().resolve()
    return cfg


def _confirm_or_create_dataset(cfg: Config) -> Path:
    raw = Path(cfg.data.raw_path).expanduser()
    if raw.exists() and raw.stat().st_size > 0:
        return raw
    console.print(f"[yellow]⚠ No dataset at[/yellow] {raw}")
    if typer.confirm("Create a small demo dataset now?", default=True):
        return create_demo_dataset(raw, schema=cfg.data.data_schema, n=12)
    raise typer.Exit(4)


def run_pipeline(
    config_path: Path,
    action: str = "pipeline",
    allow_train: bool = False,
    collator_windowing: str = "window",
    window_overlap: int = 128,
    eval_sampling: str = "off",
    real_mode_threshold: int = 1000
) -> int:
    cfg = _load_config_with_source(config_path)

    # Apply new collator and evaluation settings
    if not hasattr(cfg.data, 'collator_windowing'):
        cfg.data.collator_windowing = collator_windowing
    if not hasattr(cfg.data, 'window_overlap'):
        cfg.data.window_overlap = window_overlap
    if not hasattr(cfg.eval, 'sampling_enabled'):
        cfg.eval.sampling_enabled = eval_sampling == "on"
    if not hasattr(cfg.data, 'real_mode_threshold'):
        cfg.data.real_mode_threshold = real_mode_threshold

    # Summary log (short)
    console.rule("[bold]Starting Humigence Pipeline[/bold]")
    console.print(f"Project: {cfg.project}")
    console.print(f"Action: {action} | Training enabled: {allow_train}")
    console.print(f"Collator windowing: {cfg.data.collator_windowing} | Window overlap: {cfg.data.window_overlap}")
    console.print(f"Evaluation sampling: {'on' if cfg.eval.sampling_enabled else 'off'} | Real mode threshold: {cfg.data.real_mode_threshold}")

    # PLAN (always)
    console.print("\n[cyan]📋 Planning[/cyan]")
    # (If you have a TrainingPlanner, call it; else skip verbose planning.)

    # MODEL (ensure local)
    console.print("\n[cyan]📥 Ensuring base model[/cyan]")
    try:
        ensure_model_available(cfg)
    except Exception as e:
        console.print(f"[red]❌ Model availability check failed: {e}[/red]")
        console.print("[yellow]💡 Run: `humigence model download` or ensure network/HF auth.[/yellow]")
        raise typer.Exit(1)

    # DATA (raw presence or demo)
    console.print("\n[cyan]🧰 Validating dataset[/cyan]")
    _confirm_or_create_dataset(cfg)

    # PREPROCESS
    console.print("\n[cyan]🧪 Preprocessing[/cyan]")
    # Ensure processed data directory exists
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    try:
        DataPreprocessor(cfg).preprocess()
    except PreprocessingEmptyTrainError as e:
        console.print(f"[red]❌ Preprocessing failed: {e}[/red]")
        console.print("[yellow]💡 Choose Bundled OpenAssist demo or supply a valid dataset.[/yellow]")
        raise typer.Exit(2)
    except Exception as e:
        console.print(f"[red]❌ Preprocessing failed: {e}[/red]")
        raise typer.Exit(2)

    # Check real mode threshold if training is enabled
    if action == "pipeline" and allow_train:
        console.print("\n[cyan]🔍 Dataset Integrity Check[/cyan]")
        try:
            # Quick check of processed training data
            train_file = Path("data/processed/train.jsonl")
            if train_file.exists():
                with open(train_file) as f:
                    train_count = sum(1 for line in f if line.strip())

                if train_count < cfg.data.real_mode_threshold:
                    console.print(f"[red]❌ Insufficient training samples: {train_count} < {cfg.data.real_mode_threshold}[/red]")
                    console.print(f"[yellow]💡 Real data mode requires at least {cfg.data.real_mode_threshold} samples.[/yellow]")
                    console.print("[yellow]💡 Use --collator_windowing=window or increase max_seq_len to preserve more samples.[/yellow]")
                    raise typer.Exit(2)
                else:
                    console.print(f"[green]✓ Training samples: {train_count} >= {cfg.data.real_mode_threshold}[/green]")
            else:
                console.print("[red]❌ Processed training data not found[/red]")
                raise typer.Exit(2)
        except Exception as e:
            if not isinstance(e, typer.Exit):
                console.print(f"[red]❌ Dataset integrity check failed: {e}[/red]")
                raise typer.Exit(2)

    # TRAIN
    if action == "pipeline" and allow_train:
        console.print("\n[cyan]🚂 Training[/cyan]")
        # Ensure target directories exist before training
        runs_dir = Path("runs") / cfg.project
        runs_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir = Path(cfg.export.artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        try:
            QLoRATrainer(cfg).train()
        except Exception as e:
            console.print(f"[red]❌ Training failed: {e}[/red]")
            raise typer.Exit(3)
    else:
        if action == "pipeline":
            console.print(
                "[yellow]⚠️  Training disabled by default. Use --train or TRAIN=1 to enable.[/yellow]"
            )

    # EVAL
    if action in ("pipeline", "validate"):
        console.print("\n[cyan]📏 Evaluation[/cyan]")
        ModelEvaluator(cfg).evaluate_model()

    # PACK
    if action in ("pipeline", "validate"):
        console.print("\n[cyan]📦 Packaging[/cyan]")
        # Ensure target directories exist before packing
        artifacts_dir = Path(cfg.export.artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        ModelPacker(cfg).pack_model()

    # ACCEPTANCE
    if action in ("pipeline", "validate"):
        console.print("\n[cyan]✅ Acceptance[/cyan]")
        result = AcceptanceGates(cfg).evaluate_training_run()
        if not result.passed:
            console.print("[red]Acceptance gates failed.[/red]")
            raise typer.Exit(3)
    console.print("\n[green]✔ Done.[/green]")
    return 0


@app.command(name="pipeline", help="Run complete training pipeline directly")
def pipeline_direct(
    config: Path = typer.Option(
        str(DEFAULT_CONFIG),
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    project: str
    | None = typer.Option(
        None, "--project", "-p", help="Override project name from config"
    ),
    train: bool = typer.Option(
        False, "--train", help="Enable training (overrides TRAIN env var)"
    ),
    no_strict: bool = typer.Option(
        False, "--no-strict", help="Don't exit non-zero on acceptance gate failures"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress output"),
    collator_windowing: str = typer.Option(
        "window", "--collator_windowing", help="Collator windowing mode: window|drop (default: window)"
    ),
    window_overlap: int = typer.Option(
        128, "--window_overlap", help="Window overlap for long sequences (default: 128)"
    ),
    eval_sampling: str = typer.Option(
        "off", "--eval_sampling", help="Evaluation sampling mode: on|off (default: off)"
    ),
    real_mode_threshold: int = typer.Option(
        1000, "--real_mode_threshold", help="Minimum training samples threshold for real data mode (default: 1000)"
    ),
) -> None:
    """Run the complete training pipeline directly without wizard.

    This command is for advanced users who want to skip the interactive wizard
    and run training directly with an existing configuration file.
    """
    try:
        # Check if training is allowed
        training_allowed = train or os.getenv("TRAIN") == "1"

        if not training_allowed:
            console.print("[red]❌ Training is disabled by default for safety.[/red]")
            console.print(
                "[yellow]💡 Use --train flag or set TRAIN=1 environment variable to enable training.[/yellow]"
            )
            console.print(
                "[yellow]💡 Example: humigence pipeline --config my_config.json --train[/yellow]"
            )
            raise typer.Exit(1)

        # Load configuration
        config_path = Path(config)
        if not config_path.exists():
            console.print(f"[red]❌ Configuration file not found: {config_path}[/red]")
            console.print(
                "[yellow]💡 Please provide a valid config file or run 'humigence init' to create one.[/yellow]"
            )
            raise typer.Exit(2)

        # Load and validate config
        config_obj = load_config(config_path, project)

        # Apply new collator and evaluation settings
        if not hasattr(config_obj.data, 'collator_windowing'):
            config_obj.data.collator_windowing = collator_windowing
        if not hasattr(config_obj.data, 'window_overlap'):
            config_obj.data.window_overlap = window_overlap
        if not hasattr(config_obj.eval, 'sampling_enabled'):
            config_obj.eval.sampling_enabled = eval_sampling == "on"
        if not hasattr(config_obj.data, 'real_mode_threshold'):
            config_obj.data.real_mode_threshold = real_mode_threshold

        # Use the proper run_pipeline function that handles model downloading
        exit_code = run_pipeline(
            config_path=config,
            action="pipeline",
            allow_train=training_allowed,
            collator_windowing=collator_windowing,
            window_overlap=window_overlap,
            eval_sampling=eval_sampling,
            real_mode_threshold=real_mode_threshold
        )
        if exit_code != 0:
            raise typer.Exit(exit_code)

        console.print("\n[bold green]🎉 Pipeline completed successfully![/bold green]")

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]❌ Pipeline failed: {e}[/red]")
        raise typer.Exit(1) from None


@app.command(name="init", help="Interactive setup wizard")
def init(
    config: str = typer.Option(str(DEFAULT_CONFIG), "--config", "-c"),
    run: str
    | None = typer.Option(
        None,
        help="Post-wizard action after config: plan|validate|pipeline (default: plan).",
    ),
    train: bool = typer.Option(
        False, help="Allow training immediately (also honors TRAIN=1)."
    ),
    mode: str = typer.Option(
        None, "--mode", help="Setup mode: basic|advanced (default: interactive selection)"
    ),
) -> None:
    """Interactive setup wizard. After completion, auto-runs selected action."""

    # Parse mode if provided
    wizard_mode = None
    if mode:
        if mode.lower() == "basic":
            from .wizard import WizardMode
            wizard_mode = WizardMode.BASIC
        elif mode.lower() == "advanced":
            from .wizard import WizardMode
            wizard_mode = WizardMode.ADVANCED
        else:
            console.print(f"[red]Invalid mode: {mode}. Use 'basic' or 'advanced'.[/red]")
            raise typer.Exit(1)

    result = run_wizard(Path(config), default_action=run, train=train, mode=wizard_mode)
    if not result or result.get("next_action") is None:
        console.print("[green]Wizard complete.[/green] No action selected.")
        raise typer.Exit(0)
    cfg_path = Path(result["config_path"]).expanduser().resolve()
    action = result["next_action"]
    allow_train = (
        bool(result.get("train")) or bool(train) or (os.environ.get("TRAIN") == "1")
    )
    raise typer.Exit(run_pipeline(cfg_path, action=action, allow_train=allow_train))


@app.command(name="wizard", help="Interactive setup wizard (alias for init)")
def wizard(
    config: str = typer.Option(str(DEFAULT_CONFIG), "--config", "-c"),
    run: str | None = typer.Option(None, help="Post-wizard action after config."),
    train: bool = typer.Option(False, help="Allow training (also honors TRAIN=1)."),
) -> None:
    """Alias for init to preserve old behavior."""
    return init(config=config, run=run, train=train)


@app.command("data-demo")
def data_demo(
    out: Path = typer.Argument(...), schema: str = "chat_messages", n: int = 12
):
    """Create a demo dataset for testing."""
    from .data_utils import create_demo_dataset

    create_demo_dataset(out, schema=schema, n=n)


@app.command("data-doctor")
def data_doctor(config: Path = typer.Option("configs/humigence.basic.json")):
    """Diagnose dataset issues."""
    cfg = _load_config_with_source(config)
    from .data_utils import doctor_dataset

    info = doctor_dataset(Path(cfg.data.raw_path))
    console.print(info)


# Main entry point
if __name__ == "__main__":
    app()
