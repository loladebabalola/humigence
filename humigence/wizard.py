"""Interactive wizard for Humigence CLI configuration and setup."""

import subprocess
from enum import Enum
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import Config, save_config_atomic

console = Console()


class WizardMode(Enum):
    """Wizard setup mode selection."""
    BASIC = "basic"
    ADVANCED = "advanced"


# Try to import InquirerPy, fall back to basic prompts if not available
try:
    import sys

    from InquirerPy import inquirer

    # Check if we're in a terminal environment
    if sys.stdin.isatty():
        INQUIRER_AVAILABLE = True
    else:
        INQUIRER_AVAILABLE = False
        console.print(
            "[yellow]Warning: Not in terminal environment, using basic prompts[/yellow]"
        )
except ImportError:
    INQUIRER_AVAILABLE = False
    console.print(
        "[yellow]Warning: InquirerPy not available, using basic prompts[/yellow]"
    )


def detect_gpus() -> list:
    """Detect available GPUs using nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    parts = line.split(", ")
                    if len(parts) >= 3:
                        gpu_id = parts[0].strip()
                        gpu_name = parts[1].strip()
                        memory = parts[2].strip()
                        gpus.append(
                            {
                                "name": f"GPU{gpu_id}: {gpu_name} ({memory}MB)",
                                "value": int(gpu_id),
                                "gpu_id": gpu_id,
                                "gpu_name": gpu_name,
                                "memory_mb": int(memory) if memory.isdigit() else 0,
                            }
                        )
            return gpus
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass

    # Fallback to default GPU
    return [
        {
            "name": "GPU0: RTX_4080_16GB (default)",
            "value": 0,
            "gpu_id": "0",
            "gpu_name": "RTX_4080_16GB",
            "memory_mb": 16384,
        }
    ]


def select_wizard_mode() -> WizardMode:
    """Present mode selection to user."""
    console.print("\n[bold cyan]Choose Setup Mode:[/bold cyan]")
    console.print("[1] Basic Setup - Essential configuration only")
    console.print("[2] Advanced Setup - Full control over all parameters")

    while True:
        choice = input("\nSelect mode (1 or 2): ").strip()
        if choice == "1":
            return WizardMode.BASIC
        elif choice == "2":
            return WizardMode.ADVANCED
        else:
            console.print("[red]Invalid choice. Please enter 1 or 2.[/red]")


def get_default_config() -> dict:
    """Get the default configuration template used by the wizard."""
    return {
        "project": "humigence",
        "model": {
            "repo": "Qwen/Qwen2.5-0.5B",
            "local_path": None,
            "use_flash_attn": True
        },
        "compute": {
            "gpus": 1,
            "gpu_type": "RTX_4080_16GB"
        },
        "data": {
            "raw_path": "data/raw/oasst1_conversations.jsonl",
            "processed_dir": "data/processed",
            "schema": "chat_messages",
            "max_seq_len": 1024,
            "packing": True,
            "split": {
                "train": 0.8,
                "val": 0.1,
                "test": 0.1
            },
            "template": "qwen_chat_basic_v1"
        },
        "train": {
            "precision_mode": "qlora_nf4",
            "lora": {
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "r": 16,
                "alpha": 32,
                "dropout": 0.05
            },
            "tokens_per_step_target": 100000,
            "eval_every_steps": 500,
            "save_every_steps": 500,
            "lr": 0.0002,
            "scheduler": "cosine",
            "warmup_ratio": 0.03,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "gradient_checkpointing": True,
            "epochs": 10
        },
        "eval": {
            "curated_prompts_path": "configs/curated_eval_prompts.jsonl"
        },
        "acceptance": {
            "min_val_improvement_pct": 1.0,
            "curated_reasonable_threshold_pct": 70.0,
            "throughput_jitter_pct": 20.0
        },
        "export": {
            "formats": ["peft_adapter"]
        }
    }


def deep_merge(default_config: dict, user_config: dict) -> dict:
    """Deep merge user configuration with defaults."""
    result = default_config.copy()

    for key, value in user_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def run_basic_setup(current_config: Config | None) -> Config:
    """Run Basic Setup - ask only the 5 essential questions."""
    console.print(
        Panel(
            "[bold green]Basic Setup Mode[/bold green]\n"
            "Configure only the essential parameters. All other settings will use safe defaults.",
            title="⚡ Quick Setup",
            border_style="green",
        )
    )

    if INQUIRER_AVAILABLE:
        return run_basic_setup_inquirer(current_config)
    else:
        return run_basic_setup_basic(current_config)


def run_basic_setup_inquirer(current_config: Config | None) -> Config:
    """Basic Setup using InquirerPy."""

    # 1. GPU selection
    gpu_choices = detect_gpus()
    gpu_device = inquirer.select(
        message="Select GPU device:",
        choices=gpu_choices,
        default=current_config.compute.gpus if current_config else 1,
    ).execute()

    # 2. Base model selection
    model_choices = [
        {"name": "Qwen/Qwen2.5-0.5B (default)", "value": "Qwen/Qwen2.5-0.5B"},
        {"name": "Phi-2", "value": "microsoft/phi-2"},
        {"name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "value": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"},
        {"name": "microsoft/phi-1_5", "value": "microsoft/phi-1_5"},
    ]

    base_model = inquirer.select(
        message="Choose base model:",
        choices=model_choices,
        default=current_config.model.repo if current_config else "Qwen/Qwen2.5-0.5B",
    ).execute()

    # 3. Dataset path
    dataset_source = inquirer.select(
        message="Dataset source:",
        choices=[
            {"name": "Use existing real data (oasst1_conversations.jsonl)", "value": "existing"},
            {"name": "Local JSONL file (enter custom path)", "value": "local"},
            {"name": "Bundled OpenAssist demo (13 samples - quick test)", "value": "bundled"},
            {"name": "Generate realistic demo (1000 samples - proper training)", "value": "generate"},
        ],
        default="existing",
    ).execute()

    # Handle dataset source selection
    if dataset_source == "existing":
        # Use the real OpenAssist data that should already exist
        chosen_raw_path = "data/raw/oasst1_conversations.jsonl"
        if Path(chosen_raw_path).exists():
            console.print("[green]✓ Using existing real OpenAssist dataset[/green]")
        else:
            console.print(f"[yellow]⚠ Real dataset not found at {chosen_raw_path}[/yellow]")
            console.print("[yellow]Falling back to bundled demo...[/yellow]")
            dataset_source = "bundled"

    if dataset_source == "bundled":
        import shutil
        from importlib.resources import files

        try:
            demo_path = files("humigence.assets.datasets") / "openassist_demo.jsonl"
            raw_path = Path("data/raw/oa.jsonl")
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(demo_path, raw_path)
            chosen_raw_path = str(raw_path)
            console.print("[green]✓ Using bundled OpenAssist demo dataset[/green]")
        except Exception as e:
            console.print(f"[red]Error copying bundled dataset: {e}[/red]")
            console.print("[yellow]Falling back to generating realistic demo dataset...[/yellow]")
            dataset_source = "generate"

    elif dataset_source == "local":
        chosen_raw_path = inquirer.text(
            message="Enter path to local JSONL file:",
            default=current_config.data.raw_path if current_config else "data/raw/oasst1_conversations.jsonl",
        ).execute()

        # Validate file exists
        if not Path(chosen_raw_path).exists():
            console.print(f"[red]Error: File not found: {chosen_raw_path}[/red]")
            raise FileNotFoundError(f"Dataset file not found: {chosen_raw_path}")

    if dataset_source == "generate":
        from .data_utils import create_demo_dataset

        raw_path = Path("data/raw/oa.jsonl")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        create_demo_dataset(raw_path, schema="chat_messages", n=1000)
        chosen_raw_path = str(raw_path)
        console.print("[green]✓ Generated realistic fine-tuning dataset (1000 samples)[/green]")

    # 4. Dataset schema
    schema_choices = [
        {"name": "chat_messages (default)", "value": "chat_messages"},
        {"name": "instruction_output", "value": "instruction_output"},
    ]

    data_schema = inquirer.select(
        message="Data format/schema:",
        choices=schema_choices,
        default=current_config.data.data_schema if current_config else "chat_messages",
    ).execute()

    # 5. Fine-tuning precision
    precision_choices = [
        {"name": "qlora_nf4 (default)", "value": "qlora_nf4"},
        {"name": "lora_fp16", "value": "lora_fp16"},
        {"name": "lora_bf16", "value": "lora_bf16"},
        {"name": "lora_int8", "value": "lora_int8"},
    ]

    precision_mode = inquirer.select(
        message="Training precision mode:",
        choices=precision_choices,
        default=current_config.train.precision_mode if current_config else "qlora_nf4",
    ).execute()

    # Build basic configuration with user choices
    basic_config = {
        "project": current_config.project if current_config else "humigence",
        "model": {"repo": base_model, "local_path": None, "use_flash_attn": True},
        "compute": {"gpus": int(gpu_device) if gpu_device else 1, "gpu_type": "RTX_4080_16GB"},
        "data": {"raw_path": chosen_raw_path, "schema": data_schema},
        "train": {"precision_mode": precision_mode},
    }

    # Merge with defaults
    default_config = get_default_config()
    final_config = deep_merge(default_config, basic_config)

    return Config(**final_config)


def run_basic_setup_basic(current_config: Config | None) -> Config:
    """Basic Setup using basic input prompts."""
    console.print("[yellow]Using basic input prompts (InquirerPy not available)[/yellow]")

    # 1. GPU selection
    gpu_device = input("Select GPU device [0]: ").strip() or "0"

    # 2. Base model
    base_model = input("Base model [Qwen/Qwen2.5-0.5B]: ").strip() or "Qwen/Qwen2.5-0.5B"

    # 3. Dataset path
    print("\nDataset options:")
    print("1. Use existing real data (oasst1_conversations.jsonl)")
    print("2. Local JSONL file (enter custom path)")
    print("3. Bundled OpenAssist demo (13 samples - quick test)")
    print("4. Generate realistic demo (1000 samples - proper training)")

    dataset_choice = input("Choose dataset option [1]: ").strip() or "1"

    if dataset_choice == "1":
        # Use the real OpenAssist data
        dataset_path = "data/raw/oasst1_conversations.jsonl"
        if Path(dataset_path).exists():
            print("[green]✓ Using existing real OpenAssist dataset[/green]")
        else:
            print(f"[yellow]⚠ Real dataset not found at {dataset_path}[/yellow]")
            print("[yellow]Falling back to bundled demo...[/yellow]")
            dataset_path = "data/raw/oa.jsonl"
    elif dataset_choice == "2":
        dataset_path = input("Enter path to local JSONL file: ").strip()
        if not dataset_path:
            dataset_path = "data/raw/oasst1_conversations.jsonl"
    elif dataset_choice == "3":
        dataset_path = "data/raw/oa.jsonl"
    else:  # choice == "4"
        dataset_path = "data/raw/oa.jsonl"

    # 4. Dataset schema
    data_schema = input("Data schema [chat_messages]: ").strip() or "chat_messages"

    # 5. Precision mode
    precision_mode = input("Precision mode [qlora_nf4]: ").strip() or "qlora_nf4"

    # Build basic configuration with user choices
    basic_config = {
        "project": current_config.project if current_config else "humigence",
        "model": {"repo": base_model, "local_path": None, "use_flash_attn": True},
        "compute": {"gpus": int(gpu_device), "gpu_type": "RTX_4080_16GB"},
        "data": {"raw_path": dataset_path, "schema": data_schema},
        "train": {"precision_mode": precision_mode},
    }

    # Merge with defaults
    default_config = get_default_config()
    final_config = deep_merge(default_config, basic_config)

    return Config(**final_config)


def run_configuration_wizard(current_config: Config | None, mode: WizardMode | None = None) -> Config:
    """Run the interactive configuration prompts."""
    console.print(
        Panel(
            "[bold blue]Humigence Configuration Wizard[/bold blue]\n"
            "Configure your QLoRA fine-tuning pipeline interactively",
            title="🚀 Welcome to Humigence",
            border_style="blue",
        )
    )

    # Project configuration (always asked first)
    if INQUIRER_AVAILABLE:
        _ = inquirer.text(
            message="Project name:",
            default=current_config.project if current_config else "humigence",
        ).execute()
    else:
        _ = input("Project name [humigence]: ").strip() or "humigence"

    # Mode selection (if not provided via CLI)
    if mode is None:
        mode = select_wizard_mode()

    # Run appropriate setup based on mode
    if mode == WizardMode.BASIC:
        return run_basic_setup(current_config)
    else:  # ADVANCED
        if INQUIRER_AVAILABLE:
            return run_inquirer_wizard(current_config)
        else:
            return run_basic_wizard(current_config)


def run_inquirer_wizard(current_config: Config | None) -> Config:
    """Run wizard using InquirerPy for rich interactive experience."""

    # Project configuration
    _ = inquirer.text(
        message="Project name:",
        default=current_config.project if current_config else "humigence",
    ).execute()

    # GPU device selection
    gpu_choices = detect_gpus()
    # Add multi-GPU as disabled option
    gpu_choices.append(
        {"name": "Multi-GPU (coming soon)", "value": None, "disabled": "coming soon"}
    )

    gpu_device = inquirer.select(
        message="Select GPU device:",
        choices=gpu_choices,
        default=current_config.compute.gpus if current_config else 1,
    ).execute()

    # Base model selection
    model_choices = [
        {"name": "Qwen/Qwen2.5-0.5B (default)", "value": "Qwen/Qwen2.5-0.5B"},
        {"name": "Phi-2", "value": "microsoft/phi-2"},
        {"name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "value": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"},
        {"name": "microsoft/phi-1_5", "value": "microsoft/phi-1_5"},
        {"name": "Llama 3.1 (license gated)", "value": None, "disabled": "coming soon"},
    ]

    base_model = inquirer.select(
        message="Choose base model:",
        choices=model_choices,
        default=current_config.model.repo if current_config else "Qwen/Qwen2.5-0.5B",
    ).execute()

    # Model download confirmation
    _ = inquirer.confirm(message="Download model if missing?", default=False).execute()

    # Dataset source selection
    dataset_source = inquirer.select(
        message="Dataset source:",
        choices=[
            {"name": "Bundled OpenAssist demo (13 samples - quick test)", "value": "bundled"},
            {"name": "Local JSONL file (enter path)", "value": "local"},
            {"name": "Generate realistic demo (1000 samples - proper training)", "value": "generate"},
        ],
        default="bundled",
    ).execute()

    # Handle dataset source selection
    if dataset_source == "bundled":
        import shutil
        from importlib.resources import files

        try:
            demo_path = files("humigence.assets.datasets") / "openassist_demo.jsonl"
            raw_path = Path("data/raw/oa.jsonl")
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(demo_path, raw_path)
            chosen_raw_path = str(raw_path)
            console.print("[green]✓ Using bundled OpenAssist demo dataset[/green]")

            # Verify the file was copied successfully
            if not raw_path.exists():
                raise FileNotFoundError("Failed to copy bundled dataset")

        except Exception as e:
            console.print(f"[red]Error copying bundled dataset: {e}[/red]")
            console.print("[yellow]Falling back to generating realistic demo dataset...[/yellow]")
            from .data_utils import create_demo_dataset
            raw_path = Path("data/raw/oa.jsonl")
            create_demo_dataset(raw_path, schema="chat_messages", n=1000)
            chosen_raw_path = str(raw_path)
            console.print("[green]✓ Generated fallback fine-tuning dataset (1000 samples)[/green]")

    elif dataset_source == "local":
        chosen_raw_path = inquirer.text(
            message="Enter path to local JSONL file:",
            default=current_config.data.raw_path
            if current_config
            else "data/raw/oa.jsonl",
        ).execute()

        # Validate file exists
        if not Path(chosen_raw_path).exists():
            console.print(f"[red]Error: File not found: {chosen_raw_path}[/red]")
            raise FileNotFoundError(f"Dataset file not found: {chosen_raw_path}")

    else:  # generate
        from .data_utils import create_demo_dataset

        raw_path = Path("data/raw/oa.jsonl")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        create_demo_dataset(raw_path, schema="chat_messages", n=1000)
        chosen_raw_path = str(raw_path)
        console.print("[green]✓ Generated realistic fine-tuning dataset (1000 samples)[/green]")

    # Data schema selection
    schema_choices = [
        {"name": "chat_messages (default)", "value": "chat_messages"},
        {"name": "instruction_output", "value": "instruction_output"},
        {"name": "alpaca", "value": None, "disabled": "coming soon"},
        {"name": "sharegpt", "value": None, "disabled": "coming soon"},
        {"name": "oasst-1", "value": None, "disabled": "coming soon"},
        {"name": "parquet", "value": None, "disabled": "coming soon"},
    ]

    data_schema = inquirer.select(
        message="Data format/schema:",
        choices=schema_choices,
        default=current_config.data.data_schema if current_config else "chat_messages",
    ).execute()

    # Data splits
    train_split = inquirer.text(
        message="Training split ratio (0.0-1.0):",
        default=str(current_config.data.split["train"] if current_config else 0.8),
    ).execute()

    val_split = inquirer.text(
        message="Validation split ratio (0.0-1.0):",
        default=str(current_config.data.split["val"] if current_config else 0.1),
    ).execute()

    test_split = inquirer.text(
        message="Test split ratio (0.0-1.0):",
        default=str(current_config.data.split["test"] if current_config else 0.1),
    ).execute()

    # Sequence length
    max_seq_len = inquirer.select(
        message="Maximum sequence length:",
        choices=[
            {"name": "512", "value": 512},
            {"name": "1024 (default)", "value": 1024},
            {"name": "2048", "value": None, "disabled": "pending backend check"},
        ],
        default=current_config.data.max_seq_len if current_config else 1024,
    ).execute()

    # Data packing
    packing = inquirer.confirm(
        message="Enable data packing?",
        default=current_config.data.packing if current_config else True,
    ).execute()

    # Training precision mode
    precision_choices = [
        {"name": "qlora_nf4 (default)", "value": "qlora_nf4"},
        {"name": "lora_fp16", "value": "lora_fp16"},
        {"name": "lora_bf16", "value": "lora_bf16"},
        {"name": "lora_int8", "value": "lora_int8"},
    ]

    precision_mode = inquirer.select(
        message="Training precision mode:",
        choices=precision_choices,
        default=current_config.train.precision_mode if current_config else "qlora_nf4",
    ).execute()

    # LoRA configuration
    lora_r = inquirer.text(
        message="LoRA rank (r):",
        default=str(current_config.train.lora.r if current_config else 16),
    ).execute()

    lora_alpha = inquirer.text(
        message="LoRA alpha:",
        default=str(current_config.train.lora.alpha if current_config else 32),
    ).execute()

    lora_dropout = inquirer.text(
        message="LoRA dropout:",
        default=str(current_config.train.lora.dropout if current_config else 0.05),
    ).execute()

    # LoRA target modules
    target_module_choices = [
        {"name": "q_proj", "value": "q_proj", "enabled": True},
        {"name": "k_proj", "value": "k_proj", "enabled": True},
        {"name": "v_proj", "value": "v_proj", "enabled": True},
        {"name": "o_proj", "value": "o_proj", "enabled": True},
        {"name": "up_proj", "value": "up_proj", "enabled": True},
        {"name": "down_proj", "value": "down_proj", "enabled": True},
        {"name": "gate_proj", "value": "gate_proj", "enabled": True},
    ]

    target_modules = inquirer.checkbox(
        message="Select LoRA target modules:",
        choices=target_module_choices,
        default=current_config.train.lora.target_modules
        if current_config
        else ["q_proj", "k_proj", "v_proj", "o_proj"],
    ).execute()

    # Training parameters
    tokens_per_step = inquirer.text(
        message="Tokens per step target:",
        default=str(
            current_config.train.tokens_per_step_target if current_config else 100000
        ),
    ).execute()

    eval_every_steps = inquirer.text(
        message="Evaluate every N steps:",
        default=str(current_config.train.eval_every_steps if current_config else 500),
    ).execute()

    save_every_steps = inquirer.text(
        message="Save checkpoint every N steps:",
        default=str(current_config.train.save_every_steps if current_config else 500),
    ).execute()

    # Evaluation & Acceptance
    curated_prompts_path = inquirer.text(
        message="Curated prompts path:",
        default=current_config.eval.curated_prompts_path
        if current_config
        else "configs/curated_eval_prompts.jsonl",
    ).execute()

    min_val_loss_improvement = inquirer.text(
        message="Min validation loss improvement (%):",
        default=str(
            current_config.acceptance.min_val_improvement_pct if current_config else 1.0
        ),
    ).execute()

    curated_reasonable_threshold = inquirer.text(
        message="Curated reasonable threshold (%):",
        default=str(
            current_config.acceptance.curated_reasonable_threshold_pct
            if current_config
            else 70.0
        ),
    ).execute()

    jitter_threshold = inquirer.text(
        message="Jitter threshold (%):",
        default=str(
            current_config.acceptance.throughput_jitter_pct if current_config else 20.0
        ),
    ).execute()

    # Exports
    export_formats = inquirer.checkbox(
        message="Select export formats:",
        choices=[
            {
                "name": "peft_adapter (default)",
                "value": "peft_adapter",
                "enabled": True,
            },
            {"name": "merged_fp16", "value": "merged_fp16", "disabled": "coming soon"},
            {
                "name": "runtime_int8",
                "value": "runtime_int8",
                "disabled": "coming soon",
            },
        ],
        default=current_config.export.formats if current_config else ["peft_adapter"],
    ).execute()

    # Build the configuration
    config_data = {
        "project": "humigence",
        "model": {"repo": base_model, "local_path": None, "use_flash_attn": True},
        "data": {
            "raw_path": chosen_raw_path,
            "processed_dir": "data/processed",
            "schema": data_schema,  # Will be mapped to data_schema via alias
            "max_seq_len": max_seq_len,
            "packing": packing,
            "split": {
                "train": float(train_split),
                "val": float(val_split),
                "test": float(test_split),
            },
            "template": "qwen_chat_basic_v1",
        },
        "train": {
            "precision_mode": precision_mode,
            "lora": {
                "target_modules": target_modules,
                "r": int(lora_r),
                "alpha": int(lora_alpha),
                "dropout": float(lora_dropout),
            },
            "tokens_per_step_target": int(tokens_per_step),
            "eval_every_steps": int(eval_every_steps),
            "save_every_steps": int(save_every_steps),
            "lr": 0.0002,
            "scheduler": "cosine",
            "warmup_ratio": 0.03,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "gradient_checkpointing": True,
        },
        "compute": {
            "gpus": int(gpu_device) if gpu_device else 1,
            "gpu_type": "RTX_4080_16GB",
        },
        "eval": {"curated_prompts_path": curated_prompts_path},
        "acceptance": {
            "min_val_improvement_pct": float(min_val_loss_improvement),
            "curated_reasonable_threshold_pct": float(curated_reasonable_threshold),
            "throughput_jitter_pct": float(jitter_threshold),
        },
        "export": {"formats": export_formats},
    }

    return Config(**config_data)


def run_basic_wizard(current_config: Config | None) -> Config:
    """Fallback wizard using basic input prompts."""
    console.print(
        "[yellow]Using basic input prompts (InquirerPy not available)[/yellow]"
    )

    # Simple text-based configuration
    _ = input("Project name [humigence]: ").strip() or "humigence"
    base_model = (
        input("Base model [Qwen/Qwen2.5-0.5B]: ").strip() or "Qwen/Qwen2.5-0.5B"
    )

    # Build minimal config
    config_data = {
        "project": "humigence",
        "model": {"repo": base_model, "local_path": None, "use_flash_attn": True},
        "data": {
            "raw_path": "data/raw/oasst1_conversations.jsonl",
            "processed_dir": "data/processed",
            "schema": "chat_messages",
            "max_seq_len": 1024,
            "packing": True,
            "split": {"train": 0.8, "val": 0.1, "test": 0.1},
            "template": "qwen_chat_basic_v1",
        },
        "train": {
            "precision_mode": "qlora_nf4",
            "lora": {
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "r": 16,
                "alpha": 32,
                "dropout": 0.05,
            },
            "tokens_per_step_target": 100000,
            "eval_every_steps": 500,
            "save_every_steps": 500,
            "lr": 0.0002,
            "scheduler": "cosine",
            "warmup_ratio": 0.03,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "gradient_checkpointing": True,
        },
        "compute": {"gpus": 1, "gpu_type": "RTX_4080_16GB"},
        "eval": {"curated_prompts_path": "configs/curated_eval_prompts.jsonl"},
        "acceptance": {
            "min_val_improvement_pct": 1.0,
            "curated_reasonable_threshold_pct": 70.0,
            "throughput_jitter_pct": 20.0,
        },
        "export": {"formats": ["peft_adapter"]},
    }

    return Config(**config_data)


def print_configuration_summary(config: Config, mode: WizardMode | None = None) -> None:
    """Print a rich summary of the configuration."""
    # Show mode information
    if mode:
        mode_text = "Basic Setup" if mode == WizardMode.BASIC else "Advanced Setup"
        console.print(f"\n[bold green]Setup Mode: {mode_text}[/bold green]")
        if mode == WizardMode.BASIC:
            console.print("[yellow]Note: All other parameters set to defaults[/yellow]")

    table = Table(
        title="Configuration Summary", show_header=True, header_style="bold magenta"
    )
    table.add_column("Category", style="cyan")
    table.add_column("Setting", style="green")
    table.add_column("Value", style="yellow")

    table.add_row("Project", "Name", config.project)
    table.add_row("Model", "Repository", config.model.repo)
    table.add_row("Data", "Schema", config.data.data_schema)
    table.add_row("Data", "Max Seq Len", str(config.data.max_seq_len))
    table.add_row("Training", "Precision", config.train.precision_mode)
    table.add_row("LoRA", "Rank (r)", str(config.train.lora.r))
    table.add_row("LoRA", "Alpha", str(config.train.lora.alpha))
    table.add_row("LoRA", "Targets", ", ".join(config.train.lora.target_modules))

    console.print(table)


# These functions are no longer needed as actions are now executed in CLI
# def run_selected_action(config: Config, run: str, allow_train: bool) -> int:
# def print_next_command(run: str, allow_train: bool) -> None:


def run_wizard(
    config_path: Path, default_action: str | None = None, train: bool = False, mode: WizardMode | None = None
) -> dict:
    """Run the interactive configuration wizard.

    Args:
        config_path: Path to save/load configuration
        default_action: Default action to suggest (plan|validate|pipeline)
        train: Whether training is allowed
        mode: Wizard mode (basic|advanced) - if None, interactive selection

    Returns:
        dict: {
            "config_path": Path,
            "next_action": str,  # one of {"plan", "validate", "pipeline", None}
            "train": bool,        # derived from CLI flag or env TRAIN
            "exit_code": int      # exit code for CLI
        }
    """
    # Use default action if provided, otherwise default to plan
    run = default_action or "plan"

    try:
        # Load existing config if available
        current_config = None
        if config_path.exists():
            try:
                current_config = Config.from_file(config_path)
                console.print(f"[green]✓ Loaded existing config: {config_path}[/green]")
            except Exception as e:
                console.print(
                    f"[yellow]Warning: Could not load existing config: {e}[/yellow]"
                )

        # Run the wizard
        config = run_configuration_wizard(current_config, mode)

        # Save config atomically
        save_config_atomic(config_path, config)
        console.print(f"[green]✓ Configuration saved to: {config_path}[/green]")

        # Set _source_path for future updates
        config._source_path = config_path

        # Print configuration summary
        print_configuration_summary(config, mode)

        # Handle pipeline confirmation if training is disabled
        if run == "pipeline" and not train:
            console.print(
                "[yellow]⚠️  Training is disabled by default. Run pipeline without training?[/yellow]"
            )
            response = (
                input("Continue with pipeline (skip training)? [Y/n]: ").strip().lower()
            )
            if response in ["n", "no"]:
                console.print("[blue]Switching to validation mode...[/blue]")
                run = "validate"
            else:
                console.print(
                    "[blue]Continuing with pipeline (training will be skipped)...[/blue]"
                )
        elif run == "pipeline" and train:
            console.print(
                "[green]🚀 Training is enabled! Pipeline will run with full training.[/green]"
            )
            console.print(
                "[blue]This will execute: Plan → Preprocess → Train → Eval → Pack → Acceptance[/blue]"
            )

        # Return the wizard result
        return {
            "config_path": config_path,
            "next_action": run,
            "train": train,
            "exit_code": 0,
        }

    except KeyboardInterrupt:
        console.print("\n[yellow]Wizard cancelled by user[/yellow]")
        return {
            "config_path": config_path,
            "next_action": None,
            "train": train,
            "exit_code": 0,
        }
    except Exception as e:
        console.print(f"[red]Error in wizard: {e}[/red]")
        return {
            "config_path": config_path,
            "next_action": None,
            "train": train,
            "exit_code": 2,
        }
