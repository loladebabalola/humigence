"""Configuration management for Humigence."""

import json
import os
import shutil
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, validator


class ModelConfig(BaseModel):
    """Model configuration."""

    repo: str
    local_path: str | None = None
    use_flash_attn: bool = True


class ComputeConfig(BaseModel):
    """Compute configuration."""

    gpus: int = 1
    gpu_type: str = "RTX_4080_16GB"


class DataConfig(BaseModel):
    """Data configuration."""

    raw_path: str
    processed_dir: str
    data_schema: str = Field(
        alias="schema",
        description="Data schema: chat_messages or instruction_output",
        serialization_alias="schema",
    )
    max_seq_len: int = 1024
    packing: bool = True
    split: dict = Field(default_factory=lambda: {"train": 0.8, "val": 0.1, "test": 0.1})
    template: str = "qwen_chat_basic_v1"
    collator_windowing: str = "window"  # "window" or "drop"
    window_overlap: int = 128
    real_mode_threshold: int = 1000  # Minimum training samples for real data mode

    model_config = ConfigDict(populate_by_name=True)

    @validator("data_schema")
    def validate_schema(cls, v):
        valid_schemas = ["chat_messages", "instruction_output"]
        if v not in valid_schemas:
            raise ValueError(f"Schema must be one of {valid_schemas}")
        return v

    @validator("max_seq_len")
    def validate_max_seq_len(cls, v):
        if v <= 0 or v > 8192:
            raise ValueError("max_seq_len must be between 1 and 8192")
        return v


class LoRAConfig(BaseModel):
    """LoRA configuration."""

    target_modules: list[str]
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05

    @validator("target_modules")
    def validate_target_modules(cls, v):
        valid_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
            # GPT-2 style modules
            "c_attn",
            "c_proj",
            "c_fc",
        ]
        for module in v:
            if module not in valid_modules:
                raise ValueError(
                    f"Invalid target module: {module}. Valid: {valid_modules}"
                )
        return v


class EarlyStoppingConfig(BaseModel):
    """Early stopping configuration."""

    metric: str = "val_loss"
    patience: int = 3
    min_delta: float = 0.002


class TrainConfig(BaseModel):
    """Training configuration."""

    precision_mode: str = Field(
        ..., description="Precision mode: qlora_nf4, lora_fp16, lora_bf16, lora_int8"
    )
    lr: float = 0.0002
    scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    gradient_checkpointing: bool = True
    tokens_per_step_target: int = 100000
    eval_every_steps: int = 500
    save_every_steps: int = 500
    epochs: str | int = "auto_≈1"
    lora: LoRAConfig
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)

    @validator("precision_mode")
    def validate_precision_mode(cls, v):
        valid_modes = ["qlora_nf4", "lora_fp16", "lora_bf16", "lora_int8"]
        if v not in valid_modes:
            raise ValueError(f"Precision mode must be one of {valid_modes}")
        return v


class EvalConfig(BaseModel):
    """Evaluation configuration."""

    primary_metric: str = "val_loss"
    curated_prompts_path: str = "configs/curated_eval_prompts.jsonl"
    temperature_low: float = 0.2
    temperature_high: float = 0.7
    sampling_enabled: bool = False


class AcceptanceConfig(BaseModel):
    """Acceptance gates configuration."""

    # Accept multiple legacy keys by alias
    model_config = ConfigDict(populate_by_name=True)
    min_val_improvement_pct: float = Field(1.0, alias="min_val_loss_improvement")
    # Also accept exact legacy spelling if present:
    min_val_improvement_pct2: float | None = Field(
        None, alias="min_val_improvement_pct"
    )  # normalized below
    throughput_jitter_pct: float = Field(20.0, alias="jitter_threshold")
    curated_reasonable_threshold_pct: float = Field(70.0, alias="curated_threshold")

    @validator("min_val_improvement_pct")
    def validate_improvement_pct(cls, v):
        if v <= 0 or v > 10.0:
            raise ValueError("min_val_improvement_pct must be between 0 and 10.0")
        return v

    @validator("throughput_jitter_pct")
    def validate_jitter_pct(cls, v):
        if v <= 0 or v > 50.0:
            raise ValueError("throughput_jitter_pct must be between 0 and 50.0")
        return v

    @validator("curated_reasonable_threshold_pct")
    def validate_reasonable_threshold(cls, v):
        if v <= 0 or v > 95.0:
            raise ValueError(
                "curated_reasonable_threshold_pct must be between 0 and 95.0"
            )
        return v


class ExportConfig(BaseModel):
    """Export configuration."""

    artifacts_dir: str = "artifacts/humigence"
    formats: list[str] = Field(default_factory=lambda: ["peft_adapter"])

    @validator("formats")
    def validate_formats(cls, v):
        valid_formats = ["peft_adapter", "merged_fp16", "runtime_int8"]
        for fmt in v:
            if fmt not in valid_formats:
                raise ValueError(
                    f"Invalid export format: {fmt}. Valid: {valid_formats}"
                )
        return v


class Config(BaseModel):
    """Main configuration class."""

    project: str
    seed: int = 42
    model: ModelConfig
    compute: ComputeConfig = Field(default_factory=ComputeConfig)
    data: DataConfig
    train: TrainConfig
    eval: EvalConfig = Field(default_factory=EvalConfig)
    acceptance: AcceptanceConfig = Field(default_factory=AcceptanceConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)

    @classmethod
    def from_file(cls, config_path: str | Path) -> "Config":
        """Load configuration from JSON file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path) as f:
            config_data = json.load(f)

        return cls(**config_data)

    def save(self, config_path: str | Path) -> None:
        """Save configuration to JSON file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            json.dump(self.dict(), f, indent=2)

    def to_file(self, config_path: str | Path) -> None:
        """Alias for save method for backward compatibility."""
        self.save(config_path)

    def get_runs_dir(self) -> Path:
        """Get the runs directory for checkpoints and logs."""
        return Path("runs") / self.project

    def get_artifacts_dir(self) -> Path:
        """Get the artifacts directory for model exports."""
        return Path(self.export.artifacts_dir)

    def get_model_path(self) -> Path:
        """Get the resolved model path."""
        if self.model.local_path:
            return Path(self.model.local_path).expanduser()
        return Path(self.model.repo)

    def get_data_paths(self) -> dict:
        """Get the resolved data paths."""
        base_dir = Path(self.data.processed_dir)
        return {
            "train": base_dir / "train.jsonl",
            "val": base_dir / "val.jsonl",
            "test": base_dir / "test.jsonl",
        }

    def validate_for_training(self) -> None:
        """Validate configuration for training."""
        # Check that data file exists
        raw_path = Path(self.data.raw_path)
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw data file not found: {raw_path}")

        # Check that processed directory can be created
        processed_dir = Path(self.data.processed_dir)
        processed_dir.mkdir(parents=True, exist_ok=True)

        # Check that runs directory can be created
        runs_dir = self.get_runs_dir()
        runs_dir.mkdir(parents=True, exist_ok=True)

        # Check that artifacts directory can be created
        artifacts_dir = self.get_artifacts_dir()
        artifacts_dir.mkdir(parents=True, exist_ok=True)


def save_config_atomic(config_path: Path, config: Config) -> None:
    """Save configuration atomically with backup.

    Ensures parent directories exist and handles path expansion.
    """
    # Expand user paths and resolve to absolute paths
    config_path = Path(config_path).expanduser().resolve()

    # Ensure parent directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Create temporary file
    tmp = config_path.with_suffix(config_path.suffix + ".tmp")

    # Prepare data with proper serialization
    data = config.model_dump(by_alias=True)

    # Normalize acceptance aliases
    acc = data.get("acceptance") or {}
    if "min_val_improvement_pct" not in acc:
        if "min_val_loss_improvement" in acc:
            acc["min_val_improvement_pct"] = acc.pop("min_val_loss_improvement")
        elif "min_val_improvement_pct2" in acc:
            acc["min_val_improvement_pct"] = acc.pop("min_val_improvement_pct2")
    data["acceptance"] = acc

    # Write to temporary file
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # Create backup if original exists
    if config_path.exists():
        backup = config_path.with_suffix(config_path.suffix + ".bak")
        shutil.copy2(config_path, backup)

    # Atomic replace
    os.replace(tmp, config_path)
