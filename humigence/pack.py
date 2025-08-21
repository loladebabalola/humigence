"""
Packaging module for Humigence.
Handles exporting trained models and artifacts.
"""

import argparse
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

from .config import Config
from .utils_logging import setup_logging


class ModelPacker:
    """Handles packaging and exporting of trained models."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging()

        self.logger.info("Initializing model packer...")

        # Set up paths
        self.runs_dir = self.config.get_runs_dir()
        self.artifacts_dir = self.config.get_artifacts_dir()

        # Create artifacts directory
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def pack_model(self) -> Path:
        """Pack the trained model and related artifacts."""
        self.logger.info("Starting model packaging...")

        # Check if training run exists
        if not self.runs_dir.exists():
            raise FileNotFoundError(
                f"Training run directory not found: {self.runs_dir}"
            )

        # Copy PEFT adapter
        self._copy_adapter()

        # Copy tokenizer files
        self._copy_tokenizer()

        # Copy configuration files
        self._copy_configs()

        # Create model card
        self._create_model_card()

        # Create dataset card
        self._create_dataset_card()

        # Create metadata file
        self._create_metadata()

        self.logger.info(
            f"Model packaging completed! Artifacts saved to: {self.artifacts_dir}"
        )
        return self.artifacts_dir

    def _copy_adapter(self):
        """Copy the PEFT adapter files."""
        self.logger.info("Copying PEFT adapter...")

        adapter_files = [
            "adapter_config.json",
            "adapter_model.bin",
            "adapter_model.safetensors",
        ]

        copied_files = []
        for file_name in adapter_files:
            source_file = self.runs_dir / file_name
            if source_file.exists():
                dest_file = self.artifacts_dir / file_name
                shutil.copy2(source_file, dest_file)
                copied_files.append(file_name)

        if not copied_files:
            raise FileNotFoundError("No PEFT adapter files found in training run")

        self.logger.info(f"Copied adapter files: {', '.join(copied_files)}")

    def _copy_tokenizer(self):
        """Copy tokenizer files from the base model."""
        self.logger.info("Copying tokenizer files...")

        model_path = self.config.get_model_path()
        tokenizer_files = [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt",
        ]

        copied_files = []
        for file_name in tokenizer_files:
            source_file = model_path / file_name
            if source_file.exists():
                dest_file = self.artifacts_dir / file_name
                shutil.copy2(source_file, dest_file)
                copied_files.append(file_name)

        self.logger.info(f"Copied tokenizer files: {', '.join(copied_files)}")

    def _copy_configs(self):
        """Copy configuration files."""
        self.logger.info("Copying configuration files...")

        # Copy training configuration
        config_file = self.runs_dir / "training_results.json"
        if config_file.exists():
            dest_file = self.artifacts_dir / "training_config.json"
            shutil.copy2(config_file, dest_file)

        # Copy evaluation results
        eval_file = self.runs_dir / "eval_report.json"
        if eval_file.exists():
            dest_file = self.artifacts_dir / "evaluation_results.json"
            shutil.copy2(eval_file, dest_file)

    def _create_model_card(self):
        """Create a model card for the trained model."""
        self.logger.info("Creating model card...")

        model_card = f"""# Humigence Fine-tuned Model

## Model Description

This is a fine-tuned version of {self.config.model.repo} using QLoRA (Quantized Low-Rank Adaptation).

## Training Details

- **Base Model**: {self.config.model.repo}
- **Training Method**: {self.config.train.precision_mode}
- **LoRA Rank**: {self.config.train.lora.r}
- **LoRA Alpha**: {self.config.train.lora.alpha}
- **Learning Rate**: {self.config.train.lr}
        - **Training Data**: Custom dataset with {self.config.data.data_schema} schema
- **Max Sequence Length**: {self.config.data.max_seq_len}

## QLoRA Configuration

- **Precision Mode**: {self.config.train.precision_mode}
- **Target Modules**: {', '.join(self.config.train.lora.target_modules)}
- **Dropout**: {self.config.train.lora.dropout}

## Usage

This model can be loaded using the PEFT library:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("{self.config.model.repo}")
tokenizer = AutoTokenizer.from_pretrained("{self.config.model.repo}")

# Load adapter
model = PeftModel.from_pretrained(base_model, "path/to/adapter")
```

## Training Configuration

The model was trained with the following configuration:
- **Seed**: {self.config.seed}
- **Gradient Checkpointing**: {self.config.train.gradient_checkpointing}
- **Weight Decay**: {self.config.train.weight_decay}
- **Gradient Clipping**: {self.config.train.grad_clip}
- **Scheduler**: {self.config.train.scheduler}
- **Warmup Ratio**: {self.config.train.warmup_ratio}

## Model Performance

See `evaluation_results.json` for detailed performance metrics.

## Known Limits

- **Context Length**: Limited to {self.config.data.max_seq_len} tokens
        - **Training Data**: Trained on {self.config.data.data_schema} format data
- **Domain**: General purpose, may need domain-specific fine-tuning

## License

This model inherits the license from the base model {self.config.model.repo}.

## Citation

If you use this model, please cite the base model and the QLoRA paper.
"""

        model_card_file = self.artifacts_dir / "model_card.md"
        with open(model_card_file, "w", encoding="utf-8") as f:
            f.write(model_card)

        self.logger.info("Model card created")

    def _create_dataset_card(self):
        """Create a dataset card for the training data."""
        self.logger.info("Creating dataset card...")

        dataset_card = f"""# Humigence Training Dataset

## Dataset Description

This dataset was used to fine-tune the {self.config.model.repo} model using QLoRA.

## Dataset Structure

        - **Schema**: {self.config.data.data_schema}
- **Format**: JSONL with instruction-output pairs or chat messages
- **Source**: Custom dataset
- **Provenance**: Training data for Humigence fine-tuning pipeline

## Data Processing

The dataset underwent the following processing steps:
1. Schema validation
2. Data cleaning and filtering
3. Length filtering (max {self.config.data.max_seq_len} tokens)
4. Deduplication
5. Train/validation/test splitting

## Split Ratios

- **Train**: {self.config.data.split['train'] * 100}%
- **Validation**: {self.config.data.split['val'] * 100}%
- **Test**: {self.config.data.split['test'] * 100}%

## Data Quality

The dataset was cleaned to ensure:
- Valid schema compliance
- Minimum content length requirements
- Removal of duplicate entries
- Appropriate sequence lengths for training

## Usage Notes

- This dataset is intended for fine-tuning language models
- The data format follows standard instruction-following or chat patterns
- All data has been processed and validated for training use

## Cleaning Steps

        1. **Schema Validation**: Ensured all samples conform to {self.config.data.data_schema} format
2. **Length Filtering**: Removed samples exceeding {self.config.data.max_seq_len} tokens
3. **Deduplication**: Eliminated exact and near-duplicate entries
4. **Quality Filtering**: Removed samples with insufficient content
5. **Split Generation**: Created train/validation/test splits with specified ratios

## License

Please ensure you have appropriate rights to use the source data.
"""

        dataset_card_file = self.artifacts_dir / "dataset_card.md"
        with open(dataset_card_file, "w", encoding="utf-8") as f:
            f.write(dataset_card)

        self.logger.info("Dataset card created")

    def _create_metadata(self):
        """Create a metadata file with all relevant information."""
        self.logger.info("Creating metadata file...")

        metadata = {
            "model_info": {
                "base_model": self.config.model.repo,
                "fine_tuning_method": "QLoRA",
                "precision_mode": self.config.train.precision_mode,
                "lora_rank": self.config.train.lora.r,
                "lora_alpha": self.config.train.lora.alpha,
                "lora_dropout": self.config.train.lora.dropout,
                "lora_target_modules": self.config.train.lora.target_modules,
            },
            "training_info": {
                "learning_rate": self.config.train.lr,
                "max_sequence_length": self.config.data.max_seq_len,
                "gradient_checkpointing": self.config.train.gradient_checkpointing,
                "seed": self.config.seed,
                "epochs": getattr(self.config.train, 'epochs', 'auto'),
            },
            "data_info": {
                "schema": self.config.data.data_schema,
                "split_ratios": self.config.data.split,
                "packing": self.config.data.packing,
            },
            "packaging_info": {
                "packaged_at": datetime.now().isoformat(),
                "humigence_version": "0.1.0",
                "artifacts_directory": str(self.artifacts_dir),
            },
        }

        metadata_file = self.artifacts_dir / "metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        self.logger.info("Metadata file created")

    def get_artifacts_summary(self) -> dict:
        """Get a summary of the packaged artifacts."""
        if not self.artifacts_dir.exists():
            return {}

        artifacts = {}
        for file_path in self.artifacts_dir.iterdir():
            if file_path.is_file():
                file_size = file_path.stat().st_size
                artifacts[file_path.name] = {
                    "size_bytes": file_size,
                    "size_mb": file_size / (1024 * 1024),
                }

        return artifacts


def main():
    """Main function for the packaging CLI."""
    parser = argparse.ArgumentParser(description="Humigence Model Packaging")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration file"
    )

    args = parser.parse_args()

    try:
        # Load configuration
        config = Config.from_file(args.config)

        # Initialize packer
        packer = ModelPacker(config)

        # Pack model
        artifacts_dir = packer.pack_model()

        # Print summary
        summary = packer.get_artifacts_summary()
        print(f"\nArtifacts packaged successfully to: {artifacts_dir}")
        print("\nArtifacts summary:")
        for filename, info in summary.items():
            print(f"  {filename}: {info['size_mb']:.2f} MB")

    except Exception as e:
        logging.error(f"Packaging failed: {e}")
        raise


if __name__ == "__main__":
    main()
