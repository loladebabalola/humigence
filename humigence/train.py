"""
QLoRA training module for Humigence.
Handles model training with QLoRA fine-tuning.
"""

import argparse
import inspect
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from .training_gate import (
    TrainingReadinessError,
    validate_fsdp_config,
    validate_training_arguments_compatibility,
    validate_training_readiness,
)

# Set environment variables for RTX 4000 series compatibility
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

from .config import Config
from .utils_logging import create_run_logger, log_config_summary, log_system_info


class QLoRATrainer:
    """Handles QLoRA training for Humigence."""

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define runs_dir early to prevent AttributeError
        self.project = getattr(config, "project", "default")
        self.runs_root = Path("runs")
        self.runs_dir = (self.runs_root / self.project).resolve()
        self.runs_dir.mkdir(parents=True, exist_ok=True)

        # Set up logging
        self.logger = create_run_logger("humigence", self.runs_dir)

        # Set random seeds
        set_seed(self.config.seed)
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)

        self.logger.info("Initializing QLoRA trainer...")
        self._setup_model()
        self._setup_data()
        self._setup_training()

    def _setup_model(self):
        """Set up the model using the precision dispatcher."""
        self.logger.info("Loading base model...")

        # Load tokenizer
        model_path = self.config.get_model_path()
        if model_path.exists():
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(model_path), trust_remote_code=True, padding_side="right"
            )
        else:
            # Fallback to loading from the repo
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model.repo, trust_remote_code=True, padding_side="right"
            )

        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Use the new precision dispatcher
        from .precision import build_model_and_peft

        # Pass the tokenizer separately to avoid conflicts
        config_dict = self.config.dict()
        config_dict["_tokenizer"] = self.tokenizer  # Pass existing tokenizer

        self.model, _, _ = build_model_and_peft(config_dict)

        # Enable gradient checkpointing if configured
        if self.config.train.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.logger.info("Model setup completed")

    def _setup_data(self):
        """Set up training data."""
        self.logger.info("Loading training data...")

        # Load processed data
        data_paths = self.config.get_data_paths()

        train_data = self._load_jsonl_data(data_paths["train"])
        val_data = self._load_jsonl_data(data_paths["val"])

        # Tokenize the data
        train_data = self._tokenize_data(train_data)
        val_data = self._tokenize_data(val_data)

        # Convert to datasets
        self.train_dataset = Dataset.from_list(train_data)
        self.val_dataset = Dataset.from_list(val_data)

        self.logger.info(
            f"Loaded {len(train_data)} training samples and {len(val_data)} validation samples"
        )

        # Set up custom data collator for pre-tokenized data
        self.data_collator = self._create_custom_collator()

    def _load_jsonl_data(self, file_path: Path) -> list[dict]:
        """Load data from JSONL file."""
        data = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def _build_training_args(self, effective_tokens_per_step: int) -> TrainingArguments:
        """Create TrainingArguments with cross-version compatibility for Transformers."""
        # Get version-compatible base arguments
        compatible_args = validate_training_arguments_compatibility()

        # Map precision mode to TrainingArguments flags
        precision_mode = self.config.train.precision_mode
        fp16, bf16 = False, False

        if precision_mode == "qlora_nf4":
            # 4-bit quantization uses fp16 for compute
            fp16 = True
            bf16 = False
        elif precision_mode == "lora_fp16":
            # 16-bit float training
            fp16 = True
            bf16 = False
        elif precision_mode == "lora_bf16":
            # 16-bit bfloat training
            fp16 = False
            bf16 = True
        elif precision_mode == "lora_int8":
            # 8-bit integer training (no mixed precision)
            fp16 = False
            bf16 = False
        else:
            # Fallback to fp16
            fp16 = True
            bf16 = False

        # Add our specific configuration
        training_args = {
            **compatible_args,
            "output_dir": str(self.runs_dir),
            "overwrite_output_dir": False,
            "learning_rate": self.config.train.lr,
            "weight_decay": self.config.train.weight_decay,
            "warmup_ratio": self.config.train.warmup_ratio,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "per_device_train_batch_size": self.micro_batch_size,
            "per_device_eval_batch_size": max(1, self.micro_batch_size // 2),
            "num_train_epochs": 10.0,  # Force proper training length for convergence
            "logging_steps": 1,  # Log every step to see progress
            "save_steps": 10,  # Save more frequently
            "eval_steps": 5,  # Evaluate more frequently
            "save_total_limit": 5,  # Keep more checkpoints
            "dataloader_pin_memory": False,  # Avoid memory issues
            "remove_unused_columns": False,  # Keep all columns
            "fp16": fp16,
            "bf16": bf16,
        }

        # Add FSDP configuration with conflict resolution
        fsdp_config = validate_fsdp_config(self.config)
        training_args.update(fsdp_config)

        # Filter to only include valid parameters for this transformers version
        sig = inspect.signature(TrainingArguments.__init__)
        allowed = set(sig.parameters.keys())
        filtered = {k: v for k, v in training_args.items() if k in allowed}

        return TrainingArguments(**filtered)

    def _setup_training(self):
        """Set up training configuration with auto-VRAM fitting."""
        self.logger.info("Setting up training configuration...")

        # Auto-VRAM fitting: try different batch sizes to find optimal configuration
        self.micro_batch_size, self.gradient_accumulation_steps = self._auto_fit_vram()

        # Validate training readiness after VRAM fitting
        try:
            validate_training_readiness(
                self.config,
                self.train_dataset,
                self.val_dataset,
                self.runs_dir
            )
        except TrainingReadinessError as e:
            self.logger.error(f"Training readiness check failed: {e}")
            raise

        # Log final configuration
        effective_tokens_per_step = (
            self.micro_batch_size
            * self.gradient_accumulation_steps
            * self.config.data.max_seq_len
        )

        self.logger.info("=" * 60)
        self.logger.info("🎯 AUTO-VRAM FIT RESULTS")
        self.logger.info("=" * 60)
        self.logger.info(f"micro_batch_size: {self.micro_batch_size}")
        self.logger.info(f"grad_accum: {self.gradient_accumulation_steps}")
        self.logger.info(f"effective tokens/step: {effective_tokens_per_step:,}")
        self.logger.info("=" * 60)

        # Set up training arguments using compatibility shim
        self.training_args = self._build_training_args(effective_tokens_per_step)

        # Set up trainer
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
        )

        self.logger.info("Training configuration completed")

        # Run dry-run to get post-collator counts and compute steps summary
        self._compute_post_collator_counts()
        self._print_dataset_integrity_block()
        self._compute_steps_summary()
        self._save_dataset_stats()
        self._save_steps_summary()

        # Now add the steps monitor callback
        self.trainer.add_callback(self._create_steps_monitor_callback())

    def _auto_fit_vram(self) -> tuple[int, int]:
        """Automatically find optimal batch size and gradient accumulation for available VRAM."""
        self.logger.info("🔍 Auto-fitting VRAM configuration...")

        # Target tokens per step from config
        target_tokens_per_step = self.config.train.tokens_per_step_target

        # Try different micro-batch sizes, starting small for your GPU
        micro_batch_sizes = [4, 2, 1, 8, 16, 32]  # Start with smaller sizes
        max_seq_len = self.config.data.max_seq_len

        for micro_batch_size in micro_batch_sizes:
            try:
                self.logger.info(f"Testing micro_batch_size: {micro_batch_size}")

                # Calculate required gradient accumulation to reach target tokens/step
                required_grad_accum = max(
                    1, target_tokens_per_step // (micro_batch_size * max_seq_len)
                )

                # Test if this configuration fits in VRAM
                if self._test_vram_fit(micro_batch_size, required_grad_accum):
                    self.logger.info(
                        f"✅ VRAM fit successful: micro_batch_size={micro_batch_size}, grad_accum={required_grad_accum}"
                    )
                    return micro_batch_size, required_grad_accum

            except Exception as e:
                self.logger.warning(
                    f"❌ VRAM fit failed for micro_batch_size={micro_batch_size}: {e}"
                )
                continue

        # Fallback to minimal configuration
        self.logger.warning(
            "⚠️  All VRAM configurations failed, using fallback: micro_batch_size=1, grad_accum=1"
        )
        return 1, 1

    def _test_vram_fit(self, micro_batch_size: int, grad_accum: int) -> bool:
        """Test if a specific configuration fits in available VRAM."""
        try:
            # Create a realistic test batch using actual sequence length
            max_seq_len = self.config.data.max_seq_len
            test_batch = torch.randint(
                0, 1000, (micro_batch_size, max_seq_len), device=self.device
            )

            # Test forward pass with gradients enabled (more realistic)
            self.model.train()
            outputs = self.model(test_batch, labels=test_batch)
            loss = outputs.loss

            # Test backward pass (this is where most memory is used)
            loss.backward()

            # Clear gradients and cache
            self.model.zero_grad()
            torch.cuda.empty_cache()
            return True

        except torch.cuda.OutOfMemoryError:
            self.model.zero_grad()
            torch.cuda.empty_cache()
            return False
        except Exception:
            self.model.zero_grad()
            torch.cuda.empty_cache()
            return False

    def _tokenize_data(self, data: list[dict]) -> list[dict]:
        """Tokenize the data for training."""
        tokenized_data = []

        for item in data:
            text = item.get("text", "")
            target = item.get("target", "")

            # Combine input and target
            full_text = text + target

            # Tokenize
            encoding = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.config.data.max_seq_len,
                padding="max_length",
                return_tensors=None,
            )

            # Add labels (same as input_ids for causal LM)
            encoding["labels"] = encoding["input_ids"].copy()

            tokenized_data.append(encoding)

        return tokenized_data

    def _create_custom_collator(self):
        """Create a custom data collator for pre-tokenized data with windowing and drop counting."""

        class EnhancedDataCollator:
            def __init__(self, max_seq_len: int, windowing: str = "window", window_overlap: int = 128):
                self.max_seq_len = max_seq_len
                self.windowing = windowing
                self.window_overlap = window_overlap
                self.stats = {
                    "total_samples": 0,
                    "kept_samples": 0,
                    "dropped_samples": 0,
                    "windowed_samples": 0,
                    "drop_reasons": {
                        "empty_text": 0,
                        "empty_target": 0,
                        "too_long": 0,
                        "malformed": 0
                    }
                }

            def __call__(self, batch):
                """Make the collator callable - this is the main entry point."""
                return self.collate_fn(batch)

            def prepare_features(self, item):
                """Prepare features for a single item with windowing support."""
                try:
                    input_ids = item.get("input_ids", [])
                    attention_mask = item.get("attention_mask", [])
                    labels = item.get("labels", [])

                    # Validate required fields
                    if not input_ids or not attention_mask or not labels:
                        self.stats["drop_reasons"]["malformed"] += 1
                        return None

                    # Check for empty content
                    if not any(input_ids) or not any(labels):
                        self.stats["drop_reasons"]["empty_text"] += 1
                        return None

                    seq_len = len(input_ids)

                    if seq_len <= self.max_seq_len:
                        # Single sample fits
                        return {
                            "input_ids": torch.tensor(input_ids),
                            "attention_mask": torch.tensor(attention_mask),
                            "labels": torch.tensor(labels)
                        }
                    else:
                        # Sequence too long - apply windowing or drop
                        if self.windowing == "window":
                            return self._create_windows(input_ids, attention_mask, labels)
                        else:
                            # Drop mode
                            self.stats["drop_reasons"]["too_long"] += 1
                            return None

                except Exception:
                    self.stats["drop_reasons"]["malformed"] += 1
                    return None

            def _create_windows(self, input_ids, attention_mask, labels):
                """Create sliding windows for long sequences."""
                windows = []
                stride = self.max_seq_len - self.window_overlap

                for start in range(0, len(input_ids), stride):
                    end = start + self.max_seq_len
                    if end > len(input_ids):
                        end = len(input_ids)

                    # Ensure minimum window size
                    if end - start < self.max_seq_len // 2:
                        break

                    window_input_ids = input_ids[start:end]
                    window_attention_mask = attention_mask[start:end]
                    window_labels = labels[start:end]

                    # Pad if necessary
                    if len(window_input_ids) < self.max_seq_len:
                        pad_len = self.max_seq_len - len(window_input_ids)
                        window_input_ids.extend([0] * pad_len)
                        window_attention_mask.extend([0] * pad_len)
                        window_labels.extend([-100] * pad_len)  # -100 for padding in labels

                    windows.append({
                        "input_ids": torch.tensor(window_input_ids),
                        "attention_mask": torch.tensor(window_attention_mask),
                        "labels": torch.tensor(window_labels)
                    })

                self.stats["windowed_samples"] += len(windows) - 1  # Count additional windows
                return windows[0] if windows else None  # Return first window for collation

            def collate_fn(self, batch):
                """Collate a batch of samples."""
                self.stats["total_samples"] += len(batch)

                # Process each item
                processed_items = []
                for item in batch:
                    features = self.prepare_features(item)
                    if features is not None:
                        processed_items.append(features)

                self.stats["kept_samples"] += len(processed_items)
                self.stats["dropped_samples"] += len(batch) - len(processed_items)

                if not processed_items:
                    # Return empty batch with proper structure
                    return {
                        "input_ids": torch.empty((0, self.max_seq_len), dtype=torch.long),
                        "attention_mask": torch.empty((0, self.max_seq_len), dtype=torch.long),
                        "labels": torch.empty((0, self.max_seq_len), dtype=torch.long)
                    }

                # Stack tensors
                input_ids = torch.stack([item["input_ids"] for item in processed_items])
                attention_mask = torch.stack([item["attention_mask"] for item in processed_items])
                labels = torch.stack([item["labels"] for item in processed_items])

                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }

            def get_stats(self):
                """Get current statistics."""
                return self.stats.copy()

            def reset_stats(self):
                """Reset statistics."""
                self.stats = {
                    "total_samples": 0,
                    "kept_samples": 0,
                    "dropped_samples": 0,
                    "windowed_samples": 0,
                    "drop_reasons": {
                        "empty_text": 0,
                        "empty_target": 0,
                        "too_long": 0,
                        "malformed": 0
                    }
                }

        # Create collator with configurable options
        collator = EnhancedDataCollator(
            max_seq_len=self.config.data.max_seq_len,
            windowing=getattr(self.config.data, "collator_windowing", "window"),
            window_overlap=getattr(self.config.data, "window_overlap", 128)
        )

        return collator

    def _compute_post_collator_counts(self):
        """Run a dry-run with the collator to get post-collator counts."""
        self.logger.info("Running dry-run to compute post-collator counts...")

        # Reset collator stats
        self.data_collator.reset_stats()

        # Run collator on a sample of the actual training data
        sample_size = min(1000, len(self.train_dataset))  # Sample up to 1000 items
        sample_indices = list(range(sample_size))

        # Process samples through the collator
        for i in range(0, sample_size, 10):  # Process in batches of 10
            batch_indices = sample_indices[i:i+10]
            batch_data = [self.train_dataset[idx] for idx in batch_indices]
            self.data_collator(batch_data) # Use the __call__ method

        # Get final stats
        stats = self.data_collator.get_stats()

        # Extrapolate to full dataset
        total_items = len(self.train_dataset)
        extrapolation_factor = total_items / stats['total_samples'] if stats['total_samples'] > 0 else 1

        self.post_collator_stats = {
            "raw_count": total_items,
            "processed_count": total_items,
            "train_count_after_collator": int(stats['kept_samples'] * extrapolation_factor),
            "val_count": len(self.val_dataset),
            "test_count": 0,  # No test set in current config
            "drop_reasons": {
                "empty_text": int(stats['drop_reasons']['empty_text'] * extrapolation_factor),
                "empty_target": int(stats['drop_reasons']['empty_target'] * extrapolation_factor),
                "too_long": int(stats['drop_reasons']['too_long'] * extrapolation_factor),
                "malformed": int(stats['drop_reasons']['malformed'] * extrapolation_factor)
            },
            "windowed_samples": int(stats['windowed_samples'] * extrapolation_factor)
        }

        self.logger.info("Post-collator counts computed:")
        self.logger.info(f"Raw count: {self.post_collator_stats['raw_count']}")
        self.logger.info(f"Train count after collator: {self.post_collator_stats['train_count_after_collator']}")
        self.logger.info(f"Val count: {self.post_collator_stats['val_count']}")
        self.logger.info(f"Drop reasons: {self.post_collator_stats['drop_reasons']}")
        self.logger.info(f"Windowed samples: {self.post_collator_stats['windowed_samples']}")

        # Check if we have enough training data
        if self.post_collator_stats['train_count_after_collator'] < 1000:
            self.logger.warning(
                f"⚠️  Low training sample count after collator: {self.post_collator_stats['train_count_after_collator']} < 1000"
            )
            self.logger.warning("Consider using --collator_windowing=window or increasing max_seq_len")

    def _print_dataset_integrity_block(self):
        """Print a block summarizing dataset integrity."""
        self.logger.info("=" * 60)
        self.logger.info("🔍 DATASET INTEGRITY CHECK")
        self.logger.info("=" * 60)
        self.logger.info(f"Raw Training Data Count: {self.post_collator_stats['raw_count']}")
        self.logger.info(f"Processed Training Data Count (after collator): {self.post_collator_stats['train_count_after_collator']}")
        self.logger.info(f"Validation Data Count: {self.post_collator_stats['val_count']}")
        self.logger.info(f"Dropped Samples (empty text/target/too long): {self.post_collator_stats['drop_reasons']}")
        self.logger.info(f"Windowed Samples: {self.post_collator_stats['windowed_samples']}")
        self.logger.info("=" * 60)

    def _compute_steps_summary(self):
        """Compute and log the total number of steps for training."""
        total_samples = self.post_collator_stats['train_count_after_collator']
        global_batch_size = self.micro_batch_size * self.gradient_accumulation_steps
        steps_per_epoch = (total_samples + global_batch_size - 1) // global_batch_size  # Ceiling division
        expected_total_steps = steps_per_epoch * self.config.train.epochs

        self.steps_summary = {
            "global_batch_size": global_batch_size,
            "steps_per_epoch": steps_per_epoch,
            "expected_total_steps": expected_total_steps,
            "num_train_epochs": self.config.train.epochs,
            "total_training_samples": total_samples
        }

        self.logger.info("=" * 60)
        self.logger.info("🎯 STEPS SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Total Training Samples: {total_samples}")
        self.logger.info(f"Global Batch Size: {global_batch_size}")
        self.logger.info(f"Steps per Epoch: {steps_per_epoch}")
        self.logger.info(f"Expected Total Steps: {expected_total_steps}")
        self.logger.info(f"Number of Epochs: {self.config.train.epochs}")
        self.logger.info("=" * 60)

    def train(self):
        """Run the training loop."""
        self.logger.info("Starting training...")

        # Log system information
        log_system_info(self.logger)
        log_config_summary(self.logger, self.config.dict())

        # Print precision banner
        precision_mode = self.config.train.precision_mode
        if precision_mode == "qlora_nf4":
            dtype_str = "4bit-nf4"
        elif precision_mode == "lora_fp16":
            dtype_str = "fp16"
        elif precision_mode == "lora_bf16":
            dtype_str = "bf16"
        elif precision_mode == "lora_int8":
            dtype_str = "int8"
        else:
            dtype_str = "unknown"

        lora_targets = ", ".join(self.config.train.lora.target_modules)

        self.logger.info("=" * 80)
        self.logger.info(
            f"🎯 PRECISION MODE={precision_mode}; DTYPE={dtype_str}; LORA TARGETS=[{lora_targets}]"
        )
        self.logger.info("=" * 80)

        # Train the model
        self.logger.info("Training started...")
        train_result = self.trainer.train()

        # Save the final model
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.config.get_runs_dir())

        # Log training results
        self.logger.info("Training completed!")
        self.logger.info(f"Training loss: {train_result.training_loss:.4f}")

        # Save training results
        self._save_training_results(train_result)

        return train_result

    def _save_training_results(self, train_result):
        """Save training results and configuration."""
        results_file = self.config.get_runs_dir() / "training_results.json"

        results = {
            "training_loss": train_result.training_loss,
            "global_step": train_result.global_step,
            "config": self.config.dict(),
            "training_args": self.training_args.to_dict(),
        }

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"Training results saved to: {results_file}")

    def _save_dataset_stats(self):
        """Save dataset stats to a JSON file."""
        dataset_stats_file = self.config.get_runs_dir() / "dataset_stats.json"
        with open(dataset_stats_file, "w") as f:
            json.dump(self.post_collator_stats, f, indent=2)
        self.logger.info(f"Dataset stats saved to: {dataset_stats_file}")

    def _save_steps_summary(self):
        """Save steps summary to a JSON file."""
        steps_summary_file = self.config.get_runs_dir() / "steps_summary.json"
        with open(steps_summary_file, "w") as f:
            json.dump(self.steps_summary, f, indent=2)
        self.logger.info(f"Steps summary saved to: {steps_summary_file}")

    def _create_steps_monitor_callback(self):
        """Create a callback to monitor training steps and abort if they exceed the expected total by more than 10%."""
        class StepsMonitorCallback(TrainerCallback):
            def __init__(self, expected_total_steps: int):
                self.expected_total_steps = expected_total_steps
                self.current_step = 0
                self.logger = logging.getLogger(__name__)

            def on_step_end(self, args, state, control, **kwargs):
                self.current_step = state.global_step
                if self.current_step > self.expected_total_steps * 1.1:
                    self.logger.warning(
                        f"Training steps exceeded expected total by more than 10%. Current step: {self.current_step}, Expected total: {self.expected_total_steps}"
                    )
                    control.should_training_stop = True
                    self.logger.warning("Aborting training due to excessive steps.")

        return StepsMonitorCallback(self.steps_summary["expected_total_steps"])


def main():
    """Main function for the training CLI."""
    parser = argparse.ArgumentParser(description="Humigence QLoRA Training")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration file"
    )
    parser.add_argument(
        "--dry_run_counts_only", action="store_true",
        help="Only compute and display dataset counts without training"
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Run a short smoke test training run (limited steps)"
    )

    args = parser.parse_args()

    try:
        # Load configuration
        config = Config.from_file(args.config)

        # Initialize trainer
        trainer = QLoRATrainer(config)

        if args.dry_run_counts_only:
            # Just run the dataset integrity checks
            print("=" * 60)
            print("🔍 DATASET INTEGRITY CHECK (DRY RUN)")
            print("=" * 60)
            print(f"Raw count: {trainer.post_collator_stats['raw_count']}")
            print(f"Train count after collator: {trainer.post_collator_stats['train_count_after_collator']}")
            print(f"Val count: {trainer.post_collator_stats['val_count']}")
            print(f"Drop reasons: {trainer.post_collator_stats['drop_reasons']}")
            print(f"Windowed samples: {trainer.post_collator_stats['windowed_samples']}")
            print("=" * 60)

            print("\n🎯 STEPS SUMMARY")
            print("=" * 60)
            print(f"Global batch size: {trainer.steps_summary['global_batch_size']}")
            print(f"Steps per epoch: {trainer.steps_summary['steps_per_epoch']}")
            print(f"Expected total steps: {trainer.steps_summary['expected_total_steps']}")
            print(f"Number of epochs: {trainer.steps_summary['num_train_epochs']}")
            print("=" * 60)

            print("\n✅ Dataset integrity check completed successfully!")
            return 0

        # Apply smoke mode if requested
        if args.smoke:
            print("🔥 SMOKE MODE: Limiting training to 10 steps for testing")
            # Override max_steps for smoke test
            trainer.training_args.max_steps = 10
            trainer.training_args.eval_steps = 5
            trainer.training_args.save_steps = 10
            trainer.training_args.logging_steps = 1

        # Start training
        trainer.train()

    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
