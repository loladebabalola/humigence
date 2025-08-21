"""Tests for config path handling functionality."""

import tempfile
from pathlib import Path
from unittest.mock import patch

from humigence.config import Config, save_config_atomic


def test_save_config_atomic_creates_deep_nested_directories():
    """Test that save_config_atomic creates deep nested directories."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a deeply nested path
        config_path = (
            Path(temp_dir) / "nested" / "deeper" / "much_deeper" / "config.json"
        )

        # Create a sample config
        sample_config = Config(
            project="test_project",
            model={
                "repo": "Qwen/Qwen2.5-0.5B",
                "local_path": None,
                "use_flash_attn": True,
            },
            compute={"gpus": 1, "gpu_type": "RTX_4080_16GB"},
            data={
                "raw_path": "data/raw/test.jsonl",
                "processed_dir": "data/processed",
                "data_schema": "chat_messages",
                "max_seq_len": 1024,
                "packing": True,
                "split": {"train": 0.8, "val": 0.1, "test": 0.1},
                "template": "qwen_chat_basic_v1",
            },
            train={
                "precision_mode": "qlora_nf4",
                "lr": 0.0002,
                "scheduler": "cosine",
                "warmup_ratio": 0.03,
                "weight_decay": 0.0,
                "grad_clip": 1.0,
                "tokens_per_step_target": 100000,
                "eval_every_steps": 500,
                "save_every_steps": 500,
                "lora": {
                    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                    "r": 16,
                    "alpha": 32,
                    "dropout": 0.05,
                },
                "early_stopping": {
                    "metric": "val_loss",
                    "patience": 3,
                    "min_delta": 0.002,
                },
            },
            eval={"curated_prompts_path": "configs/curated_eval_prompts.jsonl"},
            acceptance={
                "min_val_loss_improvement": 0.01,
                "curated_reasonable_threshold": 0.7,
                "jitter_threshold": 0.2,
            },
            export={
                "formats": ["peft_adapter"],
                "artifacts_dir": "artifacts/humigence",
            },
        )

        # Save config to the deep nested path
        save_config_atomic(config_path, sample_config)

        # Verify that both the directory and file were created
        assert (
            config_path.parent.exists()
        ), f"Parent directory {config_path.parent} was not created"
        assert config_path.exists(), f"Config file {config_path} was not created"

        # Verify the content is correct
        with open(config_path) as f:
            import json

            saved_data = json.load(f)

        assert saved_data["project"] == "test_project"
        assert saved_data["model"]["repo"] == "Qwen/Qwen2.5-0.5B"


def test_save_config_atomic_expands_tilde():
    """Test that save_config_atomic expands ~ in paths."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock the home directory to be our temp directory
        with patch("pathlib.Path.expanduser") as mock_expanduser:
            mock_expanduser.return_value = Path(temp_dir) / "config.json"

            config_path = Path("~/test_config.json")
            sample_config = Config(
                project="test_project",
                model={
                    "repo": "Qwen/Qwen2.5-0.5B",
                    "local_path": None,
                    "use_flash_attn": True,
                },
                compute={"gpus": 1, "gpu_type": "RTX_4080_16GB"},
                data={
                    "raw_path": "data/raw/test.jsonl",
                    "processed_dir": "data/processed",
                    "data_schema": "chat_messages",
                    "max_seq_len": 1024,
                    "packing": True,
                    "split": {"train": 0.8, "val": 0.1, "test": 0.1},
                    "template": "qwen_chat_basic_v1",
                },
                train={
                    "precision_mode": "qlora_nf4",
                    "lr": 0.0002,
                    "scheduler": "cosine",
                    "warmup_ratio": 0.03,
                    "weight_decay": 0.0,
                    "grad_clip": 1.0,
                    "tokens_per_step_target": 100000,
                    "eval_every_steps": 500,
                    "save_every_steps": 500,
                    "lora": {
                        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                        "r": 16,
                        "alpha": 32,
                        "dropout": 0.05,
                    },
                    "early_stopping": {
                        "metric": "val_loss",
                        "patience": 3,
                        "min_delta": 0.002,
                    },
                },
                eval={"curated_prompts_path": "configs/curated_eval_prompts.jsonl"},
                acceptance={
                    "min_val_loss_improvement": 0.01,
                    "curated_reasonable_threshold": 0.7,
                    "jitter_threshold": 0.2,
                },
                export={
                    "formats": ["peft_adapter"],
                    "artifacts_dir": "artifacts/humigence",
                },
            )

            # Save config
            save_config_atomic(config_path, sample_config)

            # Verify that expanduser was called
            mock_expanduser.assert_called_once()


def test_save_config_atomic_creates_backup():
    """Test that save_config_atomic creates a backup when file exists."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "test_config.json"
        backup_path = config_path.with_suffix(".bak")

        # Create initial config
        initial_config = Config(
            project="initial_project",
            model={
                "repo": "Qwen/Qwen2.5-0.5B",
                "local_path": None,
                "use_flash_attn": True,
            },
            compute={"gpus": 1, "gpu_type": "RTX_4080_16GB"},
            data={
                "raw_path": "data/raw/test.jsonl",
                "processed_dir": "data/processed",
                "data_schema": "chat_messages",
                "max_seq_len": 1024,
                "packing": True,
                "split": {"train": 0.8, "val": 0.1, "test": 0.1},
                "template": "qwen_chat_basic_v1",
            },
            train={
                "precision_mode": "qlora_nf4",
                "lr": 0.0002,
                "scheduler": "cosine",
                "warmup_ratio": 0.03,
                "weight_decay": 0.0,
                "grad_clip": 1.0,
                "tokens_per_step_target": 100000,
                "eval_every_steps": 500,
                "save_every_steps": 500,
                "lora": {
                    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                    "r": 16,
                    "alpha": 32,
                    "dropout": 0.05,
                },
                "early_stopping": {
                    "metric": "val_loss",
                    "patience": 3,
                    "min_delta": 0.002,
                },
            },
            eval={"curated_prompts_path": "configs/curated_eval_prompts.jsonl"},
            acceptance={
                "min_val_loss_improvement": 0.01,
                "curated_reasonable_threshold": 0.7,
                "jitter_threshold": 0.2,
            },
            export={
                "formats": ["peft_adapter"],
                "artifacts_dir": "artifacts/humigence",
            },
        )

        # Save initial config
        save_config_atomic(config_path, initial_config)

        # Verify initial config was saved
        assert config_path.exists()

        # Create modified config
        modified_config = Config(
            project="modified_project",
            model={
                "repo": "Qwen/Qwen2.5-0.5B",
                "local_path": None,
                "use_flash_attn": True,
            },
            compute={"gpus": 1, "gpu_type": "RTX_4080_16GB"},
            data={
                "raw_path": "data/raw/test.jsonl",
                "processed_dir": "data/processed",
                "data_schema": "chat_messages",
                "max_seq_len": 1024,
                "packing": True,
                "split": {"train": 0.8, "val": 0.1, "test": 0.1},
                "template": "qwen_chat_basic_v1",
            },
            train={
                "precision_mode": "qlora_nf4",
                "lr": 0.0002,
                "scheduler": "cosine",
                "warmup_ratio": 0.03,
                "weight_decay": 0.0,
                "grad_clip": 1.0,
                "tokens_per_step_target": 100000,
                "eval_every_steps": 500,
                "save_every_steps": 500,
                "lora": {
                    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                    "r": 16,
                    "alpha": 32,
                    "dropout": 0.05,
                },
                "early_stopping": {
                    "metric": "val_loss",
                    "patience": 3,
                    "min_delta": 0.002,
                },
            },
            eval={"curated_prompts_path": "configs/curated_eval_prompts.jsonl"},
            acceptance={
                "min_val_loss_improvement": 0.01,
                "curated_reasonable_threshold": 0.7,
                "jitter_threshold": 0.2,
            },
            export={
                "formats": ["peft_adapter"],
                "artifacts_dir": "artifacts/humigence",
            },
        )

        # Save modified config (should create backup)
        save_config_atomic(config_path, modified_config)

        # Verify backup was created
        assert backup_path.exists()

        # Verify backup contains initial content
        with open(backup_path) as f:
            import json

            backup_data = json.load(f)
        assert backup_data["project"] == "initial_project"

        # Verify current file contains modified content
        with open(config_path) as f:
            current_data = json.load(f)
        assert current_data["project"] == "modified_project"
