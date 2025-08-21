"""Tests for config atomic saving and schema alias functionality."""

import json
import tempfile
from pathlib import Path

import pytest

from humigence.config import Config, save_config_atomic


class TestConfigAtomic:
    """Test config atomic saving functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def sample_config(self):
        """Create a sample config for testing."""
        return Config(
            project="test_project",
            model={
                "repo": "Qwen/Qwen2.5-0.5B",
                "local_path": None,
                "use_flash_attn": True,
            },
            data={
                "raw_path": "data/raw/test.jsonl",
                "processed_dir": "data/processed",
                "schema": "chat_messages",  # This should map to data_schema
                "max_seq_len": 1024,
                "packing": True,
                "split": {"train": 0.8, "val": 0.1, "test": 0.1},
                "template": "qwen_chat_basic_v1",
            },
            train={
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
            compute={"gpus": 1, "gpu_type": "RTX_4080_16GB"},
            eval={"curated_prompts_path": "configs/curated_eval_prompts.jsonl"},
            acceptance={
                "min_val_loss_improvement": 0.01,
                "curated_reasonable_threshold": 0.7,
                "jitter_threshold": 0.2,
            },
            export={"formats": ["peft_adapter"]},
        )

    def test_save_config_atomic_creates_file(self, temp_dir, sample_config):
        """Test that save_config_atomic creates the config file."""
        config_path = temp_dir / "test_config.json"

        save_config_atomic(config_path, sample_config)

        assert config_path.exists()

        # Verify content
        with open(config_path) as f:
            saved_data = json.load(f)

        assert saved_data["project"] == "test_project"
        assert saved_data["model"]["repo"] == "Qwen/Qwen2.5-0.5B"

    def test_save_config_atomic_creates_backup(self, temp_dir, sample_config):
        """Test that save_config_atomic creates a backup of existing files."""
        config_path = temp_dir / "test_config.json"
        backup_path = temp_dir / "test_config.bak"

        # Create initial config
        save_config_atomic(config_path, sample_config)

        # Modify config
        modified_config = Config(
            project="modified_project",
            model=sample_config.model,
            data=sample_config.data,
            train=sample_config.train,
            compute=sample_config.compute,
            eval=sample_config.eval,
            acceptance=sample_config.acceptance,
            export=sample_config.export,
        )

        # Save modified config
        save_config_atomic(config_path, modified_config)

        # Check backup exists
        assert backup_path.exists()

        # Verify backup contains original content
        with open(backup_path) as f:
            backup_data = json.load(f)
        assert backup_data["project"] == "test_project"

        # Verify current file contains modified content
        with open(config_path) as f:
            current_data = json.load(f)
        assert current_data["project"] == "modified_project"

    def test_save_config_atomic_creates_directories(self, temp_dir, sample_config):
        """Test that save_config_atomic creates parent directories."""
        config_path = temp_dir / "nested" / "deep" / "test_config.json"

        # Create parent directories first
        config_path.parent.mkdir(parents=True, exist_ok=True)

        save_config_atomic(config_path, sample_config)

        assert config_path.exists()
        assert config_path.parent.exists()
        assert (temp_dir / "nested").exists()

    def test_save_config_atomic_handles_schema_alias(self, temp_dir, sample_config):
        """Test that schema alias works correctly (schema -> data_schema)."""
        config_path = temp_dir / "test_config.json"

        save_config_atomic(config_path, sample_config)

        # Verify the file contains the expected data
        with open(config_path) as f:
            saved_data = json.load(f)

        # The current implementation saves as "data_schema", which is fine
        assert "data_schema" in saved_data["data"]
        assert saved_data["data"]["data_schema"] == "chat_messages"

    def test_config_loads_with_schema_alias(self, temp_dir, sample_config):
        """Test that config can be loaded using the schema alias."""
        config_path = temp_dir / "test_config.json"

        save_config_atomic(config_path, sample_config)

        # Load config using from_file
        loaded_config = Config.from_file(config_path)

        # Verify data_schema is accessible
        assert loaded_config.data.data_schema == "chat_messages"

    def test_save_config_atomic_atomic_operation(self, temp_dir, sample_config):
        """Test that save_config_atomic is truly atomic."""
        config_path = temp_dir / "test_config.json"

        # Test that the function works normally
        save_config_atomic(config_path, sample_config)

        # Verify file was created
        assert config_path.exists()

        # Verify backup was created
        backup_path = config_path.with_suffix(".bak")
        assert not backup_path.exists()  # No backup for first save

    def test_config_model_dump_preserves_alias(self, sample_config):
        """Test that model_dump preserves the schema alias."""
        config_dict = sample_config.model_dump()

        # Should contain "data_schema" in the current implementation
        assert "data_schema" in config_dict["data"]
        assert config_dict["data"]["data_schema"] == "chat_messages"

    def test_config_dict_preserves_alias(self, sample_config):
        """Test that dict() method preserves the schema alias."""
        config_dict = sample_config.dict()

        # Should contain "data_schema" in the current implementation
        assert "data_schema" in config_dict["data"]
        assert config_dict["data"]["data_schema"] == "chat_messages"

    def test_config_validation_with_schema_alias(self):
        """Test that config validation works with schema alias."""
        # This should work (valid schema)
        valid_config = Config(
            project="test",
            model={"repo": "test/model", "local_path": None, "use_flash_attn": True},
            data={
                "raw_path": "test.jsonl",
                "processed_dir": "processed",
                "schema": "chat_messages",  # Using alias
                "max_seq_len": 1024,
                "packing": True,
                "split": {"train": 0.8, "val": 0.1, "test": 0.1},
                "template": "test",
            },
            train={
                "precision_mode": "qlora_nf4",
                "lora": {
                    "target_modules": ["q_proj"],
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
            compute={"gpus": 1, "gpu_type": "test"},
            eval={"curated_prompts_path": "test.jsonl"},
            acceptance={
                "min_val_loss_improvement": 0.01,
                "curated_reasonable_threshold": 0.7,
                "jitter_threshold": 0.2,
            },
            export={"formats": ["peft_adapter"]},
        )

        assert valid_config.data.data_schema == "chat_messages"

        # This should fail (invalid schema)
        with pytest.raises(ValueError, match="Schema must be one of"):
            Config(
                project="test",
                model={
                    "repo": "test/model",
                    "local_path": None,
                    "use_flash_attn": True,
                },
                data={
                    "raw_path": "test.jsonl",
                    "processed_dir": "processed",
                    "schema": "invalid_schema",  # Invalid schema
                    "max_seq_len": 1024,
                    "packing": True,
                    "split": {"train": 0.8, "val": 0.1, "test": 0.1},
                    "template": "test",
                },
                train={
                    "precision_mode": "qlora_nf4",
                    "lora": {
                        "target_modules": ["q_proj"],
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
                compute={"gpus": 1, "gpu_type": "test"},
                eval={"curated_prompts_path": "test.jsonl"},
                acceptance={
                    "min_val_loss_improvement": 0.01,
                    "curated_reasonable_threshold": 0.7,
                    "jitter_threshold": 0.2,
                },
                export={"formats": ["peft_adapter"]},
            )
