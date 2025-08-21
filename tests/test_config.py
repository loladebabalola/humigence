"""Test configuration validation and schema."""

import json
from pathlib import Path

import pytest

from humigence.config import Config


class TestConfig:
    """Test configuration loading and validation."""

    def test_load_basic_config(self):
        """Test loading the basic configuration file."""
        config_path = Path("configs/humigence.basic.json")
        assert config_path.exists(), "Basic config file should exist"

        config = Config.from_file(config_path)
        assert config is not None
        assert config.project == "humigence"
        assert config.seed == 42

    def test_precision_mode_validation(self):
        """Test that precision_mode accepts valid values."""
        valid_modes = ["qlora_nf4", "lora_fp16", "lora_bf16", "lora_int8"]

        for mode in valid_modes:
            config_data = {
                "project": "test",
                "seed": 42,
                "model": {"repo": "test/model", "local_path": None},
                "data": {
                    "raw_path": "test.jsonl",
                    "processed_dir": "test",
                    "schema": "chat_messages",
                },
                "train": {
                    "precision_mode": mode,
                    "lora": {
                        "target_modules": ["q_proj", "v_proj"],
                        "r": 16,
                        "alpha": 32,
                        "dropout": 0.1,
                    },
                },
            }

            # Should not raise validation error
            config = Config(**config_data)
            assert config.train.precision_mode == mode

    def test_invalid_precision_mode(self):
        """Test that invalid precision_mode raises error."""
        config_data = {
            "project": "test",
            "seed": 42,
            "model": {"repo": "test/model", "local_path": None},
            "data": {
                "raw_path": "test.jsonl",
                "processed_dir": "test",
                "schema": "chat_messages",
            },
            "train": {
                "precision_mode": "invalid_mode",
                "lora": {
                    "target_modules": ["q_proj", "v_proj"],
                    "r": 16,
                    "alpha": 32,
                    "dropout": 0.1,
                },
            },
        }

        with pytest.raises(ValueError):
            Config(**config_data)

    def test_required_fields(self):
        """Test that required fields are enforced."""
        # Missing required fields
        incomplete_config = {
            "project": "test"
            # Missing seed, model, train
        }

        with pytest.raises(ValueError):
            Config(**incomplete_config)

    def test_lora_config_validation(self):
        """Test LoRA configuration validation."""
        config_data = {
            "project": "test",
            "seed": 42,
            "model": {"repo": "test/model", "local_path": None},
            "data": {
                "raw_path": "test.jsonl",
                "processed_dir": "test",
                "schema": "chat_messages",
            },
            "train": {
                "precision_mode": "qlora_nf4",
                "lora": {
                    "target_modules": ["q_proj", "v_proj"],
                    "r": 16,
                    "alpha": 32,
                    "dropout": 0.1,
                },
            },
        }

        config = Config(**config_data)
        assert config.train.lora.r == 16
        assert config.train.lora.alpha == 32
        assert config.train.lora.dropout == 0.1
        assert "q_proj" in config.train.lora.target_modules

    def test_acceptance_criteria_validation(self):
        """Test acceptance criteria configuration."""
        config_data = {
            "project": "test",
            "seed": 42,
            "model": {"repo": "test/model", "local_path": None},
            "data": {
                "raw_path": "test.jsonl",
                "processed_dir": "test",
                "schema": "chat_messages",
            },
            "train": {
                "precision_mode": "qlora_nf4",
                "lora": {
                    "target_modules": ["q_proj", "v_proj"],
                    "r": 16,
                    "alpha": 32,
                    "dropout": 0.1,
                },
            },
            "acceptance": {
                "min_val_improvement_pct": 2.0,
                "throughput_jitter_pct": 15.0,
                "curated_reasonable_threshold_pct": 80.0,
            },
        }

        config = Config(**config_data)
        assert config.acceptance.min_val_improvement_pct == 2.0
        assert config.acceptance.throughput_jitter_pct == 15.0
        assert config.acceptance.curated_reasonable_threshold_pct == 80.0

    def test_export_config_validation(self):
        """Test export configuration validation."""
        config_data = {
            "project": "test",
            "seed": 42,
            "model": {"repo": "test/model", "local_path": None},
            "data": {
                "raw_path": "test.jsonl",
                "processed_dir": "test",
                "schema": "chat_messages",
            },
            "train": {
                "precision_mode": "qlora_nf4",
                "lora": {
                    "target_modules": ["q_proj", "v_proj"],
                    "r": 16,
                    "alpha": 32,
                    "dropout": 0.1,
                },
            },
            "export": {
                "artifacts_dir": "artifacts/test",
                "formats": ["peft_adapter", "merged_fp16"],
            },
        }

        config = Config(**config_data)
        assert config.export.artifacts_dir == "artifacts/test"
        assert "peft_adapter" in config.export.formats
        assert "merged_fp16" in config.export.formats


class TestConfigSchema:
    """Test configuration schema structure."""

    def test_config_file_structure(self):
        """Test that config file has expected structure."""
        config_path = Path("configs/humigence.basic.json")
        with open(config_path) as f:
            config_data = json.load(f)

        # Check top-level sections
        required_sections = [
            "project",
            "seed",
            "model",
            "compute",
            "data",
            "train",
            "eval",
            "acceptance",
            "export",
        ]
        for section in required_sections:
            assert section in config_data, f"Missing required section: {section}"

        # Check model section
        assert "repo" in config_data["model"]
        assert "local_path" in config_data["model"]

        # Check train section
        assert "precision_mode" in config_data["train"]
        assert "lora" in config_data["train"]

        # Check LoRA config
        lora = config_data["train"]["lora"]
        assert "target_modules" in lora
        assert "r" in lora
        assert "alpha" in lora
        assert "dropout" in lora

    def test_precision_mode_options(self):
        """Test that precision_mode has valid options."""
        config_path = Path("configs/humigence.basic.json")
        with open(config_path) as f:
            config_data = json.load(f)

        precision_mode = config_data["train"]["precision_mode"]
        valid_modes = ["qlora_nf4", "lora_fp16", "lora_bf16", "lora_int8"]
        assert (
            precision_mode in valid_modes
        ), f"Invalid precision_mode: {precision_mode}"

    def test_lora_target_modules(self):
        """Test that LoRA target modules are valid."""
        config_path = Path("configs/humigence.basic.json")
        with open(config_path) as f:
            config_data = json.load(f)

        target_modules = config_data["train"]["lora"]["target_modules"]
        expected_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ]

        for module in target_modules:
            assert (
                module in expected_modules
            ), f"Unexpected LoRA target module: {module}"

    def test_acceptance_thresholds(self):
        """Test that acceptance thresholds are reasonable."""
        config_path = Path("configs/humigence.basic.json")
        with open(config_path) as f:
            config_data = json.load(f)

        acceptance = config_data["acceptance"]

        # Check thresholds are positive and reasonable
        assert acceptance["min_val_improvement_pct"] > 0
        assert acceptance["throughput_jitter_pct"] > 0
        assert acceptance["curated_reasonable_threshold_pct"] > 0

        # Check thresholds are not too strict
        assert (
            acceptance["min_val_improvement_pct"] <= 10.0
        )  # 10% max improvement requirement
        assert acceptance["throughput_jitter_pct"] <= 50.0  # 50% max jitter tolerance
        assert (
            acceptance["curated_reasonable_threshold_pct"] <= 95.0
        )  # 95% max quality requirement


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
