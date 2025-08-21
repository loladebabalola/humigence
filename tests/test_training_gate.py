"""
Tests for the training readiness gate.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from humigence.training_gate import (
    TrainingReadinessError,
    validate_fsdp_config,
    validate_training_arguments_compatibility,
    validate_training_readiness,
)


class TestTrainingReadiness:
    """Test training readiness validation."""

    def test_validate_training_readiness_success(self):
        """Test successful validation."""
        config = Mock()
        config.export.artifacts_dir = "artifacts/test"
        config.get_data_paths.return_value = {
            "train": Path("data/processed/train.jsonl"),
            "val": Path("data/processed/val.jsonl"),
            "test": Path("data/processed/test.jsonl"),
        }

        train_dataset = Mock()
        train_dataset.__len__ = Mock(return_value=100)

        eval_dataset = Mock()
        eval_dataset.__len__ = Mock(return_value=20)

        runs_dir = Path("runs/test")

        # Mock file existence
        with patch("pathlib.Path.exists", return_value=True):
            validate_training_readiness(config, train_dataset, eval_dataset, runs_dir)

    def test_validate_training_readiness_empty_train(self):
        """Test validation fails with empty training dataset."""
        config = Mock()
        config.export.artifacts_dir = "artifacts/test"

        train_dataset = Mock()
        train_dataset.__len__ = Mock(return_value=0)

        eval_dataset = Mock()
        eval_dataset.__len__ = Mock(return_value=20)

        runs_dir = Path("runs/test")

        with pytest.raises(TrainingReadinessError, match="No training samples found"):
            validate_training_readiness(config, train_dataset, eval_dataset, runs_dir)

    def test_validate_training_readiness_empty_eval(self):
        """Test validation fails with empty evaluation dataset."""
        config = Mock()
        config.export.artifacts_dir = "artifacts/test"

        train_dataset = Mock()
        train_dataset.__len__ = Mock(return_value=100)

        eval_dataset = Mock()
        eval_dataset.__len__ = Mock(return_value=0)

        runs_dir = Path("runs/test")

        with pytest.raises(TrainingReadinessError, match="No validation samples found"):
            validate_training_readiness(config, train_dataset, eval_dataset, runs_dir)


class TestFSDPConfig:
    """Test FSDP configuration validation."""

    def test_validate_fsdp_config_no_conflict(self):
        """Test FSDP config with no conflicts."""
        config = Mock()
        config.train.fsdp = True
        config.train.fsdp_full_shard = False

        result = validate_fsdp_config(config)
        assert result["fsdp"] is True
        assert result["fsdp_full_shard"] is None

    def test_validate_fsdp_config_conflict_resolution(self):
        """Test FSDP config conflict resolution."""
        config = Mock()
        config.train.fsdp = True
        config.train.fsdp_full_shard = True

        result = validate_fsdp_config(config)
        assert result["fsdp"] is None
        assert result["fsdp_full_shard"] is True


class TestTrainingArgumentsCompatibility:
    """Test training arguments compatibility detection."""

    @patch("transformers.__version__", "4.35.0")
    def test_validate_training_arguments_compatibility_modern(self):
        """Test compatibility with modern transformers version."""
        result = validate_training_arguments_compatibility()
        assert "evaluation_strategy" in result
        assert "save_strategy" in result
        assert "logging_strategy" in result

    @patch("transformers.__version__", "4.25.0")
    def test_validate_training_arguments_compatibility_older(self):
        """Test compatibility with older transformers version."""
        result = validate_training_arguments_compatibility()
        assert "eval_strategy" in result
        assert "save_strategy" in result
        assert "logging_strategy" in result
