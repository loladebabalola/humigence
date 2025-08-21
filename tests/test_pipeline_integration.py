"""Tests for the new pipeline integration functionality."""

import json
import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from humigence.cli import app, run_pipeline, validate_config_for_pipeline
from humigence.config import Config


@pytest.fixture
def runner():
    """CLI runner fixture."""
    return CliRunner()


@pytest.fixture
def valid_config(tmp_path):
    """Create a valid configuration for testing."""
    # Create a dummy dataset file
    data_file = tmp_path / "data" / "raw" / "test.jsonl"
    data_file.parent.mkdir(parents=True, exist_ok=True)

    # Create a simple test dataset
    test_data = [
        {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "I'm doing well!"},
            ]
        },
    ]

    with open(data_file, "w") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")

    # Create config
    config_data = {
        "project": "test_project",
        "model": {
            "repo": "Qwen/Qwen2.5-0.5B",
            "local_path": None,
            "use_flash_attn": True,
        },
        "compute": {"gpus": 1, "gpu_type": "RTX_4080_16GB"},
        "data": {
            "raw_path": str(data_file),
            "processed_dir": str(tmp_path / "data" / "processed"),
            "data_schema": "chat_messages",
            "max_seq_len": 1024,
            "packing": True,
            "split": {"train": 0.8, "val": 0.1, "test": 0.1},
            "template": "qwen_chat_basic_v1",
        },
        "train": {
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
        },
        "eval": {"curated_prompts_path": "configs/curated_eval_prompts.jsonl"},
        "acceptance": {
            "min_val_improvement_pct": 1.0,
            "throughput_jitter_pct": 20.0,
            "curated_reasonable_threshold_pct": 70.0,
        },
        "export": {
            "formats": ["peft_adapter"],
            "artifacts_dir": str(tmp_path / "artifacts"),
        },
    }

    return Config(**config_data)


class TestPipelineIntegration:
    """Test the new pipeline integration functionality."""

    def test_validate_config_for_pipeline_valid_config(self, valid_config):
        """Test that valid configuration passes validation."""
        is_valid, errors = validate_config_for_pipeline(valid_config)

        assert is_valid
        assert len(errors) == 0

    def test_validate_config_for_pipeline_missing_data_file(self, valid_config):
        """Test that missing data file fails validation."""
        # Remove the data file
        data_file = Path(valid_config.data.raw_path)
        if data_file.exists():
            data_file.unlink()

        is_valid, errors = validate_config_for_pipeline(valid_config)

        assert not is_valid
        assert any("Raw data file not found" in error for error in errors)

    def test_validate_config_for_pipeline_invalid_precision_mode(self, valid_config):
        """Test that invalid precision mode fails validation."""
        valid_config.train.precision_mode = "invalid_mode"

        is_valid, errors = validate_config_for_pipeline(valid_config)

        assert not is_valid
        assert any("Invalid precision mode" in error for error in errors)

    def test_validate_config_for_pipeline_invalid_lora_params(self, valid_config):
        """Test that invalid LoRA parameters fail validation."""
        valid_config.train.lora.r = -1
        valid_config.train.lora.alpha = 0

        is_valid, errors = validate_config_for_pipeline(valid_config)

        assert not is_valid
        assert any("Invalid LoRA rank" in error for error in errors)
        assert any("Invalid LoRA alpha" in error for error in errors)

    def test_run_pipeline_with_training_enabled(self, valid_config):
        """Test that pipeline runs successfully with training enabled."""
        with patch("humigence.cli.TrainingPlanner") as mock_planner, patch(
            "humigence.cli.ensure_model_available"
        ) as mock_model, patch(
            "humigence.cli.DataPreprocessor"
        ) as mock_preprocessor, patch(
            "humigence.cli.QLoRATrainer"
        ) as mock_trainer, patch(
            "humigence.cli.ModelEvaluator"
        ) as mock_evaluator, patch(
            "humigence.cli.ModelPacker"
        ) as mock_packer, patch(
            "humigence.cli.AcceptanceGates"
        ) as mock_acceptance:
            # Mock all components
            mock_planner.return_value.plan_training.return_value = {"status": "planned"}
            mock_model.return_value = Path("/tmp/model")
            mock_preprocessor.return_value.preprocess.return_value = {
                "status": "processed"
            }
            mock_trainer.return_value.train.return_value = None
            mock_evaluator.return_value.evaluate.return_value = {"status": "evaluated"}
            mock_packer.return_value.pack.return_value = {"status": "packed"}
            mock_acceptance.return_value.evaluate_training_run.return_value = Mock(
                passed=True, dict=lambda: {"passed": True}
            )

            # Run pipeline with training enabled
            result = run_pipeline(valid_config, train=True)

            # Should succeed
            assert result == 0

            # All components should be called
            mock_planner.return_value.plan_training.assert_called_once()
            mock_model.assert_called_once()
            mock_preprocessor.return_value.preprocess.assert_called_once()
            mock_trainer.return_value.train.assert_called_once()
            mock_evaluator.return_value.evaluate.assert_called_once()
            mock_packer.return_value.pack.assert_called_once()
            mock_acceptance.return_value.evaluate_training_run.assert_called_once()

    def test_run_pipeline_without_training(self, valid_config):
        """Test that pipeline runs successfully without training."""
        with patch("humigence.cli.TrainingPlanner") as mock_planner, patch(
            "humigence.cli.ensure_model_available"
        ) as mock_model, patch(
            "humigence.cli.DataPreprocessor"
        ) as mock_preprocessor, patch(
            "humigence.cli.QLoRATrainer"
        ) as mock_trainer, patch(
            "humigence.cli.ModelEvaluator"
        ) as mock_evaluator, patch(
            "humigence.cli.ModelPacker"
        ) as mock_packer, patch(
            "humigence.cli.AcceptanceGates"
        ) as mock_acceptance:
            # Mock all components
            mock_planner.return_value.plan_training.return_value = {"status": "planned"}
            mock_model.return_value = Path("/tmp/model")
            mock_preprocessor.return_value.preprocess.return_value = {
                "status": "processed"
            }
            mock_evaluator.return_value.evaluate.return_value = {"status": "evaluated"}
            mock_packer.return_value.pack.return_value = {"status": "packed"}
            mock_acceptance.return_value.evaluate_training_run.return_value = Mock(
                passed=True, dict=lambda: {"passed": True}
            )

            # Run pipeline without training
            result = run_pipeline(valid_config, train=False)

            # Should succeed
            assert result == 0

            # Training should NOT be called
            mock_trainer.return_value.train.assert_not_called()

            # Other components should be called
            mock_planner.return_value.plan_training.assert_called_once()
            mock_model.assert_called_once()
            mock_preprocessor.return_value.preprocess.assert_called_once()
            mock_evaluator.return_value.evaluate.assert_called_once()
            mock_packer.return_value.pack.assert_called_once()
            mock_acceptance.return_value.evaluate_training_run.assert_called_once()

    def test_run_pipeline_validation_failure(self, valid_config):
        """Test that pipeline fails when validation fails."""
        # Make config invalid by removing data file
        data_file = Path(valid_config.data.raw_path)
        if data_file.exists():
            data_file.unlink()

        result = run_pipeline(valid_config, train=True)

        # Should fail
        assert result == 1

    def test_run_pipeline_planning_failure(self, valid_config):
        """Test that pipeline fails when planning fails."""
        with patch("humigence.cli.TrainingPlanner") as mock_planner:
            # Mock planning to fail
            mock_planner.side_effect = Exception("Planning failed")

            result = run_pipeline(valid_config, train=True)

            # Should fail
            assert result == 1

    def test_run_pipeline_model_failure(self, valid_config):
        """Test that pipeline fails when model preparation fails."""
        with patch("humigence.cli.TrainingPlanner") as mock_planner, patch(
            "humigence.cli.ensure_model_available"
        ) as mock_model:
            # Mock planning to succeed
            mock_planner.return_value.plan_training.return_value = {"status": "planned"}

            # Mock model preparation to fail
            mock_model.side_effect = Exception("Model download failed")

            result = run_pipeline(valid_config, train=True)

            # Should fail
            assert result == 1

    def test_run_pipeline_preprocessing_failure(self, valid_config):
        """Test that pipeline fails when preprocessing fails."""
        with patch("humigence.cli.TrainingPlanner") as mock_planner, patch(
            "humigence.cli.ensure_model_available"
        ) as mock_model, patch("humigence.cli.DataPreprocessor") as mock_preprocessor:
            # Mock planning and model to succeed
            mock_planner.return_value.plan_training.return_value = {"status": "planned"}
            mock_model.return_value = Path("/tmp/model")

            # Mock preprocessing to fail
            mock_preprocessor.return_value.preprocess.side_effect = Exception(
                "Preprocessing failed"
            )

            result = run_pipeline(valid_config, train=True)

            # Should fail
            assert result == 1


class TestPipelineCLI:
    """Test the pipeline CLI command."""

    def test_pipeline_command_with_training_enabled(
        self, runner, valid_config, tmp_path
    ):
        """Test that pipeline command works with training enabled."""
        # Save config to file
        config_file = tmp_path / "test_config.json"
        with open(config_file, "w") as f:
            json.dump(valid_config.model_dump(), f)

        with patch("humigence.cli.run_pipeline") as mock_pipeline:
            mock_pipeline.return_value = 0

            result = runner.invoke(
                app, ["pipeline", "--config", str(config_file), "--train"]
            )

            # Should succeed
            assert result.exit_code == 0

            # run_pipeline should be called
            mock_pipeline.assert_called_once()
            args, kwargs = mock_pipeline.call_args
            assert args[1] is True  # train is the second positional argument

    def test_pipeline_command_without_training_flag(
        self, runner, valid_config, tmp_path
    ):
        """Test that pipeline command fails without training flag."""
        # Save config to file
        config_file = tmp_path / "test_config.json"
        with open(config_file, "w") as f:
            json.dump(valid_config.model_dump(), f)

        result = runner.invoke(app, ["pipeline", "--config", str(config_file)])

        # Should fail because training is not enabled
        assert result.exit_code == 1
        assert "Training is disabled by default for safety" in result.stdout

    def test_pipeline_command_with_train_env_var(self, runner, valid_config, tmp_path):
        """Test that pipeline command works with TRAIN=1 environment variable."""
        # Save config to file
        config_file = tmp_path / "test_config.json"
        with open(config_file, "w") as f:
            json.dump(valid_config.model_dump(), f)

        with patch("humigence.cli.run_pipeline") as mock_pipeline:
            mock_pipeline.return_value = 0

            with patch.dict(os.environ, {"TRAIN": "1"}):
                result = runner.invoke(app, ["pipeline", "--config", str(config_file)])

            # Should succeed
            assert result.exit_code == 0

            # run_pipeline should be called
            mock_pipeline.assert_called_once()

    def test_pipeline_command_missing_config(self, runner):
        """Test that pipeline command fails with missing config."""
        result = runner.invoke(
            app, ["pipeline", "--config", "nonexistent.json", "--train"]
        )

        # Should fail
        assert result.exit_code == 2
        assert "Configuration file not found" in result.stdout


class TestWizardPipelineIntegration:
    """Test the wizard pipeline integration."""

    def test_wizard_pipeline_automatic_execution(self, runner, valid_config, tmp_path):
        """Test that wizard automatically executes pipeline when training is enabled."""
        # Save config to file
        config_file = tmp_path / "test_config.json"
        with open(config_file, "w") as f:
            json.dump(valid_config.model_dump(), f)

        with patch("humigence.cli.run_wizard") as mock_wizard, patch(
            "humigence.cli.run_pipeline"
        ) as mock_pipeline:
            # Mock wizard to return pipeline action with training enabled
            mock_wizard.return_value = {
                "config_path": config_file,
                "next_action": "pipeline",
                "train": True,
                "exit_code": 0,
            }

            mock_pipeline.return_value = 0

            result = runner.invoke(
                app,
                ["init", "--config", str(config_file), "--run", "pipeline", "--train"],
            )

            # Should succeed
            assert result.exit_code == 0

            # run_pipeline should be called
            mock_pipeline.assert_called_once()
            args, kwargs = mock_pipeline.call_args
            assert args[1] is True  # train is the second positional argument

    def test_wizard_pipeline_training_disabled(self, runner, valid_config, tmp_path):
        """Test that wizard skips training when training is disabled."""
        # Save config to file
        config_file = tmp_path / "test_config.json"
        with open(config_file, "w") as f:
            json.dump(valid_config.model_dump(), f)

        with patch("humigence.cli.run_wizard") as mock_wizard, patch(
            "humigence.cli.run_pipeline"
        ) as mock_pipeline:
            # Mock wizard to return pipeline action with training disabled
            mock_wizard.return_value = {
                "config_path": config_file,
                "next_action": "pipeline",
                "train": False,
                "exit_code": 0,
            }

            mock_pipeline.return_value = 0

            result = runner.invoke(
                app, ["init", "--config", str(config_file), "--run", "pipeline"]
            )

            # Should succeed
            assert result.exit_code == 0

            # run_pipeline should be called with train=False
            mock_pipeline.assert_called_once()
            args, kwargs = mock_pipeline.call_args
            assert args[1] is False  # train is the second positional argument
