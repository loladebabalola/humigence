"""Tests for wizard pipeline functionality."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from humigence.cli import app
from humigence.config import Config


@pytest.fixture
def runner():
    """CLI runner fixture."""
    return CliRunner()


@pytest.fixture
def temp_config(tmp_path):
    """Temporary config file fixture."""
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

    config_file = tmp_path / "test_config.json"
    with open(config_file, "w") as f:
        json.dump(config_data, f)

    return config_file


class TestWizardPipeline:
    """Test wizard pipeline functionality."""

    def test_init_run_plan_does_not_train(self, runner, temp_config):
        """Test that `init --run plan` does not train and exits cleanly."""
        with patch("humigence.cli.run_wizard") as mock_wizard:
            mock_wizard.return_value = {
                "config_path": temp_config,
                "next_action": "plan",
                "train": False,
                "exit_code": 0,
            }

            with patch("humigence.cli.TrainingPlanner") as mock_planner:
                mock_plan = Mock()
                mock_planner.return_value.plan_training.return_value = mock_plan

                result = runner.invoke(
                    app, ["init", "--config", str(temp_config), "--run", "plan"]
                )

                # Should not crash
                assert result.exit_code == 0

                # TrainingPlanner should be called once
                mock_planner.return_value.plan_training.assert_called_once()

    def test_init_run_pipeline_with_train(self, runner, temp_config):
        """Test that `init --run pipeline --train` executes the full chain without raising."""
        with patch("humigence.cli.run_wizard") as mock_wizard:
            mock_wizard.return_value = {
                "config_path": temp_config,
                "next_action": "pipeline",
                "train": True,
                "exit_code": 0,
            }

            with patch("humigence.cli.run_pipeline") as mock_pipeline:
                mock_pipeline.return_value = 0

                result = runner.invoke(
                    app,
                    [
                        "init",
                        "--config",
                        str(temp_config),
                        "--run",
                        "pipeline",
                        "--train",
                    ],
                )

                # Should not crash
                assert result.exit_code == 0

                # run_pipeline should be called once with train=True
                mock_pipeline.assert_called_once()
                args, kwargs = mock_pipeline.call_args
                assert args[1] is True  # train is the second positional argument

    def test_init_run_pipeline_without_train(self, runner, temp_config):
        """Test that `init --run pipeline` (no --train) skips training."""
        with patch("humigence.cli.run_wizard") as mock_wizard:
            mock_wizard.return_value = {
                "config_path": temp_config,
                "next_action": "pipeline",
                "train": False,
                "exit_code": 0,
            }

            with patch("humigence.cli.run_pipeline") as mock_pipeline:
                mock_pipeline.return_value = 0

                result = runner.invoke(
                    app, ["init", "--config", str(temp_config), "--run", "pipeline"]
                )

                # Should not crash
                assert result.exit_code == 0

                # run_pipeline should be called once with train=False
                mock_pipeline.assert_called_once()
                args, kwargs = mock_pipeline.call_args
                assert args[1] is False  # train is the second positional argument

    def test_wizard_cancelled_returns_none_action(self, runner, temp_config):
        """Test that cancelled wizard returns None action."""
        with patch("humigence.cli.run_wizard") as mock_wizard:
            mock_wizard.return_value = {
                "config_path": temp_config,
                "next_action": None,
                "train": False,
                "exit_code": 0,
            }

            result = runner.invoke(
                app, ["init", "--config", str(temp_config), "--run", "plan"]
            )

            # Should exit cleanly
            assert result.exit_code == 0
            assert "Wizard completed without selecting an action" in result.stdout

    def test_wizard_failed_returns_error_code(self, runner, temp_config):
        """Test that failed wizard returns error code."""
        with patch("humigence.cli.run_wizard") as mock_wizard:
            mock_wizard.return_value = {
                "config_path": temp_config,
                "next_action": None,
                "train": False,
                "exit_code": 2,
            }

            result = runner.invoke(
                app, ["init", "--config", str(temp_config), "--run", "plan"]
            )

            # Should exit with error code
            assert result.exit_code == 2


class TestModelAvailability:
    """Test model availability functionality."""

    def test_ensure_model_available_downloads_if_missing(self, temp_config):
        """Test that ensure_model_available downloads model if not found."""
        from humigence.model_utils import ensure_model_available

        config = Config.from_file(temp_config)

        with patch("humigence.model_utils.snapshot_download") as mock_download:
            mock_download.return_value = "/tmp/cache/model"

            # Mock that model path doesn't exist
            with patch.object(Path, "exists", return_value=False):
                result = ensure_model_available(config)

                # Should call snapshot_download
                mock_download.assert_called_once()

                # Should return the downloaded path
                assert str(result) == "/tmp/cache/model"

                # Should update config with local path
                assert config.model.local_path == "/tmp/cache/model"

    def test_ensure_model_available_uses_existing_if_found(self, temp_config):
        """Test that ensure_model_available uses existing model if found."""
        from humigence.model_utils import ensure_model_available

        config = Config.from_file(temp_config)

        # Mock that model path exists
        with patch.object(Path, "exists", return_value=True):
            result = ensure_model_available(config)

            # Should return the existing path
            assert result == config.get_model_path()


class TestPipelineExecution:
    """Test pipeline execution functionality."""

    def test_run_pipeline_executes_all_steps(self, temp_config):
        """Test that run_pipeline executes all required steps."""
        from humigence.cli import run_pipeline

        config = Config.from_file(temp_config)

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
            # Mock all the components
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
            result = run_pipeline(config, train=True)

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

    def test_run_pipeline_skips_training_when_disabled(self, temp_config):
        """Test that run_pipeline skips training when train=False."""
        from humigence.cli import run_pipeline

        config = Config.from_file(temp_config)

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
            # Mock all the components
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
            result = run_pipeline(config, train=False)

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
