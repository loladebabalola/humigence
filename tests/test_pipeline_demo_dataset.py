"""Test pipeline execution with bundled demo dataset."""

from pathlib import Path
from unittest.mock import Mock, patch

from humigence.cli import run_pipeline


class TestPipelineDemoDataset:
    """Test pipeline execution with demo dataset."""

    def test_pipeline_with_bundled_dataset_no_training(self, tmp_path):
        """Test that pipeline runs through plan → preprocess using bundled dataset with training disabled."""
        # Create a temporary config path
        config_path = tmp_path / "test_config.json"

        # Create a minimal config file
        config_data = {
            "project": "test_project",
            "model": {
                "repo": "Qwen/Qwen2.5-0.5B",
                "local_path": None,
                "use_flash_attn": True,
            },
            "compute": {"gpus": 1, "gpu_type": "RTX_4080_16GB"},
            "data": {
                "raw_path": "data/raw/oa.jsonl",
                "processed_dir": "data/processed",
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
            "export": {"formats": ["peft_adapter"], "artifacts_dir": "artifacts/test"},
        }

        # Write config to file
        with open(config_path, "w") as f:
            import json

            json.dump(config_data, f)

        # Create the bundled dataset file
        bundled_dataset = tmp_path / "data" / "raw" / "oa.jsonl"
        bundled_dataset.parent.mkdir(parents=True, exist_ok=True)
        bundled_dataset.write_text('{"messages":[{"role":"user","content":"test"}]}')

        # Mock all the heavy components
        with patch("humigence.cli.ensure_model_available") as mock_model, patch(
            "humigence.cli.DataPreprocessor"
        ) as mock_preprocessor, patch(
            "humigence.cli.ModelEvaluator"
        ) as mock_evaluator, patch(
            "humigence.cli.ModelPacker"
        ) as mock_packer, patch(
            "humigence.cli.AcceptanceGates"
        ) as mock_acceptance:
            # Set up mock returns
            mock_model.return_value = Path("/tmp/model")
            mock_preprocessor.return_value.preprocess.return_value = {
                "status": "processed"
            }
            mock_evaluator.return_value.evaluate.return_value = {"status": "evaluated"}
            mock_packer.return_value.pack.return_value = {"status": "packed"}
            mock_acceptance.return_value.evaluate_training_run.return_value = Mock(
                passed=True, dict=lambda: {"passed": True}
            )

            # Run pipeline with training disabled
            result = run_pipeline(config_path, action="pipeline", allow_train=False)

            # Should succeed
            assert result == 0

            # Verify all components were called except training
            mock_model.assert_called_once()
            mock_preprocessor.return_value.preprocess.assert_called_once()
            mock_evaluator.return_value.evaluate.assert_called_once()
            mock_packer.return_value.pack.assert_called_once()
            mock_acceptance.return_value.evaluate_training_run.assert_called_once()

            # Verify the bundled dataset was used
            assert bundled_dataset.exists()

            # Verify processed data directory was created
            processed_dir = tmp_path / "data" / "processed"
            assert processed_dir.exists()

    def test_pipeline_with_bundled_dataset_training_enabled(self, tmp_path):
        """Test that pipeline runs through all steps including training when enabled."""
        # Create a temporary config path
        config_path = tmp_path / "test_config.json"

        # Create a minimal config file
        config_data = {
            "project": "test_project",
            "model": {
                "repo": "Qwen/Qwen2.5-0.5B",
                "local_path": None,
                "use_flash_attn": True,
            },
            "compute": {"gpus": 1, "gpu_type": "RTX_4080_16GB"},
            "data": {
                "raw_path": "data/raw/oa.jsonl",
                "processed_dir": "data/processed",
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
            "export": {"formats": ["peft_adapter"], "artifacts_dir": "artifacts/test"},
        }

        # Write config to file
        with open(config_path, "w") as f:
            import json

            json.dump(config_data, f)

        # Create the bundled dataset file
        bundled_dataset = tmp_path / "data" / "raw" / "oa.jsonl"
        bundled_dataset.parent.mkdir(parents=True, exist_ok=True)
        bundled_dataset.write_text('{"messages":[{"role":"user","content":"test"}]}')

        # Mock all the heavy components including training
        with patch("humigence.cli.ensure_model_available") as mock_model, patch(
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
            # Set up mock returns
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
            result = run_pipeline(config_path, action="pipeline", allow_train=True)

            # Should succeed
            assert result == 0

            # Verify all components were called including training
            mock_model.assert_called_once()
            mock_preprocessor.return_value.preprocess.assert_called_once()
            mock_trainer.return_value.train.assert_called_once()
            mock_evaluator.return_value.evaluate.assert_called_once()
            mock_packer.return_value.pack.assert_called_once()
            mock_acceptance.return_value.evaluate_training_run.assert_called_once()

            # Verify directories were created
            runs_dir = tmp_path / "runs" / "test_project"
            artifacts_dir = tmp_path / "artifacts" / "test"
            assert runs_dir.exists()
            assert artifacts_dir.exists()
