"""Test that QLoRATrainer initializes self.runs_dir properly."""

from pathlib import Path
from unittest.mock import Mock, patch

from humigence.config import Config
from humigence.train import QLoRATrainer


class TestTrainerRunsDir:
    """Test that trainer properly initializes runs_dir."""

    def test_trainer_initializes_runs_dir(self, tmp_path):
        """Test that trainer creates runs_dir in __init__."""
        # Create a minimal config
        config_data = {
            "project": "test_project",
            "model": {"repo": "Qwen/Qwen2.5-0.5B"},
            "data": {
                "raw_path": "data/raw/test.jsonl",
                "processed_dir": "data/processed",
                "data_schema": "chat_messages",
                "max_seq_len": 1024,
                "packing": True,
                "split": {"train": 0.8, "val": 0.1, "test": 0.1},
                "template": "qwen_chat_basic_v1",
            },
            "train": {
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
            "compute": {"gpus": 1, "gpu_type": "RTX_4080_16GB"},
            "eval": {"curated_prompts_path": "configs/curated_eval_prompts.jsonl"},
            "acceptance": {
                "min_val_improvement_pct": 1.0,
                "throughput_jitter_pct": 20.0,
                "curated_reasonable_threshold_pct": 70.0,
            },
            "export": {"formats": ["peft_adapter"], "artifacts_dir": "artifacts/test"},
        }

        config = Config(**config_data)

        # Mock the heavy dependencies
        with patch("humigence.train.create_run_logger") as mock_logger, patch(
            "humigence.train.build_model_and_peft"
        ) as mock_build, patch(
            "humigence.train.AutoTokenizer.from_pretrained"
        ) as mock_tokenizer, patch(
            "humigence.train.Dataset.from_list"
        ) as mock_dataset, patch(
            "humigence.train.DataCollatorForLanguageModeling"
        ) as mock_collator:
            mock_logger.return_value = Mock()
            mock_build.return_value = (Mock(), Mock(), Mock())
            mock_tokenizer.return_value = Mock()
            mock_dataset.return_value = Mock()
            mock_collator.return_value = Mock()

            # Create trainer - this should not raise AttributeError
            trainer = QLoRATrainer(config)

            # Verify runs_dir is properly set
            assert hasattr(trainer, "runs_dir")
            assert trainer.runs_dir == Path("runs/test_project").resolve()

            # Verify the directory was created
            assert trainer.runs_dir.exists()

            # Verify project attribute is set
            assert trainer.project == "test_project"

            # Verify runs_root is set
            assert hasattr(trainer, "runs_root")
            assert trainer.runs_root == Path("runs")
