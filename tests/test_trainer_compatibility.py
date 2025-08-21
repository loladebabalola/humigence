"""Test TrainingArguments compatibility shim."""

import inspect
from unittest.mock import Mock, patch

from transformers import TrainingArguments

from humigence.config import Config
from humigence.train import QLoRATrainer


class TestTrainerCompatibility:
    """Test that the trainer compatibility shim works correctly."""

    def test_build_training_args_compatibility(self, tmp_path):
        """Test that _build_training_args creates compatible TrainingArguments."""
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
        ) as mock_collator, patch(
            "humigence.train.QLoRATrainer._auto_fit_vram"
        ) as mock_vram:
            mock_logger.return_value = Mock()
            mock_build.return_value = (Mock(), Mock(), Mock())
            mock_tokenizer.return_value = Mock()
            mock_dataset.return_value = Mock()
            mock_collator.return_value = Mock()
            mock_vram.return_value = (4, 8)  # micro_batch_size, grad_accum

            # Create trainer
            trainer = QLoRATrainer(config)

            # Verify runs_dir is properly set
            assert hasattr(trainer, "runs_dir")
            assert trainer.runs_dir == tmp_path / "runs" / "test_project"

            # Test the compatibility shim
            training_args = trainer._build_training_args(100000)

            # Verify it's a TrainingArguments instance
            assert isinstance(training_args, TrainingArguments)

            # Verify key arguments are set correctly
            assert training_args.output_dir == str(trainer.runs_dir)
            assert training_args.do_train is True
            assert training_args.do_eval is True
            assert training_args.learning_rate == 0.0002
            assert training_args.weight_decay == 0.0
            assert training_args.warmup_ratio == 0.03

            # Verify the args only contain valid parameters for this Transformers version
            sig = inspect.signature(TrainingArguments.__init__)
            allowed_params = set(sig.parameters.keys())

            # Get the actual args that were passed
            actual_args = training_args.__dict__

            # All args should be valid
            for key in actual_args:
                if key.startswith("_"):
                    continue  # Skip private attributes
                # The key should be in the allowed parameters
                assert (
                    key in allowed_params
                ), f"Parameter {key} not allowed in TrainingArguments"

    def test_training_args_signature_inspection(self):
        """Test that we can inspect TrainingArguments signature correctly."""
        from transformers import TrainingArguments

        # This should not raise any errors
        sig = inspect.signature(TrainingArguments.__init__)
        allowed = set(sig.parameters.keys())

        # Should have some common parameters
        assert "output_dir" in allowed
        assert "do_train" in allowed
        assert "do_eval" in allowed

        # Log which strategy parameters are available
        strategy_params = []
        for param in [
            "eval_strategy",
            "evaluation_strategy",
            "save_strategy",
            "logging_strategy",
        ]:
            if param in allowed:
                strategy_params.append(param)

        print(f"Available strategy parameters: {strategy_params}")
        print(f"Total parameters: {len(allowed)}")

        # Should have at least some parameters
        assert len(allowed) > 10
