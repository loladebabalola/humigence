"""Tests for acceptance config legacy key compatibility."""

from humigence.config import Config


def test_acceptance_config_legacy_keys_load_correctly():
    """Test that configs with legacy acceptance keys load successfully."""
    # Create config with legacy keys
    config_data = {
        "project": "test_project",
        "model": {
            "repo": "Qwen/Qwen2.5-0.5B",
            "local_path": None,
            "use_flash_attn": True,
        },
        "compute": {"gpus": 1, "gpu_type": "RTX_4080_16GB"},
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
            "early_stopping": {"metric": "val_loss", "patience": 3, "min_delta": 0.002},
        },
        "eval": {"curated_prompts_path": "configs/curated_eval_prompts.jsonl"},
        "acceptance": {
            "min_val_loss_improvement": 1.2,
            "jitter_threshold": 22.0,
            "curated_threshold": 72.0,
        },
        "export": {"formats": ["peft_adapter"], "artifacts_dir": "artifacts/humigence"},
    }

    # Load config - this should not crash
    config = Config(**config_data)

    # Verify that the new attributes are accessible with correct values
    assert config.acceptance.min_val_improvement_pct == 1.2
    assert config.acceptance.throughput_jitter_pct == 22.0
    assert config.acceptance.curated_reasonable_threshold_pct == 72.0


def test_acceptance_config_new_keys_load_correctly():
    """Test that configs with new acceptance keys load successfully."""
    # Create config with new keys
    config_data = {
        "project": "test_project",
        "model": {
            "repo": "Qwen/Qwen2.5-0.5B",
            "local_path": None,
            "use_flash_attn": True,
        },
        "compute": {"gpus": 1, "gpu_type": "RTX_4080_16GB"},
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
            "early_stopping": {"metric": "val_loss", "patience": 3, "min_delta": 0.002},
        },
        "eval": {"curated_prompts_path": "configs/curated_eval_prompts.jsonl"},
        "acceptance": {
            "min_val_improvement_pct": 1.5,
            "throughput_jitter_pct": 25.0,
            "curated_reasonable_threshold_pct": 75.0,
        },
        "export": {"formats": ["peft_adapter"], "artifacts_dir": "artifacts/humigence"},
    }

    # Load config - this should not crash
    config = Config(**config_data)

    # Verify that the new attributes are accessible with correct values
    assert config.acceptance.min_val_improvement_pct == 1.5
    assert config.acceptance.throughput_jitter_pct == 25.0
    assert config.acceptance.curated_reasonable_threshold_pct == 75.0


def test_acceptance_config_mixed_keys_load_correctly():
    """Test that configs with mixed legacy and new keys load successfully."""
    # Create config with mixed keys (some legacy, some new)
    config_data = {
        "project": "test_project",
        "model": {
            "repo": "Qwen/Qwen2.5-0.5B",
            "local_path": None,
            "use_flash_attn": True,
        },
        "compute": {"gpus": 1, "gpu_type": "RTX_4080_16GB"},
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
            "early_stopping": {"metric": "val_loss", "patience": 3, "min_delta": 0.002},
        },
        "eval": {"curated_prompts_path": "configs/curated_eval_prompts.jsonl"},
        "acceptance": {
            "min_val_loss_improvement": 1.8,  # Legacy key
            "throughput_jitter_pct": 30.0,  # New key
            "curated_threshold": 80.0,  # Legacy key
        },
        "export": {"formats": ["peft_adapter"], "artifacts_dir": "artifacts/humigence"},
    }

    # Load config - this should not crash
    config = Config(**config_data)

    # Verify that the new attributes are accessible with correct values
    assert config.acceptance.min_val_improvement_pct == 1.8
    assert config.acceptance.throughput_jitter_pct == 30.0
    assert config.acceptance.curated_reasonable_threshold_pct == 80.0


def test_acceptance_config_defaults_work():
    """Test that acceptance config defaults work correctly."""
    # Create config without acceptance section (should use defaults)
    config_data = {
        "project": "test_project",
        "model": {
            "repo": "Qwen/Qwen2.5-0.5B",
            "local_path": None,
            "use_flash_attn": True,
        },
        "compute": {"gpus": 1, "gpu_type": "RTX_4080_16GB"},
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
            "early_stopping": {"metric": "val_loss", "patience": 3, "min_delta": 0.002},
        },
        "eval": {"curated_prompts_path": "configs/curated_eval_prompts.jsonl"},
        "export": {"formats": ["peft_adapter"], "artifacts_dir": "artifacts/humigence"},
    }

    # Load config - this should not crash and should use defaults
    config = Config(**config_data)

    # Verify that the default values are used
    assert config.acceptance.min_val_improvement_pct == 1.0
    assert config.acceptance.throughput_jitter_pct == 20.0
    assert config.acceptance.curated_reasonable_threshold_pct == 70.0


def test_acceptance_config_serialization_uses_new_keys():
    """Test that config serialization uses the new key names."""
    # Create config with legacy keys
    config_data = {
        "project": "test_project",
        "model": {
            "repo": "Qwen/Qwen2.5-0.5B",
            "local_path": None,
            "use_flash_attn": True,
        },
        "compute": {"gpus": 1, "gpu_type": "RTX_4080_16GB"},
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
            "early_stopping": {"metric": "val_loss", "patience": 3, "min_delta": 0.002},
        },
        "eval": {"curated_prompts_path": "configs/curated_eval_prompts.jsonl"},
        "acceptance": {
            "min_val_loss_improvement": 1.2,
            "jitter_threshold": 22.0,
            "curated_threshold": 72.0,
        },
        "export": {"formats": ["peft_adapter"], "artifacts_dir": "artifacts/humigence"},
    }

    # Load config
    config = Config(**config_data)

    # Serialize to dict
    serialized = config.model_dump()

    # Verify that the serialized version uses new keys
    assert "min_val_improvement_pct" in serialized["acceptance"]
    assert "throughput_jitter_pct" in serialized["acceptance"]
    assert "curated_reasonable_threshold_pct" in serialized["acceptance"]

    # Verify that legacy keys are NOT in the serialized version
    assert "min_val_loss_improvement" not in serialized["acceptance"]
    assert "jitter_threshold" not in serialized["acceptance"]
    assert "curated_threshold" not in serialized["acceptance"]

    # Verify the values are correct
    assert serialized["acceptance"]["min_val_improvement_pct"] == 1.2
    assert serialized["acceptance"]["throughput_jitter_pct"] == 22.0
    assert serialized["acceptance"]["curated_reasonable_threshold_pct"] == 72.0
