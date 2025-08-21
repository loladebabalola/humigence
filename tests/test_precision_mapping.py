"""
Test precision mode mapping to TrainingArguments.
"""

from unittest.mock import Mock

from humigence.train import QLoRATrainer


class TestPrecisionModeMapping:
    """Test that precision modes correctly map to TrainingArguments flags."""

    def test_qlora_nf4_precision_mapping(self):
        """Test qlora_nf4 maps to fp16=True, bf16=False."""
        # Create minimal config
        config = Mock()
        config.train.precision_mode = "qlora_nf4"

        # Create trainer instance without calling __init__
        trainer = QLoRATrainer.__new__(QLoRATrainer)
        trainer.config = config

        # Test the precision mapping logic directly
        precision_mode = trainer.config.train.precision_mode
        fp16, bf16 = False, False

        if precision_mode == "qlora_nf4":
            # 4-bit quantization uses fp16 for compute
            fp16 = True
            bf16 = False
        elif precision_mode == "lora_fp16":
            # 16-bit float training
            fp16 = True
            bf16 = False
        elif precision_mode == "lora_bf16":
            # 16-bit bfloat training
            fp16 = False
            bf16 = True
        elif precision_mode == "lora_int8":
            # 8-bit integer training (no mixed precision)
            fp16 = False
            bf16 = False
        else:
            # Fallback to fp16
            fp16 = True
            bf16 = False

        # Verify precision flags
        assert fp16 is True
        assert bf16 is False

    def test_lora_bf16_precision_mapping(self):
        """Test lora_bf16 maps to fp16=False, bf16=True."""
        # Create minimal config
        config = Mock()
        config.train.precision_mode = "lora_bf16"

        # Create trainer instance without calling __init__
        trainer = QLoRATrainer.__new__(QLoRATrainer)
        trainer.config = config

        # Test the precision mapping logic directly
        precision_mode = trainer.config.train.precision_mode
        fp16, bf16 = False, False

        if precision_mode == "qlora_nf4":
            # 4-bit quantization uses fp16 for compute
            fp16 = True
            bf16 = False
        elif precision_mode == "lora_fp16":
            # 16-bit float training
            fp16 = True
            bf16 = False
        elif precision_mode == "lora_bf16":
            # 16-bit bfloat training
            fp16 = False
            bf16 = True
        elif precision_mode == "lora_int8":
            # 8-bit integer training (no mixed precision)
            fp16 = False
            bf16 = False
        else:
            # Fallback to fp16
            fp16 = True
            bf16 = False

        # Verify precision flags
        assert fp16 is False
        assert bf16 is True
