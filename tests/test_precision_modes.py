"""
Test precision modes for Humigence.
Tests all precision modes without loading actual models.
"""

from unittest.mock import Mock, patch

import pytest
import torch

from humigence.precision import build_model_and_peft


class TestPrecisionModes:
    """Test all precision modes initialize correctly."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        return {
            "model": {
                "repo": "Qwen/Qwen2.5-0.5B",
                "local_path": "~/.cache/hf/qwen2.5-0.5b",
                "use_flash_attn": True,
            },
            "train": {
                "precision_mode": "qlora_nf4",
                "lora": {
                    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                    "r": 16,
                    "alpha": 32,
                    "dropout": 0.05,
                },
            },
            "_tokenizer": Mock(),
        }

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = Mock()
        model.named_parameters.return_value = [
            ("lora_A.weight", Mock(numel=Mock(return_value=1000))),
            ("lora_B.weight", Mock(numel=Mock(return_value=1000))),
            ("base.weight", Mock(numel=Mock(return_value=1000000))),
        ]
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer for testing."""
        tokenizer = Mock()
        tokenizer.pad_token = None
        tokenizer.eos_token = "<|endoftext|>"
        return tokenizer

    @patch("humigence.precision.AutoModelForCausalLM.from_pretrained")
    @patch("humigence.precision.AutoTokenizer.from_pretrained")
    @patch("humigence.precision.prepare_model_for_kbit_training")
    @patch("humigence.precision.get_peft_model")
    def test_qlora_nf4_mode(
        self,
        mock_get_peft,
        mock_prepare,
        mock_model_class,
        mock_tokenizer_class,
        mock_config,
        mock_model,
        mock_tokenizer,
    ):
        """Test qlora_nf4 precision mode."""
        mock_config["train"]["precision_mode"] = "qlora_nf4"
        mock_config["_tokenizer"] = mock_tokenizer

        mock_model_class.return_value = mock_model
        mock_prepare.return_value = mock_model
        mock_get_peft.return_value = mock_model

        # Should not raise any exceptions
        model, tokenizer, peft_config = build_model_and_peft(mock_config)

        assert model is not None
        assert tokenizer is not None
        assert peft_config is not None

    @patch("humigence.precision.AutoModelForCausalLM.from_pretrained")
    @patch("humigence.precision.AutoTokenizer.from_pretrained")
    @patch("humigence.precision.get_peft_model")
    def test_lora_fp16_mode(
        self,
        mock_get_peft,
        mock_model_class,
        mock_tokenizer_class,
        mock_config,
        mock_model,
        mock_tokenizer,
    ):
        """Test lora_fp16 precision mode."""
        mock_config["train"]["precision_mode"] = "lora_fp16"
        mock_config["_tokenizer"] = mock_tokenizer

        mock_model_class.return_value = mock_model
        mock_get_peft.return_value = mock_model

        # Should not raise any exceptions
        model, tokenizer, peft_config = build_model_and_peft(mock_config)

        assert model is not None
        assert tokenizer is not None
        assert peft_config is not None

    @patch("humigence.precision.AutoModelForCausalLM.from_pretrained")
    @patch("humigence.precision.AutoTokenizer.from_pretrained")
    @patch("humigence.precision.get_peft_model")
    def test_lora_bf16_mode(
        self,
        mock_get_peft,
        mock_model_class,
        mock_tokenizer_class,
        mock_config,
        mock_model,
        mock_tokenizer,
    ):
        """Test lora_bf16 precision mode."""
        mock_config["train"]["precision_mode"] = "lora_bf16"
        mock_config["_tokenizer"] = mock_tokenizer

        mock_model_class.return_value = mock_model
        mock_get_peft.return_value = mock_model

        # Mock CUDA BF16 support
        with patch("torch.cuda.is_bf16_supported", return_value=True):
            # Should not raise any exceptions
            model, tokenizer, peft_config = build_model_and_peft(mock_config)

            assert model is not None
            assert tokenizer is not None
            assert peft_config is not None

    @patch("humigence.precision.AutoModelForCausalLM.from_pretrained")
    @patch("humigence.precision.AutoTokenizer.from_pretrained")
    @patch("humigence.precision.get_peft_model")
    def test_lora_int8_mode(
        self,
        mock_get_peft,
        mock_model_class,
        mock_tokenizer_class,
        mock_config,
        mock_model,
        mock_tokenizer,
    ):
        """Test lora_int8 precision mode."""
        mock_config["train"]["precision_mode"] = "lora_int8"
        mock_config["_tokenizer"] = mock_tokenizer

        mock_model_class.return_value = mock_model
        mock_get_peft.return_value = mock_model

        # Should not raise any exceptions
        model, tokenizer, peft_config = build_model_and_peft(mock_config)

        assert model is not None
        assert tokenizer is not None
        assert peft_config is not None

    def test_invalid_precision_mode(self, mock_config):
        """Test that invalid precision mode raises ValueError."""
        mock_config["train"]["precision_mode"] = "invalid_mode"

        with pytest.raises(ValueError, match="Unsupported precision_mode"):
            build_model_and_peft(mock_config)

    @patch("humigence.precision.AutoModelForCausalLM.from_pretrained")
    @patch("humigence.precision.AutoTokenizer.from_pretrained")
    @patch("humigence.precision.get_peft_model")
    def test_bf16_not_supported(
        self,
        mock_get_peft,
        mock_model_class,
        mock_tokenizer_class,
        mock_config,
        mock_model,
        mock_tokenizer,
    ):
        """Test that BF16 mode fails gracefully when not supported."""
        mock_config["train"]["precision_mode"] = "lora_bf16"
        mock_config["_tokenizer"] = mock_tokenizer

        mock_model_class.return_value = mock_model
        mock_get_peft.return_value = mock_model

        # Mock CUDA BF16 not supported
        with patch("torch.cuda.is_bf16_supported", return_value=False):
            with pytest.raises(ValueError, match="BF16 not supported"):
                build_model_and_peft(mock_config)

    def test_precision_banner_function(self):
        """Test the precision banner function."""
        from humigence.precision import print_precision_banner

        # Should not raise any exceptions
        print_precision_banner(
            precision_mode="qlora_nf4",
            dtype=torch.float16,
            quantization="4-bit NF4",
            target_modules=["q_proj", "k_proj"],
        )
