"""Test preprocessing functionality."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from humigence.preprocess import DataPreprocessor
from humigence.utils_data import DataProcessor


class TestDataPreprocessor:
    """Test data preprocessing functionality."""

    def test_load_config(self):
        """Test configuration loading."""
        config_path = Path("configs/humigence.basic.json")
        assert config_path.exists(), "Config file should exist"

        with open(config_path) as f:
            config = json.load(f)

        assert "data" in config
        assert "raw_path" in config["data"]
        assert "processed_dir" in config["data"]
        assert "schema" in config["data"]

    def test_data_schema_validation(self):
        """Test that data schema is valid."""
        config_path = Path("configs/humigence.basic.json")
        with open(config_path) as f:
            config = json.load(f)

        schema = config["data"]["schema"]
        valid_schemas = ["chat_messages", "instruction_output"]
        assert schema in valid_schemas, f"Invalid schema: {schema}"

    def test_max_seq_len_validation(self):
        """Test that max_seq_len is reasonable."""
        config_path = Path("configs/humigence.basic.json")
        with open(config_path) as f:
            config = json.load(f)

        max_seq_len = config["data"]["max_seq_len"]
        assert max_seq_len > 0, "max_seq_len should be positive"
        assert max_seq_len <= 8192, "max_seq_len should be reasonable for RTX 4080"

    def test_split_ratios(self):
        """Test that train/val/test split ratios are valid."""
        config_path = Path("configs/humigence.basic.json")
        with open(config_path) as f:
            config = json.load(f)

        split = config["data"]["split"]
        train_ratio = split["train"]
        val_ratio = split["val"]
        test_ratio = split["test"]

        # Check ratios are positive
        assert train_ratio > 0
        assert val_ratio > 0
        assert test_ratio > 0

        # Check ratios sum to approximately 1.0
        total_ratio = train_ratio + val_ratio + test_ratio
        assert (
            abs(total_ratio - 1.0) < 0.01
        ), f"Split ratios should sum to 1.0, got {total_ratio}"

        # Check train is largest
        assert train_ratio > val_ratio
        assert train_ratio > test_ratio


class TestDataProcessor:
    """Test data processing utilities."""

    def test_estimate_token_length(self):
        """Test token length estimation."""
        mock_tokenizer = MagicMock()
        processor = DataProcessor(mock_tokenizer)

        # Test short text
        short_text = "Hello world"
        estimated_length = processor.estimate_token_length(short_text)
        assert estimated_length > 0
        assert estimated_length <= len(short_text)

        # Test longer text
        long_text = "This is a much longer piece of text that should give us a better estimate of token length based on the heuristic of approximately 4 characters per token for English text."
        estimated_length = processor.estimate_token_length(long_text)
        assert estimated_length > 0
        assert estimated_length <= len(long_text)

    def test_chat_messages_cleaning(self):
        """Test chat messages cleaning."""
        mock_tokenizer = MagicMock()
        processor = DataProcessor(mock_tokenizer)

        # Test valid chat messages
        valid_chat = {
            "messages": [
                {"role": "user", "content": "What is machine learning?"},
                {
                    "role": "assistant",
                    "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
                },
            ]
        }

        cleaned = processor._clean_chat_messages(valid_chat)
        assert cleaned is not None
        assert "messages" in cleaned
        assert len(cleaned["messages"]) == 2

        # Test invalid chat (too short)
        invalid_chat = {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ]
        }

        cleaned = processor._clean_chat_messages(invalid_chat)
        assert cleaned is None  # Should be filtered out

    def test_instruction_output_cleaning(self):
        """Test instruction-output cleaning."""
        mock_tokenizer = MagicMock()
        processor = DataProcessor(mock_tokenizer)

        # Test valid instruction-output
        valid_io = {
            "instruction": "Explain the concept of overfitting in machine learning.",
            "output": "Overfitting occurs when a machine learning model learns the training data too well, including noise and irrelevant patterns, leading to poor generalization on unseen data.",
        }

        cleaned = processor._clean_instruction_output(valid_io)
        assert cleaned is not None
        assert "instruction" in cleaned
        assert "output" in cleaned

        # Test invalid instruction-output (too short)
        invalid_io = {"instruction": "Hi", "output": "Hello"}

        cleaned = processor._clean_instruction_output(invalid_io)
        assert cleaned is None  # Should be filtered out

    def test_duplicate_removal(self):
        """Test duplicate removal functionality."""
        mock_tokenizer = MagicMock()
        processor = DataProcessor(mock_tokenizer)

        # Create test data with duplicates
        test_data = [
            {
                "messages": [
                    {"role": "user", "content": "A"},
                    {"role": "assistant", "content": "B"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "A"},
                    {"role": "assistant", "content": "B"},
                ]
            },  # Duplicate
            {
                "messages": [
                    {"role": "user", "content": "C"},
                    {"role": "assistant", "content": "D"},
                ]
            },
        ]

        deduplicated = processor.remove_duplicates(test_data, "chat_messages")
        assert len(deduplicated) == 2  # Should remove one duplicate

        # Check that unique items remain
        unique_contents = set()
        for item in deduplicated:
            content = processor._extract_chat_text(item)
            unique_contents.add(content)

        assert len(unique_contents) == 2

    def test_length_filtering(self):
        """Test length filtering functionality."""
        mock_tokenizer = MagicMock()
        processor = DataProcessor(mock_tokenizer)

        # Create test data with varying lengths
        test_data = [
            {
                "messages": [
                    {"role": "user", "content": "Short"},
                    {"role": "assistant", "content": "Response"},
                ]
            },
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Medium length question that should pass the filter",
                    },
                    {
                        "role": "assistant",
                        "content": "Medium length response that should also pass the filter",
                    },
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Very long question " * 100},
                    {"role": "assistant", "content": "Very long response " * 100},
                ]
            },  # Too long
        ]

        # Filter with reasonable max length
        filtered = processor.filter_by_length(
            test_data, max_tokens=100, schema="chat_messages"
        )

        # Should keep short and medium, filter out very long
        assert len(filtered) == 2

        # Check that filtered items are within length limit
        for item in filtered:
            text = processor._extract_chat_text(item)
            estimated_length = processor.estimate_token_length(text)
            assert estimated_length <= 100


class TestPreprocessingIntegration:
    """Test preprocessing integration."""

    @patch("humigence.preprocess.DataProcessor")
    @patch("humigence.preprocess.AutoTokenizer")
    def test_preprocessor_initialization(self, mock_tokenizer, mock_data_processor):
        """Test preprocessor initialization."""
        mock_processor = MagicMock()
        mock_data_processor.return_value = mock_processor

        # Mock the tokenizer
        mock_tok = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tok

        from humigence.config import Config

        config = Config(
            project="test",
            seed=42,
            model={"repo": "test/model", "local_path": None},
            data={
                "raw_path": "test_data.jsonl",
                "processed_dir": "test_processed",
                "schema": "chat_messages",
                "max_seq_len": 512,
                "packing": True,
            },
            train={
                "precision_mode": "qlora_nf4",
                "lora": {
                    "target_modules": ["q_proj", "v_proj"],
                    "r": 16,
                    "alpha": 32,
                    "dropout": 0.1,
                },
            },
        )

        preprocessor = DataPreprocessor(config)
        assert preprocessor.config == config
        assert preprocessor.data_processor is not None

    def test_config_validation(self):
        """Test that config validation works."""
        config_path = Path("configs/humigence.basic.json")
        assert config_path.exists(), "Config file should exist"

        # Should be able to load and validate config
        with open(config_path) as f:
            config = json.load(f)

        # Check required fields exist
        required_fields = ["data", "train", "model"]
        for field in required_fields:
            assert field in config, f"Missing required field: {field}"

        # Check data section
        data_section = config["data"]
        assert "raw_path" in data_section
        assert "processed_dir" in data_section
        assert "schema" in data_section
        assert "max_seq_len" in data_section
        assert "packing" in data_section


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
