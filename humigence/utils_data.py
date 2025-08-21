"""
Data utilities for Humigence.
Handles data loading, validation, and processing.
"""

import json
import logging
from pathlib import Path

import numpy as np
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class DataProcessor:
    """Handles data processing and validation."""

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def load_jsonl(self, file_path: str | Path) -> list[dict]:
        """Load data from a JSONL file."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        data = []
        with open(file_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    line = line.strip()
                    if line:
                        item = json.loads(line)
                        data.append(item)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error at line {line_num}: {e}")
                    logger.error(f"Line content: {line[:100]}...")
                    raise

        logger.info(f"Loaded {len(data)} items from {file_path}")
        return data

    def validate_schema(
        self, data: list[dict], schema: str = "chat_messages"
    ) -> tuple[list[dict], list[str]]:
        """
        Validate data schema and return valid items with error messages.

        Args:
            data: List of data items
            schema: Expected schema type ('chat_messages' or 'instruction_output')

        Returns:
            Tuple of (valid_items, error_messages)
        """
        valid_items = []
        errors = []

        for i, item in enumerate(data):
            try:
                if schema == "chat_messages":
                    if self._validate_chat_messages(item):
                        valid_items.append(item)
                    else:
                        errors.append(f"Item {i}: Invalid chat_messages schema")

                elif schema == "instruction_output":
                    if self._validate_instruction_output(item):
                        valid_items.append(item)
                    else:
                        errors.append(f"Item {i}: Invalid instruction_output schema")

                else:
                    errors.append(f"Item {i}: Unknown schema type '{schema}'")

            except Exception as e:
                errors.append(f"Item {i}: Validation error - {e}")

        logger.info(
            f"Schema validation: {len(valid_items)} valid, {len(errors)} errors"
        )
        return valid_items, errors

    def _validate_chat_messages(self, item: dict) -> bool:
        """Validate chat_messages schema."""
        if "messages" not in item:
            return False

        messages = item["messages"]
        if not isinstance(messages, list) or len(messages) < 2:
            return False

        # Check that we have at least one user and one assistant message
        has_user = False
        has_assistant = False

        for message in messages:
            if not isinstance(message, dict):
                return False

            role = message.get("role", "").lower()
            content = message.get("content", "")

            if role == "user" and content.strip():
                has_user = True
            elif role == "assistant" and content.strip():
                has_assistant = True

        return has_user and has_assistant

    def _validate_instruction_output(self, item: dict) -> bool:
        """Validate instruction_output schema."""
        instruction = item.get("instruction", "")
        output = item.get("output", "")

        return bool(instruction.strip() and output.strip())

    def clean_data(self, data: list[dict], schema: str = "chat_messages") -> list[dict]:
        """
        Clean and filter data based on quality criteria.

        Args:
            data: List of data items
            schema: Data schema type

        Returns:
            Cleaned data items
        """
        cleaned = []

        for item in data:
            if schema == "chat_messages":
                cleaned_item = self._clean_chat_messages(item)
            elif schema == "instruction_output":
                cleaned_item = self._clean_instruction_output(item)
            else:
                cleaned_item = item

            if cleaned_item:
                cleaned.append(cleaned_item)

        logger.info(f"Data cleaning: {len(data)} -> {len(cleaned)} items")
        return cleaned

    def _clean_chat_messages(self, item: dict) -> dict | None:
        """Clean chat_messages item."""
        messages = item.get("messages", [])
        cleaned_messages = []

        for message in messages:
            role = message.get("role", "").lower()
            content = message.get("content", "")

            # Clean content
            content = content.strip()
            if len(content) < 10:  # Skip very short messages
                continue

            cleaned_messages.append({"role": role, "content": content})

        # Must have at least user + assistant
        if len(cleaned_messages) < 2:
            return None

        return {"messages": cleaned_messages}

    def _clean_instruction_output(self, item: dict) -> dict | None:
        """Clean instruction_output item."""
        instruction = item.get("instruction", "").strip()
        output = item.get("output", "").strip()

        # Skip very short items
        if len(instruction) < 10 or len(output) < 10:
            return None

        return {"instruction": instruction, "output": output}

    def estimate_token_length(self, text: str) -> int:
        """Estimate token length without loading the full model."""
        # Simple heuristic: ~4 characters per token for English text
        return len(text) // 4

    def get_token_lengths(
        self, data: list[dict], schema: str = "chat_messages"
    ) -> list[int]:
        """Get token length estimates for all data items."""
        lengths = []

        for item in data:
            if schema == "chat_messages":
                text = self._extract_chat_text(item)
            elif schema == "instruction_output":
                text = self._extract_instruction_text(item)
            else:
                text = str(item)

            length = self.estimate_token_length(text)
            lengths.append(length)

        return lengths

    def _extract_chat_text(self, item: dict) -> str:
        """Extract text from chat_messages item."""
        messages = item.get("messages", [])
        text_parts = []

        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            text_parts.append(f"{role}: {content}")

        return " ".join(text_parts)

    def _extract_instruction_text(self, item: dict) -> str:
        """Extract text from instruction_output item."""
        instruction = item.get("instruction", "")
        output = item.get("output", "")
        return f"{instruction} {output}"

    def remove_duplicates(
        self, data: list[dict], schema: str = "chat_messages"
    ) -> list[dict]:
        """Remove duplicate data items."""
        seen = set()
        unique_items = []

        for item in data:
            if schema == "chat_messages":
                key = self._get_chat_key(item)
            elif schema == "instruction_output":
                key = self._get_instruction_key(item)
            else:
                key = json.dumps(item, sort_keys=True)

            if key not in seen:
                seen.add(key)
                unique_items.append(item)

        removed = len(data) - len(unique_items)
        logger.info(f"Removed {removed} duplicate items")

        return unique_items

    def _get_chat_key(self, item: dict) -> str:
        """Get a key for deduplication of chat items."""
        messages = item.get("messages", [])
        key_parts = []

        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "").lower().strip()
            key_parts.append(f"{role}:{content}")

        return "|".join(key_parts)

    def _get_instruction_key(self, item: dict) -> str:
        """Get a key for deduplication of instruction items."""
        instruction = item.get("instruction", "").lower().strip()
        output = item.get("output", "").lower().strip()
        return f"{instruction}|{output}"

    def filter_by_length(
        self, data: list[dict], max_tokens: int, schema: str = "chat_messages"
    ) -> list[dict]:
        """Filter data by maximum token length."""
        filtered = []
        lengths = self.get_token_lengths(data, schema)

        for item, length in zip(data, lengths, strict=False):
            if length <= max_tokens:
                filtered.append(item)

        logger.info(
            f"Length filtering: {len(data)} -> {len(filtered)} items (max {max_tokens} tokens)"
        )

        return filtered

    def get_data_stats(self, data: list[dict], schema: str = "chat_messages") -> dict:
        """Get statistics about the dataset."""
        if not data:
            return {}

        lengths = self.get_token_lengths(data, schema)

        stats = {
            "total_items": len(data),
            "total_tokens_estimate": sum(lengths),
            "avg_tokens": np.mean(lengths),
            "median_tokens": np.median(lengths),
            "min_tokens": min(lengths),
            "max_tokens": max(lengths),
            "std_tokens": np.std(lengths),
        }

        # Percentiles
        percentiles = [25, 50, 75, 90, 95, 99]
        for p in percentiles:
            stats[f"p{p}_tokens"] = np.percentile(lengths, p)

        return stats
