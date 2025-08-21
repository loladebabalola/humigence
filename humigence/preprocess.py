"""
Data preprocessing module for Humigence.
Handles data loading, cleaning, splitting, and formatting.
"""

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from .config import Config
from .templates import ChatTemplate
from .utils_data import DataProcessor
from .utils_logging import setup_logging


class PreprocessingEmptyTrainError(Exception):
    """Raised when preprocessing results in empty training dataset."""
    pass


class DataPreprocessor:
    """Handles data preprocessing pipeline."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging()

        # Load tokenizer
        self.logger.info("Loading tokenizer...")
        model_path = config.get_model_path()
        if model_path.exists():
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(model_path), trust_remote_code=True
            )
        else:
            # Fallback to loading from the repo
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.model.repo, trust_remote_code=True
            )

        # Initialize data processor
        self.data_processor = DataProcessor(self.tokenizer)

        # Initialize chat template
        self.chat_template = ChatTemplate()

    def preprocess(self) -> dict:
        """Main preprocessing method called by CLI.

        Returns:
            dict: Preprocessing report with status and statistics
        """
        try:
            result = self.preprocess_data()

            # Check if training data is empty
            if not result.get("train") or len(result["train"]) == 0:
                raise PreprocessingEmptyTrainError(
                    "Preprocessing resulted in empty training dataset. "
                    "Check your data source and split configuration."
                )

            return {
                "status": "success",
                "data": result,
                "message": "Data preprocessing completed successfully",
            }
        except Exception as e:
            self.logger.error(f"Data preprocessing failed: {e}")
            raise

    def preprocess_data(self) -> dict[str, list[dict]]:
        """Run the complete preprocessing pipeline."""
        self.logger.info("Starting data preprocessing...")

        # Load raw data
        raw_data = self._load_raw_data()

        # Validate and clean data
        clean_data = self._clean_data(raw_data)

        # Convert to training format
        formatted_data = self._format_data(clean_data)

        # Split data
        split_data = self._split_data(formatted_data)

        # Save processed data
        self._save_processed_data(split_data)

        # Generate report
        report = self._generate_report(raw_data, clean_data, formatted_data, split_data)
        self._save_report(report)

        self.logger.info("Data preprocessing completed!")
        return split_data

    def _load_raw_data(self) -> list[dict]:
        """Load raw data from the configured path."""
        raw_path = Path(self.config.data.raw_path)
        self.logger.info(f"Loading raw data from: {raw_path}")

        if not raw_path.exists():
            raise FileNotFoundError(f"Raw data file not found: {raw_path}")

        return self.data_processor.load_jsonl(raw_path)

    def _clean_data(self, raw_data: list[dict]) -> list[dict]:
        """Clean and validate raw data."""
        self.logger.info("Cleaning and validating data...")

        # Validate schema
        valid_data, errors = self.data_processor.validate_schema(
            raw_data, self.config.data.data_schema
        )

        if errors:
            self.logger.warning(f"Found {len(errors)} validation errors:")
            for error in errors[:10]:  # Show first 10 errors
                self.logger.warning(f"  {error}")
            if len(errors) > 10:
                self.logger.warning(f"  ... and {len(errors) - 10} more errors")

        # Clean data
        clean_data = self.data_processor.clean_data(
            valid_data, self.config.data.data_schema
        )

        # Remove duplicates
        clean_data = self.data_processor.remove_duplicates(
            clean_data, self.config.data.data_schema
        )

        # Filter by length
        clean_data = self.data_processor.filter_by_length(
            clean_data, self.config.data.max_seq_len, self.config.data.data_schema
        )

        return clean_data

    def _format_data(self, clean_data: list[dict]) -> list[dict]:
        """Convert data to training format."""
        self.logger.info("Formatting data for training...")

        formatted_data = []

        for item in clean_data:
            if self.config.data.data_schema == "chat_messages":
                formatted_item = self._format_chat_item(item)
            elif self.config.data.data_schema == "instruction_output":
                formatted_item = self._format_instruction_item(item)
            else:
                formatted_item = item

            if formatted_item:
                formatted_data.append(formatted_item)

        return formatted_data

    def _format_chat_item(self, item: dict) -> dict | None:
        """Format a chat item for training."""
        messages = item.get("messages", [])

        # Format the conversation
        formatted_text = self.chat_template.format_chat(
            messages, add_generation_prompt=False
        )

        # Get the target text (assistant response)
        target_text = ""
        for message in reversed(messages):
            if message.get("role", "").lower() == "assistant":
                target_text = message.get("content", "")
                break

        if not target_text:
            return None

        return {"text": formatted_text, "target": target_text}

    def _format_instruction_item(self, item: dict) -> dict | None:
        """Format an instruction item for training."""
        instruction = item.get("instruction", "")
        output = item.get("output", "")

        # Format as instruction-following prompt
        formatted_text = self.chat_template.format_instruction(
            instruction, add_generation_prompt=False
        )

        return {"text": formatted_text, "target": output}

    def _split_data(self, formatted_data: list[dict]) -> dict[str, list[dict]]:
        """Split data into train/val/test sets."""
        self.logger.info("Splitting data...")

        # Set random seed for reproducibility
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)

        # Shuffle data
        shuffled_data = formatted_data.copy()
        random.shuffle(shuffled_data)

        # Calculate split indices
        total = len(shuffled_data)
        train_end = int(total * self.config.data.split["train"])
        val_end = train_end + int(total * self.config.data.split["val"])

        # Split data
        split_data = {
            "train": shuffled_data[:train_end],
            "val": shuffled_data[train_end:val_end],
            "test": shuffled_data[val_end:],
        }

        self.logger.info(
            f"Data split: train={len(split_data['train'])}, "
            f"val={len(split_data['val'])}, test={len(split_data['test'])}"
        )

        return split_data

    def _save_processed_data(self, split_data: dict[str, list[dict]]) -> None:
        """Save processed data to files."""
        processed_dir = Path(self.config.data.processed_dir)
        processed_dir.mkdir(parents=True, exist_ok=True)

        for split_name, data in split_data.items():
            output_file = processed_dir / f"{split_name}.jsonl"

            with open(output_file, "w", encoding="utf-8") as f:
                for item in data:
                    json.dump(item, f, ensure_ascii=False)
                    f.write("\n")

            self.logger.info(f"Saved {len(data)} items to {output_file}")

    def _generate_report(
        self,
        raw_data: list[dict],
        clean_data: list[dict],
        formatted_data: list[dict],
        split_data: dict[str, list[dict]],
    ) -> dict:
        """Generate a comprehensive preprocessing report."""
        report = {
            "preprocessing_summary": {
                "raw_items": len(raw_data),
                "clean_items": len(clean_data),
                "formatted_items": len(formatted_data),
                "removed_items": len(raw_data) - len(clean_data),
                "schema": self.config.data.data_schema,
                "max_seq_len": self.config.data.max_seq_len,
            },
            "data_splits": {
                split_name: len(data) for split_name, data in split_data.items()
            },
            "data_quality": self.data_processor.get_data_stats(
                clean_data, self.config.data.data_schema
            ),
            "config": self.config.dict(),
        }

        return report

    def _save_report(self, report: dict) -> None:
        """Save the preprocessing report."""
        report_file = Path(self.config.data.processed_dir) / "preprocessing_report.json"

        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Preprocessing report saved to: {report_file}")

        # Print summary
        self._print_summary(report)

    def _print_summary(self, report: dict) -> None:
        """Print a summary of the preprocessing results."""
        summary = report["preprocessing_summary"]
        splits = report["data_splits"]
        quality = report["data_quality"]

        self.logger.info("=" * 60)
        self.logger.info("PREPROCESSING SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Raw data items: {summary['raw_items']}")
        self.logger.info(f"Clean data items: {summary['clean_items']}")
        self.logger.info(f"Formatted items: {summary['formatted_items']}")
        self.logger.info(f"Removed items: {summary['removed_items']}")
        self.logger.info(f"Schema: {summary['schema']}")
        self.logger.info(f"Max sequence length: {summary['max_seq_len']}")

        self.logger.info("\nData splits:")
        for split_name, count in splits.items():
            self.logger.info(f"  {split_name}: {count} items")

        if quality:
            self.logger.info("\nData quality:")
            self.logger.info(f"  Average tokens: {quality['avg_tokens']:.1f}")
            self.logger.info(f"  Median tokens: {quality['median_tokens']:.1f}")
            self.logger.info(f"  Min tokens: {quality['min_tokens']}")
            self.logger.info(f"  Max tokens: {quality['max_tokens']}")

        self.logger.info("=" * 60)


def main():
    """Main function for the preprocessing CLI."""
    parser = argparse.ArgumentParser(description="Humigence Data Preprocessing")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration file"
    )

    args = parser.parse_args()

    try:
        # Load configuration
        config = Config.from_file(args.config)

        # Initialize preprocessor
        preprocessor = DataPreprocessor(config)

        # Run preprocessing
        preprocessor.preprocess()

    except Exception as e:
        logging.error(f"Preprocessing failed: {e}")
        raise


if __name__ == "__main__":
    main()
