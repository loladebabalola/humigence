"""Test wizard dataset source selection functionality."""

import json
from unittest.mock import patch

from humigence.wizard import run_wizard


class TestWizardDataset:
    """Test wizard dataset source selection."""

    def test_wizard_bundled_dataset_selection(self, tmp_path):
        """Test that wizard creates data/raw/oa.jsonl when bundled demo is chosen."""
        # Create a temporary config path
        config_path = tmp_path / "test_config.json"

        # Mock the InquirerPy responses to select bundled dataset
        mock_responses = {
            "project": "test_project",
            "gpu_device": 0,
            "base_model": "Qwen/Qwen2.5-0.5B",
            "dataset_source": "bundled",  # This is the key selection
            "data_schema": "chat_messages",
            "max_seq_len": 1024,
            "packing": True,
            "precision_mode": "qlora_nf4",
            "lora_r": "16",
            "lora_alpha": "32",
            "lora_dropout": "0.05",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "tokens_per_step": "100000",
            "eval_every_steps": "500",
            "save_every_steps": "500",
            "curated_prompts_path": "configs/curated_eval_prompts.jsonl",
            "min_val_loss_improvement": "1.0",
            "curated_reasonable_threshold": "70.0",
            "jitter_threshold": "20.0",
            "export_formats": ["peft_adapter"],
        }

        # Mock the inquirer to return our test responses
        with patch("humigence.wizard.inquirer") as mock_inquirer:
            # Set up the mock to return our responses
            for key, value in mock_responses.items():
                if key == "dataset_source":
                    # Special handling for dataset source selection
                    mock_inquirer.select.return_value.execute.return_value = value
                elif key == "target_modules":
                    # Special handling for multi-select
                    mock_inquirer.checkbox.return_value.execute.return_value = value
                elif key == "export_formats":
                    # Special handling for multi-select
                    mock_inquirer.checkbox.return_value.execute.return_value = value
                else:
                    # For text inputs
                    mock_inquirer.text.return_value.execute.return_value = value

            # Mock the importlib.resources.files to return a path to our test data
            with patch("humigence.wizard.files") as mock_files, patch(
                "humigence.wizard.shutil.copyfile"
            ) as mock_copy:
                # Create a mock demo dataset path
                mock_demo_path = tmp_path / "mock_demo.jsonl"
                mock_demo_path.write_text(
                    '{"messages":[{"role":"user","content":"test"}]}'
                )
                mock_files.return_value.__truediv__.return_value = mock_demo_path

                # Run the wizard
                result = run_wizard(config_path, default_action="plan", train=False)

                # Verify the result
                assert result["exit_code"] == 0
                assert result["next_action"] == "plan"
                assert result["train"] is False

                # Verify that the bundled dataset was copied
                expected_data_path = tmp_path / "data" / "raw" / "oa.jsonl"
                assert expected_data_path.exists()

                # Verify the copy was called
                mock_copy.assert_called_once_with(mock_demo_path, expected_data_path)

                # Verify the config was saved
                assert config_path.exists()

                # Load and verify the config
                with open(config_path) as f:
                    config_data = json.load(f)

                assert config_data["data"]["raw_path"] == "data/raw/oa.jsonl"
                assert config_data["data"]["data_schema"] == "chat_messages"

    def test_wizard_local_dataset_selection(self, tmp_path):
        """Test that wizard accepts local dataset path."""
        # Create a temporary config path
        config_path = tmp_path / "test_config.json"

        # Create a test dataset file
        test_dataset = tmp_path / "test_dataset.jsonl"
        test_dataset.write_text('{"messages":[{"role":"user","content":"test"}]}')

        mock_responses = {
            "project": "test_project",
            "gpu_device": 0,
            "base_model": "Qwen/Qwen2.5-0.5B",
            "dataset_source": "local",
            "local_dataset_path": str(test_dataset),  # Path to existing file
            "data_schema": "chat_messages",
            "max_seq_len": 1024,
            "packing": True,
            "precision_mode": "qlora_nf4",
            "lora_r": "16",
            "lora_alpha": "32",
            "lora_dropout": "0.05",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "tokens_per_step": "100000",
            "eval_every_steps": "500",
            "save_every_steps": "500",
            "curated_prompts_path": "configs/curated_eval_prompts.jsonl",
            "min_val_loss_improvement": "1.0",
            "curated_reasonable_threshold": "70.0",
            "jitter_threshold": "20.0",
            "export_formats": ["peft_adapter"],
        }

        with patch("humigence.wizard.inquirer") as mock_inquirer:
            # Set up the mock responses
            for key, value in mock_responses.items():
                if key == "dataset_source":
                    mock_inquirer.select.return_value.execute.return_value = value
                elif key == "local_dataset_path":
                    # This should be prompted when local is selected
                    mock_inquirer.text.return_value.execute.return_value = value
                elif key == "target_modules":
                    mock_inquirer.checkbox.return_value.execute.return_value = value
                elif key == "export_formats":
                    mock_inquirer.checkbox.return_value.execute.return_value = value
                else:
                    mock_inquirer.text.return_value.execute.return_value = value

            # Run the wizard
            result = run_wizard(config_path, default_action="plan", train=False)

            # Verify the result
            assert result["exit_code"] == 0
            assert result["next_action"] == "plan"

            # Verify the config was saved with the local path
            with open(config_path) as f:
                config_data = json.load(f)

            assert config_data["data"]["raw_path"] == str(test_dataset)
