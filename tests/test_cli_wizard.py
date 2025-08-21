"""Tests for the CLI wizard functionality."""

import tempfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from humigence.cli import app


class TestCLIWizard:
    """Test the CLI wizard functionality."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner for testing."""
        return CliRunner()

    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary directory for config files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_init_command_help(self, runner):
        """Test that init command shows help."""
        result = runner.invoke(app, ["init", "--help"])
        assert result.exit_code == 0
        assert "Interactive setup wizard" in result.output

    def test_wizard_command_help(self, runner):
        """Test that wizard command shows help."""
        result = runner.invoke(app, ["wizard", "--help"])
        assert result.exit_code == 0
        assert "Interactive setup wizard" in result.output

    def test_init_with_invalid_run(self, runner, temp_config_dir):
        """Test init command with invalid run parameter."""
        config_path = temp_config_dir / "test_config.json"

        result = runner.invoke(
            app, ["init", "--config", str(config_path), "--run", "invalid"]
        )

        assert result.exit_code == 2
        assert "Invalid run parameter" in result.output

    def test_init_command_structure(self, runner):
        """Test that init command has the expected structure."""
        # Test that the command exists and has the right options
        result = runner.invoke(app, ["init", "--help"])
        assert result.exit_code == 0

        # Check for expected options
        output = result.output
        assert "--config" in output
        assert "--run" in output
        assert "--train" in output
        assert "plan|validate|pipeline" in output

    def test_wizard_command_structure(self, runner):
        """Test that wizard command has the expected structure."""
        # Test that the command exists and has the right options
        result = runner.invoke(app, ["wizard", "--help"])
        assert result.exit_code == 0

        # Check for expected options
        output = result.output
        assert "--config" in output
        assert "--run" in output
        assert "--train" in output
        assert "plan|validate|pipeline" in output

    def test_init_and_wizard_are_aliases(self, runner):
        """Test that init and wizard commands have identical help output."""
        init_result = runner.invoke(app, ["init", "--help"])
        wizard_result = runner.invoke(app, ["wizard", "--help"])

        assert init_result.exit_code == 0
        assert wizard_result.exit_code == 0

        # The help text will be slightly different due to command names, but the options should be identical
        # Check that both have the same options
        init_output = init_result.output
        wizard_output = wizard_result.output

        # Check for expected options in both
        for option in ["--config", "--run", "--train"]:
            assert option in init_output
            assert option in wizard_output

        # Check for expected help text
        assert "Interactive setup wizard" in init_output
        assert "Interactive setup wizard" in wizard_output

    def test_init_default_values(self, runner):
        """Test that init command has the expected default values."""
        result = runner.invoke(app, ["init", "--help"])
        assert result.exit_code == 0

        output = result.output
        # Check default config path
        assert "configs/humigence.basic.json" in output
        # Check default run value
        assert "plan" in output
        # Check that train defaults to False
        assert "Allow training" in output

    def test_wizard_default_values(self, runner):
        """Test that wizard command has the expected default values."""
        result = runner.invoke(app, ["wizard", "--help"])
        assert result.exit_code == 0

        output = result.output
        # Check default config path
        assert "configs/humigence.basic.json" in output
        # Check default run value
        assert "plan" in output
        # Check that train defaults to False
        assert "Allow training" in result.output
