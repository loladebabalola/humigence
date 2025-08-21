"""Tests for CLI root callback functionality."""

from unittest.mock import patch

from typer.testing import CliRunner

from humigence.cli import app

runner = CliRunner(mix_stderr=False)


def test_root_shows_help():
    """Test that --no-wizard shows help instead of launching wizard."""
    result = runner.invoke(app, ["--no-wizard"])
    assert result.exit_code == 0
    assert "Commands" in result.stdout


def test_root_wizard_with_env_vars(monkeypatch):
    """Test that wizard launches when environment variables are set."""
    monkeypatch.setenv("HUMIGENCE_DEFAULT_CMD", "wizard")
    monkeypatch.setenv("HUMIGENCE_WIZARD_RUN", "plan")

    # Mock the run_wizard function to avoid actual wizard execution
    with patch("humigence.cli.run_wizard") as mock_run_wizard:
        mock_run_wizard.return_value = 0

        result = runner.invoke(app, [])
        assert result.exit_code == 0


def test_root_wizard_fallback_to_help():
    """Test that non-TTY environments fall back to help."""
    # Mock non-TTY environment
    import sys

    # Save original values
    original_stdin_isatty = sys.stdin.isatty
    original_stdout_isatty = sys.stdout.isatty

    try:
        # Mock non-TTY
        sys.stdin.isatty = lambda: False
        sys.stdout.isatty = lambda: False

        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "Commands" in result.stdout
    finally:
        # Restore original values
        sys.stdin.isatty = original_stdin_isatty
        sys.stdout.isatty = original_stdout_isatty


def test_root_wizard_with_train_flag():
    """Test that --train flag is properly handled."""
    result = runner.invoke(app, ["--train", "--no-wizard"])
    assert result.exit_code == 0
    assert "Commands" in result.stdout


def test_root_wizard_with_config_override():
    """Test that --config flag is properly handled."""
    result = runner.invoke(app, ["--config", "custom_config.json", "--no-wizard"])
    assert result.exit_code == 0
    assert "Commands" in result.stdout


def test_root_wizard_with_run_override():
    """Test that --run flag is properly handled."""
    result = runner.invoke(app, ["--run", "pipeline", "--no-wizard"])
    assert result.exit_code == 0
    assert "Commands" in result.stdout


def test_root_wizard_environment_variables(monkeypatch):
    """Test that environment variables are properly respected."""
    monkeypatch.setenv("HUMIGENCE_DEFAULT_CMD", "help")

    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "Commands" in result.stdout


def test_root_wizard_training_environment_variable(monkeypatch):
    """Test that TRAIN environment variable is properly handled."""
    monkeypatch.setenv("TRAIN", "1")

    result = runner.invoke(app, ["--no-wizard"])
    assert result.exit_code == 0
    assert "Commands" in result.stdout


def test_root_wizard_wizard_run_environment_variable(monkeypatch):
    """Test that HUMIGENCE_WIZARD_RUN environment variable is properly handled."""
    monkeypatch.setenv("HUMIGENCE_WIZARD_RUN", "validate")

    result = runner.invoke(app, ["--no-wizard"])
    assert result.exit_code == 0
    assert "Commands" in result.stdout
