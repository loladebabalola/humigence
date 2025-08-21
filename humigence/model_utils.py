"""Model utilities for Humigence."""

from pathlib import Path

from rich.console import Console

from .config import Config

console = Console()


def _expand(p: str | Path | None) -> Path | None:
    if p is None:
        return None
    return Path(p).expanduser().resolve()


def ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def ensure_model_available(cfg) -> Path:
    """
    Ensures the base model is present locally.
    - If cfg.model.local_path exists → return it.
    - Else try huggingface_hub.snapshot_download to populate, update cfg, persist.
    - On failure, raise RuntimeError with the exact follow-up CLI.
    """
    local = _expand(getattr(cfg.model, "local_path", None))
    if local and local.exists():
        console.print(f"[green]✓ Model already available: {local}[/green]")
        return local

    repo = cfg.model.repo
    cache_dir = (
        _expand(getattr(cfg.model, "cache_dir", "~/.cache/huggingface/hub"))
        or Path("~/.cache/huggingface/hub").expanduser()
    )

    try:
        from huggingface_hub import snapshot_download

        console.print(f"[cyan]📥 Downloading base model[/cyan] [bold]{repo}[/bold]...")
        path = Path(
            snapshot_download(
                repo_id=repo, cache_dir=str(cache_dir), local_files_only=False
            )
        )

        # Update config and persist if possible
        if hasattr(cfg, '_source_path') and cfg._source_path:
            try:
                cfg.model.local_path = str(path)
                from .config import save_config_atomic
                save_config_atomic(cfg._source_path, cfg)
                console.print(f"[green]✓ Model downloaded and config updated: {path}[/green]")
            except Exception as save_error:
                console.print(f"[yellow]⚠️  Model downloaded but config update failed: {save_error}[/yellow]")
        else:
            console.print(f"[green]✓ Model downloaded: {path}[/green]")

        return path

    except Exception as e:
        error_msg = (
            f"Base model not available and auto-download failed: {e}\n"
            "💡 Solutions:\n"
            "  1. Check your internet connection\n"
            "  2. Verify HuggingFace authentication: `huggingface-cli login`\n"
            "  3. Try manual download: `humigence model download`\n"
            "  4. Check if the model repository exists and is accessible"
        )
        raise RuntimeError(error_msg) from None


def get_model_info(config: Config) -> dict:
    """Get information about the model.

    Args:
        config: Configuration object

    Returns:
        dict: Model information including size, status, etc.
    """
    model_path = config.get_model_path()

    if model_path.exists():
        # Calculate directory size
        total_size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())

        return {
            "status": "available",
            "path": str(model_path),
            "size_gb": round(total_size / (1024**3), 2),
            "type": "local",
        }
    else:
        return {
            "status": "needs_download",
            "repo": config.model.repo,
            "estimated_size_gb": 1.2,  # Rough estimate for Qwen2.5-0.5B
            "type": "remote",
        }
