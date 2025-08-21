"""
Planning module for Humigence.
Provides training planning without actual training execution.
"""

import argparse
import json
import logging
from pathlib import Path

from .config import Config
from .utils_logging import setup_logging


class TrainingPlanner:
    """Plans training configuration without executing training."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging()

        self.logger.info("🔍 Initializing training planner...")

    def plan_training(self) -> dict:
        """Create a comprehensive training plan."""
        self.logger.info("📋 Creating training plan...")

        # Check model availability
        model_status = self._check_model_availability()

        # Check data status
        data_status = self._check_data_status()

        # Calculate projected VRAM usage
        vram_projection = self._project_vram_usage()

        # Build training plan
        plan = {
            "model_status": model_status,
            "data_status": data_status,
            "vram_projection": vram_projection,
            "training_config": self._build_training_config(),
            "precision_config": self._build_precision_config(),
            "acceptance_criteria": self.config.acceptance.dict()
            if hasattr(self.config, "acceptance")
            else {},
        }

        return plan

    def _check_model_availability(self) -> dict:
        """Check if the model is available locally or needs downloading."""
        model_path = self.config.get_model_path()

        if model_path.exists():
            return {
                "status": "available",
                "path": str(model_path),
                "size_gb": self._get_directory_size(model_path) / (1024**3),
            }
        else:
            return {
                "status": "needs_download",
                "repo": self.config.model.repo,
                "local_path": str(model_path),
                "estimated_size_gb": 1.2,  # Rough estimate for Qwen2.5-0.5B
            }

    def _check_data_status(self) -> dict:
        """Check data availability and statistics."""
        raw_path = Path(self.config.data.raw_path)
        processed_dir = Path(self.config.data.processed_dir)

        if not raw_path.exists():
            return {
                "status": "missing",
                "raw_path": str(raw_path),
                "message": "Raw data file not found",
            }

        # Count lines in raw data
        try:
            with open(raw_path) as f:
                line_count = sum(1 for _ in f)
        except Exception:
            line_count = 0

        return {
            "status": "available",
            "raw_path": str(raw_path),
            "line_count": line_count,
            "processed_dir": str(processed_dir),
            "processed_exists": processed_dir.exists(),
        }

    def _project_vram_usage(self) -> dict:
        """Project VRAM usage for different batch sizes."""
        max_seq_len = self.config.data.max_seq_len
        target_tokens_per_step = self.config.train.tokens_per_step_target

        projections = []
        for micro_batch_size in [32, 16, 8, 4, 2, 1]:
            required_grad_accum = max(
                1, target_tokens_per_step // (micro_batch_size * max_seq_len)
            )
            effective_batch_size = micro_batch_size * required_grad_accum

            # Rough VRAM estimation (conservative)
            base_model_vram = 2.0  # GB
            per_token_vram = 0.000001  # GB per token
            batch_vram = effective_batch_size * max_seq_len * per_token_vram
            total_vram = base_model_vram + batch_vram

            projections.append(
                {
                    "micro_batch_size": micro_batch_size,
                    "grad_accum": required_grad_accum,
                    "effective_batch_size": effective_batch_size,
                    "projected_vram_gb": round(total_vram, 2),
                    "feasible": total_vram <= 15.0,  # RTX 4080 limit
                }
            )

        return {
            "projections": projections,
            "recommended_config": next(
                (p for p in projections if p["feasible"]), projections[-1]
            ),
        }

    def _build_training_config(self) -> dict:
        """Build training configuration summary."""
        return {
            "epochs": self.config.train.epochs,
            "learning_rate": self.config.train.lr,
            "scheduler": self.config.train.scheduler,
            "warmup_ratio": self.config.train.warmup_ratio,
            "weight_decay": self.config.train.weight_decay,
            "gradient_clipping": self.config.train.grad_clip,
            "gradient_checkpointing": self.config.train.gradient_checkpointing,
            "tokens_per_step_target": self.config.train.tokens_per_step_target,
            "eval_every_steps": self.config.train.eval_every_steps,
            "save_every_steps": self.config.train.save_every_steps,
        }

    def _build_precision_config(self) -> dict:
        """Build precision configuration summary."""
        return {
            "mode": self.config.train.precision_mode,
            "lora_targets": self.config.train.lora.target_modules,
            "lora_rank": self.config.train.lora.r,
            "lora_alpha": self.config.train.lora.alpha,
            "lora_dropout": self.config.train.lora.dropout,
        }

    def _get_directory_size(self, path: Path) -> int:
        """Get total size of directory in bytes."""
        total_size = 0
        try:
            for file_path in path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception:
            pass
        return total_size

    def print_plan(self, plan: dict):
        """Print the training plan in a readable format."""
        self.logger.info("=" * 80)
        self.logger.info("🎯 HUMIGENCE TRAINING PLAN")
        self.logger.info("=" * 80)

        # Model status
        model = plan["model_status"]
        if model["status"] == "available":
            self.logger.info(
                f"✅ Model: Available at {model['path']} ({model['size_gb']:.1f} GB)"
            )
        else:
            self.logger.info(
                f"📥 Model: Will download {model['repo']} to {model['local_path']}"
            )

        # Data status
        data = plan["data_status"]
        if data["status"] == "available":
            self.logger.info(f"✅ Data: {data['line_count']} samples available")
        else:
            self.logger.info(f"❌ Data: {data['message']}")

        # VRAM projection
        vram = plan["vram_projection"]
        recommended = vram["recommended_config"]
        self.logger.info(
            f"🎮 VRAM: Recommended {recommended['micro_batch_size']}x{recommended['grad_accum']} "
            f"({recommended['projected_vram_gb']} GB)"
        )

        # Precision config
        precision = plan["precision_config"]
        self.logger.info(
            f"⚡ Precision: {precision['mode']} with LoRA rank {precision['lora_rank']}"
        )
        self.logger.info(f"🎯 LoRA Targets: {', '.join(precision['lora_targets'])}")

        # Training config
        train = plan["training_config"]
        self.logger.info(
            f"🚀 Training: {train['epochs']} epochs, LR {train['learning_rate']}, "
            f"target {train['tokens_per_step_target']:,} tokens/step"
        )

        self.logger.info("=" * 80)
        self.logger.info("📋 Plan complete - use TRAIN=1 to execute training")
        self.logger.info("=" * 80)


def main():
    """Main function for the planning CLI."""
    parser = argparse.ArgumentParser(description="Humigence Training Planner")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration file"
    )

    args = parser.parse_args()

    try:
        # Load configuration
        config = Config.from_file(args.config)

        # Initialize planner
        planner = TrainingPlanner(config)

        # Create and print plan
        plan = planner.plan_training()
        planner.print_plan(plan)

        # Save plan to file
        plan_file = config.get_runs_dir() / "training_plan.json"
        plan_file.parent.mkdir(parents=True, exist_ok=True)

        with open(plan_file, "w") as f:
            json.dump(plan, f, indent=2)

        print(f"\n📄 Training plan saved to: {plan_file}")

    except Exception as e:
        logging.error(f"Planning failed: {e}")
        raise


if __name__ == "__main__":
    main()
