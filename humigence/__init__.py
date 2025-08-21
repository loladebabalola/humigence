"""Humigence - Local GPU QLoRA Training Pipeline."""

__version__ = "0.2.0"
__author__ = "Humigence Team"
__description__ = "Production-grade QLoRA fine-tuning for local GPUs"

from .acceptance import AcceptanceGates
from .config import Config
from .eval import ModelEvaluator
from .infer import ModelInferencer
from .pack import ModelPacker
from .plan import TrainingPlanner
from .precision import build_model_and_peft
from .preprocess import DataPreprocessor
from .telemetry import TrainingTelemetry
from .train import QLoRATrainer

__all__ = [
    "Config",
    "DataPreprocessor",
    "QLoRATrainer",
    "ModelEvaluator",
    "ModelInferencer",
    "ModelPacker",
    "build_model_and_peft",
    "TrainingTelemetry",
    "AcceptanceGates",
    "TrainingPlanner",
]
