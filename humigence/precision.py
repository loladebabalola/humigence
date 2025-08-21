"""Precision and quantization dispatcher for Humigence training."""

import logging
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)


def print_precision_banner(
    precision_mode: str,
    dtype: torch.dtype,
    quantization: str | None,
    target_modules: list[str],
) -> None:
    """Print a clear precision configuration banner."""
    logger.info("=" * 80)
    logger.info("🎯 PRECISION CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Mode: {precision_mode}")
    logger.info(f"Base DType: {dtype}")
    logger.info(f"Quantization: {quantization or 'None'}")
    logger.info(f"LoRA Targets: {', '.join(target_modules)}")
    logger.info("=" * 80)


def build_model_and_peft(
    config: dict,
) -> tuple[AutoModelForCausalLM, AutoTokenizer, LoraConfig | None]:
    """
    Build model and PEFT configuration based on precision mode.

    Args:
        config: Configuration dictionary with model and training settings

    Returns:
        Tuple of (model, tokenizer, peft_config)
    """
    model_repo = config["model"]["repo"]
    local_path = config["model"]["local_path"]
    precision_mode = config["train"]["precision_mode"]
    lora_config = config["train"]["lora"]

    # Use local path if available, otherwise download
    if local_path:
        model_path = Path(local_path).expanduser()
        if not model_path.exists():
            logger.warning(
                f"Local path {model_path} does not exist, falling back to repo"
            )
            model_path = model_repo
    else:
        model_path = model_repo

    logger.info(f"Loading model from: {model_path}")

    # Check if tokenizer is already provided
    if "_tokenizer" in config:
        tokenizer = config["_tokenizer"]
        logger.info("Using provided tokenizer")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        # Add pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    if precision_mode == "qlora_nf4":
        # 4-bit nf4 + double quant + PEFT LoRA adapters
        logger.info("Loading model in 4-bit NF4 with double quantization")

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            ),
            device_map={"": torch.cuda.current_device()},
        )

        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)

        # Configure LoRA for 4-bit model
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            target_modules=lora_config["target_modules"],
            r=lora_config["r"],
            lora_alpha=lora_config["alpha"],
            lora_dropout=lora_config["dropout"],
            bias="none",
        )

        model = get_peft_model(model, peft_config)

        # Ensure LoRA parameters are trainable and model is in training mode
        trainable_params = 0
        all_params = 0
        for name, param in model.named_parameters():
            all_params += param.numel()
            if "lora_" in name:
                param.requires_grad = True
                trainable_params += param.numel()

        # Enable training mode
        model.train()

        logger.info(f"Total parameters: {all_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Trainable percentage: {100 * trainable_params / all_params:.2f}%")

        print_precision_banner(
            precision_mode, torch.float16, "4-bit NF4", lora_config["target_modules"]
        )

    elif precision_mode == "lora_fp16":
        # fp16 base + PEFT LoRA
        logger.info("Loading model in FP16 with LoRA")

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            target_modules=lora_config["target_modules"],
            r=lora_config["r"],
            lora_alpha=lora_config["alpha"],
            lora_dropout=lora_config["dropout"],
            bias="none",
        )

        model = get_peft_model(model, peft_config)

        # Ensure LoRA parameters are trainable
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

        print_precision_banner(
            precision_mode, torch.float16, None, lora_config["target_modules"]
        )

    elif precision_mode == "lora_bf16":
        # bf16 base + PEFT LoRA (with availability check)
        if not torch.cuda.is_bf16_supported():
            raise ValueError("BF16 not supported on this GPU. Use FP16 instead.")

        logger.info("Loading model in BF16 with LoRA")

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            target_modules=lora_config["target_modules"],
            r=lora_config["r"],
            lora_alpha=lora_config["alpha"],
            lora_dropout=lora_config["dropout"],
            bias="none",
        )

        model = get_peft_model(model, peft_config)

        # Ensure LoRA parameters are trainable
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

        print_precision_banner(
            precision_mode, torch.bfloat16, None, lora_config["target_modules"]
        )

    elif precision_mode == "lora_int8":
        # 8-bit (bnb) base + PEFT LoRA
        logger.info("Loading model in 8-bit with LoRA")

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            load_in_8bit=True,
            device_map="auto",
        )

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            target_modules=lora_config["target_modules"],
            r=lora_config["r"],
            lora_alpha=lora_config["alpha"],
            lora_dropout=lora_config["dropout"],
            bias="none",
        )

        model = get_peft_model(model, peft_config)

        # Ensure LoRA parameters are trainable
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

        print_precision_banner(
            precision_mode, torch.float16, "8-bit", lora_config["target_modules"]
        )

    else:
        raise ValueError(
            f"Unsupported precision_mode: {precision_mode}. "
            "Supported modes: qlora_nf4, lora_fp16, lora_bf16, lora_int8"
        )

    return model, tokenizer, peft_config
