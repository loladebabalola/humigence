# Humigence

A functionality-first, local GPU training pipeline that fine-tunes TinyLlama-1.1B-Chat with QLoRA on a single RTX 4080. This is a walking skeleton that runs entirely on your machine with no external services beyond Hugging Face downloads.

## What is Humigence?

Humigence is a streamlined QLoRA fine-tuning pipeline designed for local development and experimentation. It provides a complete workflow from data preparation to model training, evaluation, and packaging - all optimized for single-GPU setups like the RTX 4080.

## Local-First Promise

- **No external services**: Everything runs on your local machine
- **Deterministic**: Set seeds, log environment & commit hash
- **Testable**: Small iterations with acceptance criteria
- **Production hygiene**: Typing, docstrings, error handling, clear logging

## Quickstart

### Using Make (Recommended for beginners)

```bash
# 1. Create and activate virtual environment
make venv

# 2. Install dependencies
make install

# 3. Verify CUDA and GPU
make gpu

# 4. Download TinyLlama-1.1B-Chat model
make model
```

### Using CLI (Advanced users and automation)

```bash
# 1. Install in development mode
pip install -e .

# 2. Check environment
humigence doctor

# 3. Interactive wizard with automatic training pipeline
humigence init --run pipeline --train

# 4. Or run pipeline directly with existing config
humigence pipeline --config configs/humigence.basic.json --train
```

## 🚀 New: Wizard-First Training Pipeline

Humigence now provides a seamless experience from configuration to training completion. The wizard automatically starts the full training pipeline after configuration.

### Two Main Workflows

#### 1. **Interactive Wizard + Automatic Training** (Recommended)
```bash
# Complete setup and immediately start training
humigence init --run pipeline --train

# This will:
# 1. Run the interactive configuration wizard
# 2. Save your configuration
# 3. Automatically execute the complete pipeline:
#    - Plan training (precision banner, batch/accum plan)
#    - Download model if missing
#    - Preprocess dataset
#    - Train model (QLoRA fine-tuning)
#    - Evaluate model
#    - Package artifacts
#    - Run acceptance gates
```

#### 2. **Direct Pipeline Execution** (Advanced Users)
```bash
# Skip wizard and run training directly with existing config
humigence pipeline --config configs/humigence.basic.json --train

# This bypasses the wizard and runs the pipeline directly
```

### Safety Features

- **Training is disabled by default** for safety
- **Use `--train` flag** or set `TRAIN=1` environment variable to enable training
- **Automatic model download** if not found locally
- **Configuration validation** before pipeline starts
- **Clear error messages** with actionable fixes
- **Idempotent execution** - reuses existing downloads/configs where possible

### Pipeline Steps

The complete pipeline executes these steps automatically:

1. **🔍 Configuration Validation** - Check all settings and paths
2. **📋 Training Planning** - Generate precision banner and batch plan
3. **🤖 Model Preparation** - Download model if missing
4. **📊 Data Preprocessing** - Clean, format, and split dataset
5. **🏋️ Training** - QLoRA fine-tuning (when enabled)
6. **📈 Evaluation** - Assess model performance
7. **📦 Packaging** - Export trained model
8. **✅ Acceptance Gates** - Quality checks and final validation

## Interactive Wizard

The Humigence wizard guides you through configuration with intelligent defaults and validation.

### Dataset Source Selection

The wizard offers three dataset options:

1. **Bundled OpenAssist Demo (Recommended)**
   - Automatically copies a pre-packaged 12-row demo dataset
   - Perfect for testing and learning the pipeline
   - Creates `data/raw/oa.jsonl` with chat_messages schema
   - No need to prepare data files beforehand

2. **Local JSONL File**
   - Use your existing dataset by providing a file path
   - Supports both `chat_messages` and `instruction_output` schemas
   - Validates file existence and readability

3. **Generate Tiny Demo**
   - Creates a minimal 12-row dataset at `data/raw/oa.jsonl`
   - Useful for quick testing without external data

### Training Safety

- **Training is disabled by default** for safety
- **Enable with `--train` flag** or set `TRAIN=1` environment variable
- **Clear warnings** when training is not enabled
- **Pipeline confirmation** for no-training mode

## Data Preparation

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Or:

```json
{"instruction": "...", "output": "..."}
```

## Training Pipeline

### Using Make

```bash
# 1. Preprocess and split data
make preprocess

# 2. Run QLoRA training
make train

# 3. Evaluate model
make eval

# 4. Package artifacts
make pack
```

### Using CLI

```bash
# 1. Preprocess and split data
humigence preprocess

# 2. Run QLoRA training (explicitly enabled)
humigence train --train

# 3. Evaluate model
humigence eval

# 4. Package artifacts
humigence pack
```

**Important**: Training is disabled by default for safety. Use `--train` flag or set `TRAIN=1` environment variable to enable training.

## Inference

```bash
# Run single-prompt inference from packaged artifacts
make infer PROMPT="Explain overfitting simply."
```

## Hardware Requirements

- **GPU**: RTX 4080 (16GB VRAM) or equivalent
- **RAM**: 32GB+ recommended
- **Storage**: 10GB+ for model and data

## Troubleshooting

### Out of Memory (OOM)
- The pipeline automatically reduces micro-batch size and increases gradient accumulation
- Check logs for effective tokens/step

### Slow Training
- Verify CUDA is properly installed
- Check GPU utilization with `nvidia-smi`
- Ensure sufficient system RAM

## Model & Dataset Cards

- **Model card**: `artifacts/humigence/model_card.md`
- **Dataset card**: `artifacts/humigence/dataset_card.md`

## CLI Reference

The Humigence CLI provides a comprehensive interface for all operations. All commands support `--help` for detailed options.

### Core Commands

```bash
# Interactive wizard with automatic pipeline execution
humigence init --run pipeline --train    # Wizard + full training pipeline
humigence init --run plan                # Wizard + training plan only
humigence init --run validate            # Wizard + validation only

# Direct pipeline execution (skip wizard)
humigence pipeline --config config.json --train  # Run pipeline directly

# Planning and validation (no training)
humigence plan                    # Create training plan (dry-run)
humigence validate                # Run validation pipeline

# Data and training
humigence preprocess              # Preprocess dataset
humigence train --train           # Run training (explicitly enabled)
humigence eval                    # Evaluate model
humigence pack                    # Package artifacts

# Model management
humigence model download          # Download base model
humigence model check             # Check model status
humigence infer --prompt "..."    # Run inference

# Utilities
humigence tokens                  # Show training metrics
humigence config view             # View configuration
humigence config set key value    # Modify configuration
humigence doctor                  # Environment diagnostics
humigence version                 # Show version info
```

### CLI Examples

```bash
# Interactive wizard with automatic training (recommended)
humigence init --run pipeline --train

# Interactive wizard with planning only
humigence init --run plan

# Direct pipeline execution with existing config
humigence pipeline --config configs/humigence.basic.json --train

# Environment variable override
TRAIN=1 humigence init --run pipeline

# Dry-run plan (no training)
humigence plan

# Validate setup (no training)
humigence validate

# Ablation example (FP16 LoRA)
humigence config set train.precision_mode lora_fp16
TRAIN=1 humigence pipeline --config configs/humigence.basic.json --train

# Quick inference
humigence infer --prompt "Explain machine learning" --temperature 0.7
```

### Environment Variables

- `TRAIN=1`: Enable training (overrides --train flag)
- `VALIDATION_ONLY=1`: Run validation only
- `HF_HOME`: Hugging Face cache directory
- `CUDA_VISIBLE_DEVICES`: GPU selection

### Exit Codes

- `0`: Success
- `2`: Bad user input/config
- `3`: Acceptance gates failed
- `4`: Environment/driver error
- `5`: Missing artifacts

## Interactive Wizard

The Humigence CLI includes an interactive setup wizard that guides you through configuration with arrow-key menus, toggles, and multi-selects. This provides a user-friendly alternative to manual config editing.

### Wizard-First UX

The wizard now automatically executes your chosen action after configuration, providing a seamless experience from setup to execution.

#### Two Main Flows

**1. Wizard + Planning (Default)**
```bash
# Complete setup and create training plan (no training)
humigence init --run plan

# This will:
# 1. Run the interactive wizard
# 2. Save your configuration
# 3. Automatically execute: humigence plan
# 4. Save training plan to runs/<project>/training_plan.json
```

**2. Wizard + Full Pipeline + Training**
```bash
# Complete setup and run full pipeline with training
humigence init --run pipeline --train

# This will:
# 1. Run the interactive wizard
# 2. Save your configuration
# 3. Automatically execute the complete pipeline:
#    - Plan training (precision banner, batch/accum plan)
#    - Download model if missing
#    - Preprocess data
#    - Train model (QLoRA fine-tuning)
#    - Evaluate model
#    - Package artifacts
#    - Run acceptance gates
```

### Wizard Commands

```bash
# Launch interactive wizard (defaults to plan mode)
humigence init

# Or use the alias
humigence wizard

# Specify what to run after configuration
humigence init --run plan      # Create training plan (default)
humigence init --run validate  # Run validation pipeline
humigence init --run pipeline  # Run complete pipeline

# Enable training (also honors TRAIN=1 environment variable)
humigence init --run pipeline --train
TRAIN=1 humigence init --run pipeline
```

### Safety Features

- **Training is disabled by default** for safety
- **Use `--train` flag** or set `TRAIN=1` environment variable to enable training
- **Pipeline confirmation**: If you choose pipeline without training, you'll be asked:
  ```
  Training is disabled by default. Run pipeline without training?
  Continue with pipeline (skip training)? [Y/n]:
  ```
- **Model auto-download**: Models are automatically downloaded if not found locally

### Wizard Features

The wizard collects all necessary configuration through interactive prompts:

- **Project Setup**: Project name and basic settings
- **Compute**: GPU device selection (auto-detected from nvidia-smi)
- **Base Model**: Curated model choices with disabled options marked "coming soon"
- **Dataset**: File paths, schema selection, split ratios, sequence length
- **Training**: Precision mode, LoRA parameters, evaluation settings
- **Evaluation & Acceptance**: Prompts path, quality thresholds
- **Exports**: Format selection with future options disabled

### Keyboard Controls

- **↑/↓**: Navigate between options
- **Space**: Toggle checkboxes and confirmations
- **Enter**: Confirm selection
- **Esc**: Abort wizard

### Disabled Options

The wizard shows future features as disabled with explanatory text:
- Multi-GPU training: "coming soon"
- Additional data schemas: "coming soon" (alpaca, sharegpt, oasst-1, parquet)
- Advanced export formats: "coming soon" (merged_fp16, runtime_int8)
- License-gated models: "license gated"

### Wizard Modes: Basic vs Advanced

Humigence now offers two setup modes to accommodate different user needs:

#### **Basic Setup Mode** ⚡
**Perfect for:** Quick start, beginners, or when you want to focus on essentials.

**Asks only 5 essential questions:**
1. **GPU Selection** - Choose your GPU device
2. **Base Model** - Select from curated model choices
3. **Dataset Path** - Choose bundled demo, local file, or generate realistic data
4. **Dataset Schema** - Select data format (chat_messages, instruction_output)
5. **Precision Mode** - Choose training precision (qlora_nf4, lora_fp16, lora_bf16, lora_int8)

**Everything else uses safe defaults:**
- LoRA configuration (r=16, alpha=32, dropout=0.05)
- Training parameters (lr=0.0002, scheduler=cosine, warmup=3%)
- Data splits (train=80%, val=10%, test=10%)
- Evaluation settings and acceptance thresholds
- Export formats

#### **Advanced Setup Mode** 🔧
**Perfect for:** Power users, custom configurations, or when you need full control.

**Provides complete control over all parameters:**
- All Basic mode questions plus:
- LoRA rank, alpha, dropout, target modules
- Training parameters (learning rate, scheduler, warmup, weight decay)
- Data configuration (sequence length, packing, splits, template)
- Evaluation settings (prompts path, thresholds)
- Export formats and acceptance criteria

#### **Mode Selection**

**Interactive Selection (Default):**
```bash
humigence init
# After project name, choose:
# [1] Basic Setup - Essential configuration only
# [2] Advanced Setup - Full control over all parameters
```

**CLI Flag Selection:**
```bash
# Basic setup (5 questions only)
humigence init --mode basic

# Advanced setup (full control)
humigence init --mode advanced
```

**Makefile Convenience:**
```bash
# Quick basic setup
make setup-basic

# Full advanced setup
make setup-advanced
```

#### **Example Basic Setup Flow**
```
Project name: MyProject
Choose setup mode:
[1] Basic Setup - Essential configuration only
[2] Advanced Setup - Full control over all parameters

Select mode (1 or 2): 1

⚡ Quick Setup
Select GPU device: GPU0: RTX_4080_16GB (default)
Choose base model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
Dataset source: Local JSONL file (enter path)
Enter path to local JSONL file: data/raw/my_data.jsonl
Data format/schema: chat_messages (default)
Training precision mode: qlora_nf4 (default)

Setup Mode: Basic Setup
Note: All other parameters set to defaults
```

### Wizard Output

After configuration, the wizard:
1. Saves config atomically with `.bak` backup
2. Displays a Rich summary table
3. Runs the selected action (plan/validate/pipeline)
4. Suggests the next command

## Configuration

Edit `configs/humigence.basic.json` to customize:
- Model parameters
- Training hyperparameters
- Data processing options
- QLoRA settings

## UI Mapping Table

For future no-code UI development, here's the mapping between UI controls and CLI/config:

| UI Control | Config Key | CLI Example | Description |
|------------|------------|-------------|-------------|
| Model Selection | `model.repo` | `humigence config set model.repo "TinyLlama/TinyLlama-1.1B-Chat-v1.0"` | Base model repository |
| Precision Mode | `train.precision_mode` | `humigence config set train.precision_mode "lora_fp16"` | Training precision (qlora_nf4, lora_fp16, lora_bf16) |
| Learning Rate | `train.lr` | `humigence config set train.lr 0.0001` | Learning rate for training |
| LoRA Rank | `lora.r` | `humigence config set lora.r 32` | LoRA rank (higher = more parameters) |
| Max Sequence Length | `data.max_seq_len` | `humigence config set data.max_seq_len 2048` | Maximum input sequence length |
| Training Epochs | `train.epochs` | `humigence config set train.epochs 3` | Number of training epochs |
| Enable Training | `--train` flag | `humigence train --train` | Explicitly enable training |
| Dataset Path | `data.raw_path` | `humigence config set data.raw_path "data/custom.jsonl"` | Path to training dataset |

## No-Code UI Contract

The following table maps UI controls to configuration fields and Make targets:

| UI Control | Config Field | Type | Make Target |
|------------|--------------|------|-------------|
| **Precision Mode** | `train.precision_mode` | Dropdown | `make plan` |
| **LoRA Rank** | `train.lora.r` | Slider | `make plan` |
| **Target Modules** | `train.lora.target_modules` | Multi-select | `make plan` |
| **Max Seq Length** | `data.max_seq_len` | Dropdown | `make plan` |
| **Split Ratios** | `data.split` | Sliders | `make plan` |
| **Learning Rate** | `train.lr` | Input | `make plan` |
| **Acceptance Thresholds** | `acceptance.*` | Inputs | `make validate` |
| **Export Formats** | `export.formats` | Checkboxes | `make pack` |

| UI Button | Action | Command |
|-----------|--------|---------|
| **Plan** | Show training configuration | `make plan` |
| **Validate** | Run validation pipeline | `make validate` |
| **Train** | Execute training | `TRAIN=1 make pipeline` |

## License

This project uses the TinyLlama-1.1B-Chat model and OpenAssistant dataset. Please review their respective licenses before use.

