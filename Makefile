# Humigence - Local GPU QLoRA Training Pipeline
# Makefile for streamlined training workflow

.PHONY: help venv install gpu model data-download preprocess train eval pack infer pipeline ablate-fp16 tokens test clean format lint plan validate

# Default target
help:
	@echo "Humigence - Local GPU QLoRA Training Pipeline"
	@echo ""
	@echo "Available targets:"
	@echo "  clean           Remove runs/ and temp files (keep artifacts/)"
	@echo "  data-download   Download sample OpenAssistant data"
	@echo "  eval            Quantitative + qualitative evaluation"
	@echo "  format          Format code with black and ruff"
	@echo "  gpu             Verify CUDA and GPU"
	@echo "  help            Show this help message"
	@echo "  infer           Run single-prompt inference from artifacts"
	@echo "  install         Install dependencies"
	@echo "  lint            Run linting checks"
	@echo "  model           Download Qwen2.5-0.5B model locally"
	@echo "  pack            Produce artifacts/"
	@echo "  pipeline        Run complete pipeline: preprocess -> train -> eval -> pack"
	@echo "  preprocess      Normalize dataset, split, pack report"
	@echo "  setup-basic     Quick setup wizard (5 essential questions only)"
	@echo "  setup-advanced  Full setup wizard (all parameters)"
	@echo "  test            Run tests"
	@echo "  train           Run QLoRA training (short run ok)"
	@echo "  venv            Create and activate virtual environment"
	@echo "  ablate-fp16     Run pipeline with precision_mode=lora_fp16 (temp patch at runtime)"
	@echo "  tokens          Print last eval's tokens/step, tok/s, VRAM_peak"
	@echo "  plan            Show training plan without executing (no training)"
	@echo "  validate        Run complete validation pipeline (no training unless TRAIN=1)"
	@echo "  cli-help        Show CLI help"
	@echo ""
	@echo "Quick start: make venv && make install && make gpu && make model"
	@echo ""
	@echo "💡 All commands now delegate to the CLI. Use 'make cli-help' for detailed CLI options."

# Virtual environment
venv:
	@echo "🐍 Creating virtual environment..."
	python3 -m venv venv
	@echo "✅ Virtual environment created. Activate with: source venv/bin/activate"

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	pip install -e .
	pip install -r requirements.txt
	@echo "✅ Dependencies installed"
	@echo "🚀 Next: make gpu"

# GPU check
gpu:
	@echo "🔍 Checking GPU and CUDA availability..."
	python3 scripts/check_gpu.py

# Download model
model:
	@echo "📥 Downloading Qwen2.5-0.5B model..."
	python3 scripts/download_model.py

# Download sample data
data-download:
	@echo "📊 Downloading sample OpenAssistant data..."
	@mkdir -p data/raw
	@if [ ! -f data/raw/oa.jsonl ]; then \
		echo "Downloading OpenAssistant dataset..."; \
		wget -O data/raw/oa.jsonl https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/data/train-00000-of-00001-abc123.jsonl; \
	else \
		echo "Sample data already exists"; \
	fi

# Preprocess data
preprocess:
	@echo "🔄 Preprocessing dataset..."
	python3 -m humigence.cli preprocess --config configs/humigence.basic.json
	@echo "✅ Preprocessing complete"
	@echo "🚀 Next: make train"

# Training
train:
	@echo "🚀 Starting QLoRA training..."
	python3 -m humigence.cli train --config configs/humigence.basic.json --train
	@echo "✅ Training complete"
	@echo "🚀 Next: make eval"

# Evaluation
eval:
	@echo "📊 Running evaluation..."
	python3 -m humigence.cli eval --config configs/humigence.basic.json
	@echo "✅ Evaluation complete"
	@echo "🚀 Next: make pack"

# Pack artifacts
pack:
	@echo "📦 Packing model artifacts..."
	python3 -m humigence.cli pack --config configs/humigence.basic.json
	@echo "✅ Packing complete"
	@echo "🚀 Next: make infer"

# Inference
infer:
	@echo "🤖 Running inference..."
	python3 -m humigence.cli infer --config configs/humigence.basic.json "Hello, how are you?"
	@echo "✅ Inference complete"

# Setup wizards
setup-basic:
	@echo "⚡ Running Basic Setup Wizard (5 essential questions only)..."
	python3 -m humigence.cli init --mode basic
	@echo "✅ Basic setup complete"

setup-advanced:
	@echo "🔧 Running Advanced Setup Wizard (full control)..."
	python3 -m humigence.cli init --mode advanced
	@echo "✅ Advanced setup complete"

# Pipeline
pipeline:
	@echo "🚀 Running complete pipeline..."
	python3 -m humigence.cli pipeline --config configs/humigence.basic.json --train
	@echo "✅ Pipeline complete"

# Ablate with FP16 precision
ablate-fp16:
	@echo "🔬 Running ablation study with FP16 precision..."
	@python3 -m humigence.cli config set train.precision_mode lora_fp16
	@echo "🔄 Running pipeline with FP16..."
	@TRAIN=1 make pipeline
	@echo "✅ FP16 ablation complete"

# Show token metrics
tokens:
	@echo "📊 Last evaluation metrics:"
	python3 -m humigence.cli tokens

# Run tests
test:
	@echo "🧪 Running tests..."
	pytest tests/ -v
	@echo "✅ Tests complete"

# Code formatting
format:
	@echo "🎨 Formatting code..."
	black .
	ruff check --fix .
	@echo "✅ Code formatted"

# Linting
lint:
	@echo "Running linting checks..."
	ruff check .
	black --check .
	@echo "✅ Linting complete"

# Plan training (no execution)
plan:
	@echo "📋 Creating training plan..."
	python3 -m humigence.cli plan --config configs/humigence.basic.json
	@echo "✅ Planning complete"

# Run validation pipeline
validate:
	@echo "🔍 Running validation pipeline..."
	@echo "💡 Set TRAIN=1 to enable training: TRAIN=1 make validate"
	@if [ "$(TRAIN)" = "1" ]; then \
		python3 -m humigence.cli validate --config configs/humigence.basic.json --train; \
	else \
		python3 -m humigence.cli validate --config configs/humigence.basic.json; \
	fi
	@echo "✅ Validation complete"

# Show CLI help
cli-help:
	@echo "🔧 Humigence CLI Help"
	@echo ""
	python3 -m humigence.cli --help

# Clean up
clean:
	@echo "🧹 Cleaning up..."
	rm -rf runs/
	rm -rf temp/
	rm -rf tmp/
	@echo "✅ Cleanup complete"
	@echo "💡 artifacts/ directory preserved"
