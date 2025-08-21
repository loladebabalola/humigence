"""
Evaluation module for Humigence.
Handles quantitative and qualitative model evaluation.
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Config
from .templates import ChatTemplate
from .utils_logging import create_run_logger


class ModelEvaluator:
    """Handles model evaluation for Humigence."""

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set up logging
        runs_dir = self.config.get_runs_dir()
        runs_dir.mkdir(parents=True, exist_ok=True)
        self.logger = create_run_logger("humigence_eval", runs_dir)

        self.logger.info("Initializing model evaluator...")
        self._setup_model()
        self._setup_templates()

    def _setup_model(self):
        """Set up the model for evaluation."""
        self.logger.info("Loading model for evaluation...")

        # Load base model
        model_path = self.config.get_model_path()
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Load LoRA adapter
        runs_dir = self.config.get_runs_dir()
        if (runs_dir / "adapter_config.json").exists():
            self.model = PeftModel.from_pretrained(
                self.base_model, runs_dir, torch_dtype=torch.float16
            )
            self.logger.info("Loaded LoRA adapter from training run")
        else:
            self.model = self.base_model
            self.logger.warning("No LoRA adapter found, using base model")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.logger.info("Model setup completed")

    def _setup_templates(self):
        """Set up chat templates for evaluation."""
        self.chat_template = ChatTemplate()

    def evaluate_model(self) -> dict:
        """Run comprehensive model evaluation."""
        self.logger.info("Starting model evaluation...")

        results = {}

        # Quantitative evaluation
        if self.config.eval.primary_metric == "val_loss":
            val_loss = self._evaluate_validation_loss()
            results["validation_loss"] = val_loss

        # Qualitative evaluation with curated prompts
        generation_results = self._evaluate_generations()
        results["generations"] = generation_results

        # Save evaluation results
        self._save_evaluation_results(results)

        self.logger.info("Model evaluation completed!")
        return results

    def _evaluate_validation_loss(self) -> float:
        """Evaluate validation loss on the validation set."""
        self.logger.info("Evaluating validation loss...")

        # Load validation data
        data_paths = self.config.get_data_paths()
        val_file = data_paths["val"]

        if not val_file.exists():
            self.logger.warning(
                "Validation file not found, skipping validation loss evaluation"
            )
            return 0.0

        # Load and prepare validation data
        val_data = self._load_jsonl_data(val_file)

        total_loss = 0.0
        total_tokens = 0

        self.model.eval()
        with torch.no_grad():
            for item in val_data:
                text = item.get("text", "")
                target = item.get("target", "")

                # Combine input and target
                full_text = text + target

                # Tokenize
                inputs = self.tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.data.max_seq_len,
                ).to(self.device)

                # Calculate loss
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss

                total_loss += loss.item() * inputs["input_ids"].numel()
                total_tokens += inputs["input_ids"].numel()

        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        self.logger.info(f"Validation loss: {avg_loss:.4f}")

        return avg_loss

    def _evaluate_generations(self) -> list:
        """Evaluate model generations on curated prompts."""
        self.logger.info("Evaluating model generations...")

        # Load validation data for generation evaluation
        data_paths = self.config.get_data_paths()
        val_data = self._load_jsonl_data(data_paths["val"])

        # Sample a few examples for generation evaluation
        sample_size = min(5, len(val_data))
        sample_data = val_data[:sample_size]

        generation_results = []
        for i, example in enumerate(sample_data):
            try:
                # Extract prompt from the example
                if "text" in example:
                    prompt = example["text"]
                elif "messages" in example:
                    # Handle chat format
                    messages = example["messages"]
                    if messages and len(messages) >= 2:
                        prompt = messages[-2]["content"]  # Second to last message
                    else:
                        prompt = "Hello, how are you?"
                else:
                    prompt = "Hello, how are you?"

                # Generate response
                response = self._generate_response(
                    prompt,
                    max_length=100,
                    temperature=0.7 if getattr(self.config.eval, 'sampling_enabled', False) else 0.0
                )

                generation_results.append({
                    "prompt": prompt,
                    "response": response,
                    "example_index": i,
                })

            except Exception as e:
                self.logger.warning(f"Generation failed for example {i}: {e}")
                generation_results.append({
                    "prompt": prompt if 'prompt' in locals() else "Error",
                    "response": f"Generation failed: {e}",
                    "example_index": i,
                    "error": str(e)
                })

        return generation_results

    def _generate_text(
        self, prompt: str, temperature: float = 0.7, max_length: int = 512
    ) -> str:
        """Generate text from a prompt."""
        # Format prompt using chat template
        formatted_prompt = self.chat_template.format_instruction(
            prompt, add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.data.max_seq_len,
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )

        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        return generated_text.strip()

    def _generate_response(self, prompt: str, max_length: int = 100, temperature: float = 0.0) -> str:
        """Generate a response to a prompt with non-finite detection."""
        try:
            # Encode the prompt
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            ).to(self.device)

            # Set generation parameters based on sampling toggle
            do_sample = getattr(self.config.eval, 'sampling_enabled', False) and temperature > 0.0

            # Generate with non-finite detection
            with torch.no_grad():
                # Check for non-finite values in input
                if not torch.isfinite(inputs["input_ids"]).all():
                    raise ValueError("Non-finite values detected in input tokens")

                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature if do_sample else None,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1 if do_sample else None,
                )

                # Check for non-finite values in output
                if not torch.isfinite(outputs).all():
                    raise ValueError("Non-finite values detected in generated tokens")

            # Decode the generated text
            generated_text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )

            return generated_text.strip()

        except Exception as e:
            error_msg = f"Generation failed: {e}"
            if "non-finite" in str(e).lower():
                error_msg += " - Check learning rate, warmup, or dtype settings"
            self.logger.error(error_msg)
            return error_msg

    def _load_jsonl_data(self, file_path: Path) -> list[dict]:
        """Load data from JSONL file."""
        data = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def _save_evaluation_results(self, results: dict):
        """Save evaluation results to file."""
        eval_file = self.config.get_runs_dir() / "eval_report.json"

        # Add metadata
        results["metadata"] = {
            "config": self.config.dict(),
            "evaluation_timestamp": str(Path().cwd()),
            "model_path": str(self.config.get_model_path()),
            "adapter_path": str(self.config.get_runs_dir()),
        }

        with open(eval_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Evaluation results saved to: {eval_file}")

        # Print summary
        self._print_evaluation_summary(results)

    def _print_evaluation_summary(self, results: dict):
        """Print a summary of evaluation results."""
        self.logger.info("=" * 60)
        self.logger.info("EVALUATION SUMMARY")
        self.logger.info("=" * 60)

        if "validation_loss" in results:
            self.logger.info(f"Validation Loss: {results['validation_loss']:.4f}")

        if "generations" in results:
            gens = results["generations"]
            if isinstance(gens, dict):
                gens = list(gens.values())
            elif not isinstance(gens, list):
                gens = []

            gen_count = len(gens)
            self.logger.info(f"Generation Evaluation: {gen_count} prompts")

            # Show a few examples safely
            for i, result in enumerate(gens[:3]):
                if isinstance(result, dict):
                    prompt = result.get("prompt", "Unknown")
                    response = result.get("response", "No response")
                    error = result.get("error", None)

                    prompt_preview = (
                        prompt[:100] + "..." if len(prompt) > 100 else prompt
                    )
                    response_preview = (
                        response[:100] + "..." if len(response) > 100 else response
                    )

                    self.logger.info(f"  Example {i+1}:")
                    self.logger.info(f"    Prompt: {prompt_preview}")
                    if error:
                        self.logger.info(f"    Error: {error}")
                    else:
                        self.logger.info(f"    Response: {response_preview}")
                else:
                    self.logger.info(f"  Example {i+1}: Invalid format - {type(result)}")

        self.logger.info("=" * 60)


def main():
    """Main function for the evaluation CLI."""
    parser = argparse.ArgumentParser(description="Humigence Model Evaluation")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration file"
    )
    parser.add_argument(
        "--greedy", action="store_true", help="Use greedy decoding (no sampling)"
    )

    args = parser.parse_args()

    try:
        # Load configuration
        config = Config.from_file(args.config)

        # Override sampling setting if greedy flag is set
        if args.greedy:
            if not hasattr(config.eval, 'sampling_enabled'):
                config.eval.sampling_enabled = False
            else:
                config.eval.sampling_enabled = False

        # Initialize evaluator
        evaluator = ModelEvaluator(config)

        # Run evaluation
        evaluator.evaluate_model()

    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
