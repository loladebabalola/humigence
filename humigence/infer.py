"""
Inference module for Humigence.
Handles single-prompt inference using trained models.
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
from .utils_logging import setup_logging


class ModelInferencer:
    """Handles model inference for Humigence."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.logger.info("Initializing model inferencer...")
        self._setup_model()
        self._setup_templates()

    def _setup_model(self):
        """Set up the model for inference."""
        self.logger.info("Loading model for inference...")

        # Load base model
        model_path = self.config.get_model_path()
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Load LoRA adapter from artifacts
        artifacts_dir = self.config.get_artifacts_dir()
        if (artifacts_dir / "adapter_config.json").exists():
            self.model = PeftModel.from_pretrained(
                self.base_model, artifacts_dir, torch_dtype=torch.float16
            )
            self.logger.info("Loaded LoRA adapter from artifacts")
        else:
            # Fallback to training run directory
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
        """Set up chat templates for inference."""
        self.chat_template = ChatTemplate()

    def generate_response(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """Generate a response to the given prompt."""
        self.logger.info(f"Generating response for prompt: {prompt[:100]}...")

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
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                length_penalty=1.0,
            )

        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        return generated_text.strip()

    def interactive_mode(self):
        """Run interactive inference mode."""
        self.logger.info("Starting interactive mode...")
        print("\n" + "=" * 60)
        print("Humigence Interactive Inference Mode")
        print("Type 'quit' to exit, 'help' for options")
        print("=" * 60)

        while True:
            try:
                user_input = input("\n🤔 You: ").strip()

                if user_input.lower() in ["quit", "exit", "q"]:
                    print("👋 Goodbye!")
                    break

                if user_input.lower() in ["help", "h"]:
                    self._show_help()
                    continue

                if not user_input:
                    continue

                # Generate response
                response = self.generate_response(user_input)
                print(f"\n🤖 Assistant: {response}")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                self.logger.error(f"Error during inference: {e}")
                print(f"\n❌ Error: {e}")

    def _show_help(self):
        """Show help information."""
        help_text = """
Available commands:
- quit/exit/q: Exit interactive mode
- help/h: Show this help message

Generation parameters can be adjusted in the configuration file.
"""
        print(help_text)

    def batch_inference(self, prompts: list, output_file: Path | None = None) -> list:
        """Run inference on a batch of prompts."""
        self.logger.info(f"Running batch inference on {len(prompts)} prompts...")

        results = []

        for i, prompt in enumerate(prompts):
            try:
                response = self.generate_response(prompt)
                results.append({"prompt": prompt, "response": response, "index": i})

                if (i + 1) % 10 == 0:
                    self.logger.info(f"Processed {i + 1}/{len(prompts)} prompts")

            except Exception as e:
                self.logger.error(f"Error processing prompt {i}: {e}")
                results.append(
                    {"prompt": prompt, "response": f"ERROR: {e}", "index": i}
                )

        # Save results if output file specified
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Batch results saved to: {output_file}")

        return results


def main():
    """Main function for the inference CLI."""
    parser = argparse.ArgumentParser(description="Humigence Model Inference")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration file"
    )
    parser.add_argument("--prompt", type=str, help="Single prompt for inference")
    parser.add_argument(
        "--interactive", action="store_true", help="Run in interactive mode"
    )
    parser.add_argument(
        "--batch-file", type=str, help="Path to file containing prompts (one per line)"
    )
    parser.add_argument("--output", type=str, help="Output file for batch results")
    parser.add_argument(
        "--max-length", type=int, default=512, help="Maximum generation length"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Generation temperature"
    )

    args = parser.parse_args()

    try:
        # Load configuration
        config = Config.from_file(args.config)

        # Initialize inferencer
        inferencer = ModelInferencer(config)

        # Run inference based on arguments
        if args.interactive:
            inferencer.interactive_mode()

        elif args.prompt:
            response = inferencer.generate_response(
                args.prompt, max_length=args.max_length, temperature=args.temperature
            )
            print(f"\nPrompt: {args.prompt}")
            print(f"Response: {response}")

        elif args.batch_file:
            batch_file = Path(args.batch_file)
            if not batch_file.exists():
                raise FileNotFoundError(f"Batch file not found: {batch_file}")

            # Load prompts
            with open(batch_file, encoding="utf-8") as f:
                prompts = [line.strip() for line in f if line.strip()]

            # Run batch inference
            output_file = Path(args.output) if args.output else None
            results = inferencer.batch_inference(prompts, output_file)

            print(f"\nProcessed {len(results)} prompts")
            if not output_file:
                print("\nFirst few results:")
                for result in results[:3]:
                    print(f"  Prompt: {result['prompt'][:100]}...")
                    print(f"  Response: {result['response'][:100]}...")
                    print()

        else:
            # Default to interactive mode
            inferencer.interactive_mode()

    except Exception as e:
        logging.error(f"Inference failed: {e}")
        raise


if __name__ == "__main__":
    main()
