"""Acceptance gates and quality checks for Humigence training."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class AcceptanceCriteria:
    """Acceptance criteria configuration."""

    min_val_improvement_pct: float = 1.0
    throughput_jitter_pct: float = 20.0
    curated_reasonable_threshold_pct: float = 70.0


@dataclass
class AcceptanceResult:
    """Result of acceptance gate evaluation."""

    passed: bool
    score: float
    details: dict
    suggestions: list[str]


class AcceptanceGates:
    """Evaluate training quality and enforce acceptance gates."""

    def __init__(self, config: dict, run_dir: Path):
        self.config = config
        self.run_dir = run_dir
        self.criteria = AcceptanceCriteria(
            min_val_improvement_pct=config.get("acceptance", {}).get(
                "min_val_improvement_pct", 1.0
            ),
            throughput_jitter_pct=config.get("acceptance", {}).get(
                "throughput_jitter_pct", 20.0
            ),
            curated_reasonable_threshold_pct=config.get("acceptance", {}).get(
                "curated_reasonable_threshold_pct", 70.0
            ),
        )

    def evaluate_training_run(self) -> AcceptanceResult:
        """Evaluate the complete training run against acceptance criteria."""
        logger.info("🔍 Evaluating training run against acceptance gates...")

        # Load metrics and evaluation results
        metrics = self._load_metrics()
        eval_results = self._load_eval_results()

        if not metrics or not eval_results:
            return AcceptanceResult(
                passed=False,
                score=0.0,
                details={"error": "Missing metrics or evaluation results"},
                suggestions=[
                    "Ensure training completed successfully and evaluation ran"
                ],
            )

        # Evaluate each gate
        val_loss_gate = self._evaluate_val_loss_gate(metrics, eval_results)
        throughput_gate = self._evaluate_throughput_gate(metrics)
        curated_gate = self._evaluate_curated_gate(eval_results)

        # Calculate overall score
        gates = [val_loss_gate, throughput_gate, curated_gate]
        passed_gates = sum(1 for gate in gates if gate["passed"])
        overall_score = passed_gates / len(gates) * 100

        # Determine if run passes
        passed = all(gate["passed"] for gate in gates)

        # Generate suggestions for failed gates
        suggestions = []
        for gate in gates:
            if not gate["passed"]:
                suggestions.extend(gate.get("suggestions", []))

        result = AcceptanceResult(
            passed=passed,
            score=overall_score,
            details={
                "val_loss_gate": val_loss_gate,
                "throughput_gate": throughput_gate,
                "curated_gate": curated_gate,
                "overall_score": overall_score,
            },
            suggestions=suggestions,
        )

        # Print acceptance report
        self._print_acceptance_report(result)

        return result

    def _evaluate_val_loss_gate(self, metrics: list[dict], eval_results: dict) -> dict:
        """Evaluate validation loss improvement gate."""
        if not metrics or len(metrics) < 2:
            return {
                "passed": False,
                "score": 0.0,
                "details": {"error": "Insufficient metrics data"},
                "suggestions": ["Ensure training runs for multiple steps"],
            }

        # Get first and last validation loss
        first_loss = None
        last_loss = None

        for metric in metrics:
            if "val_loss" in metric:
                if first_loss is None:
                    first_loss = metric["val_loss"]
                last_loss = metric["val_loss"]

        if first_loss is None or last_loss is None:
            return {
                "passed": False,
                "score": 0.0,
                "details": {"error": "No validation loss data"},
                "suggestions": ["Ensure validation loss is being computed"],
            }

        # Calculate improvement
        improvement_pct = ((first_loss - last_loss) / first_loss) * 100
        passed = improvement_pct >= self.criteria.min_val_improvement_pct

        suggestions = []
        if not passed:
            suggestions = [
                f"Validation loss improved only {improvement_pct:.1f}% (need {self.criteria.min_val_improvement_pct}%)",
                "Try increasing training steps by 50%",
                "Consider adjusting learning rate or LoRA rank",
                "Check if dataset quality is sufficient",
            ]

        return {
            "passed": passed,
            "score": min(
                improvement_pct / self.criteria.min_val_improvement_pct * 100, 100
            ),
            "details": {
                "first_loss": first_loss,
                "last_loss": last_loss,
                "improvement_pct": improvement_pct,
                "threshold": self.criteria.min_val_improvement_pct,
            },
            "suggestions": suggestions,
        }

    def _evaluate_throughput_gate(self, metrics: list[dict]) -> dict:
        """Evaluate throughput stability gate."""
        if len(metrics) < 3:
            return {
                "passed": False,
                "score": 0.0,
                "details": {"error": "Insufficient metrics for throughput analysis"},
                "suggestions": ["Ensure training runs for multiple steps"],
            }

        # Calculate throughput jitter from recent metrics
        recent_metrics = metrics[-3:]
        throughput_values = [
            m.get("tokens_per_sec", 0)
            for m in recent_metrics
            if m.get("tokens_per_sec")
        ]

        if len(throughput_values) < 2:
            return {
                "passed": False,
                "score": 0.0,
                "details": {"error": "No throughput data available"},
                "suggestions": ["Ensure telemetry is collecting throughput metrics"],
            }

        # Calculate coefficient of variation (jitter)
        mean_throughput = sum(throughput_values) / len(throughput_values)
        variance = sum((x - mean_throughput) ** 2 for x in throughput_values) / len(
            throughput_values
        )
        std_dev = variance**0.5
        jitter_pct = (std_dev / mean_throughput * 100) if mean_throughput > 0 else 0

        passed = jitter_pct <= self.criteria.throughput_jitter_pct

        suggestions = []
        if not passed:
            suggestions = [
                f"Throughput jitter is {jitter_pct:.1f}% (threshold: {self.criteria.throughput_jitter_pct}%)",
                "Check for system resource contention",
                "Consider reducing batch size for stability",
                "Monitor GPU temperature and power limits",
            ]

        return {
            "passed": passed,
            "score": max(
                0, 100 - (jitter_pct / self.criteria.throughput_jitter_pct * 100)
            ),
            "details": {
                "throughput_values": throughput_values,
                "mean_throughput": mean_throughput,
                "jitter_pct": jitter_pct,
                "threshold": self.criteria.throughput_jitter_pct,
            },
            "suggestions": suggestions,
        }

    def _evaluate_curated_gate(self, eval_results: dict) -> dict:
        """Evaluate curated evaluation quality gate."""
        if "curated_eval" not in eval_results:
            return {
                "passed": False,
                "score": 0.0,
                "details": {"error": "No curated evaluation results"},
                "suggestions": ["Run evaluation with curated prompts"],
            }

        curated_results = eval_results["curated_eval"]
        if not curated_results:
            return {
                "passed": False,
                "score": 0.0,
                "details": {"error": "Empty curated evaluation results"},
                "suggestions": ["Ensure evaluation generates responses"],
            }

        # Simple heuristic scoring
        reasonable_count = 0
        total_count = len(curated_results)

        for result in curated_results:
            if self._is_reasonable_response(result):
                reasonable_count += 1

        reasonable_pct = (reasonable_count / total_count) * 100
        passed = reasonable_pct >= self.criteria.curated_reasonable_threshold_pct

        suggestions = []
        if not passed:
            suggestions = [
                f"Only {reasonable_pct:.1f}% of responses are reasonable (threshold: {self.criteria.curated_reasonable_threshold_pct}%)",
                "Consider training for more steps",
                "Check if model is learning the task",
                "Review dataset quality and formatting",
            ]

        return {
            "passed": passed,
            "score": reasonable_pct,
            "details": {
                "reasonable_count": reasonable_count,
                "total_count": total_count,
                "reasonable_pct": reasonable_pct,
                "threshold": self.criteria.curated_reasonable_threshold_pct,
            },
            "suggestions": suggestions,
        }

    def _is_reasonable_response(self, result: dict) -> bool:
        """Simple heuristic to determine if a response is reasonable."""
        response = result.get("response", "")

        # Basic checks
        if not response or len(response.strip()) < 10:
            return False

        # Check for template artifacts (repeated tokens)
        words = response.split()
        if len(words) > 20:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
                if word_counts[word] > len(words) * 0.3:  # More than 30% repetition
                    return False

        # Check for very short or very long responses (relative to prompt)
        prompt = result.get("prompt", "")
        if len(response) < len(prompt) * 0.1:  # Response too short
            return False
        if len(response) > len(prompt) * 10:  # Response too long
            return False

        return True

    def _load_metrics(self) -> list[dict]:
        """Load training metrics from JSONL file."""
        metrics_file = self.run_dir / "metrics.jsonl"
        if not metrics_file.exists():
            return []

        metrics = []
        try:
            with open(metrics_file) as f:
                for line in f:
                    if line.strip():
                        metrics.append(json.loads(line))
        except Exception as e:
            logger.error(f"Failed to load metrics: {e}")

        return metrics

    def _load_eval_results(self) -> dict:
        """Load evaluation results."""
        eval_file = self.run_dir / "eval_report.json"
        if not eval_file.exists():
            return {}

        try:
            with open(eval_file) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load evaluation results: {e}")
            return {}

    def _print_acceptance_report(self, result: AcceptanceResult) -> None:
        """Print formatted acceptance report."""
        console.print("\n" + "=" * 80)
        console.print("🎯 ACCEPTANCE GATES REPORT")
        console.print("=" * 80)

        # Overall result
        status_style = "green" if result.passed else "red"
        status_icon = "✅" if result.passed else "❌"
        console.print(
            f"{status_icon} Overall Result: {'PASSED' if result.passed else 'FAILED'}",
            style=status_style,
        )
        console.print(f"📊 Overall Score: {result.score:.1f}%")

        # Individual gate results
        details = result.details
        for gate_name, gate_result in details.items():
            if gate_name == "overall_score":
                continue

            gate_passed = gate_result.get("passed", False)
            gate_score = gate_result.get("score", 0)
            gate_icon = "✅" if gate_passed else "❌"

            console.print(
                f"\n{gate_icon} {gate_name.replace('_', ' ').title()}: {'PASSED' if gate_passed else 'FAILED'}"
            )
            console.print(f"   Score: {gate_score:.1f}%")

            # Show suggestions for failed gates
            if not gate_passed and gate_result.get("suggestions"):
                console.print("   💡 Suggestions:")
                for suggestion in gate_result["suggestions"]:
                    console.print(f"      • {suggestion}")

        # Final suggestions
        if result.suggestions:
            console.print("\n🔧 Remediation Steps:")
            for suggestion in result.suggestions:
                console.print(f"   • {suggestion}")

        console.print("=" * 80)

        # Save acceptance report
        acceptance_file = self.run_dir / "acceptance_report.json"
        try:
            with open(acceptance_file, "w") as f:
                json.dump(
                    {
                        "passed": result.passed,
                        "score": result.score,
                        "details": result.details,
                        "suggestions": result.suggestions,
                        "criteria": {
                            "min_val_improvement_pct": self.criteria.min_val_improvement_pct,
                            "throughput_jitter_pct": self.criteria.throughput_jitter_pct,
                            "curated_reasonable_threshold_pct": self.criteria.curated_reasonable_threshold_pct,
                        },
                    },
                    f,
                    indent=2,
                )
            logger.info(f"Acceptance report saved to: {acceptance_file}")
        except Exception as e:
            logger.error(f"Failed to save acceptance report: {e}")
