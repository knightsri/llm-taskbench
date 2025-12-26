"""
Cost tracking and calculation for LLM evaluations.

This module provides functionality to calculate costs based on token usage
and track cumulative costs across multiple evaluations.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from taskbench.core.models import EvaluationResult, ModelConfig

logger = logging.getLogger(__name__)


class CostTracker:
    """
    Calculate and track costs for LLM evaluations.

    Loads model pricing information and calculates costs based on token usage.
    Maintains running totals for cost analysis.

    Example:
        ```python
        tracker = CostTracker("config/models.yaml")
        cost = tracker.calculate_cost("anthropic/claude-sonnet-4.5", 1000, 500)
        print(f"Cost: ${cost:.4f}")

        tracker.track_evaluation(result)
        print(f"Total cost: ${tracker.get_total_cost():.2f}")
        ```
    """

    def __init__(self, models_config_path: str = "config/models.yaml"):
        """
        Initialize the cost tracker.

        Args:
            models_config_path: Path to YAML file containing model pricing

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid
        """
        self.models_config_path = models_config_path
        self.models: Dict[str, ModelConfig] = {}
        self.evaluations: List[EvaluationResult] = []
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0

        # Load model pricing
        self._load_models_config()

    def _load_models_config(self) -> None:
        """Load model pricing from YAML configuration."""
        path = Path(self.models_config_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Models configuration file not found: {self.models_config_path}"
            )

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            models_list = data.get("models", [])

            for model_data in models_list:
                model_config = ModelConfig(**model_data)
                self.models[model_config.model_id] = model_config

            logger.info(f"Loaded pricing for {len(self.models)} models")

        except Exception as e:
            raise ValueError(
                f"Failed to load models configuration: {str(e)}"
            ) from e

    def calculate_cost(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        inline_cost: Optional[float] = None
    ) -> float:
        """
        Calculate cost for a specific API call.

        Args:
            model_id: Model identifier (e.g., "anthropic/claude-sonnet-4.5")
            input_tokens: Number of input tokens consumed
            output_tokens: Number of output tokens generated

        Returns:
            Cost in USD, rounded to $0.01 precision

        Raises:
            ValueError: If model_id is not found in pricing database

        Example:
            ```python
            cost = tracker.calculate_cost(
                "anthropic/claude-sonnet-4.5",
                input_tokens=1000,
                output_tokens=500
            )
            # cost = (1000/1M * $3.00) + (500/1M * $15.00) = $0.0105
            ```
        """
        if inline_cost is not None:
            return round(float(inline_cost), 4)

        if model_id not in self.models:
            raise ValueError(
                f"Model '{model_id}' not found in pricing database. "
                f"Available models: {', '.join(self.models.keys())}"
            )

        model = self.models[model_id]

        input_cost = (input_tokens / 1_000_000) * model.input_price_per_1m
        output_cost = (output_tokens / 1_000_000) * model.output_price_per_1m
        total_cost = input_cost + output_cost

        return round(total_cost, 4)

    def track_evaluation(self, result: EvaluationResult) -> None:
        """
        Track an evaluation result for cost analysis.

        Args:
            result: EvaluationResult to track
        """
        self.evaluations.append(result)
        self.total_input_tokens += result.input_tokens
        self.total_output_tokens += result.output_tokens
        logger.debug(
            f"Tracked evaluation: {result.model_name}, "
            f"cost=${result.cost_usd:.4f}"
        )

    def get_total_cost(self) -> float:
        """
        Get total cost of all tracked evaluations.

        Returns:
            Total cost in USD
        """
        return round(sum(eval.cost_usd for eval in self.evaluations), 2)

    def get_cost_breakdown(self) -> Dict[str, float]:
        """
        Get per-model cost breakdown.

        Returns:
            Dictionary mapping model names to their total costs and token usage

        Example:
            ```python
            breakdown = tracker.get_cost_breakdown()
            # {"claude-sonnet-4.5": 0.36, "gpt-4o": 0.42}
            ```
        """
        breakdown: Dict[str, Dict[str, Any]] = {}

        for eval in self.evaluations:
            model_name = eval.model_name
            if model_name not in breakdown:
                breakdown[model_name] = {
                    "cost": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "evaluations": 0
                }
            breakdown[model_name]["cost"] += eval.cost_usd
            breakdown[model_name]["input_tokens"] += eval.input_tokens
            breakdown[model_name]["output_tokens"] += eval.output_tokens
            breakdown[model_name]["evaluations"] += 1

        # Round cost values
        return {k: {**v, "cost": round(v["cost"], 4)} for k, v in breakdown.items()}

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive cost statistics.

        Returns:
            Dictionary with total cost, tokens, averages, and breakdown

        Example:
            ```python
            stats = tracker.get_statistics()
            print(f"Total cost: ${stats['total_cost']:.2f}")
            print(f"Average per eval: ${stats['avg_cost_per_eval']:.2f}")
            ```
        """
        if not self.evaluations:
            return {
                "total_cost": 0.0,
                "total_tokens": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_evaluations": 0,
                "avg_cost_per_eval": 0.0,
                "avg_tokens_per_eval": 0,
                "cost_by_model": {}
            }

        total_tokens = sum(eval.total_tokens for eval in self.evaluations)
        total_cost = self.get_total_cost()
        num_evals = len(self.evaluations)

        return {
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_evaluations": num_evals,
            "avg_cost_per_eval": round(total_cost / num_evals, 4),
            "avg_tokens_per_eval": int(total_tokens / num_evals),
            "cost_by_model": self.get_cost_breakdown()
        }

    def get_model_config(self, model_id: str) -> Optional[ModelConfig]:
        """
        Get configuration for a specific model.

        Args:
            model_id: Model identifier

        Returns:
            ModelConfig if found, None otherwise
        """
        return self.models.get(model_id)

    def list_models(self) -> List[ModelConfig]:
        """
        Get list of all available models.

        Returns:
            List of ModelConfig objects
        """
        return list(self.models.values())
