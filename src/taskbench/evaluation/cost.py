"""
Cost calculation and tracking for LLM TaskBench.

This module provides utilities for calculating and tracking costs of LLM
API usage based on token consumption and model pricing.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from taskbench.core.models import EvaluationResult, ModelConfig

logger = logging.getLogger(__name__)


class CostTracker:
    """
    Track and calculate costs for LLM API usage.

    This class loads model pricing from a YAML configuration file and provides
    methods to calculate costs based on token usage.

    Example:
        >>> tracker = CostTracker()
        >>> cost = tracker.calculate_cost("anthropic/claude-sonnet-4.5", 1000, 500)
        >>> print(f"Cost: ${cost:.4f}")
        >>> tracker.track_evaluation(eval_result)
        >>> total = tracker.get_total_cost()
        >>> print(f"Total cost: ${total:.2f}")
    """

    def __init__(self, models_config_path: Optional[str] = None):
        """
        Initialize the cost tracker.

        Args:
            models_config_path: Path to models.yaml configuration file.
                If not provided, looks for config/models.yaml relative to project root.

        Raises:
            FileNotFoundError: If the models configuration file is not found
            ValueError: If the configuration file is malformed
        """
        # Determine config path
        if models_config_path is None:
            # Try to find config/models.yaml relative to this file
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent.parent
            models_config_path = project_root / "config" / "models.yaml"
        else:
            models_config_path = Path(models_config_path)

        # Load model pricing
        self.models: Dict[str, ModelConfig] = {}
        self._load_models(models_config_path)

        # Track evaluations
        self.evaluations: List[EvaluationResult] = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0

        logger.info(f"CostTracker initialized with {len(self.models)} models")

    def _load_models(self, config_path: Path) -> None:
        """
        Load model configurations from YAML file.

        Args:
            config_path: Path to the models YAML configuration

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is malformed
        """
        if not config_path.exists():
            raise FileNotFoundError(
                f"Models configuration file not found: {config_path}. "
                "Please ensure config/models.yaml exists."
            )

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse models configuration: {e}") from e

        # Validate structure
        if not isinstance(data, dict) or 'models' not in data:
            raise ValueError(
                "Invalid models configuration: must contain 'models' key with list of models"
            )

        # Load each model
        models_list = data['models']
        if not isinstance(models_list, list):
            raise ValueError("'models' must be a list")

        for model_data in models_list:
            try:
                model_config = ModelConfig(**model_data)
                self.models[model_config.model_id] = model_config
                logger.debug(f"Loaded pricing for {model_config.model_id}")
            except Exception as e:
                logger.warning(f"Failed to load model config: {e}")
                continue

        if not self.models:
            raise ValueError("No valid model configurations found")

        logger.info(f"Loaded {len(self.models)} model configurations from {config_path}")

    def get_model_config(self, model_id: str) -> Optional[ModelConfig]:
        """
        Get the configuration for a specific model.

        Args:
            model_id: Model identifier

        Returns:
            ModelConfig object if found, None otherwise

        Example:
            >>> tracker = CostTracker()
            >>> config = tracker.get_model_config("anthropic/claude-sonnet-4.5")
            >>> if config:
            ...     print(f"Input price: ${config.input_price_per_1m}")
        """
        return self.models.get(model_id)

    def calculate_cost(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """
        Calculate cost for a given number of tokens.

        Args:
            model_id: Model identifier (e.g., "anthropic/claude-sonnet-4.5")
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Total cost in USD, rounded to $0.01 precision

        Raises:
            ValueError: If model_id is not found in configuration

        Example:
            >>> tracker = CostTracker()
            >>> cost = tracker.calculate_cost("anthropic/claude-sonnet-4.5", 1000, 500)
            >>> print(f"Cost: ${cost:.2f}")
        """
        model_config = self.models.get(model_id)

        if model_config is None:
            raise ValueError(
                f"Model '{model_id}' not found in configuration. "
                f"Available models: {list(self.models.keys())}"
            )

        # Calculate cost using ModelConfig method
        cost = model_config.calculate_cost(input_tokens, output_tokens)

        logger.debug(
            f"Cost calculation: model={model_id}, input={input_tokens}, "
            f"output={output_tokens}, cost=${cost:.4f}"
        )

        return cost

    def track_evaluation(self, evaluation: EvaluationResult) -> None:
        """
        Track an evaluation result and update cost statistics.

        Args:
            evaluation: EvaluationResult to track

        Example:
            >>> tracker = CostTracker()
            >>> result = EvaluationResult(...)
            >>> tracker.track_evaluation(result)
            >>> print(f"Total tracked: ${tracker.get_total_cost():.2f}")
        """
        self.evaluations.append(evaluation)
        self.total_input_tokens += evaluation.input_tokens
        self.total_output_tokens += evaluation.output_tokens
        self.total_cost += evaluation.cost_usd

        logger.debug(
            f"Tracked evaluation: model={evaluation.model_name}, "
            f"cost=${evaluation.cost_usd:.4f}, "
            f"total_cost=${self.total_cost:.4f}"
        )

    def get_total_cost(self) -> float:
        """
        Get the total cost of all tracked evaluations.

        Returns:
            Total cost in USD, rounded to $0.01 precision

        Example:
            >>> tracker = CostTracker()
            >>> # ... track some evaluations ...
            >>> total = tracker.get_total_cost()
            >>> print(f"Total spent: ${total:.2f}")
        """
        return round(self.total_cost, 2)

    def get_cost_breakdown(self) -> Dict[str, Dict[str, float]]:
        """
        Get cost breakdown by model.

        Returns:
            Dictionary mapping model names to their cost statistics:
            {
                "model_name": {
                    "cost": total_cost,
                    "input_tokens": total_input_tokens,
                    "output_tokens": total_output_tokens,
                    "evaluations": count
                }
            }

        Example:
            >>> tracker = CostTracker()
            >>> # ... track some evaluations ...
            >>> breakdown = tracker.get_cost_breakdown()
            >>> for model, stats in breakdown.items():
            ...     print(f"{model}: ${stats['cost']:.2f}")
        """
        breakdown: Dict[str, Dict[str, float]] = {}

        for eval_result in self.evaluations:
            model = eval_result.model_name

            if model not in breakdown:
                breakdown[model] = {
                    "cost": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "evaluations": 0
                }

            breakdown[model]["cost"] += eval_result.cost_usd
            breakdown[model]["input_tokens"] += eval_result.input_tokens
            breakdown[model]["output_tokens"] += eval_result.output_tokens
            breakdown[model]["evaluations"] += 1

        # Round costs to 2 decimal places
        for model in breakdown:
            breakdown[model]["cost"] = round(breakdown[model]["cost"], 2)

        return breakdown

    def get_statistics(self) -> Dict[str, any]:
        """
        Get comprehensive cost statistics.

        Returns:
            Dictionary containing:
            - total_cost: Total cost in USD
            - total_evaluations: Number of evaluations tracked
            - total_input_tokens: Total input tokens
            - total_output_tokens: Total output tokens
            - total_tokens: Total tokens (input + output)
            - average_cost_per_evaluation: Average cost per evaluation
            - models_used: Number of unique models used
            - breakdown: Cost breakdown by model

        Example:
            >>> tracker = CostTracker()
            >>> # ... track some evaluations ...
            >>> stats = tracker.get_statistics()
            >>> print(f"Total: ${stats['total_cost']:.2f}")
            >>> print(f"Evaluations: {stats['total_evaluations']}")
            >>> print(f"Avg per eval: ${stats['average_cost_per_evaluation']:.4f}")
        """
        total_evaluations = len(self.evaluations)
        total_tokens = self.total_input_tokens + self.total_output_tokens

        # Calculate average cost per evaluation
        avg_cost = (
            round(self.total_cost / total_evaluations, 4)
            if total_evaluations > 0
            else 0.0
        )

        # Get unique models
        models_used = len(set(eval.model_name for eval in self.evaluations))

        statistics = {
            "total_cost": round(self.total_cost, 2),
            "total_evaluations": total_evaluations,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": total_tokens,
            "average_cost_per_evaluation": avg_cost,
            "models_used": models_used,
            "breakdown": self.get_cost_breakdown()
        }

        return statistics

    def reset(self) -> None:
        """
        Reset all tracked evaluations and statistics.

        Example:
            >>> tracker = CostTracker()
            >>> # ... track some evaluations ...
            >>> tracker.reset()  # Clear all tracked data
            >>> assert tracker.get_total_cost() == 0.0
        """
        self.evaluations.clear()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        logger.info("CostTracker reset: all statistics cleared")

    def export_summary(self) -> str:
        """
        Export a formatted summary of cost statistics.

        Returns:
            Formatted string with cost summary

        Example:
            >>> tracker = CostTracker()
            >>> # ... track some evaluations ...
            >>> print(tracker.export_summary())
        """
        stats = self.get_statistics()

        summary = []
        summary.append("=" * 60)
        summary.append("COST SUMMARY")
        summary.append("=" * 60)
        summary.append(f"Total Cost:           ${stats['total_cost']:.2f}")
        summary.append(f"Total Evaluations:    {stats['total_evaluations']}")
        summary.append(f"Average Cost/Eval:    ${stats['average_cost_per_evaluation']:.4f}")
        summary.append(f"Models Used:          {stats['models_used']}")
        summary.append(f"Total Tokens:         {stats['total_tokens']:,}")
        summary.append(f"  - Input:            {stats['total_input_tokens']:,}")
        summary.append(f"  - Output:           {stats['total_output_tokens']:,}")
        summary.append("")
        summary.append("BREAKDOWN BY MODEL:")
        summary.append("-" * 60)

        breakdown = stats['breakdown']
        for model, model_stats in sorted(breakdown.items(), key=lambda x: x[1]['cost'], reverse=True):
            summary.append(f"\n{model}:")
            summary.append(f"  Cost:         ${model_stats['cost']:.2f}")
            summary.append(f"  Evaluations:  {int(model_stats['evaluations'])}")
            summary.append(f"  Tokens:       {int(model_stats['input_tokens'] + model_stats['output_tokens']):,}")

        summary.append("=" * 60)

        return "\n".join(summary)

    def __str__(self) -> str:
        """String representation of CostTracker."""
        return (
            f"CostTracker(models={len(self.models)}, "
            f"evaluations={len(self.evaluations)}, "
            f"total_cost=${self.total_cost:.2f})"
        )

    def __repr__(self) -> str:
        """Repr of CostTracker."""
        return self.__str__()
