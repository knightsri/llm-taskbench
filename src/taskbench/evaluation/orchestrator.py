"""
LLM Orchestrator for LLM TaskBench.

This module provides the LLMOrchestrator class for intelligently selecting
models based on task characteristics and budget constraints.
"""

import logging
from typing import List, Optional

from taskbench.api.client import OpenRouterClient
from taskbench.core.models import TaskDefinition
from taskbench.evaluation.cost import CostTracker

logger = logging.getLogger(__name__)


class LLMOrchestrator:
    """
    Intelligent model selection orchestrator.

    This class analyzes task definitions and budget constraints to suggest
    the most appropriate models for evaluation. It uses simple heuristics
    based on task characteristics like input type, output format, and constraints.

    This is a simple rule-based implementation suitable for MVP. Future versions
    could leverage ML-based recommendations or historical performance data.

    Example:
        >>> from taskbench.api.client import OpenRouterClient
        >>> from taskbench.core.task import TaskParser
        >>>
        >>> client = OpenRouterClient()
        >>> orchestrator = LLMOrchestrator(client)
        >>>
        >>> # Load task definition
        >>> parser = TaskParser()
        >>> task = parser.load_from_yaml("tasks/lecture_analysis.yaml")
        >>>
        >>> # Get recommended models
        >>> model_ids = await orchestrator.create_evaluation_plan(task, budget=0.50)
        >>> print(f"Recommended models: {model_ids}")
    """

    # Default model recommendations for different scenarios
    DEFAULT_MODELS = [
        "anthropic/claude-sonnet-4.5",
        "openai/gpt-4o",
        "google/gemini-2.0-flash-exp",
    ]

    # Models with large context windows (good for transcripts, long documents)
    LARGE_CONTEXT_MODELS = [
        "anthropic/claude-sonnet-4.5",  # 200k context
        "google/gemini-2.0-flash-exp",  # 1M context
        "openai/gpt-4-turbo",           # 128k context
    ]

    # Budget-friendly models
    BUDGET_MODELS = [
        "google/gemini-2.0-flash-exp",
        "anthropic/claude-haiku-3.5",
        "openai/gpt-4o-mini",
    ]

    # High-quality models for production
    PREMIUM_MODELS = [
        "anthropic/claude-sonnet-4.5",
        "openai/gpt-4o",
        "google/gemini-pro-1.5",
    ]

    def __init__(self, api_client: OpenRouterClient):
        """
        Initialize the LLM orchestrator.

        Args:
            api_client: OpenRouterClient instance for API calls

        Example:
            >>> client = OpenRouterClient()
            >>> orchestrator = LLMOrchestrator(client)
        """
        self.api_client = api_client
        self.cost_tracker = CostTracker()
        logger.info("LLMOrchestrator initialized")

    async def create_evaluation_plan(
        self,
        task: TaskDefinition,
        budget: Optional[float] = None
    ) -> List[str]:
        """
        Create an evaluation plan by suggesting appropriate models.

        This method analyzes the task definition and budget constraints to
        recommend a list of models that are well-suited for the task. It uses
        simple heuristics based on:
        - Input type (e.g., transcript ’ large context models)
        - Output format (e.g., JSON ’ models with good structured output)
        - Constraints (e.g., duration limits ’ fast models)
        - Budget (e.g., low budget ’ cheaper models)

        Args:
            task: TaskDefinition containing task specifications
            budget: Optional maximum budget per evaluation in USD

        Returns:
            List of recommended model IDs, sorted by suitability

        Example:
            >>> orchestrator = LLMOrchestrator(client)
            >>> task = TaskDefinition(
            ...     name="lecture_analysis",
            ...     input_type="transcript",
            ...     output_format="csv",
            ...     ...
            ... )
            >>> models = await orchestrator.create_evaluation_plan(task, budget=0.10)
            >>> print(f"Recommended: {models}")
        """
        logger.info(
            f"Creating evaluation plan for task '{task.name}' "
            f"(input={task.input_type}, output={task.output_format}, budget={budget})"
        )

        recommended_models = []

        # Heuristic 1: Check input type
        if task.input_type == "transcript":
            # Transcripts are typically long, need large context windows
            logger.debug("Task uses transcript input, preferring large context models")
            recommended_models.extend(self.LARGE_CONTEXT_MODELS)

        elif task.input_type in ["text", "csv", "json"]:
            # Regular text processing, use default models
            logger.debug("Task uses standard text input, using default models")
            recommended_models.extend(self.DEFAULT_MODELS)

        # Heuristic 2: Check output format
        if task.output_format == "json":
            # JSON output benefits from models with good structured output
            logger.debug("Task requires JSON output, prioritizing structured output models")
            # OpenAI and Claude are good at JSON
            if "openai/gpt-4o" not in recommended_models:
                recommended_models.insert(0, "openai/gpt-4o")
            if "anthropic/claude-sonnet-4.5" not in recommended_models:
                recommended_models.insert(0, "anthropic/claude-sonnet-4.5")

        # Heuristic 3: Check budget constraints
        if budget is not None:
            logger.debug(f"Budget constraint: ${budget:.4f} per evaluation")

            # Filter models that fit within budget
            # Estimate costs (rough approximation: 2000 input tokens, 500 output tokens)
            estimated_input_tokens = 2000
            estimated_output_tokens = 500

            budget_friendly_models = []
            for model_id in recommended_models:
                try:
                    estimated_cost = self.cost_tracker.calculate_cost(
                        model_id,
                        estimated_input_tokens,
                        estimated_output_tokens
                    )

                    if estimated_cost <= budget:
                        budget_friendly_models.append(model_id)
                        logger.debug(
                            f"Model {model_id} fits budget: ${estimated_cost:.4f} <= ${budget:.4f}"
                        )
                    else:
                        logger.debug(
                            f"Model {model_id} exceeds budget: ${estimated_cost:.4f} > ${budget:.4f}"
                        )
                except ValueError:
                    # Model not found in pricing, skip it
                    logger.warning(f"Model {model_id} not found in pricing config, skipping")
                    continue

            if budget_friendly_models:
                recommended_models = budget_friendly_models
            else:
                # No models fit budget with current list, try budget models
                logger.warning(
                    f"No recommended models fit budget ${budget:.4f}, "
                    "trying budget-friendly alternatives"
                )
                for model_id in self.BUDGET_MODELS:
                    try:
                        estimated_cost = self.cost_tracker.calculate_cost(
                            model_id,
                            estimated_input_tokens,
                            estimated_output_tokens
                        )
                        if estimated_cost <= budget:
                            recommended_models.append(model_id)
                    except ValueError:
                        continue

        # If still no models recommended, use defaults
        if not recommended_models:
            logger.warning("No models matched criteria, using defaults")
            recommended_models = self.DEFAULT_MODELS.copy()

        # Remove duplicates while preserving order
        seen = set()
        unique_models = []
        for model in recommended_models:
            if model not in seen:
                seen.add(model)
                unique_models.append(model)

        # Limit to top 5 models to keep evaluation time reasonable
        final_models = unique_models[:5]

        logger.info(
            f"Evaluation plan created: {len(final_models)} models recommended - {final_models}"
        )

        return final_models

    def get_model_category(self, model_id: str) -> str:
        """
        Categorize a model based on its characteristics.

        Args:
            model_id: Model identifier

        Returns:
            Category string: 'premium', 'budget', 'large_context', or 'default'

        Example:
            >>> orchestrator = LLMOrchestrator(client)
            >>> category = orchestrator.get_model_category("anthropic/claude-sonnet-4.5")
            >>> print(category)
            'premium'
        """
        if model_id in self.PREMIUM_MODELS:
            return "premium"
        elif model_id in self.BUDGET_MODELS:
            return "budget"
        elif model_id in self.LARGE_CONTEXT_MODELS:
            return "large_context"
        else:
            return "default"

    def estimate_cost_range(
        self,
        model_ids: List[str],
        estimated_input_tokens: int = 2000,
        estimated_output_tokens: int = 500
    ) -> dict[str, float]:
        """
        Estimate cost range for a list of models.

        Args:
            model_ids: List of model identifiers
            estimated_input_tokens: Estimated input tokens (default: 2000)
            estimated_output_tokens: Estimated output tokens (default: 500)

        Returns:
            Dictionary with 'min', 'max', and 'average' cost estimates

        Example:
            >>> orchestrator = LLMOrchestrator(client)
            >>> models = ["anthropic/claude-sonnet-4.5", "openai/gpt-4o"]
            >>> cost_range = orchestrator.estimate_cost_range(models)
            >>> print(f"Estimated cost: ${cost_range['min']:.4f} - ${cost_range['max']:.4f}")
        """
        costs = []

        for model_id in model_ids:
            try:
                cost = self.cost_tracker.calculate_cost(
                    model_id,
                    estimated_input_tokens,
                    estimated_output_tokens
                )
                costs.append(cost)
            except ValueError:
                logger.warning(f"Could not estimate cost for {model_id}")
                continue

        if not costs:
            return {"min": 0.0, "max": 0.0, "average": 0.0}

        return {
            "min": min(costs),
            "max": max(costs),
            "average": sum(costs) / len(costs)
        }

    def __str__(self) -> str:
        """String representation of LLMOrchestrator."""
        return "LLMOrchestrator()"

    def __repr__(self) -> str:
        """Repr of LLMOrchestrator."""
        return self.__str__()
