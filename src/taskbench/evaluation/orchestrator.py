"""
LLM Orchestrator for LLM TaskBench.

Provides intelligent model selection using either:
1. Dynamic LLM-powered selection via ModelSelector (recommended)
2. Heuristic-based selection for quick/offline scenarios
"""

import logging
import os
from typing import Any, Dict, List, Optional

from taskbench.api.client import OpenRouterClient
from taskbench.core.models import TaskDefinition
from taskbench.evaluation.cost import CostTracker

logger = logging.getLogger(__name__)


class LLMOrchestrator:
    """
    Intelligent model selection orchestrator.

    Supports two modes:
    1. Dynamic mode (default): Uses ModelSelector for LLM-powered analysis
    2. Heuristic mode: Uses predefined model lists for quick selection

    Example:
        ```python
        orchestrator = LLMOrchestrator(api_client)

        # Dynamic selection (recommended for production)
        models = await orchestrator.select_models_dynamically(
            use_case_description="Extract concepts from lectures",
            tiers=["quality", "value", "budget"]
        )

        # Heuristic selection (quick/offline)
        models = await orchestrator.create_evaluation_plan(task)
        ```
    """

    # Fallback model lists for heuristic mode (when dynamic selection unavailable)
    DEFAULT_MODELS = [
        "anthropic/claude-sonnet-4",
        "openai/gpt-4o",
        "google/gemini-2.5-flash",
    ]

    LARGE_CONTEXT_MODELS = [
        "anthropic/claude-sonnet-4",
        "google/gemini-2.5-flash",
        "openai/gpt-4-turbo",
    ]

    BUDGET_MODELS = [
        "google/gemini-2.5-flash",
        "anthropic/claude-3.5-haiku",
        "openai/gpt-4o-mini",
    ]

    PREMIUM_MODELS = [
        "anthropic/claude-sonnet-4",
        "openai/gpt-4o",
        "google/gemini-2.5-pro",
    ]

    def __init__(self, api_client: OpenRouterClient):
        self.api_client = api_client
        self.cost_tracker = CostTracker()
        self._model_selector = None  # Lazy-loaded
        logger.info("LLMOrchestrator initialized")

    async def select_models_dynamically(
        self,
        use_case_description: str,
        tiers: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Select models using LLM-powered analysis (recommended).

        This method uses the ModelSelector to:
        1. Analyze task requirements via LLM
        2. Filter 350+ OpenRouter models programmatically
        3. Rank candidates by tier via LLM

        Args:
            use_case_description: Description of the use case/task
            tiers: List of tiers to include ["quality", "value", "budget", "speed"]
                   Default: ["quality", "value", "budget"]

        Returns:
            Dict with:
                - task_analysis: LLM's analysis of requirements
                - models: Ranked list with costs and rationale
                - suggested_test_order: Recommended evaluation order
        """
        from taskbench.evaluation.model_selector import ModelSelector

        if self._model_selector is None:
            self._model_selector = ModelSelector(self.api_client)

        return await self._model_selector.recommend_models(use_case_description, tiers=tiers)

    def get_recommended_model_ids(self, selection_result: Dict[str, Any]) -> List[str]:
        """
        Extract model IDs from a dynamic selection result.

        Args:
            selection_result: Result from select_models_dynamically()

        Returns:
            List of model IDs in suggested order
        """
        if "suggested_test_order" in selection_result:
            return selection_result["suggested_test_order"]

        models = selection_result.get("models", [])
        return [m.get("model_id") for m in models if m.get("model_id")]

    async def create_evaluation_plan(
        self,
        task: TaskDefinition,
        budget: Optional[float] = None
    ) -> List[str]:
        logger.info(
            f"Creating evaluation plan for task '{task.name}' "
            f"(input={task.input_type}, output={task.output_format}, budget={budget})"
        )

        recommended_models: List[str] = []

        if task.input_type == "transcript":
            recommended_models.extend(self.LARGE_CONTEXT_MODELS)
        elif task.input_type in ["text", "csv", "json"]:
            recommended_models.extend(self.DEFAULT_MODELS)

        if task.output_format == "json":
            if "openai/gpt-4o" not in recommended_models:
                recommended_models.insert(0, "openai/gpt-4o")
            if "anthropic/claude-sonnet-4.5" not in recommended_models:
                recommended_models.insert(0, "anthropic/claude-sonnet-4.5")

        if budget is not None:
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
                except ValueError:
                    continue
            if budget_friendly_models:
                recommended_models = budget_friendly_models
            else:
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

        if not recommended_models:
            recommended_models = self.DEFAULT_MODELS.copy()

        seen = set()
        unique_models: List[str] = []
        for model in recommended_models:
            if model not in seen:
                seen.add(model)
                unique_models.append(model)

        return unique_models[:5]

    def get_model_category(self, model_id: str) -> str:
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
                continue

        if not costs:
            return {"min": 0.0, "max": 0.0, "average": 0.0}

        return {
            "min": min(costs),
            "max": max(costs),
            "average": sum(costs) / len(costs)
        }

    async def recommend_for_usecase_dynamic(
        self,
        usecase_goal: str,
        prioritize_cost: bool = False
    ) -> Dict[str, Any]:
        """
        Recommend models for a use case using LLM-powered analysis (async).

        This is the recommended method for production use. It analyzes the use case
        description and selects optimal models from 350+ available on OpenRouter.

        Args:
            usecase_goal: Description of the use case goal
            prioritize_cost: If True, prioritize budget/value tiers over quality

        Returns:
            Dict with task_analysis, models list, and suggested_test_order
        """
        tiers = ["budget", "value", "speed"] if prioritize_cost else ["quality", "value", "budget"]
        return await self.select_models_dynamically(usecase_goal, tiers=tiers)

    def recommend_for_usecase(
        self,
        usecase_goal: str,
        require_large_context: bool = True,
        prioritize_cost: bool = False
    ) -> List[str]:
        """
        Recommend models using heuristic-based selection (sync, fallback).

        Use recommend_for_usecase_dynamic() for production scenarios.
        This method is provided for quick/offline scenarios where LLM-based
        selection is not available or desired.

        Args:
            usecase_goal: Description of the use case (used for logging)
            require_large_context: Filter to models with 120K+ context
            prioritize_cost: Sort by cost (lowest first)

        Returns:
            List of recommended model IDs (max 5)
        """
        tracker = CostTracker()
        models = tracker.list_models()

        candidates: List[str] = []
        for m in models:
            if require_large_context and m.context_window < 120000:
                continue
            candidates.append(m.model_id)

        if not candidates:
            candidates = [m.model_id for m in models]

        if prioritize_cost:
            candidates.sort(
                key=lambda mid: (
                    tracker.models[mid].input_price_per_1m,
                    tracker.models[mid].output_price_per_1m
                )
            )
        else:
            # Updated fallback priority list (current frontier models)
            priority = [
                "anthropic/claude-sonnet-4",
                "openai/gpt-4o",
                "google/gemini-2.5-flash",
                "deepseek/deepseek-chat"
            ]
            ordered = [m for m in priority if m in candidates]
            ordered += [m for m in candidates if m not in ordered]
            candidates = ordered

        return candidates[:5]

    def __str__(self) -> str:
        return "LLMOrchestrator()"

    def __repr__(self) -> str:
        return self.__str__()
