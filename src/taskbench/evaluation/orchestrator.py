"""
LLM Orchestrator for LLM TaskBench.

Provides model selection heuristics and cost estimation.
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
    """

    DEFAULT_MODELS = [
        "anthropic/claude-sonnet-4.5",
        "openai/gpt-4o",
        "google/gemini-2.0-flash-exp",
    ]

    LARGE_CONTEXT_MODELS = [
        "anthropic/claude-sonnet-4.5",
        "google/gemini-2.0-flash-exp",
        "openai/gpt-4-turbo",
    ]

    BUDGET_MODELS = [
        "google/gemini-2.0-flash-exp",
        "anthropic/claude-haiku-3.5",
        "openai/gpt-4o-mini",
    ]

    PREMIUM_MODELS = [
        "anthropic/claude-sonnet-4.5",
        "openai/gpt-4o",
        "google/gemini-pro-1.5",
    ]

    def __init__(self, api_client: OpenRouterClient):
        self.api_client = api_client
        self.cost_tracker = CostTracker()
        logger.info("LLMOrchestrator initialized")

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

    def recommend_for_usecase(
        self,
        usecase_goal: str,
        require_large_context: bool = True,
        prioritize_cost: bool = False
    ) -> List[str]:
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
            priority = [
                "anthropic/claude-sonnet-4.5",
                "openai/gpt-4o",
                "google/gemini-2.0-flash-exp",
                "qwen/qwen-2.5-72b-instruct"
            ]
            ordered = [m for m in priority if m in candidates]
            ordered += [m for m in candidates if m not in ordered]
            candidates = ordered

        return candidates[:5]

    def __str__(self) -> str:
        return "LLMOrchestrator()"

    def __repr__(self) -> str:
        return self.__str__()
