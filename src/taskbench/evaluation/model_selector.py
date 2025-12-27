"""
LLM Model Selector using tool calling to fetch and analyze OpenRouter models.

Uses an LLM with tool calling to:
1. Fetch current model catalog from OpenRouter
2. Analyze task requirements
3. Recommend 5-10 best-fit models ranked by Performance > Cost > Quality
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from taskbench.api.client import OpenRouterClient

logger = logging.getLogger(__name__)

# Cache settings (configurable via environment)
CACHE_DIR = Path(__file__).parent.parent.parent.parent / ".cache"
MODELS_CACHE_FILE = CACHE_DIR / "openrouter_models.json"
# Default 24 hours, configurable via TASKBENCH_MODELS_CACHE_TTL (in hours)
CACHE_TTL_SECONDS = int(os.getenv("TASKBENCH_MODELS_CACHE_TTL", "24")) * 60 * 60


# Tool definition for fetching OpenRouter models
GET_MODELS_TOOL = {
    "type": "function",
    "function": {
        "name": "get_openrouter_models",
        "description": "Fetches the current catalog of available LLM models from OpenRouter with pricing and capabilities",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
}


def _load_cached_models() -> Optional[List[Dict[str, Any]]]:
    """Load models from cache if valid."""
    try:
        if MODELS_CACHE_FILE.exists():
            cache_data = json.loads(MODELS_CACHE_FILE.read_text())
            cached_at = cache_data.get("cached_at", 0)
            if time.time() - cached_at < CACHE_TTL_SECONDS:
                logger.info(f"Using cached models ({len(cache_data.get('models', []))} models)")
                return cache_data.get("models", [])
    except Exception as e:
        logger.warning(f"Failed to load cache: {e}")
    return None


def _save_models_cache(models: List[Dict[str, Any]]) -> None:
    """Save models to cache."""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_data = {
            "cached_at": time.time(),
            "models": models
        }
        MODELS_CACHE_FILE.write_text(json.dumps(cache_data, indent=2))
        logger.info(f"Cached {len(models)} models to {MODELS_CACHE_FILE}")
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")


async def fetch_openrouter_models(force_refresh: bool = False) -> List[Dict[str, Any]]:
    """
    Fetch current model catalog from OpenRouter API with caching.

    Args:
        force_refresh: If True, bypass cache and fetch fresh data

    Returns:
        List of model dictionaries with id, name, pricing, context_length, etc.
    """
    # Try cache first
    if not force_refresh:
        cached = _load_cached_models()
        if cached:
            return cached

    try:
        logger.info("Fetching fresh model catalog from OpenRouter...")
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://openrouter.ai/api/v1/models",
                timeout=30.0
            )

            if response.status_code == 200:
                data = response.json()
                models = data.get("data", [])

                # Filter and format models
                formatted = []
                for m in models:
                    # Skip models without pricing
                    pricing = m.get("pricing", {})
                    if not pricing:
                        continue

                    formatted.append({
                        "id": m.get("id"),
                        "name": m.get("name"),
                        "context_length": m.get("context_length", 0),
                        "input_cost_per_1m": float(pricing.get("prompt", 0)) * 1_000_000,
                        "output_cost_per_1m": float(pricing.get("completion", 0)) * 1_000_000,
                        "description": m.get("description", ""),
                        "top_provider": m.get("top_provider", {}).get("is_moderated", False)
                    })

                # Save to cache
                _save_models_cache(formatted)

                return formatted
            else:
                logger.warning(f"Failed to fetch models: {response.status_code}")
                return []

    except Exception as e:
        logger.error(f"Error fetching OpenRouter models: {e}")
        return []


# Phase 1: Task Analysis Prompt (no model hints)
TASK_ANALYSIS_PROMPT = """Analyze this task and determine the requirements for selecting LLM models.

## Task Description
{user_task}

## Output exactly this JSON:
{{
  "task_type": "extraction | summarization | coding | reasoning | creative | chat | analysis",
  "complexity": "simple | moderate | complex",
  "min_context_tokens": <number - estimate minimum context window needed>,
  "quality_sensitivity": "low | medium | high",
  "cost_sensitivity": "low | medium | high",
  "required_capabilities": ["list of specific capabilities needed"],
  "reasoning": "Brief explanation of your analysis"
}}"""

# Phase 2: Model Ranking Prompt (works on pre-filtered candidates)
MODEL_RANKING_PROMPT = """You are selecting LLM models for benchmarking a specific task.

## Task Requirements
{task_requirements}

## Candidate Models (pre-filtered for compatibility)
{candidate_models}

## Selection Rules
Select models based on requested tiers: {tiers}

Tier definitions:
- QUALITY: Premium flagship models (>$25/1M total) - Claude Opus, o1, GPT-4-turbo
- VALUE: Mid-tier models ($3-25/1M total) - Claude Sonnet 4.5, GPT-4o, Gemini Pro, Claude Haiku
- BUDGET: Cheapest options (<$3/1M total) including free models
- SPEED: Models known for fast response times - Gemini Flash, GPT-4o-mini, Claude Haiku, small Llama models

Select 2-3 models per requested tier. The purpose is benchmarking different options.

## Output exactly this JSON:
{{
  "models": [
    {{
      "rank": 1,
      "model_id": "exact model_id from the list",
      "why": "One sentence rationale",
      "best_for": "quality | value | budget | speed"
    }}
  ],
  "suggested_test_order": ["1 model from first tier", "1 from second", "1 from third"]
}}

IMPORTANT:
- QUALITY: claude-opus-4.5, openai/o1, gpt-4-turbo
- VALUE: claude-sonnet-4.5, gpt-4o, gemini-2.5-pro (mid-tier, not premium)
- BUDGET: free models, cheap models like llama, mistral-small
- SPEED: gemini-flash, gpt-4o-mini, claude-haiku, small models"""


class ModelSelector:
    """
    Two-phase model selector:
    1. LLM analyzes task requirements
    2. Programmatic filtering of models
    3. LLM ranks filtered candidates

    Example:
        ```python
        selector = ModelSelector(api_client)
        result = await selector.recommend_models(
            "Extract concepts from lecture transcripts, 4-8 min chunks, low-cost priority"
        )
        print(result["suggested_test_order"])
        ```
    """

    def __init__(self, api_client: OpenRouterClient):
        self.api_client = api_client
        self.selector_model = os.getenv("MODEL_SELECTOR_LLM", "openai/gpt-4o")

    # Valid tier options
    VALID_TIERS = ["quality", "value", "budget", "speed"]

    async def recommend_models(
        self,
        task_description: str,
        tiers: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Two-phase model selection:
        1. Analyze task requirements
        2. Filter models programmatically
        3. LLM ranks the candidates

        Args:
            task_description: Description of the task/use case
            tiers: List of tiers to include. Options: "quality", "value", "budget", "speed"
                   Default: ["quality", "value", "budget"]

        Returns:
            Dict with task_analysis, models list, and suggested_test_order
        """
        # Default tiers if not specified
        if tiers is None:
            tiers = ["quality", "value", "budget"]

        # Validate tiers
        tiers = [t.lower() for t in tiers if t.lower() in self.VALID_TIERS]
        if not tiers:
            tiers = ["quality", "value", "budget"]

        try:
            # Phase 1: Analyze task requirements
            logger.info("Phase 1: Analyzing task requirements...")
            task_reqs = await self._analyze_task(task_description)
            logger.info(f"Task requirements: {task_reqs}")

            # Phase 2: Fetch and filter models programmatically
            logger.info("Phase 2: Fetching and filtering models...")
            all_models = await fetch_openrouter_models()
            candidates = self._filter_models(all_models, task_reqs)
            logger.info(f"Filtered to {len(candidates)} candidate models")

            # Phase 3: LLM ranks the candidates
            logger.info(f"Phase 3: Ranking candidates for tiers: {tiers}...")
            ranking = await self._rank_models(task_reqs, candidates, tiers)

            # Combine results
            return {
                "task_analysis": {
                    "type": task_reqs.get("task_type", "analysis"),
                    "complexity": task_reqs.get("complexity", "moderate"),
                    "context_size": self._context_category(task_reqs.get("min_context_tokens", 32000)),
                    "required_capabilities": task_reqs.get("required_capabilities", [])
                },
                "tiers_requested": tiers,
                "models": self._enrich_rankings(ranking.get("models", []), candidates),
                "suggested_test_order": ranking.get("suggested_test_order", [])
            }

        except Exception as e:
            logger.error(f"Model selection failed: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_recommendation()

    async def _analyze_task(self, task_description: str) -> Dict[str, Any]:
        """Phase 1: LLM analyzes task to determine requirements."""
        prompt = TASK_ANALYSIS_PROMPT.format(user_task=task_description)

        response = await self._call_llm(prompt)
        return self._parse_json(response)

    # Flagship models that should always be considered (if available)
    FLAGSHIP_MODELS = [
        # Anthropic
        "anthropic/claude-opus-4.5",
        "anthropic/claude-sonnet-4.5",
        "anthropic/claude-sonnet-4",
        "anthropic/claude-3.5-sonnet",
        "anthropic/claude-3.5-haiku",
        # OpenAI
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "openai/gpt-4-turbo",
        "openai/o1",
        "openai/o1-mini",
        # Google
        "google/gemini-2.5-pro",
        "google/gemini-2.5-flash",
        "google/gemini-2.0-flash-001",
        # Meta
        "meta-llama/llama-3.3-70b-instruct",
        "meta-llama/llama-3.1-405b-instruct",
        # Others
        "deepseek/deepseek-r1",
        "deepseek/deepseek-chat",
        "mistralai/mistral-large-2411",
    ]

    def _filter_models(self, models: List[Dict], task_reqs: Dict) -> List[Dict]:
        """Phase 2: Programmatically filter models based on requirements."""
        min_context = task_reqs.get("min_context_tokens", 32000)
        quality_sens = task_reqs.get("quality_sensitivity", "medium")
        cost_sens = task_reqs.get("cost_sensitivity", "medium")

        # Create lookup map
        model_map = {m["id"]: m for m in models}

        # Start with flagship models that meet context requirements
        candidates = []
        seen = set()

        for model_id in self.FLAGSHIP_MODELS:
            if model_id in model_map:
                m = model_map[model_id]
                if m.get("context_length", 0) >= min_context:
                    candidates.append(m)
                    seen.add(model_id)

        # Filter remaining models by context
        filtered = [m for m in models if m.get("context_length", 0) >= min_context and m["id"] not in seen]

        # Categorize remaining models into tiers
        # Thresholds designed so:
        #   - quality: Opus ($30), o1 ($75+), GPT-4-turbo ($40)
        #   - value: Sonnet 4.5 ($18), GPT-4o ($12.50), Gemini Pro ($11.25), Haiku ($4.80)
        #   - budget: GPT-4o-mini ($0.75), Llama, Mistral small, etc.
        #   - free: Free tier models
        tiers = {"quality": [], "value": [], "budget": [], "free": []}

        for m in filtered:
            total_cost = m.get("input_cost_per_1m", 0) + m.get("output_cost_per_1m", 0)

            if total_cost == 0:
                tiers["free"].append(m)
            elif total_cost < 3:  # < $3 per 1M tokens total
                tiers["budget"].append(m)
            elif total_cost < 25:  # < $25 per 1M tokens total
                tiers["value"].append(m)
            else:
                tiers["quality"].append(m)

        # Sort each tier by cost (ascending)
        for tier in tiers.values():
            tier.sort(key=lambda m: m.get("input_cost_per_1m", 0) + m.get("output_cost_per_1m", 0))

        # Add more candidates from each tier
        candidates.extend(tiers["quality"][:3])
        candidates.extend(tiers["value"][:3])
        candidates.extend(tiers["budget"][:3])

        # Free tier (if cost sensitive)
        if cost_sens in ["medium", "high"]:
            candidates.extend(tiers["free"][:3])

        # Deduplicate
        unique = []
        seen = set()
        for m in candidates:
            if m["id"] not in seen:
                seen.add(m["id"])
                unique.append(m)

        logger.info(f"Candidates include {len([m for m in unique if m['id'] in self.FLAGSHIP_MODELS])} flagship models")
        return unique[:30]  # Max 30 candidates for ranking

    async def _rank_models(
        self,
        task_reqs: Dict,
        candidates: List[Dict],
        tiers: List[str]
    ) -> Dict[str, Any]:
        """Phase 3: LLM ranks the filtered candidates."""
        # Format candidates for the prompt
        candidate_lines = []
        for m in candidates:
            candidate_lines.append(
                f"- {m['id']}: context={m['context_length']:,}, "
                f"cost=${m['input_cost_per_1m']:.2f}/${m['output_cost_per_1m']:.2f} per 1M tokens"
            )

        # Format tiers for display
        tiers_str = ", ".join([t.upper() for t in tiers])

        prompt = MODEL_RANKING_PROMPT.format(
            task_requirements=json.dumps(task_reqs, indent=2),
            candidate_models="\n".join(candidate_lines),
            tiers=tiers_str
        )

        response = await self._call_llm(prompt)
        return self._parse_json(response)

    def _enrich_rankings(self, rankings: List[Dict], candidates: List[Dict]) -> List[Dict]:
        """Add cost/context info back to the rankings."""
        candidate_map = {m["id"]: m for m in candidates}

        enriched = []
        for r in rankings:
            model_id = r.get("model_id", "")
            if model_id in candidate_map:
                m = candidate_map[model_id]
                enriched.append({
                    "rank": r.get("rank"),
                    "model_id": model_id,
                    "why": r.get("why", ""),
                    "input_cost_per_1m": m.get("input_cost_per_1m", 0),
                    "output_cost_per_1m": m.get("output_cost_per_1m", 0),
                    "context_length": m.get("context_length", 0),
                    "best_for": r.get("best_for", "value")
                })

        return enriched

    def _context_category(self, tokens: int) -> str:
        """Convert token count to category."""
        if tokens < 8000:
            return "small"
        elif tokens < 32000:
            return "medium"
        elif tokens < 128000:
            return "large"
        else:
            return "very_large"

    async def _call_llm(self, prompt: str) -> str:
        """Make a simple LLM call (no tools)."""
        api_key = os.getenv("OPENROUTER_API_KEY")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.selector_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2000,
                    "temperature": 0.3
                },
                timeout=60.0
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                raise Exception(f"API error: {response.status_code} - {response.text}")

    def _parse_json(self, content: str) -> Dict[str, Any]:
        """Parse JSON from LLM response."""
        content = content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        return json.loads(content.strip())

    def _parse_recommendation(self, content: str) -> Dict[str, Any]:
        """Parse LLM response into structured recommendation."""
        try:
            # Extract JSON from response
            content = content.strip()

            # Handle markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            return json.loads(content.strip())

        except (json.JSONDecodeError, IndexError) as e:
            logger.warning(f"Failed to parse recommendation: {e}")
            return self._fallback_recommendation()

    def _fallback_recommendation(self) -> Dict[str, Any]:
        """Return fallback recommendation if LLM fails."""
        return {
            "task_analysis": {
                "type": "analysis",
                "complexity": "moderate",
                "context_size": "large",
                "required_capabilities": ["text processing", "structured output"]
            },
            "models": [
                {
                    "rank": 1,
                    "model_id": "google/gemini-2.5-flash",
                    "why": "Cost-effective with large context window",
                    "input_cost_per_1m": 0.10,
                    "output_cost_per_1m": 0.40,
                    "context_length": 1000000,
                    "best_for": "primary"
                },
                {
                    "rank": 2,
                    "model_id": "anthropic/claude-sonnet-4.5",
                    "why": "Excellent analytical capabilities",
                    "input_cost_per_1m": 3.00,
                    "output_cost_per_1m": 15.00,
                    "context_length": 200000,
                    "best_for": "quality"
                },
                {
                    "rank": 3,
                    "model_id": "openai/gpt-4o",
                    "why": "Reliable structured output",
                    "input_cost_per_1m": 2.50,
                    "output_cost_per_1m": 10.00,
                    "context_length": 128000,
                    "best_for": "quality"
                }
            ],
            "suggested_test_order": [
                "google/gemini-2.5-flash",
                "anthropic/claude-sonnet-4.5",
                "openai/gpt-4o"
            ]
        }


async def select_models_for_task(
    task_description: str,
    tiers: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Convenience function to select models for a task.

    Args:
        task_description: Description of the task/use case
        tiers: List of tiers to include. Options: "quality", "value", "budget", "speed"
               Default: ["quality", "value", "budget"]

    Returns:
        Model recommendation dict
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.warning("OPENROUTER_API_KEY not set, using fallback")
        return ModelSelector(None)._fallback_recommendation()

    async with OpenRouterClient(api_key) as client:
        selector = ModelSelector(client)
        return await selector.recommend_models(task_description, tiers=tiers)
