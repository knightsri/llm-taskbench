"""LLM-powered quality check generation.

This module implements the key innovation of LLM TaskBench: automatically
generating task-specific quality checks by analyzing the task description.
"""

import json
import logging
from typing import Any

import httpx

from app.core.config import settings
from app.schemas.task import QualityCheckSchema

logger = logging.getLogger(__name__)


async def generate_quality_checks(
    task_description: str,
    domain: str | None = None,
    output_format: str = "json"
) -> list[dict[str, Any]]:
    """
    Use LLM to analyze task and generate validation rules.

    This is the core innovation: the framework analyzes your task description
    and automatically generates appropriate quality checks.

    Args:
        task_description: Description of the task to analyze
        domain: Optional domain (healthcare, education, etc.) for context
        output_format: Expected output format

    Returns:
        List of quality check dictionaries with name, description, validation_function, severity

    Example:
        checks = await generate_quality_checks(
            "Extract medical concepts from clinical transcripts",
            domain="healthcare",
            output_format="json"
        )
        # Returns checks like:
        # - No PHI (patient identifiable info) leaked
        # - Medical terminology preserved
        # - Minimum 2 minutes per concept
    """
    prompt = f"""You are an expert at analyzing tasks and generating quality validation rules.

Task Description: {task_description}
{f"Domain: {domain}" if domain else ""}
Output Format: {output_format}

Generate 5-8 specific quality checks for evaluating outputs from this task.
Focus on:
1. Output format constraints (structure, required fields)
2. Domain-specific requirements (terminology, accuracy)
3. Common failure modes (hallucinations, omissions, formatting errors)
4. Completeness checks (all required information present)

For each check, provide:
- name: Short identifier (snake_case)
- description: What this check validates
- validation_function: How to check it (describe the logic clearly)
- severity: "critical", "warning", or "info"

Return ONLY valid JSON array, no other text:
[
  {{
    "name": "check_name",
    "description": "What it checks",
    "validation_function": "How to validate",
    "severity": "critical"
  }}
]"""

    try:
        # Use Anthropic API for quality check generation
        api_key = settings.ANTHROPIC_API_KEY or settings.OPENROUTER_API_KEY
        if not api_key:
            raise ValueError("No API key configured for quality check generation")

        # Use OpenRouter with Claude if no direct Anthropic key
        if settings.ANTHROPIC_API_KEY:
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": settings.ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            payload = {
                "model": "claude-sonnet-4.5",
                "max_tokens": 2000,
                "temperature": 0.3,
                "messages": [{"role": "user", "content": prompt}]
            }
        else:
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://llm-taskbench.com",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "anthropic/claude-sonnet-4.5",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 2000
            }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

            # Extract content based on API
            if settings.ANTHROPIC_API_KEY:
                content = data["content"][0]["text"]
            else:
                content = data["choices"][0]["message"]["content"]

            # Parse JSON response
            # Remove markdown code blocks if present
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            checks = json.loads(content)
            logger.info(f"Generated {len(checks)} quality checks for task")
            return checks

    except httpx.HTTPError as e:
        logger.error(f"HTTP error generating quality checks: {e}")
        # Return default checks as fallback
        return _default_quality_checks(output_format)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse quality checks JSON: {e}")
        return _default_quality_checks(output_format)
    except Exception as e:
        logger.error(f"Unexpected error generating quality checks: {e}")
        return _default_quality_checks(output_format)


def _default_quality_checks(output_format: str) -> list[dict[str, Any]]:
    """Fallback quality checks if LLM generation fails."""
    checks = [
        {
            "name": "output_not_empty",
            "description": "Output must not be empty",
            "validation_function": "len(output.strip()) > 0",
            "severity": "critical"
        },
        {
            "name": "valid_format",
            "description": f"Output must be valid {output_format}",
            "validation_function": f"validate_{output_format}_format(output)",
            "severity": "critical"
        },
        {
            "name": "reasonable_length",
            "description": "Output length is reasonable (not too short or too long)",
            "validation_function": "100 < len(output) < 100000",
            "severity": "warning"
        }
    ]
    logger.warning("Using default quality checks as fallback")
    return checks


async def recommend_metrics(
    task_description: str,
    domain: str | None = None,
    has_direct_api: bool = False
) -> dict[str, Any]:
    """
    LLM analyzes task and recommends optional metrics to enable.

    Args:
        task_description: Task description
        domain: Optional domain
        has_direct_api: Whether user has direct API keys (affects latency metric)

    Returns:
        Dictionary with metric recommendations and reasoning
    """
    prompt = f"""Analyze this task and recommend which optional evaluation metrics should be enabled.

Task: {task_description}
{f"Domain: {domain}" if domain else ""}
Direct API Access: {"Yes" if has_direct_api else "No (using OpenRouter)"}

Core metrics (always enabled):
- Accuracy
- Hallucination Rate
- Completeness
- Cost
- Instruction Following
- Consistency

Optional metrics to consider:
1. Safety/Toxicity: User-facing content, potential harm
2. Bias & Fairness: Demographics, sensitive attributes
3. Factuality: Verifiable claims required
4. Latency: Real-time requirements (ONLY if direct APIs available)
5. Robustness: Noisy/imperfect inputs expected
6. Contextual Relevance: RAG or multi-turn conversations

For each optional metric, return:
- enabled: true/false
- reasoning: Why it should/shouldn't be enabled
- severity: "high", "medium", "low"

Return ONLY valid JSON, no other text:
{{
  "safety": {{"enabled": false, "reasoning": "...", "severity": "low"}},
  "bias": {{"enabled": false, "reasoning": "...", "severity": "low"}},
  "factuality": {{"enabled": true, "reasoning": "...", "severity": "high"}},
  "latency": {{"enabled": false, "reasoning": "Only enable with direct APIs"}},
  "robustness": {{"enabled": false, "reasoning": "...", "severity": "low"}},
  "contextual_relevance": {{"enabled": false, "reasoning": "...", "severity": "low"}}
}}"""

    try:
        api_key = settings.ANTHROPIC_API_KEY or settings.OPENROUTER_API_KEY
        if not api_key:
            return _default_metric_recommendations()

        if settings.ANTHROPIC_API_KEY:
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": settings.ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            payload = {
                "model": "claude-sonnet-4.5",
                "max_tokens": 1500,
                "temperature": 0.3,
                "messages": [{"role": "user", "content": prompt}]
            }
        else:
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://llm-taskbench.com",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "anthropic/claude-sonnet-4.5",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 1500
            }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

            if settings.ANTHROPIC_API_KEY:
                content = data["content"][0]["text"]
            else:
                content = data["choices"][0]["message"]["content"]

            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            recommendations = json.loads(content)
            logger.info("Generated metric recommendations")
            return recommendations

    except Exception as e:
        logger.error(f"Error generating metric recommendations: {e}")
        return _default_metric_recommendations()


def _default_metric_recommendations() -> dict[str, Any]:
    """Default metric recommendations as fallback."""
    return {
        "safety": {"enabled": False, "reasoning": "Not user-facing content", "severity": "low"},
        "bias": {"enabled": False, "reasoning": "No demographic considerations", "severity": "low"},
        "factuality": {"enabled": False, "reasoning": "No verifiable claims required", "severity": "low"},
        "latency": {"enabled": False, "reasoning": "Only enable with direct API keys", "severity": "low"},
        "robustness": {"enabled": False, "reasoning": "Clean inputs expected", "severity": "low"},
        "contextual_relevance": {"enabled": False, "reasoning": "Single-turn task", "severity": "low"}
    }
