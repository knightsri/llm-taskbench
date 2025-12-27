"""
LLM-as-judge evaluator for scoring model outputs.

This module implements LLM-based evaluation of model outputs,
including scoring, violation detection, and comparison logic.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console
from rich.table import Table

from taskbench.api.client import OpenRouterClient
from taskbench.core.models import EvaluationResult, JudgeScore, TaskDefinition
from taskbench.evaluation.comparison import ModelComparison
from taskbench.usecase import UseCase

logger = logging.getLogger(__name__)
console = Console()

# Load configurable defaults from environment
DEFAULT_JUDGE_MAX_TOKENS = int(os.getenv("TASKBENCH_JUDGE_MAX_TOKENS", "2000"))
DEFAULT_INPUT_PREVIEW_LEN = int(os.getenv("TASKBENCH_INPUT_PREVIEW_LEN", "5000"))


def extract_timestamp_range(text: str) -> Optional[Tuple[str, str]]:
    """
    Extract the first and last timestamps from transcript text.

    Looks for patterns like [00:22:14], [22:14], 00:22:14, 22:14

    Returns:
        Tuple of (first_timestamp, last_timestamp) or None if no timestamps found
    """
    # Match timestamps in various formats: [HH:MM:SS], [MM:SS], HH:MM:SS, MM:SS
    pattern = r'\[?(\d{1,2}:\d{2}(?::\d{2})?)\]?'
    matches = re.findall(pattern, text)

    if not matches:
        return None

    # Normalize to HH:MM:SS format
    def normalize_ts(ts: str) -> str:
        parts = ts.split(':')
        if len(parts) == 2:
            return f"00:{parts[0].zfill(2)}:{parts[1]}"
        elif len(parts) == 3:
            return f"{parts[0].zfill(2)}:{parts[1]}:{parts[2]}"
        return ts

    first_ts = normalize_ts(matches[0])
    last_ts = normalize_ts(matches[-1])

    return (first_ts, last_ts)


class LLMJudge:
    """
    Use an LLM to evaluate model outputs.

    Implements LLM-as-judge pattern using Claude Sonnet 4.5 (or specified model)
    to score outputs based on task criteria.

    Example:
        ```python
        judge = LLMJudge(api_client)

        score = await judge.evaluate(
            task=task_definition,
            result=evaluation_result,
            input_data=original_input
        )

        print(f"Overall score: {score.overall_score}/100")
        print(f"Violations: {score.violations}")
        ```
    """

    def __init__(
        self,
        api_client: OpenRouterClient,
        judge_model: str = None
    ):
        """
        Initialize the LLM judge.

        Args:
            api_client: OpenRouterClient for making API calls
            judge_model: Model to use as judge (default: from GENERAL_TASK_LLM or Claude Sonnet 4.5)
        """
        self.api_client = api_client
        default_judge = os.getenv("GENERAL_TASK_LLM", "anthropic/claude-sonnet-4.5")
        self.judge_model = judge_model or default_judge

    def build_judge_prompt(
        self,
        task: TaskDefinition,
        model_output: str,
        input_data: str,
        usecase: Optional[UseCase] = None
    ) -> str:
        """
        Build evaluation prompt for the judge model.

        Args:
            task: TaskDefinition with evaluation criteria
            model_output: The output to evaluate
            input_data: Original input data for context
            usecase: Optional UseCase with goals and format specs

        Returns:
            Complete judge prompt
        """
        prompt_parts = [
            "You are an expert evaluator assessing an LLM's performance on a specific task.",
            "",
            "# Task Being Evaluated",
            f"**Task**: {task.name}",
            f"**Description**: {task.description}",
            "",
        ]

        # Include input format hints (critical for evaluating timestamp handling)
        if task.input_format_hint:
            prompt_parts.extend([
                "# Input Format Rules (the model should have followed these)",
                task.input_format_hint,
                "",
            ])

        prompt_parts.append("# Evaluation Criteria")

        for criterion in task.evaluation_criteria:
            prompt_parts.append(f"- {criterion}")

        prompt_parts.extend([
            "",
            "# Constraints That Must Be Met",
        ])

        for key, value in task.constraints.items():
            prompt_parts.append(f"- **{key}**: {value}")

        if usecase:
            prompt_parts.extend([
                "",
                "# Use Case Goal",
                usecase.goal,
            ])

            # Include LLM notes if present
            if usecase.llm_notes:
                prompt_parts.extend([
                    "",
                    "# LLM Notes (context for evaluation)",
                    usecase.llm_notes,
                ])

            # Include expected output format spec
            if usecase.output_format:
                prompt_parts.extend([
                    "",
                    "# Expected Output Format",
                    f"Format Type: {usecase.output_format.format_type.upper()}",
                    "Required Fields:",
                ])
                for field in usecase.output_format.fields:
                    prompt_parts.append(
                        f"  - **{field.get('name')}** ({field.get('type')}): {field.get('description', '')}"
                    )
                if usecase.output_format.example:
                    prompt_parts.extend([
                        "",
                        "Example of correct output:",
                        f"```{usecase.output_format.example.strip()}```",
                    ])
                if usecase.output_format.notes:
                    prompt_parts.append(f"Format Notes: {usecase.output_format.notes}")

            if usecase.chunk_min_minutes or usecase.chunk_max_minutes or usecase.coverage_required:
                prompt_parts.append("")
                prompt_parts.append("## Coverage/Chunk Rules")
                if usecase.chunk_min_minutes:
                    prompt_parts.append(f"- Chunk minimum: {usecase.chunk_min_minutes} minutes")
                if usecase.chunk_max_minutes:
                    prompt_parts.append(f"- Chunk maximum: {usecase.chunk_max_minutes} minutes")
                if usecase.coverage_required:
                    prompt_parts.append("- No gaps across the time range.")
            if getattr(usecase, "notes", None):
                prompt_parts.append("")
                prompt_parts.append("## Additional Notes")
                for note in usecase.notes:
                    prompt_parts.append(f"- {note}")

        # Extract timestamp range from input for validation
        ts_range = extract_timestamp_range(input_data)
        input_len = len(input_data)
        input_preview_len = DEFAULT_INPUT_PREVIEW_LEN
        is_truncated = input_len > input_preview_len

        prompt_parts.extend([
            "",
            "# Input Data Context",
        ])

        if ts_range:
            prompt_parts.extend([
                f"**VALID TIMESTAMP RANGE**: {ts_range[0]} to {ts_range[1]}",
                f"**Input Length**: {input_len:,} characters",
            ])
            if is_truncated:
                prompt_parts.extend([
                    "",
                    f"**NOTE**: The input below is a PREVIEW (first {input_preview_len:,} chars). The full transcript is {input_len:,} chars.",
                    f"The model processed the FULL transcript spanning {ts_range[0]} to {ts_range[1]}.",
                    "Do NOT assume content beyond the preview is fabricated - only flag timestamps OUTSIDE the valid range.",
                    "",
                ])
            else:
                prompt_parts.extend([
                    "Any timestamps in the model's output OUTSIDE this range are FABRICATED and should be marked as violations.",
                    "",
                ])

        prompt_parts.extend([
            "```",
            input_data[:input_preview_len] + ("..." if is_truncated else ""),
            "```",
            "",
            "# Model Output to Evaluate",
            "```",
            model_output,
            "```",
            "",
            "# Your Evaluation Task",
            "",
            task.judge_instructions,
        ])

        if ts_range:
            prompt_parts.extend([
                "",
                f"**CRITICAL**: Timestamps must fall within {ts_range[0]} to {ts_range[1]}.",
            ])
            if is_truncated:
                prompt_parts.append("Since the input preview is truncated, focus on timestamp range validity, format compliance, and duration constraints rather than content verification.")

        prompt_parts.extend([
            "",
            "# Response Format",
            "You MUST respond with ONLY valid JSON in this exact format:",
            "```json",
            "{",
            '  "accuracy_score": <0-100>,',
            '  "format_score": <0-100>,',
            '  "compliance_score": <0-100>,',
            '  "overall_score": <0-100>,',
            '  "violations": ["list of specific violations found"],',
            '  "reasoning": "Detailed explanation of your scoring"',
            "}",
            "```",
            "",
            "Provide ONLY the JSON, no other text."
        ])

        return "\n".join(prompt_parts)

    async def evaluate(
        self,
        task: TaskDefinition,
        result: EvaluationResult,
        input_data: str,
        usecase: Any = None
    ) -> JudgeScore:
        """
        Evaluate a model's output using LLM-as-judge.

        Args:
            task: TaskDefinition with evaluation criteria
            result: EvaluationResult to evaluate
            input_data: Original input data

        Returns:
            JudgeScore with scores and violations

        Raises:
            Exception: If judge fails to return valid JSON
        """
        logger.info(f"Evaluating {result.model_name} output with judge {self.judge_model}")

        # Build judge prompt
        prompt = self.build_judge_prompt(task, result.output, input_data, usecase=usecase)

        # Get judge evaluation
        response = await self.api_client.complete_with_json(
            model=self.judge_model,
            prompt=prompt,
            max_tokens=DEFAULT_JUDGE_MAX_TOKENS,
            temperature=0.3  # Lower temperature for more consistent judging
        )

        # Parse JSON response
        try:
            eval_data = json.loads(response.content)

            # Create JudgeScore
            score = JudgeScore(
                model_evaluated=result.model_name,
                accuracy_score=eval_data["accuracy_score"],
                format_score=eval_data["format_score"],
                compliance_score=eval_data["compliance_score"],
                overall_score=eval_data["overall_score"],
                violations=eval_data.get("violations", []),
                reasoning=eval_data["reasoning"]
            )

            logger.info(
                f"Judge evaluation complete for {result.model_name}: "
                f"overall={score.overall_score}/100, violations={len(score.violations)}"
            )

            return score

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse judge response: {str(e)}")
            logger.error(f"Response content: {response.content}")
            raise Exception(f"Judge returned invalid response: {str(e)}") from e

    def parse_violations(self, violations: List[str]) -> Dict[str, List[str]]:
        """
        Categorize violations by type.

        Args:
            violations: List of violation strings

        Returns:
            Dictionary mapping violation types to specific violations
        """
        categorized = {
            "under_min": [],
            "over_max": [],
            "format": [],
            "missing_field": [],
            "other": []
        }

        for violation in violations:
            violation_lower = violation.lower()

            if "under" in violation_lower or "too short" in violation_lower or "below" in violation_lower:
                categorized["under_min"].append(violation)
            elif "over" in violation_lower or "too long" in violation_lower or "exceed" in violation_lower:
                categorized["over_max"].append(violation)
            elif "format" in violation_lower or "invalid" in violation_lower:
                categorized["format"].append(violation)
            elif "missing" in violation_lower:
                categorized["missing_field"].append(violation)
            else:
                categorized["other"].append(violation)

        return categorized

    def count_violations_by_type(self, violations: List[str]) -> Dict[str, int]:
        """
        Count violations by category.

        Args:
            violations: List of violation strings

        Returns:
            Dictionary mapping violation types to counts
        """
        categorized = self.parse_violations(violations)
        return {k: len(v) for k, v in categorized.items()}

    def get_violation_summary(self, scores: List[JudgeScore]) -> str:
        """
        Generate text summary of violations across all models.

        Args:
            scores: List of JudgeScores

        Returns:
            Human-readable summary string
        """
        if not scores:
            return "No evaluations to summarize."

        models_with_violations = sum(1 for s in scores if s.violations)
        total_violations = sum(len(s.violations) for s in scores)

        if total_violations == 0:
            return " No violations found across all models!"

        # Count by type
        all_violations = []
        for score in scores:
            all_violations.extend(score.violations)

        counts = self.count_violations_by_type(all_violations)

        summary_parts = [
            f"  {models_with_violations}/{len(scores)} models had violations. Total: {total_violations}",
            ""
        ]

        for vtype, count in counts.items():
            if count > 0:
                summary_parts.append(f"  - {vtype.replace('_', ' ').title()}: {count}")

        return "\n".join(summary_parts)

    async def aggregate_analysis(
        self,
        usecase: UseCase,
        all_runs_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze results across multiple runs to provide aggregate recommendations.

        Args:
            usecase: The UseCase being evaluated
            all_runs_data: List of run data dicts, each with 'results' and 'scores'

        Returns:
            Dict with aggregate analysis including:
            - best_overall_model: Best performing model across all runs
            - best_value_model: Best cost/performance ratio
            - model_rankings: Ordered list of models by performance
            - recommendation: Natural language recommendation
            - excluded_runs: Runs excluded from analysis (too narrow)
        """
        # Filter out narrow runs (e.g., only 1-2 models or very short input)
        valid_runs = []
        excluded_runs = []

        for i, run_data in enumerate(all_runs_data):
            results = run_data.get("results", [])
            scores = run_data.get("scores", [])
            valid_scores = [s for s in scores if s is not None]

            if len(valid_scores) < 2:
                excluded_runs.append({
                    "run_index": i,
                    "reason": f"Too few valid results ({len(valid_scores)} models)"
                })
            else:
                valid_runs.append(run_data)

        if not valid_runs:
            return {
                "best_overall_model": None,
                "best_value_model": None,
                "model_rankings": [],
                "recommendation": "No valid runs to analyze. All runs were too narrow.",
                "excluded_runs": excluded_runs
            }

        # Aggregate scores by model
        model_scores: Dict[str, List[int]] = {}
        model_costs: Dict[str, List[float]] = {}

        for run_data in valid_runs:
            results = run_data.get("results", [])
            scores = run_data.get("scores", [])

            for result, score in zip(results, scores):
                if score is None:
                    continue
                model_name = result.get("model_name", "unknown")
                if model_name not in model_scores:
                    model_scores[model_name] = []
                    model_costs[model_name] = []
                model_scores[model_name].append(score.get("overall_score", 0))
                model_costs[model_name].append(result.get("cost_usd", 0.0))

        # Calculate averages
        model_rankings = []
        for model_name in model_scores:
            avg_score = sum(model_scores[model_name]) / len(model_scores[model_name])
            avg_cost = sum(model_costs[model_name]) / len(model_costs[model_name])
            value_ratio = avg_score / avg_cost if avg_cost > 0 else 0

            model_rankings.append({
                "model": model_name,
                "avg_score": round(avg_score, 1),
                "avg_cost": round(avg_cost, 4),
                "value_ratio": round(value_ratio, 1),
                "run_count": len(model_scores[model_name])
            })

        # Sort by score
        model_rankings.sort(key=lambda x: x["avg_score"], reverse=True)

        best_overall = model_rankings[0] if model_rankings else None
        best_value = max(model_rankings, key=lambda x: x["value_ratio"]) if model_rankings else None

        # Generate recommendation
        recommendation_parts = []
        if best_overall:
            recommendation_parts.append(
                f"Best overall: {best_overall['model']} with average score {best_overall['avg_score']}/100 "
                f"across {best_overall['run_count']} run(s)."
            )
        if best_value and best_value != best_overall:
            recommendation_parts.append(
                f"Best value: {best_value['model']} with {best_value['value_ratio']} pts/$ "
                f"(score: {best_value['avg_score']}, cost: ${best_value['avg_cost']:.4f})."
            )
        if excluded_runs:
            recommendation_parts.append(
                f"Note: {len(excluded_runs)} run(s) excluded from analysis due to insufficient data."
            )

        return {
            "best_overall_model": best_overall["model"] if best_overall else None,
            "best_value_model": best_value["model"] if best_value else None,
            "model_rankings": model_rankings,
            "recommendation": " ".join(recommendation_parts),
            "excluded_runs": excluded_runs,
            "total_runs_analyzed": len(valid_runs)
        }
