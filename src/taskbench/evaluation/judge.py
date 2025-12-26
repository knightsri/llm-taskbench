"""
LLM-as-judge evaluator for scoring model outputs.

This module implements LLM-based evaluation of model outputs,
including scoring, violation detection, and comparison logic.
"""

import json
import logging
import os
from typing import Any, Dict, List

from rich.console import Console
from rich.table import Table

from taskbench.api.client import OpenRouterClient
from taskbench.core.models import EvaluationResult, JudgeScore, TaskDefinition
from taskbench.evaluation.comparison import ModelComparison

logger = logging.getLogger(__name__)
console = Console()


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
        usecase: Any = None
    ) -> str:
        """
        Build evaluation prompt for the judge model.

        Args:
            task: TaskDefinition with evaluation criteria
            model_output: The output to evaluate
            input_data: Original input data for context

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
            "# Evaluation Criteria",
        ]

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
            if usecase.chunk_min_minutes or usecase.chunk_max_minutes or usecase.coverage_required:
                prompt_parts.append("## Coverage/Chunk Rules")
                if usecase.chunk_min_minutes:
                    prompt_parts.append(f"- Chunk minimum: {usecase.chunk_min_minutes} minutes")
                if usecase.chunk_max_minutes:
                    prompt_parts.append(f"- Chunk maximum: {usecase.chunk_max_minutes} minutes")
                if usecase.coverage_required:
                    prompt_parts.append("- No gaps across the time range.")
            if getattr(usecase, "notes", None):
                prompt_parts.append("## Additional Notes")
                for note in usecase.notes:
                    prompt_parts.append(f"- {note}")

        prompt_parts.extend([
            "",
            "# Input Data (for context)",
            "```",
            input_data[:2000] + ("..." if len(input_data) > 2000 else ""),
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
            max_tokens=2000,
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


