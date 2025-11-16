"""
LLM-as-Judge evaluator for LLM TaskBench.

This module provides the LLMJudge class that uses a powerful LLM to evaluate
the quality of outputs from other models, providing detailed scores and
identifying constraint violations.
"""

import json
import logging
from typing import Dict, List, Any

from taskbench.api.client import OpenRouterClient, OpenRouterAPIError
from taskbench.core.models import TaskDefinition, EvaluationResult, JudgeScore

logger = logging.getLogger(__name__)


class LLMJudge:
    """
    LLM-as-Judge evaluator for scoring model outputs.

    This class uses a powerful LLM (typically Claude Sonnet 4.5) to evaluate
    the quality of outputs from other models. It provides detailed scoring
    across multiple dimensions and identifies specific constraint violations.

    Example:
        >>> judge = LLMJudge(api_client, judge_model="anthropic/claude-sonnet-4.5")
        >>> task = TaskDefinition(...)
        >>> result = EvaluationResult(...)
        >>> input_data = "lecture transcript..."
        >>>
        >>> score = await judge.evaluate(task, result, input_data)
        >>> print(f"Overall score: {score.overall_score}")
        >>> print(f"Violations: {score.violations}")
    """

    def __init__(
        self,
        api_client: OpenRouterClient,
        judge_model: str = "anthropic/claude-sonnet-4.5"
    ):
        """
        Initialize the LLM Judge.

        Args:
            api_client: OpenRouterClient instance for API calls
            judge_model: Model to use for judging (default: Claude Sonnet 4.5)

        Example:
            >>> client = OpenRouterClient()
            >>> judge = LLMJudge(client)
        """
        self.api_client = api_client
        self.judge_model = judge_model
        logger.info(f"LLMJudge initialized with model: {judge_model}")

    def build_judge_prompt(
        self,
        task: TaskDefinition,
        model_output: str,
        input_data: str
    ) -> str:
        """
        Build a detailed evaluation prompt for the judge model.

        This method creates a comprehensive prompt that includes:
        - Task description and requirements
        - Original input data for context
        - Model's output to evaluate
        - Specific evaluation criteria from task definition
        - Constraint checking requirements
        - JSON response format specification

        Args:
            task: TaskDefinition containing evaluation criteria
            model_output: The output produced by the model being evaluated
            input_data: The original input data provided to the model

        Returns:
            Formatted prompt string for the judge model

        Example:
            >>> judge = LLMJudge(client)
            >>> prompt = judge.build_judge_prompt(task, output, input_data)
        """
        prompt_parts = []

        # Header
        prompt_parts.append("# LLM Output Evaluation Task")
        prompt_parts.append("")
        prompt_parts.append(
            "You are an expert evaluator assessing the quality of an LLM's output. "
            "Your task is to carefully evaluate the model's response against specific "
            "criteria and constraints, then provide detailed scores and identify any violations."
        )
        prompt_parts.append("")

        # Task Context
        prompt_parts.append("## Task Description")
        prompt_parts.append(f"**Task Name**: {task.name}")
        prompt_parts.append(f"**Description**: {task.description}")
        prompt_parts.append(f"**Expected Output Format**: {task.output_format.upper()}")
        prompt_parts.append("")

        # Evaluation Criteria
        prompt_parts.append("## Evaluation Criteria")
        prompt_parts.append("Evaluate the model's output based on these criteria:")
        for i, criterion in enumerate(task.evaluation_criteria, 1):
            prompt_parts.append(f"{i}. {criterion}")
        prompt_parts.append("")

        # Constraints
        if task.constraints:
            prompt_parts.append("## Constraints to Check")
            prompt_parts.append("The model MUST comply with these constraints:")
            prompt_parts.append("")
            for key, value in task.constraints.items():
                readable_key = key.replace("_", " ").title()
                prompt_parts.append(f"- **{readable_key}**: {value}")
            prompt_parts.append("")

        # Judge Instructions (scoring rubric)
        prompt_parts.append("## Scoring Rubric")
        prompt_parts.append(task.judge_instructions)
        prompt_parts.append("")

        # Original Input Data
        prompt_parts.append("## Original Input Data")
        prompt_parts.append("This was the input provided to the model:")
        prompt_parts.append("```")
        prompt_parts.append(input_data[:2000])  # Limit to 2000 chars to avoid token bloat
        if len(input_data) > 2000:
            prompt_parts.append("... (truncated)")
        prompt_parts.append("```")
        prompt_parts.append("")

        # Model Output to Evaluate
        prompt_parts.append("## Model Output to Evaluate")
        prompt_parts.append("This is the output produced by the model:")
        prompt_parts.append("```")
        prompt_parts.append(model_output)
        prompt_parts.append("```")
        prompt_parts.append("")

        # Response Format Instructions
        prompt_parts.append("## Your Response Format")
        prompt_parts.append(
            "You MUST respond with a valid JSON object containing the following fields:"
        )
        prompt_parts.append("")
        prompt_parts.append("```json")
        prompt_parts.append("{")
        prompt_parts.append('  "accuracy_score": <integer 0-100>,')
        prompt_parts.append('  "format_score": <integer 0-100>,')
        prompt_parts.append('  "compliance_score": <integer 0-100>,')
        prompt_parts.append('  "overall_score": <integer 0-100>,')
        prompt_parts.append('  "violations": [<list of specific violation strings>],')
        prompt_parts.append('  "reasoning": "<detailed explanation of your scores>"')
        prompt_parts.append("}")
        prompt_parts.append("```")
        prompt_parts.append("")

        # Scoring Guidelines
        prompt_parts.append("### Scoring Guidelines:")
        prompt_parts.append("")
        prompt_parts.append("**accuracy_score (0-100)**: How accurate and correct is the content?")
        prompt_parts.append("- 90-100: Excellent accuracy, all information correct")
        prompt_parts.append("- 70-89: Good accuracy, minor errors")
        prompt_parts.append("- 50-69: Fair accuracy, several errors")
        prompt_parts.append("- 0-49: Poor accuracy, major errors or incorrect")
        prompt_parts.append("")
        prompt_parts.append("**format_score (0-100)**: Does it match the required format?")
        prompt_parts.append("- 100: Perfect format compliance")
        prompt_parts.append("- 70-99: Minor format issues")
        prompt_parts.append("- 50-69: Significant format problems")
        prompt_parts.append("- 0-49: Wrong format or malformed")
        prompt_parts.append("")
        prompt_parts.append("**compliance_score (0-100)**: Does it meet all constraints?")
        prompt_parts.append("- 100: All constraints met")
        prompt_parts.append("- 70-99: Minor constraint violations")
        prompt_parts.append("- 50-69: Several constraint violations")
        prompt_parts.append("- 0-49: Major constraint violations")
        prompt_parts.append("")
        prompt_parts.append("**overall_score (0-100)**: Weighted average considering all factors")
        prompt_parts.append("")
        prompt_parts.append("**violations**: List specific violations found, e.g.:")
        prompt_parts.append('- "Duration 00:08:30 exceeds max_duration_minutes: 7"')
        prompt_parts.append('- "Missing required field: end_time"')
        prompt_parts.append('- "Format is not valid CSV"')
        prompt_parts.append("")
        prompt_parts.append(
            "Be thorough, objective, and specific. Your evaluation will be used to "
            "compare different LLM models."
        )

        prompt = "\n".join(prompt_parts)
        logger.debug(f"Built judge prompt ({len(prompt)} characters)")
        return prompt

    async def evaluate(
        self,
        task: TaskDefinition,
        result: EvaluationResult,
        input_data: str
    ) -> JudgeScore:
        """
        Evaluate a model's output using the judge model.

        This method:
        1. Builds a comprehensive evaluation prompt
        2. Calls the judge model with JSON mode enabled
        3. Parses the JSON response into a JudgeScore object
        4. Handles parsing errors gracefully

        Args:
            task: TaskDefinition containing evaluation criteria
            result: EvaluationResult containing the model's output
            input_data: Original input data for context

        Returns:
            JudgeScore object with detailed scoring and violations

        Raises:
            OpenRouterAPIError: If the API call fails
            ValueError: If the response cannot be parsed

        Example:
            >>> judge = LLMJudge(client)
            >>> score = await judge.evaluate(task, result, input_data)
            >>> print(f"Score: {score.overall_score}")
            >>> if score.violations:
            ...     print(f"Violations found: {len(score.violations)}")
        """
        logger.info(
            f"Evaluating output from '{result.model_name}' for task '{task.name}'"
        )

        # Build the judge prompt
        prompt = self.build_judge_prompt(task, result.output, input_data)

        try:
            # Call judge model with JSON mode
            response = await self.api_client.complete_with_json(
                model=self.judge_model,
                prompt=prompt,
                temperature=0.3,  # Low temperature for consistent evaluation
                max_tokens=2000
            )

            logger.debug(f"Judge response received: {len(response.content)} characters")

            # Parse JSON response
            try:
                score_data = json.loads(response.content)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse judge response as JSON: {e}")
                logger.error(f"Response content: {response.content[:500]}")
                raise ValueError(
                    f"Judge model returned invalid JSON: {e}. "
                    f"Response: {response.content[:200]}"
                ) from e

            # Validate required fields
            required_fields = [
                "accuracy_score", "format_score", "compliance_score",
                "overall_score", "reasoning"
            ]
            missing_fields = [
                field for field in required_fields
                if field not in score_data
            ]
            if missing_fields:
                raise ValueError(
                    f"Judge response missing required fields: {missing_fields}"
                )

            # Ensure violations is a list
            if "violations" not in score_data:
                score_data["violations"] = []
            elif not isinstance(score_data["violations"], list):
                score_data["violations"] = []

            # Create JudgeScore object
            judge_score = JudgeScore(
                model_evaluated=result.model_name,
                accuracy_score=int(score_data["accuracy_score"]),
                format_score=int(score_data["format_score"]),
                compliance_score=int(score_data["compliance_score"]),
                overall_score=int(score_data["overall_score"]),
                violations=score_data["violations"],
                reasoning=score_data["reasoning"]
            )

            logger.info(
                f"Evaluation complete: {result.model_name} scored {judge_score.overall_score}, "
                f"{len(judge_score.violations)} violations"
            )

            return judge_score

        except OpenRouterAPIError as e:
            logger.error(f"API error during evaluation: {e}")
            raise

        except Exception as e:
            logger.error(f"Unexpected error during evaluation: {e}", exc_info=True)
            raise ValueError(f"Failed to evaluate model output: {e}") from e

    def parse_violations(self, violations: List[str]) -> Dict[str, List[str]]:
        """
        Categorize violations by type.

        This method analyzes violation strings and categorizes them into:
        - under_min: Values below minimum constraints
        - over_max: Values exceeding maximum constraints
        - format: Format-related violations
        - missing_field: Missing required fields
        - other: Other types of violations

        Args:
            violations: List of violation strings from JudgeScore

        Returns:
            Dictionary mapping violation types to lists of specific violations

        Example:
            >>> judge = LLMJudge(client)
            >>> violations = [
            ...     "Duration 00:08:30 exceeds max_duration_minutes: 7",
            ...     "Missing required field: end_time",
            ...     "Format is not valid CSV"
            ... ]
            >>> categorized = judge.parse_violations(violations)
            >>> print(categorized)
            {
                'over_max': ['Duration 00:08:30 exceeds max_duration_minutes: 7'],
                'missing_field': ['Missing required field: end_time'],
                'format': ['Format is not valid CSV'],
                'under_min': [],
                'other': []
            }
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

            # Check for specific violation types
            if "missing" in violation_lower and "field" in violation_lower:
                categorized["missing_field"].append(violation)
            elif "exceeds" in violation_lower or "over" in violation_lower or "greater than max" in violation_lower:
                categorized["over_max"].append(violation)
            elif "below" in violation_lower or "less than min" in violation_lower or "under" in violation_lower:
                categorized["under_min"].append(violation)
            elif "format" in violation_lower:
                categorized["format"].append(violation)
            else:
                categorized["other"].append(violation)

        logger.debug(f"Categorized {len(violations)} violations into {len(categorized)} types")
        return categorized

    def count_violations_by_type(self, violations: List[str]) -> Dict[str, int]:
        """
        Count violations by category.

        Args:
            violations: List of violation strings

        Returns:
            Dictionary mapping violation types to counts

        Example:
            >>> judge = LLMJudge(client)
            >>> violations = ["Duration exceeds max", "Missing field: name"]
            >>> counts = judge.count_violations_by_type(violations)
            >>> print(counts)
            {'over_max': 1, 'missing_field': 1, 'format': 0, 'under_min': 0, 'other': 0}
        """
        categorized = self.parse_violations(violations)
        counts = {
            category: len(violation_list)
            for category, violation_list in categorized.items()
        }

        logger.debug(f"Violation counts: {counts}")
        return counts

    def get_violation_summary(self, scores: List[JudgeScore]) -> str:
        """
        Generate a text summary of violations across all models.

        This method aggregates violations from multiple JudgeScore objects
        and creates a human-readable summary showing:
        - Total violations
        - Breakdown by violation type
        - Most common violations

        Args:
            scores: List of JudgeScore objects to analyze

        Returns:
            Formatted string summarizing all violations

        Example:
            >>> judge = LLMJudge(client)
            >>> scores = [score1, score2, score3]
            >>> summary = judge.get_violation_summary(scores)
            >>> print(summary)
        """
        if not scores:
            return "No evaluations to summarize."

        # Collect all violations
        all_violations = []
        for score in scores:
            all_violations.extend(score.violations)

        if not all_violations:
            return "No violations found across all models. Excellent!"

        # Count by type
        total_counts = self.count_violations_by_type(all_violations)

        # Count frequency of each unique violation
        violation_frequency: Dict[str, int] = {}
        for violation in all_violations:
            violation_frequency[violation] = violation_frequency.get(violation, 0) + 1

        # Sort by frequency
        sorted_violations = sorted(
            violation_frequency.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Build summary
        summary_parts = []
        summary_parts.append("=" * 70)
        summary_parts.append("VIOLATION SUMMARY")
        summary_parts.append("=" * 70)
        summary_parts.append(f"Total Violations: {len(all_violations)}")
        summary_parts.append("")
        summary_parts.append("Breakdown by Type:")
        summary_parts.append(f"  - Exceeding Maximum: {total_counts['over_max']}")
        summary_parts.append(f"  - Below Minimum: {total_counts['under_min']}")
        summary_parts.append(f"  - Format Issues: {total_counts['format']}")
        summary_parts.append(f"  - Missing Fields: {total_counts['missing_field']}")
        summary_parts.append(f"  - Other: {total_counts['other']}")
        summary_parts.append("")
        summary_parts.append("Most Common Violations:")

        # Show top 10 most common violations
        for i, (violation, count) in enumerate(sorted_violations[:10], 1):
            summary_parts.append(f"  {i}. [{count}x] {violation}")

        summary_parts.append("=" * 70)

        summary = "\n".join(summary_parts)
        logger.debug("Generated violation summary")
        return summary

    def __str__(self) -> str:
        """String representation of LLMJudge."""
        return f"LLMJudge(model='{self.judge_model}')"

    def __repr__(self) -> str:
        """Repr of LLMJudge."""
        return self.__str__()
