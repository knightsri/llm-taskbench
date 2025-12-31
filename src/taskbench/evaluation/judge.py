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
from taskbench.evaluation.rubric_generator import RubricGenerator
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


def timestamp_to_seconds(ts: str) -> int:
    """Convert HH:MM:SS or MM:SS timestamp to seconds."""
    parts = ts.split(':')
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    return 0


def normalize_timestamp(ts: str) -> str:
    """Normalize timestamp to HH:MM:SS format."""
    parts = ts.split(':')
    if len(parts) == 2:
        return f"00:{parts[0].zfill(2)}:{parts[1].zfill(2)}"
    elif len(parts) == 3:
        return f"{parts[0].zfill(2)}:{parts[1].zfill(2)}:{parts[2].zfill(2)}"
    return ts


class ProgrammaticValidator:
    """
    Validates model outputs programmatically before/alongside LLM judge scoring.

    Performs structural checks that don't require LLM interpretation:
    - JSON/CSV parsing validation (based on expected output format)
    - Timestamp range validation
    - Duration constraint checking
    - Required field presence

    This helps catch clear errors (hallucinated timestamps, invalid JSON) that
    might otherwise receive inconsistent scores from the LLM judge.
    """

    def __init__(
        self,
        input_data: str,
        constraints: Optional[Dict[str, Any]] = None,
        output_format: str = "json"
    ):
        """
        Initialize the validator.

        Args:
            input_data: The input data (used to extract valid timestamp range)
            constraints: Task constraints (e.g., min/max duration)
            output_format: Expected output format ("json", "csv", or "markdown")
        """
        self.input_data = input_data
        self.constraints = constraints or {}
        self.output_format = output_format.lower()
        self.ts_range = extract_timestamp_range(input_data)
        self.violations: List[str] = []

    def validate_json(self, output: str) -> Tuple[bool, Optional[Any], List[str]]:
        """
        Validate that output is valid JSON.

        Returns:
            Tuple of (is_valid, parsed_data, violations)
        """
        violations = []
        try:
            content = output.strip()

            # Handle chunked output format: /* chunk N */\n...
            # This happens when executor couldn't merge JSON arrays
            if content.startswith("/* chunk"):
                # Try to extract and merge JSON from all chunks
                chunk_pattern = r'/\* chunk \d+ \*/\n'
                chunks = re.split(chunk_pattern, content)
                # Filter out empty strings from the split
                chunks = [c.strip() for c in chunks if c.strip()]

                if len(chunks) == 1:
                    # Single chunk - just use that content
                    content = chunks[0]
                else:
                    # Multiple chunks - try to merge as arrays
                    merged_items = []
                    for chunk in chunks:
                        chunk_content = self._strip_markdown_blocks(chunk)
                        try:
                            chunk_data = json.loads(chunk_content)
                            if isinstance(chunk_data, list):
                                merged_items.extend(chunk_data)
                            else:
                                # Single object chunk - add to list
                                merged_items.append(chunk_data)
                        except json.JSONDecodeError:
                            pass  # Skip unparseable chunks

                    if merged_items:
                        return True, merged_items, violations
                    # If no items parsed, fall through to try the whole content

            # Strip markdown code blocks
            content = self._strip_markdown_blocks(content)

            parsed = json.loads(content)
            return True, parsed, violations
        except json.JSONDecodeError as e:
            violations.append(f"Invalid JSON output: {str(e)[:100]}")
            return False, None, violations

    def _strip_markdown_blocks(self, content: str) -> str:
        """Strip markdown code block markers from content."""
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        return content.strip()

    def validate_csv(self, output: str) -> Tuple[bool, Optional[List[Dict[str, str]]], List[str]]:
        """
        Validate that output is valid CSV.

        Returns:
            Tuple of (is_valid, parsed_rows_as_dicts, violations)
        """
        import csv
        import io

        violations = []
        try:
            content = output.strip()

            # Handle chunked output format: /* chunk N */\n...
            if content.startswith("/* chunk"):
                # Extract and merge CSV from all chunks
                chunk_pattern = r'/\* chunk \d+ \*/\n'
                chunks = re.split(chunk_pattern, content)
                chunks = [c.strip() for c in chunks if c.strip()]

                all_rows = []
                header = None
                for chunk in chunks:
                    chunk_content = self._strip_csv_markdown_blocks(chunk)
                    try:
                        reader = csv.DictReader(io.StringIO(chunk_content))
                        chunk_rows = list(reader)
                        if chunk_rows:
                            if header is None:
                                header = reader.fieldnames
                            all_rows.extend(chunk_rows)
                    except csv.Error:
                        pass  # Skip unparseable chunks

                if all_rows:
                    return True, all_rows, violations
                # Fall through if no rows parsed

            # Strip markdown code blocks
            content = self._strip_csv_markdown_blocks(content)

            # Try to parse as CSV
            reader = csv.DictReader(io.StringIO(content))
            rows = list(reader)

            if not rows:
                violations.append("CSV output is empty (no data rows)")
                return False, None, violations

            # Check that we have expected columns for timestamp-based tasks
            if rows:
                fieldnames = reader.fieldnames or []
                # For lecture/concept extraction, expect start_time and end_time
                has_timestamps = any(
                    f in fieldnames for f in ['start_time', 'end_time', 'start', 'end']
                )

            return True, rows, violations
        except csv.Error as e:
            violations.append(f"Invalid CSV output: {str(e)[:100]}")
            return False, None, violations
        except Exception as e:
            violations.append(f"Failed to parse CSV: {str(e)[:100]}")
            return False, None, violations

    def _strip_csv_markdown_blocks(self, content: str) -> str:
        """Strip markdown code block markers from CSV content."""
        content = content.strip()
        if content.startswith("```csv"):
            content = content[6:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        return content.strip()

    def validate_timestamp_range(
        self,
        parsed_data: Any,
        start_field: str = "start_time",
        end_field: str = "end_time"
    ) -> List[str]:
        """
        Validate that all timestamps in output are within valid range.

        Args:
            parsed_data: Parsed JSON output (list of items)
            start_field: Field name for start timestamp
            end_field: Field name for end timestamp

        Returns:
            List of violation strings
        """
        violations = []

        if not self.ts_range:
            # Can't validate without knowing valid range
            return violations

        valid_start_secs = timestamp_to_seconds(self.ts_range[0])
        valid_end_secs = timestamp_to_seconds(self.ts_range[1])

        if not isinstance(parsed_data, list):
            return violations

        for i, item in enumerate(parsed_data):
            if not isinstance(item, dict):
                continue

            start_ts = item.get(start_field, "")
            end_ts = item.get(end_field, "")

            if start_ts:
                start_normalized = normalize_timestamp(str(start_ts))
                start_secs = timestamp_to_seconds(start_normalized)
                if start_secs < valid_start_secs - 60:  # 1 min tolerance
                    violations.append(
                        f"Item {i+1}: start_time '{start_ts}' is before valid input range ({self.ts_range[0]})"
                    )
                if start_secs > valid_end_secs + 60:
                    violations.append(
                        f"Item {i+1}: start_time '{start_ts}' is after valid input range ({self.ts_range[1]})"
                    )

            if end_ts:
                end_normalized = normalize_timestamp(str(end_ts))
                end_secs = timestamp_to_seconds(end_normalized)
                if end_secs > valid_end_secs + 60:  # 1 min tolerance
                    violations.append(
                        f"Item {i+1}: end_time '{end_ts}' is after valid input range ({self.ts_range[1]})"
                    )

        return violations

    def validate_duration_constraints(
        self,
        parsed_data: Any,
        start_field: str = "start_time",
        end_field: str = "end_time"
    ) -> List[str]:
        """
        Validate that segment durations meet constraints.

        Args:
            parsed_data: Parsed JSON output (list of items)
            start_field: Field name for start timestamp
            end_field: Field name for end timestamp

        Returns:
            List of violation strings
        """
        violations = []

        min_duration_mins = self.constraints.get("min_segment_duration_minutes") or \
                           self.constraints.get("min_duration_minutes") or \
                           self.constraints.get("chunk_min_minutes")
        max_duration_mins = self.constraints.get("max_segment_duration_minutes") or \
                           self.constraints.get("max_duration_minutes") or \
                           self.constraints.get("chunk_max_minutes")

        if not isinstance(parsed_data, list):
            return violations

        for i, item in enumerate(parsed_data):
            if not isinstance(item, dict):
                continue

            start_ts = item.get(start_field, "")
            end_ts = item.get(end_field, "")

            if start_ts and end_ts:
                try:
                    start_secs = timestamp_to_seconds(normalize_timestamp(str(start_ts)))
                    end_secs = timestamp_to_seconds(normalize_timestamp(str(end_ts)))
                    duration_mins = (end_secs - start_secs) / 60

                    if min_duration_mins and duration_mins < min_duration_mins:
                        violations.append(
                            f"Item {i+1}: duration {duration_mins:.1f}min < minimum {min_duration_mins}min"
                        )
                    if max_duration_mins and duration_mins > max_duration_mins:
                        violations.append(
                            f"Item {i+1}: duration {duration_mins:.1f}min > maximum {max_duration_mins}min"
                        )
                except Exception:
                    pass  # Skip if we can't parse timestamps

        return violations

    def validate(
        self,
        output: str,
        start_field: str = "start_time",
        end_field: str = "end_time"
    ) -> Dict[str, Any]:
        """
        Run all programmatic validations on model output.

        Returns:
            Dict with:
                - is_valid_format: bool (valid JSON or CSV depending on output_format)
                - is_valid_json: bool (kept for backwards compatibility)
                - parsed_data: parsed data (list of dicts) or None
                - violations: list of all violations
                - validation_score: 0-100 score based on violation count
        """
        all_violations = []
        is_valid_format = False
        parsed_data = None

        # Validate based on expected output format
        if self.output_format == "json":
            # JSON validation
            is_valid_format, parsed_data, format_violations = self.validate_json(output)
            all_violations.extend(format_violations)

        elif self.output_format == "csv":
            # CSV validation
            is_valid_format, parsed_data, format_violations = self.validate_csv(output)
            all_violations.extend(format_violations)

        else:
            # For markdown or other formats, skip format validation
            # Just consider it valid and let the LLM judge do semantic evaluation
            is_valid_format = True
            parsed_data = None

        # Run timestamp and duration validation if we have parsed data
        if is_valid_format and parsed_data:
            # Timestamp range validation
            ts_violations = self.validate_timestamp_range(parsed_data, start_field, end_field)
            all_violations.extend(ts_violations)

            # Duration constraint validation
            duration_violations = self.validate_duration_constraints(parsed_data, start_field, end_field)
            all_violations.extend(duration_violations)

        # Calculate a programmatic score (100 - penalty for violations)
        if not is_valid_format:
            validation_score = 0  # Invalid format = 0
        else:
            item_count = len(parsed_data) if isinstance(parsed_data, list) else 1
            violation_penalty = min(len(all_violations) * 10, 100)
            validation_score = max(0, 100 - violation_penalty)

        return {
            "is_valid_format": is_valid_format,
            "is_valid_json": is_valid_format if self.output_format == "json" else True,
            "parsed_data": parsed_data,
            "violations": all_violations,
            "validation_score": validation_score,
            "item_count": len(parsed_data) if isinstance(parsed_data, list) else 0
        }


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
        self.rubric_generator = RubricGenerator(api_client)

    async def build_judge_prompt(
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

        # Add dynamic use-case-aware rubric if use case is provided
        if usecase:
            dynamic_rubric = await self.rubric_generator.generate_judge_prompt_section_async(usecase)
            prompt_parts.extend([
                "",
                "# USE CASE SPECIFIC EVALUATION (CRITICAL)",
                "",
                dynamic_rubric,
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
            "**SCORING GUIDANCE:**",
            "- compliance_score: Calculate based on ACTUAL violation count, not overall impression",
            "- If multiple segments violate duration constraints, compliance_score MUST reflect this",
            "- Example: 8 of 11 segments over max = (11-8)/11 * 100 = 27% compliance, NOT 75%",
            "- overall_score should be weighted: accuracy*0.3 + format*0.2 + compliance*0.5",
            "- A model with many compliance violations CANNOT score above 60 overall",
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
        Evaluate a model's output using LLM-as-judge with programmatic pre-validation.

        The evaluation process:
        1. Run programmatic validation (JSON parsing, timestamp range, duration constraints)
        2. Call LLM judge for semantic evaluation
        3. Merge programmatic violations with LLM-identified violations
        4. Adjust scores if programmatic checks found issues LLM missed

        Args:
            task: TaskDefinition with evaluation criteria
            result: EvaluationResult to evaluate
            input_data: Original input data
            usecase: Optional UseCase for additional context

        Returns:
            JudgeScore with scores and violations

        Raises:
            Exception: If judge fails to return valid JSON
        """
        logger.info(f"Evaluating {result.model_name} output with judge {self.judge_model}")

        # Step 1: Run programmatic validation first
        constraints = dict(task.constraints) if task.constraints else {}
        if usecase:
            if hasattr(usecase, 'chunk_min_minutes') and usecase.chunk_min_minutes:
                constraints['chunk_min_minutes'] = usecase.chunk_min_minutes
            if hasattr(usecase, 'chunk_max_minutes') and usecase.chunk_max_minutes:
                constraints['chunk_max_minutes'] = usecase.chunk_max_minutes

        validator = ProgrammaticValidator(
            input_data,
            constraints,
            output_format=task.output_format
        )
        prog_validation = validator.validate(result.output)

        logger.info(
            f"Programmatic validation for {result.model_name}: "
            f"valid_format={prog_validation['is_valid_format']}, "
            f"output_format={task.output_format}, "
            f"violations={len(prog_validation['violations'])}"
        )

        # Build judge prompt (async to allow LLM-based rubric generation)
        prompt = await self.build_judge_prompt(task, result.output, input_data, usecase=usecase)

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

            # Handle scores that might be nested in a 'metrics' dict
            metrics = eval_data.get("metrics", {})

            # Extract scores with fallbacks (handle floats, nested values)
            def get_score(key: str, fallback_key: str = None) -> int:
                """Get score as int, checking main dict and metrics dict."""
                val = eval_data.get(key)
                if val is None and fallback_key:
                    val = eval_data.get(fallback_key)
                if val is None:
                    val = metrics.get(key)
                if val is None and fallback_key:
                    val = metrics.get(fallback_key)
                if val is None:
                    return 0
                return int(round(float(val)))

            accuracy = get_score("accuracy_score")
            format_sc = get_score("format_score")
            compliance = get_score("compliance_score")
            overall = get_score("overall_score", "final_score")

            # Handle violations that might be objects instead of strings
            raw_violations = eval_data.get("violations", [])
            violations = []
            for v in raw_violations:
                if isinstance(v, str):
                    violations.append(v)
                elif isinstance(v, dict):
                    # Extract description or format as string
                    desc = v.get("description", v.get("message", str(v)))
                    severity = v.get("severity", "")
                    if severity:
                        violations.append(f"{severity}: {desc}")
                    else:
                        violations.append(desc)

            # Get reasoning with fallbacks
            reasoning = eval_data.get("reasoning") or eval_data.get("summary") or ""
            if not reasoning:
                # Try to construct from strengths/weaknesses
                strengths = eval_data.get("strengths", [])
                weaknesses = eval_data.get("weaknesses", [])
                if strengths or weaknesses:
                    parts = []
                    if strengths:
                        parts.append("Strengths: " + "; ".join(strengths[:3]))
                    if weaknesses:
                        parts.append("Weaknesses: " + "; ".join(weaknesses[:3]))
                    reasoning = " | ".join(parts)
                else:
                    reasoning = "Evaluation complete."

            # Step 3: Merge programmatic violations with LLM-identified violations
            prog_violations = prog_validation.get("violations", [])
            all_violations = list(violations)  # Start with LLM violations

            # Add programmatic violations that aren't already identified
            for pv in prog_violations:
                # Check if this violation is already covered by LLM
                already_covered = any(
                    pv.lower() in v.lower() or v.lower() in pv.lower()
                    for v in violations
                )
                if not already_covered:
                    all_violations.append(f"[PROGRAMMATIC] {pv}")

            # Step 4: Adjust scores if programmatic validation found issues
            final_format_score = format_sc
            final_compliance_score = compliance
            final_overall = overall

            if not prog_validation["is_valid_format"]:
                # Invalid format (JSON/CSV depending on expected) should heavily penalize format score
                final_format_score = min(format_sc, 20)
                final_compliance_score = min(compliance, 30)
                final_overall = min(overall, 25)
                logger.warning(f"{result.model_name}: Invalid {task.output_format.upper()} output - scores adjusted")

            elif prog_violations:
                # If programmatic validation found violations, ensure scores reflect this
                prog_violation_count = len(prog_violations)
                if prog_violation_count > 3:
                    # Many violations - cap compliance score
                    max_compliance = max(30, 100 - (prog_violation_count * 8))
                    if compliance > max_compliance:
                        final_compliance_score = max_compliance
                        # Recalculate overall
                        final_overall = int(accuracy * 0.3 + format_sc * 0.2 + final_compliance_score * 0.5)
                        logger.info(
                            f"{result.model_name}: Compliance capped at {max_compliance} "
                            f"due to {prog_violation_count} programmatic violations"
                        )

            # Append programmatic validation info to reasoning
            prog_reasoning = ""
            if prog_violations:
                prog_reasoning = f" [Programmatic checks found {len(prog_violations)} additional violations]"

            # Create JudgeScore
            score = JudgeScore(
                model_evaluated=result.model_name,
                accuracy_score=accuracy,
                format_score=final_format_score,
                compliance_score=final_compliance_score,
                overall_score=final_overall,
                violations=all_violations,
                reasoning=reasoning + prog_reasoning
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
