"""Metric calculation functions for model evaluation.

Implements the 6 core metrics as specified in CLAUDE.md:
1. Accuracy (Precision)
2. Hallucination Rate
3. Completeness (Recall)
4. Cost
5. Instruction Following
6. Consistency
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def parse_output(output: str, format_type: str = "json") -> Any:
    """
    Parse model output based on expected format.

    Args:
        output: Raw model output string
        format_type: Expected format (json, csv, markdown)

    Returns:
        Parsed output structure
    """
    try:
        if format_type == "json":
            return json.loads(output)
        elif format_type == "csv":
            lines = output.strip().split("\n")
            return [line.split(",") for line in lines]
        else:
            return output
    except Exception as e:
        logger.warning(f"Failed to parse output as {format_type}: {e}")
        return output


def calculate_accuracy(output: str, gold_data: dict[str, Any], format_type: str = "json") -> float:
    """
    Calculate accuracy (precision): TP / (TP + FP).

    Measures how many of the extracted items are correct.

    Args:
        output: Model output
        gold_data: Gold standard data with expected results
        format_type: Output format

    Returns:
        Accuracy score between 0.0 and 1.0
    """
    try:
        extracted = parse_output(output, format_type)

        # Handle different output structures
        if isinstance(extracted, dict) and "items" in extracted:
            extracted_items = set(str(item) for item in extracted["items"])
        elif isinstance(extracted, list):
            extracted_items = set(str(item) for item in extracted)
        else:
            # If can't parse, return 0
            return 0.0

        # Get gold items
        if isinstance(gold_data, dict) and "items" in gold_data:
            gold_items = set(str(item) for item in gold_data["items"])
        elif isinstance(gold_data, list):
            gold_items = set(str(item) for item in gold_data)
        else:
            gold_items = set()

        if not extracted_items:
            return 0.0

        true_positives = len(extracted_items & gold_items)
        false_positives = len(extracted_items - gold_items)

        if true_positives + false_positives == 0:
            return 0.0

        accuracy = true_positives / (true_positives + false_positives)
        return round(accuracy, 4)

    except Exception as e:
        logger.error(f"Error calculating accuracy: {e}")
        return 0.0


def calculate_hallucination_rate(output: str, gold_data: dict[str, Any], format_type: str = "json") -> float:
    """
    Calculate hallucination rate: FP / (TP + FP) = 1 - accuracy.

    Measures how many extracted items are incorrect/fabricated.

    Args:
        output: Model output
        gold_data: Gold standard data
        format_type: Output format

    Returns:
        Hallucination rate between 0.0 and 1.0
    """
    accuracy = calculate_accuracy(output, gold_data, format_type)
    return round(1.0 - accuracy, 4)


def calculate_completeness(output: str, gold_data: dict[str, Any], format_type: str = "json") -> float:
    """
    Calculate completeness (recall): TP / (TP + FN).

    Measures how much of the expected content is captured.

    Args:
        output: Model output
        gold_data: Gold standard data
        format_type: Output format

    Returns:
        Completeness score between 0.0 and 1.0
    """
    try:
        extracted = parse_output(output, format_type)

        if isinstance(extracted, dict) and "items" in extracted:
            extracted_items = set(str(item) for item in extracted["items"])
        elif isinstance(extracted, list):
            extracted_items = set(str(item) for item in extracted)
        else:
            return 0.0

        if isinstance(gold_data, dict) and "items" in gold_data:
            gold_items = set(str(item) for item in gold_data["items"])
        elif isinstance(gold_data, list):
            gold_items = set(str(item) for item in gold_data)
        else:
            gold_items = set()

        if not gold_items:
            return 1.0  # If no expected items, consider complete

        true_positives = len(extracted_items & gold_items)
        false_negatives = len(gold_items - extracted_items)

        if true_positives + false_negatives == 0:
            return 0.0

        completeness = true_positives / (true_positives + false_negatives)
        return round(completeness, 4)

    except Exception as e:
        logger.error(f"Error calculating completeness: {e}")
        return 0.0


def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    model_id: str,
    pricing: dict[str, dict[str, float]] | None = None
) -> float:
    """
    Calculate cost based on token usage and model pricing.

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model_id: Model identifier
        pricing: Optional pricing dictionary (loaded from config if not provided)

    Returns:
        Cost in USD
    """
    # Default pricing for common models (per 1M tokens)
    default_pricing = {
        "anthropic/claude-sonnet-4.5": {"input": 3.0, "output": 15.0},
        "openai/gpt-4o": {"input": 5.0, "output": 15.0},
        "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "google/gemini-2.0-flash-exp:free": {"input": 0.0, "output": 0.0},
        "qwen/qwen-2.5-72b-instruct": {"input": 0.36, "output": 0.36},
    }

    pricing_db = pricing or default_pricing

    # Get pricing for model
    if model_id in pricing_db:
        model_pricing = pricing_db[model_id]
    else:
        # Default fallback pricing
        logger.warning(f"No pricing found for {model_id}, using defaults")
        model_pricing = {"input": 1.0, "output": 3.0}

    input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
    output_cost = (output_tokens / 1_000_000) * model_pricing["output"]

    total_cost = input_cost + output_cost
    return round(total_cost, 6)


def check_instruction_following(
    output: str,
    constraints: dict[str, Any],
    format_type: str = "json"
) -> float:
    """
    Check if output follows instructions and constraints.

    Args:
        output: Model output
        constraints: Constraints to check
        format_type: Expected format

    Returns:
        Score between 0.0 and 1.0 representing % of constraints followed
    """
    if not constraints:
        return 1.0  # No constraints to check

    violations = 0
    total_checks = 0

    try:
        parsed = parse_output(output, format_type)

        # Check format compliance
        total_checks += 1
        if format_type == "json":
            try:
                json.loads(output)
            except json.JSONDecodeError:
                violations += 1

        # Check specific constraints
        for key, value in constraints.items():
            total_checks += 1

            if key.startswith("min_") and isinstance(parsed, (list, dict)):
                # Check minimum length/size
                actual = len(parsed)
                if actual < value:
                    violations += 1

            elif key.startswith("max_") and isinstance(parsed, (list, dict)):
                # Check maximum length/size
                actual = len(parsed)
                if actual > value:
                    violations += 1

            elif key == "required_fields" and isinstance(parsed, dict):
                # Check required fields present
                for field in value:
                    if field not in parsed:
                        violations += 1
                        total_checks += 1

        if total_checks == 0:
            return 1.0

        score = 1.0 - (violations / total_checks)
        return round(max(0.0, score), 4)

    except Exception as e:
        logger.error(f"Error checking instruction following: {e}")
        return 0.0


def calculate_consistency_std(scores: list[float]) -> float:
    """
    Calculate standard deviation of consistency scores.

    Lower is better - indicates more consistent outputs across runs.

    Args:
        scores: List of scores from multiple runs

    Returns:
        Standard deviation
    """
    if len(scores) < 2:
        return 0.0

    mean = sum(scores) / len(scores)
    variance = sum((x - mean) ** 2 for x in scores) / len(scores)
    std = variance ** 0.5

    return round(std, 4)


def apply_quality_checks(
    output: str,
    quality_checks: list[dict[str, Any]]
) -> list[str]:
    """
    Apply quality checks to model output.

    Args:
        output: Model output to check
        quality_checks: List of quality check definitions

    Returns:
        List of violation messages (empty if all pass)
    """
    violations = []

    for check in quality_checks:
        try:
            name = check.get("name", "unknown")
            validation = check.get("validation_function", "")
            severity = check.get("severity", "warning")

            # Simple validation checks (can be extended)
            if "not_empty" in validation.lower():
                if not output or len(output.strip()) == 0:
                    violations.append(f"[{severity}] {name}: Output is empty")

            elif "length" in validation.lower():
                # Extract length constraints from validation
                if ">" in validation and "<" in validation:
                    # Extract numbers (simplified)
                    if len(output) < 100:
                        violations.append(f"[{severity}] {name}: Output too short")
                    elif len(output) > 100000:
                        violations.append(f"[{severity}] {name}: Output too long")

            elif "valid_json" in validation.lower() or "json" in validation.lower():
                try:
                    json.loads(output)
                except json.JSONDecodeError:
                    violations.append(f"[{severity}] {name}: Invalid JSON format")

        except Exception as e:
            logger.warning(f"Error applying quality check {check.get('name')}: {e}")
            continue

    return violations
