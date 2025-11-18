"""Tests for metric calculation functions."""

import pytest
from app.core.metrics import (
    calculate_accuracy,
    calculate_hallucination_rate,
    calculate_completeness,
    calculate_cost,
    check_instruction_following,
)


def test_calculate_accuracy():
    """Test accuracy calculation."""

    output = '{"items": ["A", "B", "C"]}'
    gold_data = {"items": ["A", "B", "C", "D"]}

    accuracy = calculate_accuracy(output, gold_data, "json")

    # TP = 3, FP = 0, accuracy = 3/3 = 1.0
    assert accuracy == 1.0


def test_calculate_hallucination_rate():
    """Test hallucination rate calculation."""

    output = '{"items": ["A", "B", "X"]}'  # X is hallucinated
    gold_data = {"items": ["A", "B"]}

    hallucination = calculate_hallucination_rate(output, gold_data, "json")

    # TP = 2, FP = 1, accuracy = 2/3 = 0.667, hallucination = 1 - 0.667 = 0.333
    assert 0.33 <= hallucination <= 0.34


def test_calculate_completeness():
    """Test completeness (recall) calculation."""

    output = '{"items": ["A", "B"]}'
    gold_data = {"items": ["A", "B", "C", "D"]}

    completeness = calculate_completeness(output, gold_data, "json")

    # TP = 2, FN = 2, completeness = 2/4 = 0.5
    assert completeness == 0.5


def test_calculate_cost():
    """Test cost calculation."""

    cost = calculate_cost(
        input_tokens=1000,
        output_tokens=500,
        model_id="anthropic/claude-sonnet-4.5"
    )

    # Input: 1000 tokens at $3/1M = $0.003
    # Output: 500 tokens at $15/1M = $0.0075
    # Total: $0.0105
    assert 0.010 <= cost <= 0.011


def test_check_instruction_following():
    """Test instruction following check."""

    output = '{"name": "Test", "value": 123}'
    constraints = {
        "required_fields": ["name", "value"]
    }

    score = check_instruction_following(output, constraints, "json")

    # All constraints met
    assert score == 1.0


def test_check_instruction_following_violations():
    """Test instruction following with violations."""

    output = '{"name": "Test"}'  # Missing "value" field
    constraints = {
        "required_fields": ["name", "value"]
    }

    score = check_instruction_following(output, constraints, "json")

    # Some constraints violated
    assert score < 1.0
