"""
Tests for LLMJudge.

This module tests the LLMJudge class including:
- Initialization
- Prompt building for judge evaluations
- Model output evaluation
- Violation parsing and categorization
- Violation summaries
- Error handling
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from taskbench.core.models import TaskDefinition, EvaluationResult, JudgeScore, CompletionResponse
from taskbench.evaluation.judge import LLMJudge
from taskbench.api.client import OpenRouterAPIError


@pytest.fixture
def sample_task():
    """Create a sample task definition for testing."""
    return TaskDefinition(
        name="test_task",
        description="Test task for judging",
        input_type="text",
        output_format="json",
        evaluation_criteria=["Accuracy", "Completeness", "Format compliance"],
        constraints={"max_length": 100, "min_items": 3},
        examples=[],
        judge_instructions="Evaluate based on accuracy, format, and compliance with constraints."
    )


@pytest.fixture
def sample_result():
    """Create a sample evaluation result."""
    return EvaluationResult(
        model_name="test-model",
        task_name="test_task",
        output='{"items": ["a", "b", "c"]}',
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        cost_usd=0.01,
        latency_ms=1000.0,
        status="success"
    )


@pytest.fixture
def mock_api_client():
    """Create a mock API client."""
    return AsyncMock()


class TestLLMJudge:
    """Test suite for LLMJudge."""

    def test_init_default_model(self, mock_api_client):
        """Test initialization with default judge model."""
        judge = LLMJudge(mock_api_client)
        assert judge.judge_model == "anthropic/claude-sonnet-4.5"
        assert judge.api_client is mock_api_client

    def test_init_custom_model(self, mock_api_client):
        """Test initialization with custom judge model."""
        judge = LLMJudge(mock_api_client, judge_model="openai/gpt-4o")
        assert judge.judge_model == "openai/gpt-4o"

    def test_build_judge_prompt(self, mock_api_client, sample_task, sample_result):
        """Test building judge evaluation prompt."""
        judge = LLMJudge(mock_api_client)
        input_data = "Sample input data"

        prompt = judge.build_judge_prompt(sample_task, sample_result.output, input_data)

        # Verify prompt contains key sections
        assert "LLM Output Evaluation Task" in prompt
        assert "test_task" in prompt
        assert "Test task for judging" in prompt
        assert "Accuracy" in prompt
        assert "Completeness" in prompt
        assert "Format compliance" in prompt
        assert "Max Length" in prompt  # Constraints are formatted with title case
        assert "Min Items" in prompt
        assert sample_result.output in prompt
        assert "Sample input data" in prompt
        assert "accuracy_score" in prompt
        assert "format_score" in prompt
        assert "compliance_score" in prompt
        assert "overall_score" in prompt

    def test_build_judge_prompt_truncates_long_input(self, mock_api_client, sample_task):
        """Test that very long input is truncated."""
        judge = LLMJudge(mock_api_client)
        long_input = "x" * 5000  # More than 2000 chars

        prompt = judge.build_judge_prompt(sample_task, "output", long_input)

        # Should contain truncation indicator
        assert "truncated" in prompt.lower()

    def test_build_judge_prompt_with_constraints(self, mock_api_client, sample_task):
        """Test prompt includes constraints section."""
        judge = LLMJudge(mock_api_client)
        prompt = judge.build_judge_prompt(sample_task, "output", "input")

        # Verify constraints are formatted
        assert "Constraints to Check" in prompt
        assert "Max Length" in prompt
        assert "Min Items" in prompt

    def test_build_judge_prompt_without_constraints(self, mock_api_client):
        """Test prompt generation when task has no constraints."""
        task = TaskDefinition(
            name="simple_task",
            description="Simple task",
            input_type="text",
            output_format="json",
            evaluation_criteria=["Accuracy"],
            constraints={},
            examples=[],
            judge_instructions="Evaluate"
        )

        judge = LLMJudge(mock_api_client)
        prompt = judge.build_judge_prompt(task, "output", "input")

        # Should still have main sections but no constraints
        assert "Evaluation Criteria" in prompt
        assert "Constraints to Check" not in prompt

    @pytest.mark.asyncio
    async def test_evaluate_success(self, mock_api_client, sample_task, sample_result):
        """Test successful evaluation."""
        judge = LLMJudge(mock_api_client)

        # Mock judge response
        judge_response_data = {
            "accuracy_score": 95,
            "format_score": 100,
            "compliance_score": 90,
            "overall_score": 95,
            "violations": [],
            "reasoning": "Excellent output with minor compliance issues"
        }

        mock_response = CompletionResponse(
            content=json.dumps(judge_response_data),
            model="anthropic/claude-sonnet-4.5",
            input_tokens=500,
            output_tokens=200,
            total_tokens=700,
            latency_ms=2000.0
        )

        mock_api_client.complete_with_json.return_value = mock_response

        # Execute evaluation
        score = await judge.evaluate(sample_task, sample_result, "input data")

        # Verify score
        assert isinstance(score, JudgeScore)
        assert score.model_evaluated == "test-model"
        assert score.accuracy_score == 95
        assert score.format_score == 100
        assert score.compliance_score == 90
        assert score.overall_score == 95
        assert len(score.violations) == 0
        assert score.reasoning == "Excellent output with minor compliance issues"

    @pytest.mark.asyncio
    async def test_evaluate_with_violations(self, mock_api_client, sample_task, sample_result):
        """Test evaluation with constraint violations."""
        judge = LLMJudge(mock_api_client)

        judge_response_data = {
            "accuracy_score": 80,
            "format_score": 85,
            "compliance_score": 60,
            "overall_score": 75,
            "violations": [
                "Output length 120 exceeds max_length: 100",
                "Missing required field: timestamp"
            ],
            "reasoning": "Good accuracy but has compliance violations"
        }

        mock_response = CompletionResponse(
            content=json.dumps(judge_response_data),
            model="anthropic/claude-sonnet-4.5",
            input_tokens=500,
            output_tokens=200,
            total_tokens=700,
            latency_ms=2000.0
        )

        mock_api_client.complete_with_json.return_value = mock_response

        score = await judge.evaluate(sample_task, sample_result, "input data")

        # Verify violations
        assert len(score.violations) == 2
        assert "exceeds max_length" in score.violations[0]
        assert "Missing required field" in score.violations[1]
        assert score.compliance_score == 60

    @pytest.mark.asyncio
    async def test_evaluate_invalid_json(self, mock_api_client, sample_task, sample_result):
        """Test evaluation with invalid JSON response."""
        judge = LLMJudge(mock_api_client)

        # Mock invalid JSON response
        mock_response = CompletionResponse(
            content="This is not valid JSON",
            model="anthropic/claude-sonnet-4.5",
            input_tokens=500,
            output_tokens=200,
            total_tokens=700,
            latency_ms=2000.0
        )

        mock_api_client.complete_with_json.return_value = mock_response

        # Should raise ValueError
        with pytest.raises(ValueError, match="invalid JSON"):
            await judge.evaluate(sample_task, sample_result, "input data")

    @pytest.mark.asyncio
    async def test_evaluate_missing_required_fields(self, mock_api_client, sample_task, sample_result):
        """Test evaluation with missing required fields in response."""
        judge = LLMJudge(mock_api_client)

        # Mock response missing required fields
        incomplete_data = {
            "accuracy_score": 90,
            # Missing other required fields
        }

        mock_response = CompletionResponse(
            content=json.dumps(incomplete_data),
            model="anthropic/claude-sonnet-4.5",
            input_tokens=500,
            output_tokens=200,
            total_tokens=700,
            latency_ms=2000.0
        )

        mock_api_client.complete_with_json.return_value = mock_response

        # Should raise ValueError
        with pytest.raises(ValueError, match="missing required fields"):
            await judge.evaluate(sample_task, sample_result, "input data")

    @pytest.mark.asyncio
    async def test_evaluate_missing_violations_field(self, mock_api_client, sample_task, sample_result):
        """Test that missing violations field defaults to empty list."""
        judge = LLMJudge(mock_api_client)

        judge_response_data = {
            "accuracy_score": 95,
            "format_score": 100,
            "compliance_score": 90,
            "overall_score": 95,
            # violations field is missing
            "reasoning": "Good output"
        }

        mock_response = CompletionResponse(
            content=json.dumps(judge_response_data),
            model="anthropic/claude-sonnet-4.5",
            input_tokens=500,
            output_tokens=200,
            total_tokens=700,
            latency_ms=2000.0
        )

        mock_api_client.complete_with_json.return_value = mock_response

        score = await judge.evaluate(sample_task, sample_result, "input data")

        # Should default to empty list
        assert score.violations == []

    @pytest.mark.asyncio
    async def test_evaluate_api_error(self, mock_api_client, sample_task, sample_result):
        """Test evaluation with API error."""
        judge = LLMJudge(mock_api_client)

        mock_api_client.complete_with_json.side_effect = OpenRouterAPIError(
            "API error", status_code=500
        )

        # Should raise OpenRouterAPIError
        with pytest.raises(OpenRouterAPIError):
            await judge.evaluate(sample_task, sample_result, "input data")

    def test_parse_violations_over_max(self, mock_api_client):
        """Test parsing violations that exceed maximum."""
        judge = LLMJudge(mock_api_client)
        violations = [
            "Duration 00:08:30 exceeds max_duration_minutes: 7",
            "Length 150 exceeds max_length: 100"
        ]

        categorized = judge.parse_violations(violations)

        assert len(categorized["over_max"]) == 2
        assert violations[0] in categorized["over_max"]
        assert violations[1] in categorized["over_max"]

    def test_parse_violations_under_min(self, mock_api_client):
        """Test parsing violations below minimum."""
        judge = LLMJudge(mock_api_client)
        violations = [
            "Count 2 is below min_count: 3",
            "Value less than min_value: 10"
        ]

        categorized = judge.parse_violations(violations)

        assert len(categorized["under_min"]) == 2

    def test_parse_violations_missing_field(self, mock_api_client):
        """Test parsing missing field violations."""
        judge = LLMJudge(mock_api_client)
        violations = [
            "Missing required field: timestamp",
            "Missing field: user_id"
        ]

        categorized = judge.parse_violations(violations)

        assert len(categorized["missing_field"]) == 2

    def test_parse_violations_format(self, mock_api_client):
        """Test parsing format violations."""
        judge = LLMJudge(mock_api_client)
        violations = [
            "Format is not valid CSV",
            "Invalid JSON format"
        ]

        categorized = judge.parse_violations(violations)

        assert len(categorized["format"]) == 2

    def test_parse_violations_other(self, mock_api_client):
        """Test parsing uncategorized violations."""
        judge = LLMJudge(mock_api_client)
        violations = [
            "Invalid data type for field x",
            "Unexpected behavior in output"
        ]

        categorized = judge.parse_violations(violations)

        assert len(categorized["other"]) == 2

    def test_parse_violations_mixed(self, mock_api_client):
        """Test parsing mixed violation types."""
        judge = LLMJudge(mock_api_client)
        violations = [
            "Duration exceeds max_duration: 10",
            "Missing required field: name",
            "Format is invalid",
            "Count below minimum: 5",
            "Unknown violation"
        ]

        categorized = judge.parse_violations(violations)

        assert len(categorized["over_max"]) == 1
        assert len(categorized["missing_field"]) == 1
        assert len(categorized["format"]) == 1
        assert len(categorized["under_min"]) == 1
        assert len(categorized["other"]) == 1

    def test_count_violations_by_type(self, mock_api_client):
        """Test counting violations by category."""
        judge = LLMJudge(mock_api_client)
        violations = [
            "Exceeds max: 100",
            "Exceeds max: 200",
            "Missing field: x",
            "Format error"
        ]

        counts = judge.count_violations_by_type(violations)

        assert counts["over_max"] == 2
        assert counts["missing_field"] == 1
        assert counts["format"] == 1
        assert counts["under_min"] == 0
        assert counts["other"] == 0

    def test_get_violation_summary_no_scores(self, mock_api_client):
        """Test violation summary with no scores."""
        judge = LLMJudge(mock_api_client)
        summary = judge.get_violation_summary([])

        assert "No evaluations to summarize" in summary

    def test_get_violation_summary_no_violations(self, mock_api_client):
        """Test violation summary with no violations."""
        judge = LLMJudge(mock_api_client)

        scores = [
            JudgeScore(
                model_evaluated="model-1",
                accuracy_score=95,
                format_score=100,
                compliance_score=90,
                overall_score=95,
                violations=[],
                reasoning="Perfect"
            )
        ]

        summary = judge.get_violation_summary(scores)

        assert "No violations found" in summary
        assert "Excellent" in summary

    def test_get_violation_summary_with_violations(self, mock_api_client):
        """Test violation summary with violations."""
        judge = LLMJudge(mock_api_client)

        scores = [
            JudgeScore(
                model_evaluated="model-1",
                accuracy_score=80,
                format_score=90,
                compliance_score=70,
                overall_score=80,
                violations=["Exceeds max: 100", "Missing field: x"],
                reasoning="Good but has issues"
            ),
            JudgeScore(
                model_evaluated="model-2",
                accuracy_score=75,
                format_score=85,
                compliance_score=60,
                overall_score=73,
                violations=["Exceeds max: 100", "Format error"],
                reasoning="Needs improvement"
            )
        ]

        summary = judge.get_violation_summary(scores)

        # Verify summary contains expected information
        assert "VIOLATION SUMMARY" in summary
        assert "Total Violations: 4" in summary
        assert "Exceeding Maximum:" in summary
        assert "Missing Fields:" in summary
        assert "Format Issues:" in summary
        assert "Most Common Violations:" in summary
        # "Exceeds max: 100" appears twice
        assert "[2x]" in summary

    def test_str_representation(self, mock_api_client):
        """Test string representation of LLMJudge."""
        judge = LLMJudge(mock_api_client, judge_model="test-model")
        assert str(judge) == "LLMJudge(model='test-model')"
        assert repr(judge) == "LLMJudge(model='test-model')"


class TestJudgePromptGeneration:
    """Test suite for judge prompt generation."""

    def test_prompt_includes_all_criteria(self, mock_api_client, sample_task):
        """Test that prompt includes all evaluation criteria."""
        judge = LLMJudge(mock_api_client)
        prompt = judge.build_judge_prompt(sample_task, "output", "input")

        for criterion in sample_task.evaluation_criteria:
            assert criterion in prompt

    def test_prompt_includes_judge_instructions(self, mock_api_client, sample_task):
        """Test that prompt includes judge instructions."""
        judge = LLMJudge(mock_api_client)
        prompt = judge.build_judge_prompt(sample_task, "output", "input")

        assert sample_task.judge_instructions in prompt

    def test_prompt_includes_scoring_guidelines(self, mock_api_client, sample_task):
        """Test that prompt includes scoring guidelines."""
        judge = LLMJudge(mock_api_client)
        prompt = judge.build_judge_prompt(sample_task, "output", "input")

        # Verify scoring rubric is present
        assert "Scoring Guidelines" in prompt or "Scoring Rubric" in prompt
        assert "accuracy_score" in prompt
        assert "format_score" in prompt
        assert "compliance_score" in prompt
        assert "overall_score" in prompt
