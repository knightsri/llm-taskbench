"""
Tests for Pydantic data models.
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from taskbench.core.models import (
    CompletionResponse,
    EvaluationResult,
    JudgeScore,
    ModelConfig,
    TaskDefinition,
)


class TestTaskDefinition:
    """Tests for TaskDefinition model."""

    def test_valid_task_definition(self):
        """Test creating a valid task definition."""
        task = TaskDefinition(
            name="test_task",
            description="A test task",
            input_type="transcript",
            output_format="csv",
            evaluation_criteria=["Accuracy", "Format"],
            constraints={"min_duration": 2},
            examples=[{"input": "test", "output": "result"}],
            judge_instructions="Evaluate based on criteria"
        )
        assert task.name == "test_task"
        assert task.input_type == "transcript"
        assert task.output_format == "csv"
        assert len(task.evaluation_criteria) == 2

    def test_invalid_input_type(self):
        """Test that invalid input_type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            TaskDefinition(
                name="test_task",
                description="A test task",
                input_type="invalid_type",
                output_format="csv",
                evaluation_criteria=["Accuracy"],
                judge_instructions="Test"
            )
        assert "input_type must be one of" in str(exc_info.value)

    def test_invalid_output_format(self):
        """Test that invalid output_format raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            TaskDefinition(
                name="test_task",
                description="A test task",
                input_type="transcript",
                output_format="invalid_format",
                evaluation_criteria=["Accuracy"],
                judge_instructions="Test"
            )
        assert "output_format must be one of" in str(exc_info.value)

    def test_optional_fields(self):
        """Test that optional fields work correctly."""
        task = TaskDefinition(
            name="test_task",
            description="A test task",
            input_type="text",
            output_format="json",
            evaluation_criteria=["Accuracy"],
            judge_instructions="Test"
        )
        assert task.constraints == {}
        assert task.examples == []

    def test_serialization(self):
        """Test model serialization to dict/JSON."""
        task = TaskDefinition(
            name="test_task",
            description="A test task",
            input_type="transcript",
            output_format="csv",
            evaluation_criteria=["Accuracy"],
            judge_instructions="Test"
        )
        task_dict = task.model_dump()
        assert task_dict["name"] == "test_task"
        assert isinstance(task_dict, dict)


class TestCompletionResponse:
    """Tests for CompletionResponse model."""

    def test_valid_completion_response(self):
        """Test creating a valid completion response."""
        response = CompletionResponse(
            content="Test response",
            model="anthropic/claude-sonnet-4.5",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            latency_ms=1500.5
        )
        assert response.content == "Test response"
        assert response.model == "anthropic/claude-sonnet-4.5"
        assert response.total_tokens == 150
        assert isinstance(response.timestamp, datetime)

    def test_default_timestamp(self):
        """Test that timestamp defaults to current time."""
        response = CompletionResponse(
            content="Test",
            model="test-model",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            latency_ms=1000
        )
        assert isinstance(response.timestamp, datetime)
        # Check it's within the last few seconds
        assert (datetime.now() - response.timestamp).total_seconds() < 5


class TestEvaluationResult:
    """Tests for EvaluationResult model."""

    def test_valid_evaluation_result(self):
        """Test creating a valid evaluation result."""
        result = EvaluationResult(
            model_name="anthropic/claude-sonnet-4.5",
            task_name="lecture_analysis",
            output="concept,start,end\nIntro,00:00:00,00:03:00",
            input_tokens=1000,
            output_tokens=500,
            total_tokens=1500,
            cost_usd=0.36,
            latency_ms=2500
        )
        assert result.model_name == "anthropic/claude-sonnet-4.5"
        assert result.status == "success"
        assert result.error is None

    def test_failed_evaluation_result(self):
        """Test creating a failed evaluation result."""
        result = EvaluationResult(
            model_name="test-model",
            task_name="test_task",
            output="",
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            cost_usd=0.0,
            latency_ms=0,
            status="failed",
            error="API timeout"
        )
        assert result.status == "failed"
        assert result.error == "API timeout"


class TestJudgeScore:
    """Tests for JudgeScore model."""

    def test_valid_judge_score(self):
        """Test creating a valid judge score."""
        score = JudgeScore(
            model_evaluated="anthropic/claude-sonnet-4.5",
            accuracy_score=95,
            format_score=100,
            compliance_score=98,
            overall_score=97,
            violations=[],
            reasoning="Excellent performance"
        )
        assert score.overall_score == 97
        assert len(score.violations) == 0

    def test_judge_score_with_violations(self):
        """Test judge score with violations."""
        score = JudgeScore(
            model_evaluated="test-model",
            accuracy_score=85,
            format_score=90,
            compliance_score=70,
            overall_score=80,
            violations=["Segment under 2 minutes", "Overlapping timestamps"],
            reasoning="Some issues found"
        )
        assert len(score.violations) == 2

    def test_score_range_validation(self):
        """Test that scores must be 0-100."""
        with pytest.raises(ValidationError):
            JudgeScore(
                model_evaluated="test-model",
                accuracy_score=150,  # Invalid
                format_score=100,
                compliance_score=100,
                overall_score=100,
                violations=[],
                reasoning="Test"
            )

        with pytest.raises(ValidationError):
            JudgeScore(
                model_evaluated="test-model",
                accuracy_score=95,
                format_score=-10,  # Invalid
                compliance_score=100,
                overall_score=95,
                violations=[],
                reasoning="Test"
            )


class TestModelConfig:
    """Tests for ModelConfig model."""

    def test_valid_model_config(self):
        """Test creating a valid model config."""
        config = ModelConfig(
            model_id="anthropic/claude-sonnet-4.5",
            display_name="Claude Sonnet 4.5",
            input_price_per_1m=3.00,
            output_price_per_1m=15.00,
            context_window=200000,
            provider="Anthropic"
        )
        assert config.model_id == "anthropic/claude-sonnet-4.5"
        assert config.provider == "Anthropic"

    def test_calculate_cost(self):
        """Test cost calculation method."""
        config = ModelConfig(
            model_id="anthropic/claude-sonnet-4.5",
            display_name="Claude Sonnet 4.5",
            input_price_per_1m=3.00,
            output_price_per_1m=15.00,
            context_window=200000,
            provider="Anthropic"
        )

        # Test with 100K input + 10K output tokens
        cost = config.calculate_cost(100_000, 10_000)
        expected = (100_000 / 1_000_000) * 3.00 + (10_000 / 1_000_000) * 15.00
        assert cost == round(expected, 2)
        assert cost == 0.45

    def test_calculate_cost_rounding(self):
        """Test that cost is rounded to $0.01 precision."""
        config = ModelConfig(
            model_id="test-model",
            display_name="Test Model",
            input_price_per_1m=2.50,
            output_price_per_1m=10.00,
            context_window=100000,
            provider="Test"
        )

        # Test with odd numbers that might have rounding issues
        cost = config.calculate_cost(12_345, 6_789)
        assert isinstance(cost, float)
        assert len(str(cost).split('.')[-1]) <= 2  # Max 2 decimal places
