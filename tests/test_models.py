"""
Tests for Pydantic data models.

This module tests all data models for proper instantiation, validation,
and serialization.
"""

import json
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
        """Test creating a valid TaskDefinition."""
        task = TaskDefinition(
            name="test_task",
            description="A test task",
            input_type="transcript",
            output_format="csv",
            evaluation_criteria=["Accuracy", "Format"],
            constraints={"min_duration": 2, "max_duration": 7},
            examples=[{"input": "test", "output": "result"}],
            judge_instructions="Evaluate carefully"
        )
        assert task.name == "test_task"
        assert task.input_type == "transcript"
        assert task.output_format == "csv"
        assert len(task.evaluation_criteria) == 2
        assert task.constraints["min_duration"] == 2

    def test_invalid_input_type(self):
        """Test that invalid input_type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            TaskDefinition(
                name="test_task",
                description="A test task",
                input_type="invalid_type",
                output_format="csv",
                evaluation_criteria=["Accuracy"],
                judge_instructions="Evaluate carefully"
            )
        assert "input_type must be one of" in str(exc_info.value)

    def test_invalid_output_format(self):
        """Test that invalid output_format raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            TaskDefinition(
                name="test_task",
                description="A test task",
                input_type="text",
                output_format="invalid_format",
                evaluation_criteria=["Accuracy"],
                judge_instructions="Evaluate carefully"
            )
        assert "output_format must be one of" in str(exc_info.value)

    def test_optional_fields(self):
        """Test that optional fields work correctly."""
        task = TaskDefinition(
            name="minimal_task",
            description="Minimal task",
            input_type="text",
            output_format="json",
            evaluation_criteria=["Accuracy"],
            judge_instructions="Evaluate"
        )
        assert task.constraints == {}
        assert task.examples == []

    def test_serialization_to_dict(self):
        """Test model serialization to dict."""
        task = TaskDefinition(
            name="test_task",
            description="A test task",
            input_type="text",
            output_format="json",
            evaluation_criteria=["Accuracy"],
            judge_instructions="Evaluate"
        )
        task_dict = task.model_dump()
        assert isinstance(task_dict, dict)
        assert task_dict["name"] == "test_task"

    def test_serialization_to_json(self):
        """Test model serialization to JSON."""
        task = TaskDefinition(
            name="test_task",
            description="A test task",
            input_type="text",
            output_format="json",
            evaluation_criteria=["Accuracy"],
            judge_instructions="Evaluate"
        )
        task_json = task.model_dump_json()
        assert isinstance(task_json, str)
        parsed = json.loads(task_json)
        assert parsed["name"] == "test_task"


class TestCompletionResponse:
    """Tests for CompletionResponse model."""

    def test_valid_completion_response(self):
        """Test creating a valid CompletionResponse."""
        response = CompletionResponse(
            content="Test response",
            model="test-model",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            latency_ms=1234.56
        )
        assert response.content == "Test response"
        assert response.model == "test-model"
        assert response.total_tokens == 150
        assert response.latency_ms == 1234.56
        assert isinstance(response.timestamp, datetime)

    def test_timestamp_defaults_to_now(self):
        """Test that timestamp defaults to current time."""
        before = datetime.now()
        response = CompletionResponse(
            content="Test",
            model="test-model",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            latency_ms=100.0
        )
        after = datetime.now()
        assert before <= response.timestamp <= after

    def test_serialization(self):
        """Test CompletionResponse serialization."""
        response = CompletionResponse(
            content="Test",
            model="test-model",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            latency_ms=100.0
        )
        response_dict = response.model_dump()
        assert response_dict["content"] == "Test"
        assert response_dict["total_tokens"] == 150


class TestEvaluationResult:
    """Tests for EvaluationResult model."""

    def test_valid_evaluation_result(self):
        """Test creating a valid EvaluationResult."""
        result = EvaluationResult(
            model_name="test-model",
            task_name="test-task",
            output="test output",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost_usd=0.25,
            latency_ms=1234.56
        )
        assert result.model_name == "test-model"
        assert result.task_name == "test-task"
        assert result.status == "success"
        assert result.error is None

    def test_failed_status_with_error(self):
        """Test EvaluationResult with failed status and error message."""
        result = EvaluationResult(
            model_name="test-model",
            task_name="test-task",
            output="",
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            cost_usd=0.0,
            latency_ms=0.0,
            status="failed",
            error="API timeout"
        )
        assert result.status == "failed"
        assert result.error == "API timeout"

    def test_serialization(self):
        """Test EvaluationResult serialization."""
        result = EvaluationResult(
            model_name="test-model",
            task_name="test-task",
            output="output",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost_usd=0.25,
            latency_ms=1234.56
        )
        result_dict = result.model_dump()
        assert result_dict["model_name"] == "test-model"
        assert result_dict["cost_usd"] == 0.25


class TestJudgeScore:
    """Tests for JudgeScore model."""

    def test_valid_judge_score(self):
        """Test creating a valid JudgeScore."""
        score = JudgeScore(
            model_evaluated="test-model",
            accuracy_score=95,
            format_score=100,
            compliance_score=90,
            overall_score=95,
            violations=["Minor issue"],
            reasoning="Good performance overall"
        )
        assert score.model_evaluated == "test-model"
        assert score.accuracy_score == 95
        assert score.overall_score == 95
        assert len(score.violations) == 1

    def test_score_validation_upper_bound(self):
        """Test that scores above 100 raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            JudgeScore(
                model_evaluated="test-model",
                accuracy_score=101,
                format_score=100,
                compliance_score=100,
                overall_score=100,
                violations=[],
                reasoning="Test"
            )
        assert "Score must be between 0 and 100" in str(exc_info.value)

    def test_score_validation_lower_bound(self):
        """Test that scores below 0 raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            JudgeScore(
                model_evaluated="test-model",
                accuracy_score=50,
                format_score=-1,
                compliance_score=50,
                overall_score=50,
                violations=[],
                reasoning="Test"
            )
        assert "Score must be between 0 and 100" in str(exc_info.value)

    def test_empty_violations(self):
        """Test JudgeScore with no violations."""
        score = JudgeScore(
            model_evaluated="test-model",
            accuracy_score=100,
            format_score=100,
            compliance_score=100,
            overall_score=100,
            reasoning="Perfect score"
        )
        assert score.violations == []

    def test_serialization(self):
        """Test JudgeScore serialization."""
        score = JudgeScore(
            model_evaluated="test-model",
            accuracy_score=95,
            format_score=100,
            compliance_score=90,
            overall_score=95,
            violations=[],
            reasoning="Good"
        )
        score_dict = score.model_dump()
        assert score_dict["overall_score"] == 95


class TestModelConfig:
    """Tests for ModelConfig model."""

    def test_valid_model_config(self):
        """Test creating a valid ModelConfig."""
        config = ModelConfig(
            model_id="anthropic/claude-sonnet-4.5",
            display_name="Claude Sonnet 4.5",
            input_price_per_1m=3.00,
            output_price_per_1m=15.00,
            context_window=200000,
            provider="Anthropic"
        )
        assert config.model_id == "anthropic/claude-sonnet-4.5"
        assert config.display_name == "Claude Sonnet 4.5"
        assert config.input_price_per_1m == 3.00
        assert config.output_price_per_1m == 15.00
        assert config.context_window == 200000
        assert config.provider == "Anthropic"

    def test_serialization(self):
        """Test ModelConfig serialization."""
        config = ModelConfig(
            model_id="test/model",
            display_name="Test Model",
            input_price_per_1m=1.00,
            output_price_per_1m=2.00,
            context_window=100000,
            provider="Test"
        )
        config_dict = config.model_dump()
        assert config_dict["model_id"] == "test/model"
        assert config_dict["input_price_per_1m"] == 1.00
