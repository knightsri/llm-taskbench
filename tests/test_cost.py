"""
Tests for CostTracker.

This module tests the CostTracker class including:
- Initialization and model loading
- Cost calculation
- Evaluation tracking
- Statistics generation
- Cost breakdown by model
- Reset functionality
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from taskbench.core.models import EvaluationResult, ModelConfig
from taskbench.evaluation.cost import CostTracker


@pytest.fixture
def temp_models_config():
    """Create a temporary models configuration file."""
    models_data = {
        "models": [
            {
                "model_id": "test/model-1",
                "display_name": "Test Model 1",
                "input_price_per_1m": 3.0,
                "output_price_per_1m": 15.0,
                "context_window": 200000,
                "provider": "TestProvider"
            },
            {
                "model_id": "test/model-2",
                "display_name": "Test Model 2",
                "input_price_per_1m": 0.5,
                "output_price_per_1m": 1.5,
                "context_window": 100000,
                "provider": "TestProvider"
            },
            {
                "model_id": "test/expensive-model",
                "display_name": "Expensive Model",
                "input_price_per_1m": 10.0,
                "output_price_per_1m": 30.0,
                "context_window": 500000,
                "provider": "TestProvider"
            }
        ]
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(models_data, f)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink()


@pytest.fixture
def sample_evaluation_result():
    """Create a sample evaluation result."""
    return EvaluationResult(
        model_name="test/model-1",
        task_name="test_task",
        output="test output",
        input_tokens=1000,
        output_tokens=500,
        total_tokens=1500,
        cost_usd=0.01,
        latency_ms=1000.0,
        status="success"
    )


class TestCostTracker:
    """Test suite for CostTracker."""

    def test_init_with_config_path(self, temp_models_config):
        """Test initialization with custom config path."""
        tracker = CostTracker(models_config_path=temp_models_config)

        assert len(tracker.models) == 3
        assert "test/model-1" in tracker.models
        assert "test/model-2" in tracker.models
        assert "test/expensive-model" in tracker.models

    def test_init_without_config_raises_error(self):
        """Test that initialization fails with non-existent config."""
        with pytest.raises(FileNotFoundError, match="Models configuration file not found"):
            CostTracker(models_config_path="/nonexistent/path/models.yaml")

    def test_init_with_invalid_yaml(self):
        """Test initialization with invalid YAML."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Failed to parse"):
                CostTracker(models_config_path=temp_path)
        finally:
            Path(temp_path).unlink()

    def test_init_with_missing_models_key(self):
        """Test initialization with config missing 'models' key."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({"wrong_key": []}, f)
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="must contain 'models' key"):
                CostTracker(models_config_path=temp_path)
        finally:
            Path(temp_path).unlink()

    def test_load_models(self, temp_models_config):
        """Test that models are loaded correctly."""
        tracker = CostTracker(models_config_path=temp_models_config)

        model = tracker.models["test/model-1"]
        assert model.display_name == "Test Model 1"
        assert model.input_price_per_1m == 3.0
        assert model.output_price_per_1m == 15.0
        assert model.context_window == 200000
        assert model.provider == "TestProvider"

    def test_get_model_config_exists(self, temp_models_config):
        """Test retrieving existing model config."""
        tracker = CostTracker(models_config_path=temp_models_config)

        config = tracker.get_model_config("test/model-1")
        assert config is not None
        assert config.model_id == "test/model-1"
        assert isinstance(config, ModelConfig)

    def test_get_model_config_not_exists(self, temp_models_config):
        """Test retrieving non-existent model config."""
        tracker = CostTracker(models_config_path=temp_models_config)

        config = tracker.get_model_config("nonexistent/model")
        assert config is None

    def test_calculate_cost(self, temp_models_config):
        """Test cost calculation."""
        tracker = CostTracker(models_config_path=temp_models_config)

        # Model 1: $3 per 1M input, $15 per 1M output
        # 1000 input tokens = $0.003
        # 500 output tokens = $0.0075
        # Total = $0.0105, rounded to $0.01
        cost = tracker.calculate_cost("test/model-1", 1000, 500)
        assert cost == 0.01

    def test_calculate_cost_expensive_model(self, temp_models_config):
        """Test cost calculation for expensive model."""
        tracker = CostTracker(models_config_path=temp_models_config)

        # Expensive model: $10 per 1M input, $30 per 1M output
        # 10000 input tokens = $0.10
        # 5000 output tokens = $0.15
        # Total = $0.25
        cost = tracker.calculate_cost("test/expensive-model", 10000, 5000)
        assert cost == 0.25

    def test_calculate_cost_unknown_model(self, temp_models_config):
        """Test cost calculation for unknown model raises error."""
        tracker = CostTracker(models_config_path=temp_models_config)

        with pytest.raises(ValueError, match="not found in configuration"):
            tracker.calculate_cost("unknown/model", 1000, 500)

    def test_track_evaluation(self, temp_models_config, sample_evaluation_result):
        """Test tracking a single evaluation."""
        tracker = CostTracker(models_config_path=temp_models_config)

        tracker.track_evaluation(sample_evaluation_result)

        assert len(tracker.evaluations) == 1
        assert tracker.total_input_tokens == 1000
        assert tracker.total_output_tokens == 500
        assert tracker.total_cost == 0.01

    def test_track_multiple_evaluations(self, temp_models_config):
        """Test tracking multiple evaluations."""
        tracker = CostTracker(models_config_path=temp_models_config)

        result1 = EvaluationResult(
            model_name="test/model-1",
            task_name="task1",
            output="output1",
            input_tokens=1000,
            output_tokens=500,
            total_tokens=1500,
            cost_usd=0.01,
            latency_ms=1000.0,
            status="success"
        )

        result2 = EvaluationResult(
            model_name="test/model-2",
            task_name="task2",
            output="output2",
            input_tokens=2000,
            output_tokens=1000,
            total_tokens=3000,
            cost_usd=0.02,
            latency_ms=1500.0,
            status="success"
        )

        tracker.track_evaluation(result1)
        tracker.track_evaluation(result2)

        assert len(tracker.evaluations) == 2
        assert tracker.total_input_tokens == 3000
        assert tracker.total_output_tokens == 1500
        assert tracker.total_cost == 0.03

    def test_get_total_cost(self, temp_models_config, sample_evaluation_result):
        """Test getting total cost."""
        tracker = CostTracker(models_config_path=temp_models_config)

        assert tracker.get_total_cost() == 0.0

        tracker.track_evaluation(sample_evaluation_result)
        assert tracker.get_total_cost() == 0.01

    def test_get_cost_breakdown(self, temp_models_config):
        """Test cost breakdown by model."""
        tracker = CostTracker(models_config_path=temp_models_config)

        result1 = EvaluationResult(
            model_name="test/model-1",
            task_name="task",
            output="output",
            input_tokens=1000,
            output_tokens=500,
            total_tokens=1500,
            cost_usd=0.01,
            latency_ms=1000.0,
            status="success"
        )

        result2 = EvaluationResult(
            model_name="test/model-1",
            task_name="task",
            output="output",
            input_tokens=2000,
            output_tokens=1000,
            total_tokens=3000,
            cost_usd=0.02,
            latency_ms=1500.0,
            status="success"
        )

        result3 = EvaluationResult(
            model_name="test/model-2",
            task_name="task",
            output="output",
            input_tokens=500,
            output_tokens=250,
            total_tokens=750,
            cost_usd=0.005,
            latency_ms=800.0,
            status="success"
        )

        tracker.track_evaluation(result1)
        tracker.track_evaluation(result2)
        tracker.track_evaluation(result3)

        breakdown = tracker.get_cost_breakdown()

        assert "test/model-1" in breakdown
        assert "test/model-2" in breakdown

        model1_stats = breakdown["test/model-1"]
        assert model1_stats["cost"] == 0.03
        assert model1_stats["input_tokens"] == 3000
        assert model1_stats["output_tokens"] == 1500
        assert model1_stats["evaluations"] == 2

        model2_stats = breakdown["test/model-2"]
        assert model2_stats["cost"] == 0.01  # Rounded from 0.005
        assert model2_stats["evaluations"] == 1

    def test_get_statistics(self, temp_models_config):
        """Test comprehensive statistics."""
        tracker = CostTracker(models_config_path=temp_models_config)

        # Track multiple evaluations
        for i in range(3):
            result = EvaluationResult(
                model_name=f"test/model-{(i % 2) + 1}",
                task_name=f"task{i}",
                output=f"output{i}",
                input_tokens=1000,
                output_tokens=500,
                total_tokens=1500,
                cost_usd=0.01,
                latency_ms=1000.0,
                status="success"
            )
            tracker.track_evaluation(result)

        stats = tracker.get_statistics()

        assert stats["total_cost"] == 0.03
        assert stats["total_evaluations"] == 3
        assert stats["total_input_tokens"] == 3000
        assert stats["total_output_tokens"] == 1500
        assert stats["total_tokens"] == 4500
        assert stats["average_cost_per_evaluation"] == 0.01
        assert stats["models_used"] == 2
        assert "breakdown" in stats

    def test_get_statistics_empty(self, temp_models_config):
        """Test statistics with no evaluations."""
        tracker = CostTracker(models_config_path=temp_models_config)

        stats = tracker.get_statistics()

        assert stats["total_cost"] == 0.0
        assert stats["total_evaluations"] == 0
        assert stats["average_cost_per_evaluation"] == 0.0
        assert stats["models_used"] == 0

    def test_reset(self, temp_models_config, sample_evaluation_result):
        """Test resetting tracker."""
        tracker = CostTracker(models_config_path=temp_models_config)

        tracker.track_evaluation(sample_evaluation_result)
        assert tracker.get_total_cost() == 0.01
        assert len(tracker.evaluations) == 1

        tracker.reset()

        assert tracker.get_total_cost() == 0.0
        assert len(tracker.evaluations) == 0
        assert tracker.total_input_tokens == 0
        assert tracker.total_output_tokens == 0
        assert tracker.total_cost == 0.0

    def test_export_summary(self, temp_models_config):
        """Test exporting summary."""
        tracker = CostTracker(models_config_path=temp_models_config)

        result1 = EvaluationResult(
            model_name="test/model-1",
            task_name="task",
            output="output",
            input_tokens=1000,
            output_tokens=500,
            total_tokens=1500,
            cost_usd=0.01,
            latency_ms=1000.0,
            status="success"
        )

        result2 = EvaluationResult(
            model_name="test/model-2",
            task_name="task",
            output="output",
            input_tokens=2000,
            output_tokens=1000,
            total_tokens=3000,
            cost_usd=0.02,
            latency_ms=1500.0,
            status="success"
        )

        tracker.track_evaluation(result1)
        tracker.track_evaluation(result2)

        summary = tracker.export_summary()

        # Verify summary contains key information
        assert isinstance(summary, str)
        assert "COST SUMMARY" in summary
        assert "$0.03" in summary  # Total cost
        assert "2" in summary  # Total evaluations
        assert "test/model-1" in summary
        assert "test/model-2" in summary
        assert "BREAKDOWN BY MODEL" in summary

    def test_export_summary_empty(self, temp_models_config):
        """Test exporting summary with no evaluations."""
        tracker = CostTracker(models_config_path=temp_models_config)

        summary = tracker.export_summary()

        assert "COST SUMMARY" in summary
        assert "$0.00" in summary
        assert "0" in summary

    def test_str_representation(self, temp_models_config):
        """Test string representation."""
        tracker = CostTracker(models_config_path=temp_models_config)

        str_repr = str(tracker)
        assert "CostTracker" in str_repr
        assert "models=3" in str_repr
        assert "evaluations=0" in str_repr
        assert "$0.00" in str_repr

    def test_repr(self, temp_models_config):
        """Test repr."""
        tracker = CostTracker(models_config_path=temp_models_config)

        repr_str = repr(tracker)
        assert repr_str == str(tracker)


class TestModelConfig:
    """Test suite for ModelConfig."""

    def test_calculate_cost(self):
        """Test cost calculation in ModelConfig."""
        config = ModelConfig(
            model_id="test/model",
            display_name="Test Model",
            input_price_per_1m=3.0,
            output_price_per_1m=15.0,
            context_window=200000,
            provider="TestProvider"
        )

        # 1000 input tokens = $0.003
        # 500 output tokens = $0.0075
        # Total = $0.0105, rounded to $0.01
        cost = config.calculate_cost(1000, 500)
        assert cost == 0.01

    def test_calculate_cost_zero_tokens(self):
        """Test cost calculation with zero tokens."""
        config = ModelConfig(
            model_id="test/model",
            display_name="Test Model",
            input_price_per_1m=3.0,
            output_price_per_1m=15.0,
            context_window=200000,
            provider="TestProvider"
        )

        cost = config.calculate_cost(0, 0)
        assert cost == 0.0

    def test_calculate_cost_large_numbers(self):
        """Test cost calculation with large token counts."""
        config = ModelConfig(
            model_id="test/model",
            display_name="Test Model",
            input_price_per_1m=5.0,
            output_price_per_1m=10.0,
            context_window=200000,
            provider="TestProvider"
        )

        # 1,000,000 input tokens = $5.00
        # 500,000 output tokens = $5.00
        # Total = $10.00
        cost = config.calculate_cost(1_000_000, 500_000)
        assert cost == 10.0

    def test_str_representation(self):
        """Test string representation."""
        config = ModelConfig(
            model_id="test/model",
            display_name="Test Model",
            input_price_per_1m=3.0,
            output_price_per_1m=15.0,
            context_window=200000,
            provider="TestProvider"
        )

        str_repr = str(config)
        assert "test/model" in str_repr
        assert "TestProvider" in str_repr
