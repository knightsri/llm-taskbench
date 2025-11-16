"""
Tests for ModelExecutor.

This module tests the ModelExecutor class including:
- Prompt building from task definitions
- Single model execution
- Multiple model execution
- Cost tracking
- Error handling
- Progress display
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from taskbench.core.models import TaskDefinition, EvaluationResult, CompletionResponse
from taskbench.evaluation.executor import ModelExecutor
from taskbench.evaluation.cost import CostTracker
from taskbench.api.client import OpenRouterAPIError


@pytest.fixture
def sample_task():
    """Create a sample task definition for testing."""
    return TaskDefinition(
        name="test_task",
        description="Test task description",
        input_type="text",
        output_format="json",
        evaluation_criteria=["Accuracy", "Completeness"],
        constraints={"max_length": 100, "min_items": 3},
        examples=[],
        judge_instructions="Evaluate based on criteria"
    )


@pytest.fixture
def sample_task_with_examples():
    """Create a task with examples."""
    return TaskDefinition(
        name="test_task",
        description="Test task with examples",
        input_type="text",
        output_format="csv",
        evaluation_criteria=["Accuracy"],
        constraints={"max_rows": 10},
        examples=[
            {
                "input": "Sample input",
                "expected_output": "Sample output",
                "notes": "Example note"
            }
        ],
        judge_instructions="Evaluate based on criteria"
    )


class TestModelExecutor:
    """Test suite for ModelExecutor."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        executor = ModelExecutor()
        assert executor.api_key is None
        assert executor.timeout == 120.0
        assert isinstance(executor.cost_tracker, CostTracker)

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        executor = ModelExecutor(api_key="test-key")
        assert executor.api_key == "test-key"

    def test_init_with_cost_tracker(self):
        """Test initialization with custom cost tracker."""
        tracker = CostTracker()
        executor = ModelExecutor(cost_tracker=tracker)
        assert executor.cost_tracker is tracker

    def test_init_with_timeout(self):
        """Test initialization with custom timeout."""
        executor = ModelExecutor(timeout=60.0)
        assert executor.timeout == 60.0

    def test_build_prompt_basic(self, sample_task):
        """Test building a basic prompt."""
        executor = ModelExecutor()
        input_data = "Test input data"

        prompt = executor.build_prompt(sample_task, input_data)

        # Verify prompt contains key elements
        assert "test_task" in prompt
        assert "Test task description" in prompt
        assert "JSON" in prompt.upper()
        assert "Test input data" in prompt
        assert "max_length" in prompt
        assert "min_items" in prompt

    def test_build_prompt_with_examples(self, sample_task_with_examples):
        """Test building a prompt with examples."""
        executor = ModelExecutor()
        input_data = "Test input"

        prompt = executor.build_prompt(sample_task_with_examples, input_data)

        # Verify examples are included
        assert "Examples" in prompt
        assert "Sample input" in prompt
        assert "Sample output" in prompt
        assert "Example note" in prompt

    def test_build_prompt_with_constraints(self, sample_task):
        """Test that constraints are prominently featured."""
        executor = ModelExecutor()
        prompt = executor.build_prompt(sample_task, "input")

        # Verify constraints section is present and emphasized
        assert "CONSTRAINTS" in prompt.upper()
        assert "MUST FOLLOW" in prompt.upper()
        assert "Max Length" in prompt
        assert "Min Items" in prompt

    @pytest.mark.asyncio
    async def test_execute_success(self, sample_task):
        """Test successful execution on a single model."""
        executor = ModelExecutor(api_key="test-key")

        # Mock the OpenRouterClient
        mock_response = CompletionResponse(
            content='{"result": "success"}',
            model="test-model",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            latency_ms=1000.0
        )

        with patch('taskbench.evaluation.executor.OpenRouterClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.complete.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            result = await executor.execute("test-model", sample_task, "input data")

            # Verify result
            assert isinstance(result, EvaluationResult)
            assert result.model_name == "test-model"
            assert result.task_name == "test_task"
            assert result.status == "success"
            assert result.output == '{"result": "success"}'
            assert result.input_tokens == 100
            assert result.output_tokens == 50
            assert result.total_tokens == 150
            assert result.latency_ms == 1000.0
            assert result.cost_usd >= 0.0
            assert result.error is None

    @pytest.mark.asyncio
    async def test_execute_api_error(self, sample_task):
        """Test execution with API error."""
        executor = ModelExecutor(api_key="test-key")

        with patch('taskbench.evaluation.executor.OpenRouterClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.complete.side_effect = OpenRouterAPIError("API error", status_code=500)
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            result = await executor.execute("test-model", sample_task, "input data")

            # Verify failed result
            assert result.status == "failed"
            assert "API error" in result.error
            assert result.cost_usd == 0.0
            assert result.total_tokens == 0

    @pytest.mark.asyncio
    async def test_execute_unexpected_error(self, sample_task):
        """Test execution with unexpected error."""
        executor = ModelExecutor(api_key="test-key")

        with patch('taskbench.evaluation.executor.OpenRouterClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.complete.side_effect = Exception("Unexpected error")
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            result = await executor.execute("test-model", sample_task, "input data")

            # Verify failed result
            assert result.status == "failed"
            assert "Unexpected error" in result.error

    @pytest.mark.asyncio
    async def test_execute_cost_calculation_error(self, sample_task):
        """Test execution when cost calculation fails."""
        executor = ModelExecutor(api_key="test-key")

        mock_response = CompletionResponse(
            content="output",
            model="unknown-model",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            latency_ms=1000.0
        )

        with patch('taskbench.evaluation.executor.OpenRouterClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.complete.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            # Mock cost tracker to raise ValueError
            mock_tracker = MagicMock()
            mock_tracker.calculate_cost.side_effect = ValueError("Unknown model")
            executor.cost_tracker = mock_tracker

            result = await executor.execute("unknown-model", sample_task, "input data")

            # Should still succeed but with 0 cost
            assert result.status == "success"
            assert result.cost_usd == 0.0

    @pytest.mark.asyncio
    async def test_evaluate_multiple_success(self, sample_task):
        """Test evaluation on multiple models."""
        executor = ModelExecutor(api_key="test-key")

        # Mock responses for different models
        responses = {
            "model-1": CompletionResponse(
                content="output1", model="model-1",
                input_tokens=100, output_tokens=50, total_tokens=150, latency_ms=1000.0
            ),
            "model-2": CompletionResponse(
                content="output2", model="model-2",
                input_tokens=200, output_tokens=100, total_tokens=300, latency_ms=2000.0
            ),
            "model-3": CompletionResponse(
                content="output3", model="model-3",
                input_tokens=150, output_tokens=75, total_tokens=225, latency_ms=1500.0
            )
        }

        async def mock_execute(model_id, task, input_data):
            if model_id in responses:
                return EvaluationResult(
                    model_name=model_id,
                    task_name=task.name,
                    output=responses[model_id].content,
                    input_tokens=responses[model_id].input_tokens,
                    output_tokens=responses[model_id].output_tokens,
                    total_tokens=responses[model_id].total_tokens,
                    cost_usd=0.01,
                    latency_ms=responses[model_id].latency_ms,
                    status="success"
                )

        with patch.object(executor, 'execute', side_effect=mock_execute):
            model_ids = ["model-1", "model-2", "model-3"]
            results = await executor.evaluate_multiple(
                model_ids, sample_task, "input data", show_progress=False
            )

            # Verify results
            assert len(results) == 3
            assert all(r.status == "success" for r in results)
            assert results[0].model_name == "model-1"
            assert results[1].model_name == "model-2"
            assert results[2].model_name == "model-3"

    @pytest.mark.asyncio
    async def test_evaluate_multiple_with_failures(self, sample_task):
        """Test evaluation with some failures."""
        executor = ModelExecutor(api_key="test-key")

        async def mock_execute(model_id, task, input_data):
            if model_id == "failing-model":
                return EvaluationResult(
                    model_name=model_id,
                    task_name=task.name,
                    output="",
                    input_tokens=0,
                    output_tokens=0,
                    total_tokens=0,
                    cost_usd=0.0,
                    latency_ms=0.0,
                    status="failed",
                    error="API error"
                )
            else:
                return EvaluationResult(
                    model_name=model_id,
                    task_name=task.name,
                    output="success",
                    input_tokens=100,
                    output_tokens=50,
                    total_tokens=150,
                    cost_usd=0.01,
                    latency_ms=1000.0,
                    status="success"
                )

        with patch.object(executor, 'execute', side_effect=mock_execute):
            model_ids = ["good-model", "failing-model", "another-good-model"]
            results = await executor.evaluate_multiple(
                model_ids, sample_task, "input data", show_progress=False
            )

            # Verify results
            assert len(results) == 3
            successful = [r for r in results if r.status == "success"]
            failed = [r for r in results if r.status == "failed"]
            assert len(successful) == 2
            assert len(failed) == 1
            assert failed[0].model_name == "failing-model"

    @pytest.mark.asyncio
    async def test_evaluate_multiple_tracks_cost(self, sample_task):
        """Test that evaluate_multiple tracks costs."""
        executor = ModelExecutor(api_key="test-key")

        async def mock_execute(model_id, task, input_data):
            return EvaluationResult(
                model_name=model_id,
                task_name=task.name,
                output="output",
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
                cost_usd=0.05,
                latency_ms=1000.0,
                status="success"
            )

        with patch.object(executor, 'execute', side_effect=mock_execute):
            model_ids = ["model-1", "model-2"]
            results = await executor.evaluate_multiple(
                model_ids, sample_task, "input data", show_progress=False
            )

            # Verify cost tracking
            total_cost = executor.cost_tracker.get_total_cost()
            assert total_cost == 0.10  # 2 models × $0.05

    def test_get_cost_summary(self):
        """Test cost summary retrieval."""
        executor = ModelExecutor()

        # Track some evaluations
        result1 = EvaluationResult(
            model_name="model-1",
            task_name="task",
            output="output",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost_usd=0.05,
            latency_ms=1000.0,
            status="success"
        )
        executor.cost_tracker.track_evaluation(result1)

        summary = executor.get_cost_summary()

        # Verify summary is a string and contains cost info
        assert isinstance(summary, str)
        assert "COST SUMMARY" in summary
        assert "$0.05" in summary

    def test_reset_tracker(self):
        """Test resetting the cost tracker."""
        executor = ModelExecutor()

        # Track an evaluation
        result = EvaluationResult(
            model_name="model-1",
            task_name="task",
            output="output",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost_usd=0.05,
            latency_ms=1000.0,
            status="success"
        )
        executor.cost_tracker.track_evaluation(result)
        assert executor.cost_tracker.get_total_cost() == 0.05

        # Reset
        executor.reset_tracker()
        assert executor.cost_tracker.get_total_cost() == 0.0
        assert len(executor.cost_tracker.evaluations) == 0


class TestPromptGeneration:
    """Test suite specifically for prompt generation logic."""

    def test_prompt_structure(self, sample_task):
        """Test overall prompt structure."""
        executor = ModelExecutor()
        prompt = executor.build_prompt(sample_task, "input")

        # Verify all major sections are present
        assert "# Task:" in prompt
        assert "## Output Format" in prompt
        assert "## IMPORTANT CONSTRAINTS" in prompt
        assert "## Input Data" in prompt
        assert "## Instructions" in prompt

    def test_prompt_emphasizes_format(self):
        """Test that output format is properly emphasized."""
        task = TaskDefinition(
            name="csv_task",
            description="CSV output task",
            input_type="text",
            output_format="csv",
            evaluation_criteria=["Accuracy"],
            constraints={},
            examples=[],
            judge_instructions="Evaluate"
        )

        executor = ModelExecutor()
        prompt = executor.build_prompt(task, "input")

        # CSV should be in uppercase and emphasized
        assert "CSV" in prompt
        assert "MUST return" in prompt or "MUST" in prompt

    def test_prompt_without_constraints(self):
        """Test prompt generation when no constraints are defined."""
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

        executor = ModelExecutor()
        prompt = executor.build_prompt(task, "input")

        # Should still have basic structure even without constraints
        assert "# Task:" in prompt
        assert "Simple task" in prompt
        assert "JSON" in prompt

    def test_prompt_without_examples(self, sample_task):
        """Test prompt generation when no examples are provided."""
        executor = ModelExecutor()
        prompt = executor.build_prompt(sample_task, "input")

        # Should not have examples section
        assert "## Examples" not in prompt
