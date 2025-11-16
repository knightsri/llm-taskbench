"""
Pydantic data models for LLM TaskBench.

This module defines all core data structures used throughout the application,
providing type safety and validation using Pydantic.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class TaskDefinition(BaseModel):
    """
    Represents a user-defined evaluation task.

    A task defines what the LLM should accomplish, the expected output format,
    evaluation criteria, and constraints that must be met.

    Example:
        ```python
        task = TaskDefinition(
            name="lecture_concept_extraction",
            description="Extract teaching concepts from lecture transcripts",
            input_type="transcript",
            output_format="csv",
            evaluation_criteria=["Timestamp accuracy", "Duration compliance"],
            constraints={"min_duration_minutes": 2, "max_duration_minutes": 7},
            examples=[],
            judge_instructions="Evaluate based on accuracy and format..."
        )
        ```
    """

    name: str = Field(..., description="Unique identifier for the task")
    description: str = Field(..., description="Human-readable task description")
    input_type: str = Field(..., description="Type of input data (transcript, text, csv, json)")
    output_format: str = Field(..., description="Expected output format (csv, json, markdown)")
    evaluation_criteria: List[str] = Field(
        ...,
        description="List of criteria used to evaluate model outputs"
    )
    constraints: Dict[str, Any] = Field(
        default_factory=dict,
        description="Constraints that outputs must satisfy"
    )
    examples: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Example inputs and expected outputs"
    )
    judge_instructions: str = Field(
        ...,
        description="Detailed instructions for the LLM-as-judge evaluator"
    )

    @field_validator("input_type")
    @classmethod
    def validate_input_type(cls, v: str) -> str:
        """Validate that input_type is one of the supported types."""
        allowed = ["transcript", "text", "csv", "json"]
        if v not in allowed:
            raise ValueError(f"input_type must be one of {allowed}, got '{v}'")
        return v

    @field_validator("output_format")
    @classmethod
    def validate_output_format(cls, v: str) -> str:
        """Validate that output_format is one of the supported formats."""
        allowed = ["csv", "json", "markdown"]
        if v not in allowed:
            raise ValueError(f"output_format must be one of {allowed}, got '{v}'")
        return v

    def __str__(self) -> str:
        return f"TaskDefinition(name='{self.name}', input_type='{self.input_type}')"

    def __repr__(self) -> str:
        return self.__str__()


class CompletionResponse(BaseModel):
    """
    API response from an LLM completion.

    Captures the response content, token usage, and performance metrics
    from a single API call to a language model.

    Example:
        ```python
        response = CompletionResponse(
            content="Extracted concepts: ...",
            model="anthropic/claude-sonnet-4.5",
            input_tokens=1500,
            output_tokens=500,
            total_tokens=2000,
            latency_ms=2345.67
        )
        ```
    """

    content: str = Field(..., description="The model's response text")
    model: str = Field(..., description="Model identifier (e.g., 'anthropic/claude-sonnet-4.5')")
    input_tokens: int = Field(..., description="Number of input tokens consumed")
    output_tokens: int = Field(..., description="Number of output tokens generated")
    total_tokens: int = Field(..., description="Total tokens (input + output)")
    latency_ms: float = Field(..., description="Response latency in milliseconds")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the response was received"
    )

    def __str__(self) -> str:
        return f"CompletionResponse(model='{self.model}', tokens={self.total_tokens})"

    def __repr__(self) -> str:
        return self.__str__()


class EvaluationResult(BaseModel):
    """
    Single model evaluation result.

    Contains the complete result of evaluating one model on a task, including
    the output, token usage, cost, and any errors that occurred.

    Example:
        ```python
        result = EvaluationResult(
            model_name="claude-sonnet-4.5",
            task_name="lecture_concept_extraction",
            output="concept,start_time,end_time\\n...",
            input_tokens=1500,
            output_tokens=500,
            total_tokens=2000,
            cost_usd=0.36,
            latency_ms=2345.67,
            status="success"
        )
        ```
    """

    model_name: str = Field(..., description="Model identifier")
    task_name: str = Field(..., description="Task identifier")
    output: str = Field(..., description="Model's output for the task")
    input_tokens: int = Field(..., description="Input tokens consumed")
    output_tokens: int = Field(..., description="Output tokens generated")
    total_tokens: int = Field(..., description="Total tokens used")
    cost_usd: float = Field(..., description="Cost in USD for this evaluation")
    latency_ms: float = Field(..., description="Execution latency in milliseconds")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the evaluation was performed"
    )
    status: str = Field(default="success", description="Evaluation status (success/failed)")
    error: Optional[str] = Field(default=None, description="Error message if status is failed")

    def __str__(self) -> str:
        return f"EvaluationResult(model='{self.model_name}', status='{self.status}')"

    def __repr__(self) -> str:
        return self.__str__()


class JudgeScore(BaseModel):
    """
    LLM-as-judge scoring result.

    Contains detailed scoring and reasoning from the LLM judge, including
    subscores for different evaluation dimensions and identified violations.

    Example:
        ```python
        score = JudgeScore(
            model_evaluated="claude-sonnet-4.5",
            accuracy_score=95,
            format_score=100,
            compliance_score=90,
            overall_score=95,
            violations=["One segment slightly over 7 minutes"],
            reasoning="Excellent extraction with minor duration issue..."
        )
        ```
    """

    model_evaluated: str = Field(..., description="Model that was evaluated")
    accuracy_score: int = Field(..., description="Accuracy score (0-100)")
    format_score: int = Field(..., description="Format compliance score (0-100)")
    compliance_score: int = Field(..., description="Constraint compliance score (0-100)")
    overall_score: int = Field(..., description="Overall score (0-100)")
    violations: List[str] = Field(
        default_factory=list,
        description="List of constraint violations found"
    )
    reasoning: str = Field(..., description="Detailed explanation of the scores")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the evaluation was performed"
    )

    @field_validator("accuracy_score", "format_score", "compliance_score", "overall_score")
    @classmethod
    def validate_score_range(cls, v: int) -> int:
        """Validate that all scores are in the range 0-100."""
        if not 0 <= v <= 100:
            raise ValueError(f"Score must be between 0 and 100, got {v}")
        return v

    def __str__(self) -> str:
        return f"JudgeScore(model='{self.model_evaluated}', overall={self.overall_score})"

    def __repr__(self) -> str:
        return self.__str__()


class ModelConfig(BaseModel):
    """
    Model pricing and configuration.

    Contains pricing information and metadata for a specific LLM model,
    used for cost calculation and model selection.

    Example:
        ```python
        config = ModelConfig(
            model_id="anthropic/claude-sonnet-4.5",
            display_name="Claude Sonnet 4.5",
            input_price_per_1m=3.00,
            output_price_per_1m=15.00,
            context_window=200000,
            provider="Anthropic"
        )
        ```
    """

    model_id: str = Field(..., description="Unique model identifier for API calls")
    display_name: str = Field(..., description="Human-readable model name")
    input_price_per_1m: float = Field(..., description="Input price per 1M tokens in USD")
    output_price_per_1m: float = Field(..., description="Output price per 1M tokens in USD")
    context_window: int = Field(..., description="Maximum context window size in tokens")
    provider: str = Field(..., description="Model provider (e.g., 'Anthropic', 'OpenAI')")

    def __str__(self) -> str:
        return f"ModelConfig(id='{self.model_id}', provider='{self.provider}')"

    def __repr__(self) -> str:
        return self.__str__()
