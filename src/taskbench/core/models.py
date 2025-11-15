"""
Pydantic data models for LLM TaskBench.

This module defines all core data structures using Pydantic for type safety
and validation.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class TaskDefinition(BaseModel):
    """
    Represents a user-defined evaluation task.

    A task definition specifies what the LLM should do, how its output should
    be formatted, and how it should be evaluated.

    Example:
        >>> task = TaskDefinition(
        ...     name="lecture_analysis",
        ...     description="Extract teaching concepts from lecture transcripts",
        ...     input_type="transcript",
        ...     output_format="csv",
        ...     evaluation_criteria=["Accuracy", "Format compliance"],
        ...     constraints={"min_duration_minutes": 2, "max_duration_minutes": 7},
        ...     examples=[],
        ...     judge_instructions="Evaluate based on accuracy and format"
        ... )
    """

    name: str = Field(..., description="Unique name for this task")
    description: str = Field(..., description="Human-readable description of the task")
    input_type: str = Field(
        ...,
        description="Type of input data (transcript, text, csv, json)"
    )
    output_format: str = Field(
        ...,
        description="Expected output format (csv, json, markdown)"
    )
    evaluation_criteria: List[str] = Field(
        ...,
        description="List of criteria for evaluating model outputs"
    )
    constraints: Dict[str, Any] = Field(
        default_factory=dict,
        description="Task-specific constraints (e.g., duration limits, field requirements)"
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
        """Validate that input_type is one of the allowed values."""
        allowed = ["transcript", "text", "csv", "json"]
        if v not in allowed:
            raise ValueError(f"input_type must be one of {allowed}, got {v}")
        return v

    @field_validator("output_format")
    @classmethod
    def validate_output_format(cls, v: str) -> str:
        """Validate that output_format is one of the allowed values."""
        allowed = ["csv", "json", "markdown"]
        if v not in allowed:
            raise ValueError(f"output_format must be one of {allowed}, got {v}")
        return v

    def __str__(self) -> str:
        return f"TaskDefinition(name='{self.name}', input_type='{self.input_type}', output_format='{self.output_format}')"

    def __repr__(self) -> str:
        return self.__str__()


class CompletionResponse(BaseModel):
    """
    API response from LLM completion.

    Captures the model's response along with token usage and performance metrics.

    Example:
        >>> response = CompletionResponse(
        ...     content="Extracted concepts...",
        ...     model="anthropic/claude-sonnet-4.5",
        ...     input_tokens=1000,
        ...     output_tokens=500,
        ...     total_tokens=1500,
        ...     latency_ms=2500
        ... )
    """

    content: str = Field(..., description="The model's generated text response")
    model: str = Field(..., description="Model identifier used for generation")
    input_tokens: int = Field(..., description="Number of input tokens consumed")
    output_tokens: int = Field(..., description="Number of output tokens generated")
    total_tokens: int = Field(..., description="Total tokens (input + output)")
    latency_ms: float = Field(..., description="Response latency in milliseconds")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the response was received"
    )

    def __str__(self) -> str:
        return f"CompletionResponse(model='{self.model}', tokens={self.total_tokens}, latency={self.latency_ms}ms)"

    def __repr__(self) -> str:
        return self.__str__()


class EvaluationResult(BaseModel):
    """
    Single model evaluation result.

    Contains all information about one model's performance on a task,
    including output, token usage, cost, and any errors.

    Example:
        >>> result = EvaluationResult(
        ...     model_name="anthropic/claude-sonnet-4.5",
        ...     task_name="lecture_analysis",
        ...     output="concept,start_time,end_time\\nIntroduction,00:00:00,00:03:15",
        ...     input_tokens=1000,
        ...     output_tokens=500,
        ...     total_tokens=1500,
        ...     cost_usd=0.36,
        ...     latency_ms=2500
        ... )
    """

    model_name: str = Field(..., description="Name/ID of the model evaluated")
    task_name: str = Field(..., description="Name of the task being evaluated")
    output: str = Field(..., description="The model's generated output")
    input_tokens: int = Field(..., description="Number of input tokens used")
    output_tokens: int = Field(..., description="Number of output tokens generated")
    total_tokens: int = Field(..., description="Total tokens consumed")
    cost_usd: float = Field(..., description="Cost in USD for this evaluation")
    latency_ms: float = Field(..., description="Time taken in milliseconds")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the evaluation was performed"
    )
    status: str = Field(
        default="success",
        description="Evaluation status (success, failed, timeout)"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if evaluation failed"
    )

    def __str__(self) -> str:
        return f"EvaluationResult(model='{self.model_name}', status='{self.status}', cost=${self.cost_usd:.4f})"

    def __repr__(self) -> str:
        return self.__str__()


class JudgeScore(BaseModel):
    """
    LLM-as-judge scoring result.

    Contains detailed scores and reasoning from the judge model's evaluation
    of another model's output.

    Example:
        >>> score = JudgeScore(
        ...     model_evaluated="anthropic/claude-sonnet-4.5",
        ...     accuracy_score=95,
        ...     format_score=100,
        ...     compliance_score=98,
        ...     overall_score=97,
        ...     violations=[],
        ...     reasoning="Excellent performance with accurate concept extraction"
        ... )
    """

    model_evaluated: str = Field(..., description="Name of the model being judged")
    accuracy_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Accuracy score (0-100)"
    )
    format_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Format compliance score (0-100)"
    )
    compliance_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Constraint compliance score (0-100)"
    )
    overall_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Overall weighted score (0-100)"
    )
    violations: List[str] = Field(
        default_factory=list,
        description="List of constraint violations found"
    )
    reasoning: str = Field(..., description="Detailed reasoning for the scores")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the judging was performed"
    )

    @field_validator("accuracy_score", "format_score", "compliance_score", "overall_score")
    @classmethod
    def validate_score_range(cls, v: int) -> int:
        """Validate that scores are in the range 0-100."""
        if not 0 <= v <= 100:
            raise ValueError(f"Score must be between 0 and 100, got {v}")
        return v

    def __str__(self) -> str:
        return f"JudgeScore(model='{self.model_evaluated}', overall={self.overall_score}, violations={len(self.violations)})"

    def __repr__(self) -> str:
        return self.__str__()


class ModelConfig(BaseModel):
    """
    Model pricing and configuration.

    Contains pricing information and metadata for a specific LLM model.

    Example:
        >>> config = ModelConfig(
        ...     model_id="anthropic/claude-sonnet-4.5",
        ...     display_name="Claude Sonnet 4.5",
        ...     input_price_per_1m=3.00,
        ...     output_price_per_1m=15.00,
        ...     context_window=200000,
        ...     provider="Anthropic"
        ... )
    """

    model_id: str = Field(..., description="Unique identifier for the model")
    display_name: str = Field(..., description="Human-readable name for display")
    input_price_per_1m: float = Field(
        ...,
        description="Price per 1 million input tokens in USD"
    )
    output_price_per_1m: float = Field(
        ...,
        description="Price per 1 million output tokens in USD"
    )
    context_window: int = Field(..., description="Maximum context window in tokens")
    provider: str = Field(..., description="Model provider (e.g., Anthropic, OpenAI)")

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate the cost for a given number of tokens.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Total cost in USD, rounded to 2 decimal places
        """
        input_cost = (input_tokens / 1_000_000) * self.input_price_per_1m
        output_cost = (output_tokens / 1_000_000) * self.output_price_per_1m
        total = input_cost + output_cost
        return round(total, 2)

    def __str__(self) -> str:
        return f"ModelConfig(model='{self.model_id}', provider='{self.provider}')"

    def __repr__(self) -> str:
        return self.__str__()
