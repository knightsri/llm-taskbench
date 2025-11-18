"""Pydantic schemas for evaluation-related API requests/responses."""

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Model configuration schema."""

    id: str  # e.g., "gpt-4o" or "custom-model-1"
    endpoint: str = "https://openrouter.ai/api/v1"
    api_key: Optional[str] = None  # User-provided API key
    provider: str = "openrouter"  # openrouter, anthropic, openai, custom


class EvaluationCreate(BaseModel):
    """Request schema for creating an evaluation run."""

    task_id: UUID
    models: list[ModelConfig]
    metrics: list[str] = Field(
        default=["accuracy", "hallucination", "completeness", "cost", "instruction_following", "consistency"]
    )
    consistency_runs: int = Field(default=10, ge=5, le=100)
    budget_limit: Optional[float] = None  # Optional budget limit in USD


class ModelResultResponse(BaseModel):
    """Response schema for model result."""

    id: UUID
    model_id: str
    output: Optional[str]
    status: str
    accuracy: Optional[float]
    hallucination_rate: Optional[float]
    completeness: Optional[float]
    cost: float
    consistency_std: Optional[float]
    instruction_following: Optional[float]
    quality_violations: list[str]
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: Optional[float]
    created_at: datetime
    completed_at: Optional[datetime]
    error_message: Optional[str]

    class Config:
        from_attributes = True


class EvaluationResponse(BaseModel):
    """Response schema for evaluation run."""

    id: UUID
    task_id: UUID
    status: str
    models: list[dict[str, Any]]
    metrics: list[str]
    estimated_cost: float
    actual_cost: float
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error_message: Optional[str]
    results: list[ModelResultResponse] = Field(default_factory=list)

    class Config:
        from_attributes = True


class EvaluationProgress(BaseModel):
    """Progress update for evaluation run."""

    evaluation_id: UUID
    status: str
    progress_percent: float  # 0-100
    completed_models: int
    total_models: int
    current_cost: float
    estimated_remaining_cost: float
    message: str
