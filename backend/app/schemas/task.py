"""Pydantic schemas for task-related API requests/responses."""

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class QualityCheckSchema(BaseModel):
    """Quality check schema."""

    name: str
    description: str
    validation_function: str
    severity: str  # critical, warning, info

    class Config:
        from_attributes = True


class TaskCreate(BaseModel):
    """Request schema for creating a task."""

    name: str = Field(..., min_length=1, max_length=255)
    description: str = Field(..., min_length=10)
    domain: Optional[str] = None
    input_format: str = Field(default="text")
    output_format: str = Field(default="json")
    gold_data: Optional[dict[str, Any]] = None
    constraints: dict[str, Any] = Field(default_factory=dict)
    evaluation_criteria: list[str] = Field(default_factory=list)


class TaskResponse(BaseModel):
    """Response schema for task."""

    id: UUID
    name: str
    description: str
    domain: Optional[str]
    input_format: str
    output_format: str
    gold_data: Optional[dict[str, Any]]
    quality_checks: list[dict[str, Any]]  # LLM-generated
    constraints: dict[str, Any]
    evaluation_criteria: list[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class TaskUpdate(BaseModel):
    """Request schema for updating a task."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, min_length=10)
    domain: Optional[str] = None
    input_format: Optional[str] = None
    output_format: Optional[str] = None
    gold_data: Optional[dict[str, Any]] = None
    quality_checks: Optional[list[dict[str, Any]]] = None
    constraints: Optional[dict[str, Any]] = None
    evaluation_criteria: Optional[list[str]] = None
