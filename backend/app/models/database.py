"""SQLAlchemy database models."""

from datetime import datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import JSON, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Task(Base):
    """Task definition stored in database."""

    __tablename__ = "tasks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False, unique=True, index=True)
    description = Column(Text, nullable=False)
    domain = Column(String(100), nullable=True)  # healthcare, education, legal, etc.
    input_format = Column(String(50), nullable=False)
    output_format = Column(String(50), nullable=False)
    gold_data = Column(JSON, nullable=True)  # Gold standard data for evaluation
    quality_checks = Column(JSON, nullable=False)  # LLM-generated quality checks
    constraints = Column(JSON, default={})
    evaluation_criteria = Column(JSON, default=[])
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    evaluations = relationship("EvaluationRun", back_populates="task", cascade="all, delete-orphan")


class EvaluationRun(Base):
    """Evaluation run instance."""

    __tablename__ = "evaluation_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    task_id = Column(UUID(as_uuid=True), ForeignKey("tasks.id"), nullable=False)
    status = Column(String(20), nullable=False, default="pending")  # pending, running, completed, failed
    models = Column(JSON, nullable=False)  # List of model configurations
    metrics = Column(JSON, nullable=False)  # Metrics to calculate
    estimated_cost = Column(Float, default=0.0)
    actual_cost = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)

    # Relationships
    task = relationship("Task", back_populates="evaluations")
    results = relationship("ModelResult", back_populates="evaluation", cascade="all, delete-orphan")


class ModelResult(Base):
    """Result from a single model evaluation."""

    __tablename__ = "model_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    evaluation_id = Column(UUID(as_uuid=True), ForeignKey("evaluation_runs.id"), nullable=False)
    model_id = Column(String(255), nullable=False)
    output = Column(Text, nullable=True)
    status = Column(String(20), nullable=False, default="pending")  # pending, running, completed, failed

    # Metrics
    accuracy = Column(Float, nullable=True)
    hallucination_rate = Column(Float, nullable=True)
    completeness = Column(Float, nullable=True)
    cost = Column(Float, default=0.0)
    consistency_std = Column(Float, nullable=True)  # Standard deviation across consistency runs
    instruction_following = Column(Float, nullable=True)

    # Quality violations
    quality_violations = Column(JSON, default=[])

    # Token usage
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)

    # Timing
    latency_ms = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)

    # Relationships
    evaluation = relationship("EvaluationRun", back_populates="results")


class QualityCheck(Base):
    """Generated quality check stored separately for reusability."""

    __tablename__ = "quality_checks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    validation_function = Column(Text, nullable=False)  # Python code or rule
    severity = Column(String(20), nullable=False)  # critical, warning, info
    task_type = Column(String(100), nullable=True)  # Category for reuse
    created_at = Column(DateTime, default=datetime.utcnow)
