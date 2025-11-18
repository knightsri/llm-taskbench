"""Evaluation management API endpoints."""

import logging
from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.models.database import EvaluationRun, ModelResult, Task
from app.schemas.evaluation import (
    EvaluationCreate,
    EvaluationProgress,
    EvaluationResponse,
)
from app.workers.tasks import run_evaluation_async

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/", response_model=EvaluationResponse, status_code=status.HTTP_201_CREATED)
async def create_evaluation(
    evaluation: EvaluationCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Start a new evaluation run.

    This endpoint:
    1. Creates evaluation run record
    2. Kicks off async Celery task for execution
    3. Returns evaluation ID for progress tracking

    The actual evaluation runs asynchronously via Celery workers.
    """
    try:
        # Verify task exists
        result = await db.execute(select(Task).where(Task.id == evaluation.task_id))
        task = result.scalar_one_or_none()

        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {evaluation.task_id} not found"
            )

        # Estimate cost (simplified - actual calculation in worker)
        num_models = len(evaluation.models)
        estimated_cost = num_models * 0.50  # Rough estimate: $0.50 per model

        # Create evaluation run
        db_evaluation = EvaluationRun(
            task_id=evaluation.task_id,
            status="pending",
            models=[model.model_dump() for model in evaluation.models],
            metrics=evaluation.metrics,
            estimated_cost=estimated_cost
        )

        db.add(db_evaluation)
        await db.flush()
        await db.refresh(db_evaluation)

        # Kick off async evaluation task
        logger.info(f"Starting evaluation {db_evaluation.id} with {num_models} models")
        run_evaluation_async.delay(str(db_evaluation.id))

        return db_evaluation

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error creating evaluation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create evaluation"
        )


@router.get("/", response_model=List[EvaluationResponse])
async def list_evaluations(
    task_id: UUID | None = None,
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
):
    """
    List evaluation runs with optional filtering by task.

    Args:
        task_id: Optional task ID to filter by
        skip: Number of records to skip
        limit: Maximum number of records to return
    """
    try:
        query = select(EvaluationRun).options(selectinload(EvaluationRun.results))

        if task_id:
            query = query.where(EvaluationRun.task_id == task_id)

        query = query.order_by(EvaluationRun.created_at.desc()).offset(skip).limit(limit)

        result = await db.execute(query)
        evaluations = result.scalars().all()

        return evaluations

    except Exception as e:
        logger.exception(f"Error listing evaluations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list evaluations"
        )


@router.get("/{evaluation_id}", response_model=EvaluationResponse)
async def get_evaluation(
    evaluation_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Get a specific evaluation with all results."""
    try:
        result = await db.execute(
            select(EvaluationRun)
            .options(selectinload(EvaluationRun.results))
            .where(EvaluationRun.id == evaluation_id)
        )
        evaluation = result.scalar_one_or_none()

        if not evaluation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Evaluation {evaluation_id} not found"
            )

        return evaluation

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting evaluation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get evaluation"
        )


@router.get("/{evaluation_id}/progress", response_model=EvaluationProgress)
async def get_evaluation_progress(
    evaluation_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Get real-time progress of an evaluation.

    Returns current status, completion percentage, and cost tracking.
    """
    try:
        result = await db.execute(
            select(EvaluationRun)
            .options(selectinload(EvaluationRun.results))
            .where(EvaluationRun.id == evaluation_id)
        )
        evaluation = result.scalar_one_or_none()

        if not evaluation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Evaluation {evaluation_id} not found"
            )

        # Calculate progress
        total_models = len(evaluation.models)
        completed_models = sum(
            1 for r in evaluation.results
            if r.status in ("completed", "failed")
        )

        progress_percent = (completed_models / total_models * 100) if total_models > 0 else 0

        # Calculate costs
        current_cost = sum(r.cost for r in evaluation.results)
        estimated_remaining = evaluation.estimated_cost - current_cost

        # Determine message
        if evaluation.status == "completed":
            message = "Evaluation completed"
        elif evaluation.status == "failed":
            message = f"Evaluation failed: {evaluation.error_message}"
        elif evaluation.status == "running":
            message = f"Evaluating model {completed_models + 1}/{total_models}"
        else:
            message = "Evaluation pending"

        return EvaluationProgress(
            evaluation_id=evaluation.id,
            status=evaluation.status,
            progress_percent=progress_percent,
            completed_models=completed_models,
            total_models=total_models,
            current_cost=current_cost,
            estimated_remaining_cost=max(0, estimated_remaining),
            message=message
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting evaluation progress: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get evaluation progress"
        )


@router.delete("/{evaluation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_evaluation(
    evaluation_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Delete an evaluation and all its results."""
    try:
        result = await db.execute(
            select(EvaluationRun).where(EvaluationRun.id == evaluation_id)
        )
        evaluation = result.scalar_one_or_none()

        if not evaluation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Evaluation {evaluation_id} not found"
            )

        await db.delete(evaluation)
        logger.info(f"Deleted evaluation: {evaluation_id}")
        return None

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error deleting evaluation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete evaluation"
        )
