"""Task management API endpoints."""

import logging
from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.quality_gen import generate_quality_checks
from app.models.database import Task
from app.schemas.task import TaskCreate, TaskResponse, TaskUpdate

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/", response_model=TaskResponse, status_code=status.HTTP_201_CREATED)
async def create_task(
    task: TaskCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new evaluation task with LLM-generated quality checks.

    This endpoint:
    1. Receives task definition from user
    2. Uses LLM to analyze task and generate quality checks
    3. Stores task with generated checks in database
    4. Returns complete task definition

    Example:
        POST /api/v1/tasks
        {
            "name": "concept_extraction",
            "description": "Extract teaching concepts from lecture transcripts",
            "domain": "education",
            "input_format": "text",
            "output_format": "json",
            "constraints": {"min_concepts": 3, "max_concepts": 10}
        }
    """
    try:
        # Check if task name already exists
        result = await db.execute(select(Task).where(Task.name == task.name))
        existing_task = result.scalar_one_or_none()
        if existing_task:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Task with name '{task.name}' already exists"
            )

        # Generate quality checks using LLM
        logger.info(f"Generating quality checks for task: {task.name}")
        quality_checks = await generate_quality_checks(
            task_description=task.description,
            domain=task.domain,
            output_format=task.output_format
        )

        # Create task in database
        db_task = Task(
            name=task.name,
            description=task.description,
            domain=task.domain,
            input_format=task.input_format,
            output_format=task.output_format,
            gold_data=task.gold_data,
            quality_checks=quality_checks,
            constraints=task.constraints,
            evaluation_criteria=task.evaluation_criteria
        )

        db.add(db_task)
        await db.flush()
        await db.refresh(db_task)

        logger.info(f"Created task: {db_task.id} with {len(quality_checks)} quality checks")
        return db_task

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error creating task: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create task"
        )


@router.get("/", response_model=List[TaskResponse])
async def list_tasks(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
):
    """
    List all tasks with pagination.

    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return
    """
    try:
        result = await db.execute(
            select(Task)
            .order_by(Task.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        tasks = result.scalars().all()
        return tasks

    except Exception as e:
        logger.exception(f"Error listing tasks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list tasks"
        )


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(
    task_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Get a specific task by ID."""
    try:
        result = await db.execute(select(Task).where(Task.id == task_id))
        task = result.scalar_one_or_none()

        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found"
            )

        return task

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting task: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get task"
        )


@router.patch("/{task_id}", response_model=TaskResponse)
async def update_task(
    task_id: UUID,
    task_update: TaskUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update a task."""
    try:
        result = await db.execute(select(Task).where(Task.id == task_id))
        task = result.scalar_one_or_none()

        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found"
            )

        # Update fields if provided
        update_data = task_update.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(task, field, value)

        await db.flush()
        await db.refresh(task)

        logger.info(f"Updated task: {task_id}")
        return task

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error updating task: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update task"
        )


@router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_task(
    task_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Delete a task."""
    try:
        result = await db.execute(select(Task).where(Task.id == task_id))
        task = result.scalar_one_or_none()

        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found"
            )

        await db.delete(task)
        logger.info(f"Deleted task: {task_id}")
        return None

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error deleting task: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete task"
        )


@router.post("/{task_id}/regenerate-quality-checks", response_model=TaskResponse)
async def regenerate_quality_checks(
    task_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Regenerate quality checks for a task using LLM.

    Useful if user wants to refresh the auto-generated checks.
    """
    try:
        result = await db.execute(select(Task).where(Task.id == task_id))
        task = result.scalar_one_or_none()

        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found"
            )

        # Regenerate quality checks
        logger.info(f"Regenerating quality checks for task: {task_id}")
        quality_checks = await generate_quality_checks(
            task_description=task.description,
            domain=task.domain,
            output_format=task.output_format
        )

        task.quality_checks = quality_checks
        await db.flush()
        await db.refresh(task)

        logger.info(f"Regenerated {len(quality_checks)} quality checks for task: {task_id}")
        return task

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error regenerating quality checks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to regenerate quality checks"
        )
