"""Celery tasks for async evaluation execution."""

import asyncio
import logging
from datetime import datetime
from typing import Any
from uuid import UUID

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from app.core.config import settings
from app.core.metrics import (
    calculate_accuracy,
    calculate_completeness,
    calculate_cost,
    calculate_hallucination_rate,
    check_instruction_following,
    apply_quality_checks,
)
from app.models.database import EvaluationRun, ModelResult, Task
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

# Create separate engine for worker
worker_engine = create_async_engine(settings.database_url, echo=False)
WorkerSessionLocal = async_sessionmaker(worker_engine, class_=AsyncSession, expire_on_commit=False)


async def get_worker_db():
    """Get database session for worker."""
    async with WorkerSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@celery_app.task(bind=True, name="run_evaluation")
def run_evaluation_async(self, evaluation_id: str):
    """
    Celery task to run evaluation asynchronously.

    This task:
    1. Loads evaluation and task from database
    2. Executes each model sequentially
    3. Calculates metrics for each result
    4. Updates database with results
    5. Marks evaluation as completed

    Args:
        evaluation_id: UUID of evaluation to run
    """
    try:
        # Run async evaluation in event loop
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        loop.run_until_complete(_run_evaluation(evaluation_id))

    except Exception as e:
        logger.exception(f"Evaluation {evaluation_id} failed: {e}")
        # Update evaluation status to failed
        loop = asyncio.get_event_loop()
        loop.run_until_complete(_mark_evaluation_failed(evaluation_id, str(e)))
        raise


async def _run_evaluation(evaluation_id: str):
    """Execute evaluation (async implementation)."""
    async with WorkerSessionLocal() as db:
        try:
            # Load evaluation and task
            result = await db.execute(
                select(EvaluationRun).where(EvaluationRun.id == UUID(evaluation_id))
            )
            evaluation = result.scalar_one_or_none()

            if not evaluation:
                raise ValueError(f"Evaluation {evaluation_id} not found")

            result = await db.execute(select(Task).where(Task.id == evaluation.task_id))
            task = result.scalar_one_or_none()

            if not task:
                raise ValueError(f"Task {evaluation.task_id} not found")

            # Update status to running
            evaluation.status = "running"
            evaluation.started_at = datetime.utcnow()
            await db.commit()

            logger.info(f"Starting evaluation {evaluation_id} for task {task.name}")

            # Execute each model
            total_cost = 0.0
            for model_config in evaluation.models:
                model_id = model_config["id"]
                logger.info(f"Evaluating model: {model_id}")

                # Create result record
                result_record = ModelResult(
                    evaluation_id=evaluation.id,
                    model_id=model_id,
                    status="running"
                )
                db.add(result_record)
                await db.commit()
                await db.refresh(result_record)

                try:
                    # Execute model
                    output, input_tokens, output_tokens, latency = await _execute_model(
                        model_config=model_config,
                        task=task,
                        input_data=task.gold_data.get("input", "") if task.gold_data else ""
                    )

                    # Calculate metrics
                    accuracy = calculate_accuracy(
                        output,
                        task.gold_data or {},
                        task.output_format
                    )
                    hallucination = calculate_hallucination_rate(
                        output,
                        task.gold_data or {},
                        task.output_format
                    )
                    completeness = calculate_completeness(
                        output,
                        task.gold_data or {},
                        task.output_format
                    )
                    cost = calculate_cost(input_tokens, output_tokens, model_id)
                    instruction_following = check_instruction_following(
                        output,
                        task.constraints,
                        task.output_format
                    )
                    violations = apply_quality_checks(output, task.quality_checks)

                    # Update result
                    result_record.output = output
                    result_record.status = "completed"
                    result_record.accuracy = accuracy
                    result_record.hallucination_rate = hallucination
                    result_record.completeness = completeness
                    result_record.cost = cost
                    result_record.instruction_following = instruction_following
                    result_record.quality_violations = violations
                    result_record.input_tokens = input_tokens
                    result_record.output_tokens = output_tokens
                    result_record.total_tokens = input_tokens + output_tokens
                    result_record.latency_ms = latency
                    result_record.completed_at = datetime.utcnow()

                    total_cost += cost
                    await db.commit()

                    logger.info(
                        f"Model {model_id} completed: "
                        f"accuracy={accuracy:.2f}, cost=${cost:.4f}"
                    )

                except Exception as e:
                    logger.exception(f"Model {model_id} failed: {e}")
                    result_record.status = "failed"
                    result_record.error_message = str(e)
                    result_record.completed_at = datetime.utcnow()
                    await db.commit()
                    continue

            # Mark evaluation as completed
            evaluation.status = "completed"
            evaluation.actual_cost = total_cost
            evaluation.completed_at = datetime.utcnow()
            await db.commit()

            logger.info(
                f"Evaluation {evaluation_id} completed. "
                f"Total cost: ${total_cost:.4f}"
            )

        except Exception as e:
            logger.exception(f"Error running evaluation: {e}")
            if evaluation:
                evaluation.status = "failed"
                evaluation.error_message = str(e)
                evaluation.completed_at = datetime.utcnow()
                await db.commit()
            raise


async def _execute_model(
    model_config: dict[str, Any],
    task: Task,
    input_data: str
) -> tuple[str, int, int, float]:
    """
    Execute a single model on the task.

    Args:
        model_config: Model configuration dict
        task: Task definition
        input_data: Input data to process

    Returns:
        Tuple of (output, input_tokens, output_tokens, latency_ms)
    """
    model_id = model_config["id"]
    endpoint = model_config.get("endpoint", "https://openrouter.ai/api/v1")
    provider = model_config.get("provider", "openrouter")

    # Build prompt
    prompt = f"""# Task: {task.name}

{task.description}

## Output Format
You MUST provide output in {task.output_format.upper()} format.

## Constraints
"""
    if task.constraints:
        for key, value in task.constraints.items():
            prompt += f"- **{key}**: {value}\n"

    prompt += f"\n## Input Data\n\n{input_data}\n\n"
    prompt += "Please provide your response following the format and constraints above."

    # Make API call
    api_key = model_config.get("api_key") or settings.OPENROUTER_API_KEY

    if not api_key:
        raise ValueError("No API key configured")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://llm-taskbench.com",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
    }

    start_time = datetime.utcnow()

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{endpoint}/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        data = response.json()

    end_time = datetime.utcnow()
    latency = (end_time - start_time).total_seconds() * 1000  # Convert to ms

    # Extract response
    output = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)

    return output, input_tokens, output_tokens, latency


async def _mark_evaluation_failed(evaluation_id: str, error_message: str):
    """Mark evaluation as failed in database."""
    async with WorkerSessionLocal() as db:
        try:
            result = await db.execute(
                select(EvaluationRun).where(EvaluationRun.id == UUID(evaluation_id))
            )
            evaluation = result.scalar_one_or_none()

            if evaluation:
                evaluation.status = "failed"
                evaluation.error_message = error_message
                evaluation.completed_at = datetime.utcnow()
                await db.commit()

        except Exception as e:
            logger.exception(f"Failed to mark evaluation as failed: {e}")
