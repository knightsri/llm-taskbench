"""
Model executor for LLM TaskBench.

This module provides the ModelExecutor class that handles executing tasks
on multiple LLM models, building prompts, tracking costs, and collecting results.
"""

import asyncio
import logging
from typing import List, Optional

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from taskbench.api.client import OpenRouterClient, OpenRouterAPIError
from taskbench.core.models import TaskDefinition, EvaluationResult, CompletionResponse
from taskbench.evaluation.cost import CostTracker

logger = logging.getLogger(__name__)


class ModelExecutor:
    """
    Execute evaluation tasks on multiple LLM models.

    This class handles:
    - Building comprehensive prompts from task definitions
    - Executing tasks on one or more models via OpenRouter API
    - Tracking token usage and costs
    - Handling errors gracefully
    - Displaying progress with Rich progress bars

    Example:
        >>> executor = ModelExecutor()
        >>> task = TaskDefinition(...)
        >>> input_data = "lecture transcript..."
        >>>
        >>> # Execute on a single model
        >>> result = await executor.execute("anthropic/claude-sonnet-4.5", task, input_data)
        >>> print(f"Cost: ${result.cost_usd:.4f}")
        >>>
        >>> # Execute on multiple models
        >>> models = ["anthropic/claude-sonnet-4.5", "openai/gpt-4o"]
        >>> results = await executor.evaluate_multiple(models, task, input_data)
        >>> for result in results:
        ...     print(f"{result.model_name}: {result.status}")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cost_tracker: Optional[CostTracker] = None,
        timeout: float = 120.0
    ):
        """
        Initialize the model executor.

        Args:
            api_key: OpenRouter API key (optional, reads from env if not provided)
            cost_tracker: CostTracker instance for cost calculation (creates new if not provided)
            timeout: Request timeout in seconds (default: 120)
        """
        self.api_key = api_key
        self.timeout = timeout
        self.cost_tracker = cost_tracker or CostTracker()
        logger.info("ModelExecutor initialized")

    def build_prompt(self, task: TaskDefinition, input_data: str) -> str:
        """
        Build a comprehensive prompt from task definition and input data.

        This method creates a detailed prompt that includes:
        - Task description and objectives
        - Input data
        - Expected output format with examples
        - Constraints (emphasized for clarity)
        - Any example inputs/outputs

        Args:
            task: TaskDefinition containing task specifications
            input_data: The input data to process (e.g., transcript text)

        Returns:
            Formatted prompt string ready to send to the model

        Example:
            >>> executor = ModelExecutor()
            >>> task = TaskDefinition(name="concept_extraction", ...)
            >>> input_data = "Lecture transcript content..."
            >>> prompt = executor.build_prompt(task, input_data)
        """
        prompt_parts = []

        # Task description
        prompt_parts.append(f"# Task: {task.name}")
        prompt_parts.append("")
        prompt_parts.append(task.description)
        prompt_parts.append("")

        # Output format instructions
        prompt_parts.append(f"## Output Format")
        prompt_parts.append(f"You MUST return your response in **{task.output_format.upper()}** format.")
        prompt_parts.append("")

        # Constraints - make them VERY clear
        if task.constraints:
            prompt_parts.append("## IMPORTANT CONSTRAINTS - YOU MUST FOLLOW THESE:")
            prompt_parts.append("")
            for key, value in task.constraints.items():
                # Format constraint keys to be more readable
                readable_key = key.replace("_", " ").title()
                prompt_parts.append(f"- **{readable_key}**: {value}")
            prompt_parts.append("")
            prompt_parts.append("**CRITICAL**: Violating these constraints will result in a failed evaluation.")
            prompt_parts.append("")

        # Examples (if provided)
        if task.examples:
            prompt_parts.append("## Examples")
            prompt_parts.append("")
            for i, example in enumerate(task.examples, 1):
                prompt_parts.append(f"### Example {i}")
                if "input" in example:
                    prompt_parts.append(f"Input: {example['input']}")
                if "expected_output" in example:
                    prompt_parts.append(f"Expected Output: {example['expected_output']}")
                if "notes" in example:
                    prompt_parts.append(f"Notes: {example['notes']}")
                prompt_parts.append("")

        # Input data
        prompt_parts.append("## Input Data")
        prompt_parts.append("")
        prompt_parts.append(input_data)
        prompt_parts.append("")

        # Final instructions
        prompt_parts.append("## Instructions")
        prompt_parts.append(f"Process the input data according to the task description above.")
        prompt_parts.append(f"Return ONLY the {task.output_format.upper()} output - no explanations or additional text.")
        prompt_parts.append("Ensure all constraints are strictly followed.")

        prompt = "\n".join(prompt_parts)

        logger.debug(f"Built prompt for task '{task.name}' ({len(prompt)} characters)")
        return prompt

    async def execute(
        self,
        model_id: str,
        task: TaskDefinition,
        input_data: str
    ) -> EvaluationResult:
        """
        Execute a task on a single model.

        This method:
        1. Builds the prompt from task and input data
        2. Calls the OpenRouter API with the specified model
        3. Parses the response
        4. Calculates cost using the cost tracker
        5. Returns an EvaluationResult

        If an error occurs, the result status is set to "failed" and the error
        message is included.

        Args:
            model_id: Model identifier (e.g., "anthropic/claude-sonnet-4.5")
            task: TaskDefinition to execute
            input_data: Input data for the task

        Returns:
            EvaluationResult containing output, tokens, cost, and status

        Example:
            >>> executor = ModelExecutor()
            >>> task = TaskDefinition(...)
            >>> result = await executor.execute("anthropic/claude-sonnet-4.5", task, "input")
            >>> if result.status == "success":
            ...     print(f"Output: {result.output}")
            ... else:
            ...     print(f"Error: {result.error}")
        """
        logger.info(f"Executing task '{task.name}' on model '{model_id}'")

        try:
            # Build prompt
            prompt = self.build_prompt(task, input_data)

            # Create API client and execute
            async with OpenRouterClient(api_key=self.api_key, timeout=self.timeout) as client:
                response: CompletionResponse = await client.complete(
                    model=model_id,
                    prompt=prompt,
                    temperature=0.3  # Lower temperature for more consistent outputs
                )

            # Calculate cost
            try:
                cost = self.cost_tracker.calculate_cost(
                    model_id,
                    response.input_tokens,
                    response.output_tokens
                )
            except ValueError as e:
                logger.warning(f"Could not calculate cost for {model_id}: {e}")
                cost = 0.0

            # Create evaluation result
            result = EvaluationResult(
                model_name=model_id,
                task_name=task.name,
                output=response.content,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                total_tokens=response.total_tokens,
                cost_usd=cost,
                latency_ms=response.latency_ms,
                status="success",
                error=None
            )

            logger.info(
                f"Task '{task.name}' completed on '{model_id}': "
                f"{result.total_tokens} tokens, ${result.cost_usd:.4f}, "
                f"{result.latency_ms:.0f}ms"
            )

            return result

        except OpenRouterAPIError as e:
            # API-specific error
            logger.error(f"API error executing task '{task.name}' on '{model_id}': {e}")
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
                error=str(e)
            )

        except Exception as e:
            # Unexpected error
            logger.error(
                f"Unexpected error executing task '{task.name}' on '{model_id}': {e}",
                exc_info=True
            )
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
                error=f"Unexpected error: {str(e)}"
            )

    async def evaluate_multiple(
        self,
        model_ids: List[str],
        task: TaskDefinition,
        input_data: str,
        show_progress: bool = True
    ) -> List[EvaluationResult]:
        """
        Execute a task on multiple models sequentially.

        This method runs the same task on multiple models and collects all results,
        even if some models fail. Progress is displayed using Rich progress bars.

        Args:
            model_ids: List of model identifiers to evaluate
            task: TaskDefinition to execute
            input_data: Input data for the task
            show_progress: Whether to show progress bar (default: True)

        Returns:
            List of EvaluationResult objects, one per model

        Example:
            >>> executor = ModelExecutor()
            >>> models = [
            ...     "anthropic/claude-sonnet-4.5",
            ...     "openai/gpt-4o",
            ...     "google/gemini-2.0-flash-exp"
            ... ]
            >>> task = TaskDefinition(...)
            >>> results = await executor.evaluate_multiple(models, task, "input data")
            >>>
            >>> # Check results
            >>> for result in results:
            ...     if result.status == "success":
            ...         print(f"{result.model_name}: ${result.cost_usd:.4f}")
            ...     else:
            ...         print(f"{result.model_name}: FAILED - {result.error}")
        """
        logger.info(
            f"Starting evaluation of task '{task.name}' on {len(model_ids)} models"
        )

        results: List[EvaluationResult] = []

        if show_progress:
            # Create Rich progress bar
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("[cyan]{task.fields[status]}"),
            ) as progress:
                progress_task = progress.add_task(
                    "[cyan]Evaluating models...",
                    total=len(model_ids),
                    status=""
                )

                for i, model_id in enumerate(model_ids, 1):
                    # Update progress bar
                    progress.update(
                        progress_task,
                        description=f"[cyan]Evaluating {i}/{len(model_ids)}: {model_id}",
                        status="Running..."
                    )

                    # Execute task
                    result = await self.execute(model_id, task, input_data)
                    results.append(result)

                    # Track cost
                    if result.status == "success":
                        self.cost_tracker.track_evaluation(result)

                    # Update progress status
                    status_emoji = "" if result.status == "success" else ""
                    status_text = (
                        f"{status_emoji} {result.total_tokens} tokens, ${result.cost_usd:.4f}"
                        if result.status == "success"
                        else f"{status_emoji} FAILED"
                    )
                    progress.update(progress_task, advance=1, status=status_text)

                progress.update(
                    progress_task,
                    description=f"[green]Completed {len(model_ids)} evaluations",
                    status=f"Total: ${self.cost_tracker.get_total_cost():.4f}"
                )
        else:
            # No progress bar - just execute sequentially
            for model_id in model_ids:
                result = await self.execute(model_id, task, input_data)
                results.append(result)

                if result.status == "success":
                    self.cost_tracker.track_evaluation(result)

        # Log summary
        successful = sum(1 for r in results if r.status == "success")
        failed = len(results) - successful
        total_cost = self.cost_tracker.get_total_cost()

        logger.info(
            f"Evaluation complete: {successful} successful, {failed} failed, "
            f"total cost: ${total_cost:.4f}"
        )

        return results

    def get_cost_summary(self) -> str:
        """
        Get a formatted cost summary.

        Returns:
            Formatted string with cost statistics

        Example:
            >>> executor = ModelExecutor()
            >>> # ... run some evaluations ...
            >>> print(executor.get_cost_summary())
        """
        return self.cost_tracker.export_summary()

    def reset_tracker(self) -> None:
        """
        Reset the cost tracker statistics.

        Example:
            >>> executor = ModelExecutor()
            >>> # ... run some evaluations ...
            >>> executor.reset_tracker()  # Clear stats for next run
        """
        self.cost_tracker.reset()
        logger.info("Cost tracker reset")
