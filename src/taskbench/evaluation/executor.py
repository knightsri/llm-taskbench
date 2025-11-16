"""
Model executor for running evaluations.

This module handles executing tasks on LLM models, building prompts,
and collecting results.
"""

import logging
from typing import List

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from taskbench.api.client import OpenRouterClient
from taskbench.core.models import EvaluationResult, TaskDefinition
from taskbench.evaluation.cost import CostTracker

logger = logging.getLogger(__name__)
console = Console()


class ModelExecutor:
    """
    Execute tasks on LLM models and collect results.

    Handles prompt building, API calls, error handling, and result collection
    for single and multiple model evaluations.

    Example:
        ```python
        executor = ModelExecutor(api_client, cost_tracker)

        result = await executor.execute(
            model_id="anthropic/claude-sonnet-4.5",
            task=task_definition,
            input_data=transcript
        )

        results = await executor.evaluate_multiple(
            model_ids=["claude-sonnet-4.5", "gpt-4o"],
            task=task_definition,
            input_data=transcript
        )
        ```
    """

    def __init__(self, api_client: OpenRouterClient, cost_tracker: CostTracker):
        """
        Initialize the model executor.

        Args:
            api_client: OpenRouterClient for making API calls
            cost_tracker: CostTracker for calculating costs
        """
        self.api_client = api_client
        self.cost_tracker = cost_tracker

    def build_prompt(self, task: TaskDefinition, input_data: str) -> str:
        """
        Build a comprehensive prompt from task definition and input data.

        Args:
            task: TaskDefinition describing the task
            input_data: Input data to process

        Returns:
            Complete prompt string to send to the model

        Example:
            The prompt includes:
            - Task description
            - Output format requirements
            - Constraints (EMPHASIZED)
            - Examples
            - Input data
        """
        # Start with task description
        prompt_parts = [
            f"# Task: {task.name}",
            "",
            task.description,
            "",
            "## Output Format",
            f"You MUST provide output in {task.output_format.upper()} format.",
            ""
        ]

        # Add constraints with emphasis
        if task.constraints:
            prompt_parts.extend([
                "## CRITICAL CONSTRAINTS",
                "You MUST follow these constraints strictly:",
                ""
            ])

            for key, value in task.constraints.items():
                constraint_line = f"- **{key}**: {value}"
                prompt_parts.append(constraint_line)

            prompt_parts.append("")

        # Add examples if available
        if task.examples:
            prompt_parts.extend([
                "## Examples",
                "Here are examples of good output:",
                ""
            ])

            for i, example in enumerate(task.examples[:3], 1):  # Limit to 3 examples
                prompt_parts.append(f"### Example {i}")
                if "input" in example:
                    prompt_parts.append(f"Input: {example['input'][:200]}...")
                if "expected_output" in example:
                    prompt_parts.append(f"Expected output: {example['expected_output']}")
                if "notes" in example:
                    prompt_parts.append(f"Notes: {example['notes']}")
                prompt_parts.append("")

        # Add evaluation criteria
        if task.evaluation_criteria:
            prompt_parts.extend([
                "## Evaluation Criteria",
                "Your output will be evaluated on:",
                ""
            ])

            for criterion in task.evaluation_criteria:
                prompt_parts.append(f"- {criterion}")

            prompt_parts.append("")

        # Add input data
        prompt_parts.extend([
            "## Input Data",
            "",
            input_data,
            "",
            "## Your Task",
            f"Process the above input and provide output in {task.output_format.upper()} format.",
            "Ensure you follow ALL constraints listed above.",
            "Provide ONLY the output, no explanations."
        ])

        return "\n".join(prompt_parts)

    async def execute(
        self,
        model_id: str,
        task: TaskDefinition,
        input_data: str,
        max_tokens: int = 2000,
        temperature: float = 0.7
    ) -> EvaluationResult:
        """
        Execute a task on a single model.

        Args:
            model_id: Model identifier (e.g., "anthropic/claude-sonnet-4.5")
            task: TaskDefinition describing the task
            input_data: Input data to process
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            EvaluationResult with model output and metadata

        Example:
            ```python
            result = await executor.execute(
                model_id="anthropic/claude-sonnet-4.5",
                task=lecture_task,
                input_data=transcript_text
            )
            print(f"Cost: ${result.cost_usd:.4f}")
            ```
        """
        logger.info(f"Executing task '{task.name}' on model '{model_id}'")

        try:
            # Build prompt
            prompt = self.build_prompt(task, input_data)

            # Make API call
            response = await self.api_client.complete(
                model=model_id,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )

            # Calculate cost
            cost = self.cost_tracker.calculate_cost(
                model_id,
                response.input_tokens,
                response.output_tokens
            )

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
                timestamp=response.timestamp,
                status="success"
            )

            # Track the evaluation
            self.cost_tracker.track_evaluation(result)

            logger.info(
                f"Successfully executed on {model_id}: "
                f"{result.total_tokens} tokens, ${result.cost_usd:.4f}"
            )

            return result

        except Exception as e:
            logger.error(f"Failed to execute on {model_id}: {type(e).__name__}: {str(e)}")

            # Create failed result
            result = EvaluationResult(
                model_name=model_id,
                task_name=task.name,
                output="",
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                cost_usd=0.0,
                latency_ms=0.0,
                status="failed",
                error=f"{type(e).__name__}: {str(e)}"
            )

            return result

    async def evaluate_multiple(
        self,
        model_ids: List[str],
        task: TaskDefinition,
        input_data: str,
        max_tokens: int = 2000,
        temperature: float = 0.7
    ) -> List[EvaluationResult]:
        """
        Execute a task on multiple models with progress tracking.

        Args:
            model_ids: List of model identifiers
            task: TaskDefinition describing the task
            input_data: Input data to process
            max_tokens: Maximum tokens to generate per model
            temperature: Sampling temperature

        Returns:
            List of EvaluationResults, one per model

        Example:
            ```python
            results = await executor.evaluate_multiple(
                model_ids=["claude-sonnet-4.5", "gpt-4o", "qwen-2.5-72b"],
                task=lecture_task,
                input_data=transcript_text
            )

            for result in results:
                if result.status == "success":
                    print(f"{result.model_name}: ${result.cost_usd:.4f}")
            ```
        """
        console.print(f"\n[bold blue]Evaluating {len(model_ids)} models on task '{task.name}'[/bold blue]\n")

        results = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            eval_task = progress.add_task(
                f"[cyan]Evaluating models...",
                total=len(model_ids)
            )

            for model_id in model_ids:
                progress.update(
                    eval_task,
                    description=f"[cyan]Evaluating {model_id}..."
                )

                result = await self.execute(
                    model_id=model_id,
                    task=task,
                    input_data=input_data,
                    max_tokens=max_tokens,
                    temperature=temperature
                )

                results.append(result)

                # Show result status
                if result.status == "success":
                    console.print(
                        f" [green]{model_id}[/green]: "
                        f"{result.total_tokens:,} tokens, "
                        f"${result.cost_usd:.4f}, "
                        f"{result.latency_ms:.0f}ms"
                    )
                else:
                    console.print(
                        f" [red]{model_id}[/red]: "
                        f"{result.error}"
                    )

                progress.advance(eval_task)

        # Summary
        successful = sum(1 for r in results if r.status == "success")
        total_cost = sum(r.cost_usd for r in results)

        console.print(
            f"\n[bold green]Evaluation complete![/bold green] "
            f"{successful}/{len(model_ids)} successful, "
            f"Total cost: ${total_cost:.4f}\n"
        )

        return results
