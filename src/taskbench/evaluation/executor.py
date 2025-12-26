"""
Model executor for running evaluations.

This module handles executing tasks on LLM models, building prompts,
and collecting results.
"""

import asyncio
import json
import logging
import os
from typing import List, Optional, Any, Tuple

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
        self.max_concurrency = int(
            os.getenv("TASKBENCH_MAX_CONCURRENCY", "5")
        )
        self.use_generation_lookup = os.getenv("TASKBENCH_USE_GENERATION_LOOKUP", "true").lower() == "true"
        # Cache for computed chunk sizes keyed by model set
        self._chunk_cache: dict[str, Tuple[int, int]] = {}

    def _chunk_text(
        self,
        text: str,
        max_chars: int,
        overlap_chars: int
    ) -> List[str]:
        """
        Split text into overlapping character chunks.

        No preprocessing of content is performed; the raw text is sliced.
        """
        if max_chars <= 0:
            return [text]

        chunks: List[str] = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + max_chars, text_len)
            chunks.append(text[start:end])
            if end == text_len:
                break
            start = max(0, end - overlap_chars)

        return chunks

    def _compute_chunk_params(
        self,
        model_ids: List[str],
        default_chunk_chars: int,
        default_overlap: int
    ) -> Tuple[int, int]:
        """
        Compute a conservative shared chunk size from model context windows.

        - Uses the minimum context window across selected models.
        - Reserves ~40% for prompt/system, ~20% for output, leaving ~40% for input.
        - Converts tokens to chars with a coarse 4 chars/token ratio.
        - Falls back to defaults if model config is missing.
        """
        cache_key = ",".join(sorted(model_ids))
        if cache_key in self._chunk_cache:
            return self._chunk_cache[cache_key]

        min_ctx = None
        for mid in model_ids:
            cfg = self.cost_tracker.get_model_config(mid)
            if cfg and cfg.context_window:
                if min_ctx is None or cfg.context_window < min_ctx:
                    min_ctx = cfg.context_window

        if min_ctx is None:
            chunk_chars = default_chunk_chars
            overlap = default_overlap
        else:
            # Reserve 60% (40% system+prompt, 20% output), use 40% for input
            input_tokens_budget = int(min_ctx * 0.4)
            chars_per_token = 4  # coarse heuristic
            chunk_chars = max(2000, input_tokens_budget * chars_per_token)
            overlap = max(int(chunk_chars * 0.03), 200)

        self._chunk_cache[cache_key] = (chunk_chars, overlap)
        return chunk_chars, overlap

    def build_prompt(self, task: TaskDefinition, input_data: str, usecase: Optional[Any] = None) -> str:
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
        # Start with task description and use-case context
        prompt_parts = [
            f"# Task: {task.name}",
            "",
            task.description,
            "",
        ]

        if usecase:
            prompt_parts.extend([
                "## Use Case Goal",
                usecase.goal,
                ""
            ])

        prompt_parts.extend([
            "## Output Format",
            f"You MUST provide output in {task.output_format.upper()} format.",
            ""
        ])

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

        # Add use case notes
        if usecase and getattr(usecase, "notes", None):
            prompt_parts.extend([
                "## Additional Notes",
                *[f"- {note}" for note in usecase.notes],
                ""
            ])

        # Add chunk/coverage constraints
        if usecase and (usecase.chunk_min_minutes or usecase.chunk_max_minutes or usecase.coverage_required):
            prompt_parts.extend(["## Coverage and Chunk Requirements"])
            if usecase.chunk_min_minutes:
                prompt_parts.append(f"- Chunk minimum: {usecase.chunk_min_minutes} minutes")
            if usecase.chunk_max_minutes:
                prompt_parts.append(f"- Chunk maximum: {usecase.chunk_max_minutes} minutes")
            if usecase.coverage_required:
                prompt_parts.append("- Cover the full time range with no gaps.")
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
        usecase: Optional[Any] = None,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        chunk_mode: bool = False,
        chunk_chars: int = 20000,
        chunk_overlap: int = 500,
        dynamic_chunk: bool = False
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
            # Resolve dynamic chunk sizing if requested
            if chunk_mode and dynamic_chunk:
                chunk_chars, chunk_overlap = self._compute_chunk_params(
                    [model_id],
                    default_chunk_chars=chunk_chars,
                    default_overlap=chunk_overlap
                )

            if not chunk_mode:
                # Single-shot prompt
                prompt = self.build_prompt(task, input_data, usecase=usecase)
                responses = [await self.api_client.complete(
                    model=model_id,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )]
            else:
                # Multi-chunk prompt pipeline: slice raw text without preprocessing
                raw_chunks = self._chunk_text(input_data, chunk_chars, chunk_overlap)
                responses = []
                for idx, chunk in enumerate(raw_chunks, 1):
                    chunk_header = (
                        f"### Chunk {idx}/{len(raw_chunks)}\n"
                        "Process ONLY the content in this chunk. "
                        "Do not invent or summarize content beyond this chunk. "
                        "Return a JSON array of concepts for this chunk.\n\n"
                    )
                    chunk_prompt = self.build_prompt(
                        task,
                        input_data=f"{chunk_header}{chunk}",
                        usecase=usecase
                    )
                    resp = await self.api_client.complete(
                        model=model_id,
                        prompt=chunk_prompt,
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                    responses.append(resp)

            # Aggregate responses
            total_input_tokens = sum(r.input_tokens for r in responses)
            total_output_tokens = sum(r.output_tokens for r in responses)
            total_tokens = sum(r.total_tokens for r in responses)
            total_latency = sum(r.latency_ms for r in responses)

            # Inline cost aggregation
            billed_cost = sum((r.billed_cost_usd or 0.0) for r in responses)

            # Fallback cost calculation per chunk if inline missing
            calc_costs = []
            for r in responses:
                calc_costs.append(
                    self.cost_tracker.calculate_cost(
                        model_id,
                        r.input_tokens,
                        r.output_tokens,
                        inline_cost=r.billed_cost_usd
                    )
                )
            cost = sum(calc_costs)

            # Try to merge JSON arrays if possible
            merged_output = ""
            merge_failed = False
            aggregated_items: List[Any] = []
            for r in responses:
                try:
                    chunk_data = json.loads(r.content)
                    if isinstance(chunk_data, list):
                        aggregated_items.extend(chunk_data)
                    else:
                        merge_failed = True
                        break
                except Exception:
                    merge_failed = True
                    break

            if not merge_failed:
                merged_output = json.dumps(aggregated_items, indent=2)
            else:
                # Fall back to concatenating raw chunk outputs with separators
                merged_output = "\n\n".join(
                    [f"/* chunk {i+1} */\n{r.content}" for i, r in enumerate(responses)]
                )

            result = EvaluationResult(
                model_name=model_id,
                task_name=task.name,
                output=merged_output,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                total_tokens=total_tokens,
                cost_usd=cost,
                billed_cost_usd=billed_cost if billed_cost > 0 else None,
                latency_ms=total_latency,
                timestamp=responses[-1].timestamp if responses else None,
                status="success",
                generation_id=responses[-1].generation_id if len(responses) == 1 else None
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
        usecase: Optional[Any] = None,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        chunk_mode: bool = False,
        chunk_chars: int = 20000,
        chunk_overlap: int = 500,
        dynamic_chunk: bool = False
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

        results: List[EvaluationResult] = []
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def run_model(model_id: str) -> EvaluationResult:
            async with semaphore:
                return await self.execute(
                    model_id=model_id,
                    task=task,
                    input_data=input_data,
                    usecase=usecase,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    chunk_mode=chunk_mode,
                    chunk_chars=chunk_chars,
                    chunk_overlap=chunk_overlap,
                    dynamic_chunk=dynamic_chunk
                )

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

            tasks = {asyncio.create_task(run_model(mid)): mid for mid in model_ids}
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                model_id = result.model_name

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

        successful = sum(1 for r in results if r.status == "success")
        total_cost = sum(r.cost_usd for r in results)

        console.print(
            f"\n[bold green]Evaluation complete![/bold green] "
            f"{successful}/{len(model_ids)} successful, "
            f"Total cost: ${total_cost:.4f}\n"
        )

        return results
