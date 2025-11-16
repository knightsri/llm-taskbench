"""
CLI interface for LLM TaskBench.

This module provides a command-line interface using Typer for evaluating
LLMs on custom tasks, managing models, and viewing results.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from taskbench.core.task import TaskParser
from taskbench.core.models import EvaluationResult, TaskDefinition, JudgeScore
from taskbench.evaluation.executor import ModelExecutor
from taskbench.evaluation.cost import CostTracker
from taskbench.evaluation.comparison import ModelComparison
from taskbench.evaluation.recommender import RecommendationEngine
from taskbench.utils.logging import setup_logging

# Create Typer app
app = typer.Typer(
    name="taskbench",
    help="LLM TaskBench - Evaluate LLMs on custom tasks",
    add_completion=False
)

# Rich console for beautiful output
console = Console()

# Global logger (configured on first command)
logger = logging.getLogger(__name__)


def setup_cli_logging(verbose: bool = False) -> None:
    """
    Setup logging for CLI commands.

    Args:
        verbose: Enable verbose (DEBUG) logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    setup_logging(level=level)


@app.command()
def evaluate(
    task_yaml: str = typer.Argument(
        ...,
        help="Path to task definition YAML file"
    ),
    models: Optional[str] = typer.Option(
        None,
        "--models", "-m",
        help="Comma-separated list of model IDs to evaluate (e.g., 'anthropic/claude-sonnet-4.5,openai/gpt-4o')"
    ),
    input_file: Optional[str] = typer.Option(
        None,
        "--input-file", "-i",
        help="Path to input data file (if not specified, uses stdin)"
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output", "-o",
        help="Path to save results JSON file"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose logging"
    )
) -> None:
    """
    Evaluate one or more models on a task.

    This command loads a task definition, runs it on specified models,
    displays results in a beautiful table, and optionally saves to JSON.

    Example:

        # Evaluate two models on a task
        taskbench evaluate tasks/lecture_analysis.yaml \\
            --models "anthropic/claude-sonnet-4.5,openai/gpt-4o" \\
            --input-file tests/fixtures/sample_transcript.txt \\
            --output results.json

        # Use default models (from task definition or all available)
        taskbench evaluate tasks/lecture_analysis.yaml \\
            --input-file transcript.txt
    """
    setup_cli_logging(verbose)

    try:
        # Load task definition
        console.print(f"\n[cyan]Loading task definition from {task_yaml}...[/cyan]")
        parser = TaskParser()
        task = parser.load_from_yaml(task_yaml)

        # Validate task
        is_valid, errors = parser.validate_task(task)
        if not is_valid:
            console.print("\n[red]Task validation failed:[/red]")
            for error in errors:
                console.print(f"  [red][/red] {error}")
            raise typer.Exit(code=1)

        console.print(f"[green][/green] Task loaded: {task.name}")

        # Load input data
        if input_file:
            console.print(f"\n[cyan]Loading input data from {input_file}...[/cyan]")
            input_path = Path(input_file)
            if not input_path.exists():
                console.print(f"[red]Error: Input file not found: {input_file}[/red]")
                raise typer.Exit(code=1)
            input_data = input_path.read_text(encoding='utf-8')
            console.print(f"[green][/green] Loaded {len(input_data)} characters of input data")
        else:
            console.print("\n[yellow]Reading input data from stdin (Ctrl+D when done)...[/yellow]")
            input_data = sys.stdin.read()

        if not input_data.strip():
            console.print("[red]Error: Input data is empty[/red]")
            raise typer.Exit(code=1)

        # Parse model IDs
        if models:
            model_ids = [m.strip() for m in models.split(",")]
        else:
            # Default to a few common models
            console.print("[yellow]No models specified, using default models[/yellow]")
            model_ids = [
                "anthropic/claude-sonnet-4.5",
                "openai/gpt-4o",
                "google/gemini-2.0-flash-exp"
            ]

        console.print(f"\n[cyan]Evaluating {len(model_ids)} model(s):[/cyan]")
        for model_id in model_ids:
            console.print(f"  " {model_id}")

        # Execute evaluation
        console.print()
        executor = ModelExecutor()
        results = asyncio.run(
            executor.evaluate_multiple(model_ids, task, input_data, show_progress=True)
        )

        # Display results table
        console.print("\n")
        display_results_table(results)

        # Display cost summary
        console.print()
        cost_summary = executor.get_cost_summary()
        console.print(Panel(cost_summary, title="Cost Summary", border_style="green"))

        # Save results if output path specified
        if output:
            save_results(results, output)
            console.print(f"\n[green][/green] Results saved to {output}")

        # Exit with error code if any evaluations failed
        failed_count = sum(1 for r in results if r.status != "success")
        if failed_count > 0:
            console.print(f"\n[yellow]Warning: {failed_count} evaluation(s) failed[/yellow]")
            raise typer.Exit(code=1)

    except KeyboardInterrupt:
        console.print("\n[yellow]Evaluation cancelled by user[/yellow]")
        raise typer.Exit(code=130)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=verbose)
        console.print(f"\n[red]Error: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(code=1)


@app.command()
def models(
    list_models: bool = typer.Option(
        False,
        "--list", "-l",
        help="List all available models with pricing"
    ),
    info: Optional[str] = typer.Option(
        None,
        "--info",
        help="Show detailed information about a specific model"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose logging"
    )
) -> None:
    """
    Show available models and pricing information.

    Example:

        # List all models
        taskbench models --list

        # Get info about a specific model
        taskbench models --info "anthropic/claude-sonnet-4.5"
    """
    setup_cli_logging(verbose)

    try:
        tracker = CostTracker()

        if info:
            # Show detailed info about a specific model
            model_config = tracker.get_model_config(info)
            if not model_config:
                console.print(f"[red]Error: Model '{info}' not found[/red]")
                console.print(f"\nAvailable models: {', '.join(tracker.models.keys())}")
                raise typer.Exit(code=1)

            # Display model info in a panel
            info_text = f"""
[bold]{model_config.display_name}[/bold]
Provider: {model_config.provider}
Model ID: {model_config.model_id}

Pricing:
  Input:  ${model_config.input_price_per_1m:.2f} per 1M tokens
  Output: ${model_config.output_price_per_1m:.2f} per 1M tokens

Context Window: {model_config.context_window:,} tokens
"""
            console.print(Panel(info_text, title=f"Model Info: {info}", border_style="cyan"))

        elif list_models:
            # Display table of all models
            table = Table(title="Available Models", box=box.ROUNDED)
            table.add_column("Model ID", style="cyan", no_wrap=True)
            table.add_column("Display Name", style="white")
            table.add_column("Provider", style="yellow")
            table.add_column("Input Price\n(per 1M tokens)", justify="right", style="green")
            table.add_column("Output Price\n(per 1M tokens)", justify="right", style="green")
            table.add_column("Context Window", justify="right", style="blue")

            for model_config in sorted(tracker.models.values(), key=lambda m: m.display_name):
                table.add_row(
                    model_config.model_id,
                    model_config.display_name,
                    model_config.provider,
                    f"${model_config.input_price_per_1m:.2f}",
                    f"${model_config.output_price_per_1m:.2f}",
                    f"{model_config.context_window:,}"
                )

            console.print()
            console.print(table)
            console.print()
            console.print(f"[cyan]Total models available: {len(tracker.models)}[/cyan]")

        else:
            # No option specified, show help
            console.print("[yellow]Please specify --list or --info <model_id>[/yellow]")
            console.print("\nExamples:")
            console.print("  taskbench models --list")
            console.print("  taskbench models --info 'anthropic/claude-sonnet-4.5'")

    except Exception as e:
        logger.error(f"Failed to load models: {e}", exc_info=verbose)
        console.print(f"\n[red]Error: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(code=1)


@app.command()
def validate(
    task_yaml: str = typer.Argument(
        ...,
        help="Path to task definition YAML file to validate"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose logging"
    )
) -> None:
    """
    Validate a task definition YAML file.

    This command checks that the task definition is valid and all required
    fields are present and correctly formatted.

    Example:

        taskbench validate tasks/lecture_analysis.yaml
    """
    setup_cli_logging(verbose)

    try:
        console.print(f"\n[cyan]Validating task definition: {task_yaml}[/cyan]\n")

        # Load task
        parser = TaskParser()
        task = parser.load_from_yaml(task_yaml)

        console.print(f"[green][/green] Task YAML loaded successfully")
        console.print(f"  Name: {task.name}")
        console.print(f"  Input Type: {task.input_type}")
        console.print(f"  Output Format: {task.output_format}")

        # Validate task
        is_valid, errors = parser.validate_task(task)

        if is_valid:
            console.print(f"\n[green] Task validation passed![/green]\n")

            # Show task details
            details = f"""
Task: {task.name}
Description: {task.description}
Input Type: {task.input_type}
Output Format: {task.output_format}

Evaluation Criteria ({len(task.evaluation_criteria)}):
"""
            for criterion in task.evaluation_criteria:
                details += f"  " {criterion}\n"

            if task.constraints:
                details += f"\nConstraints ({len(task.constraints)}):\n"
                for key, value in task.constraints.items():
                    details += f"  " {key}: {value}\n"

            if task.examples:
                details += f"\nExamples: {len(task.examples)}\n"

            console.print(Panel(details, title="Task Details", border_style="green"))
            raise typer.Exit(code=0)
        else:
            console.print(f"\n[red] Task validation failed:[/red]\n")
            for error in errors:
                console.print(f"  [red][/red] {error}")
            console.print()
            raise typer.Exit(code=1)

    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=verbose)
        console.print(f"\n[red]Error: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(code=1)


@app.command()
def results(
    results_file: str = typer.Argument(
        "results.json",
        help="Path to results JSON file"
    ),
    format: str = typer.Option(
        "table",
        "--format", "-f",
        help="Output format: table, json, or csv"
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output", "-o",
        help="Save output to file instead of displaying"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose logging"
    )
) -> None:
    """
    Display evaluation results from a JSON file.

    This command loads previously saved evaluation results and displays
    them in various formats (table, JSON, or CSV).

    Example:

        # Display as table
        taskbench results results.json

        # Export as CSV
        taskbench results results.json --format csv --output results.csv

        # Show as JSON
        taskbench results results.json --format json
    """
    setup_cli_logging(verbose)

    try:
        # Load results
        results_path = Path(results_file)
        if not results_path.exists():
            console.print(f"[red]Error: Results file not found: {results_file}[/red]")
            raise typer.Exit(code=1)

        console.print(f"\n[cyan]Loading results from {results_file}...[/cyan]")
        results = load_results(results_file)
        console.print(f"[green][/green] Loaded {len(results)} result(s)")

        # Format output
        if format == "table":
            output_text = None
            if not output:
                # Display directly to console
                console.print()
                display_results_table(results)
                console.print()
            else:
                # This doesn't make much sense for table format, but support it anyway
                console.print("[yellow]Note: Table format is best viewed in terminal, not saved to file[/yellow]")
                display_results_table(results)

        elif format == "json":
            output_text = format_results_json(results)
            if output:
                Path(output).write_text(output_text, encoding='utf-8')
                console.print(f"\n[green][/green] Results saved to {output}")
            else:
                console.print("\n" + output_text)

        elif format == "csv":
            output_text = format_results_csv(results)
            if output:
                Path(output).write_text(output_text, encoding='utf-8')
                console.print(f"\n[green][/green] Results saved to {output}")
            else:
                console.print("\n" + output_text)

        else:
            console.print(f"[red]Error: Invalid format '{format}'. Must be: table, json, or csv[/red]")
            raise typer.Exit(code=1)

    except Exception as e:
        logger.error(f"Failed to display results: {e}", exc_info=verbose)
        console.print(f"\n[red]Error: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(code=1)


@app.command()
def recommend(
    results_file: str = typer.Option(
        "results.json",
        "--results-file", "-r",
        help="Path to evaluation results JSON file"
    ),
    scores_file: str = typer.Option(
        "scores.json",
        "--scores-file", "-s",
        help="Path to judge scores JSON file"
    ),
    budget: Optional[float] = typer.Option(
        None,
        "--budget", "-b",
        help="Maximum budget per request in USD (e.g., 0.10)"
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output", "-o",
        help="Save recommendations to JSON file"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose logging"
    )
) -> None:
    """
    Generate model recommendations from evaluation results.

    This command analyzes evaluation results and judge scores to provide
    intelligent recommendations about which models to use based on:
    - Overall performance (accuracy, format, compliance)
    - Cost efficiency (best value for money)
    - Budget constraints
    - Different use cases (production, development, etc.)

    Example:

        # Generate recommendations from results
        taskbench recommend --results-file results.json --scores-file scores.json

        # Filter by budget (max 10 cents per request)
        taskbench recommend --budget 0.10

        # Save recommendations to JSON
        taskbench recommend --output recommendations.json
    """
    setup_cli_logging(verbose)

    try:
        # Load evaluation results
        results_path = Path(results_file)
        if not results_path.exists():
            console.print(f"[red]Error: Results file not found: {results_file}[/red]")
            console.print("[yellow]Tip: Run 'taskbench evaluate' first to generate results[/yellow]")
            raise typer.Exit(code=1)

        console.print(f"\n[cyan]Loading evaluation results from {results_file}...[/cyan]")
        results = load_results(results_file)
        console.print(f"[green]✓[/green] Loaded {len(results)} evaluation result(s)")

        # Load judge scores
        scores_path = Path(scores_file)
        if not scores_path.exists():
            console.print(f"\n[yellow]Warning: Scores file not found: {scores_file}[/yellow]")
            console.print("[yellow]Recommendations will be limited without judge scores[/yellow]")
            console.print("[yellow]Tip: Use the judge functionality to generate scores[/yellow]")
            scores = []
        else:
            console.print(f"\n[cyan]Loading judge scores from {scores_file}...[/cyan]")
            scores = load_scores(scores_file)
            console.print(f"[green]✓[/green] Loaded {len(scores)} judge score(s)")

        # Filter by budget if specified
        if budget is not None:
            console.print(f"\n[cyan]Filtering models by budget: ${budget:.4f} per request[/cyan]")
            original_count = len(results)
            results = [r for r in results if r.cost_usd <= budget]
            filtered_count = original_count - len(results)

            if filtered_count > 0:
                console.print(f"[yellow]→[/yellow] Filtered out {filtered_count} model(s) exceeding budget")

            if not results:
                console.print(f"\n[red]Error: No models found within budget of ${budget:.4f}[/red]")
                console.print("[yellow]Try increasing the budget or running more evaluations[/yellow]")
                raise typer.Exit(code=1)

        if not results:
            console.print("\n[red]Error: No evaluation results available[/red]")
            raise typer.Exit(code=1)

        # Create comparison data
        console.print("\n[cyan]Analyzing results and generating comparison...[/cyan]")
        comparison = ModelComparison()
        comparison_data = comparison.compare_results(results, scores)
        console.print(f"[green]✓[/green] Comparison complete: {comparison_data['successful_models']} successful, {comparison_data['failed_models']} failed")

        # Display comparison table
        console.print()
        comparison_table = comparison.generate_comparison_table(comparison_data)
        console.print(comparison_table)

        # Generate recommendations
        console.print("\n[cyan]Generating intelligent recommendations...[/cyan]")
        engine = RecommendationEngine()
        recommendations = engine.generate_recommendations(comparison_data)
        console.print(f"[green]✓[/green] Recommendations generated")

        # Display formatted recommendations
        console.print()
        formatted_recs = engine.format_recommendations(recommendations)
        console.print(formatted_recs)

        # Save to JSON if requested
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            export_data = engine.export_recommendations_json(recommendations)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, default=str)

            console.print(f"\n[green]✓[/green] Recommendations saved to {output}")

    except Exception as e:
        logger.error(f"Failed to generate recommendations: {e}", exc_info=verbose)
        console.print(f"\n[red]Error: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(code=1)


# Helper functions

def display_results_table(results: List[EvaluationResult]) -> None:
    """
    Display evaluation results in a Rich table.

    Args:
        results: List of EvaluationResult objects to display
    """
    table = Table(title="Evaluation Results", box=box.ROUNDED)
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Status", style="white")
    table.add_column("Input\nTokens", justify="right", style="blue")
    table.add_column("Output\nTokens", justify="right", style="blue")
    table.add_column("Total\nTokens", justify="right", style="blue")
    table.add_column("Cost\n(USD)", justify="right", style="green")
    table.add_column("Latency\n(ms)", justify="right", style="yellow")

    for result in results:
        # Status with emoji
        if result.status == "success":
            status = "[green] Success[/green]"
        else:
            status = f"[red] {result.status}[/red]"

        table.add_row(
            result.model_name,
            status,
            f"{result.input_tokens:,}" if result.status == "success" else "",
            f"{result.output_tokens:,}" if result.status == "success" else "",
            f"{result.total_tokens:,}" if result.status == "success" else "",
            f"${result.cost_usd:.4f}" if result.status == "success" else "",
            f"{result.latency_ms:.0f}" if result.status == "success" else ""
        )

    console.print(table)

    # Show errors if any
    failed_results = [r for r in results if r.status != "success"]
    if failed_results:
        console.print("\n[yellow]Failed Evaluations:[/yellow]")
        for result in failed_results:
            console.print(f"  [red][/red] {result.model_name}: {result.error}")


def save_results(results: List[EvaluationResult], output_path: str) -> None:
    """
    Save evaluation results to a JSON file.

    Args:
        results: List of EvaluationResult objects
        output_path: Path to save the JSON file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Convert to JSON-serializable format
    results_data = [result.model_dump() for result in results]

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, default=str)

    logger.info(f"Saved {len(results)} results to {output_path}")


def load_results(results_path: str) -> List[EvaluationResult]:
    """
    Load evaluation results from a JSON file.

    Args:
        results_path: Path to the JSON file

    Returns:
        List of EvaluationResult objects
    """
    with open(results_path, 'r', encoding='utf-8') as f:
        results_data = json.load(f)

    results = [EvaluationResult(**data) for data in results_data]
    logger.info(f"Loaded {len(results)} results from {results_path}")
    return results


def load_scores(scores_path: str) -> List[JudgeScore]:
    """
    Load judge scores from a JSON file.

    Args:
        scores_path: Path to the JSON file

    Returns:
        List of JudgeScore objects
    """
    with open(scores_path, 'r', encoding='utf-8') as f:
        scores_data = json.load(f)

    scores = [JudgeScore(**data) for data in scores_data]
    logger.info(f"Loaded {len(scores)} scores from {scores_path}")
    return scores


def format_results_json(results: List[EvaluationResult]) -> str:
    """
    Format results as JSON string.

    Args:
        results: List of EvaluationResult objects

    Returns:
        Formatted JSON string
    """
    results_data = [result.model_dump() for result in results]
    return json.dumps(results_data, indent=2, default=str)


def format_results_csv(results: List[EvaluationResult]) -> str:
    """
    Format results as CSV string.

    Args:
        results: List of EvaluationResult objects

    Returns:
        CSV formatted string
    """
    lines = []
    # Header
    lines.append("model_name,task_name,status,input_tokens,output_tokens,total_tokens,cost_usd,latency_ms,error")

    # Data rows
    for result in results:
        error_msg = result.error.replace('"', '""') if result.error else ""
        lines.append(
            f'"{result.model_name}","{result.task_name}","{result.status}",'
            f'{result.input_tokens},{result.output_tokens},{result.total_tokens},'
            f'{result.cost_usd},{result.latency_ms:.2f},"{error_msg}"'
        )

    return "\n".join(lines)


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
