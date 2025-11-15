"""
CLI interface for LLM TaskBench.

Provides command-line interface for running evaluations, viewing results,
and getting recommendations.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import List, Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from taskbench.api.client import OpenRouterClient
from taskbench.core.task import TaskParser
from taskbench.evaluation.cost import CostTracker
from taskbench.evaluation.executor import ModelExecutor
from taskbench.evaluation.judge import LLMJudge, ModelComparison

# Load environment variables
load_dotenv()

app = typer.Typer(
    name="taskbench",
    help="LLM TaskBench - Task-specific LLM evaluation framework",
    add_completion=False
)
console = Console()


@app.command()
def evaluate(
    task_yaml: str = typer.Argument(..., help="Path to task definition YAML file"),
    models: str = typer.Option(
        "anthropic/claude-sonnet-4.5,openai/gpt-4o,qwen/qwen-2.5-72b-instruct",
        "--models",
        "-m",
        help="Comma-separated list of model IDs to evaluate"
    ),
    input_file: Optional[str] = typer.Option(
        None,
        "--input-file",
        "-i",
        help="Path to input data file"
    ),
    output: str = typer.Option(
        "results/evaluation_results.json",
        "--output",
        "-o",
        help="Output file for results"
    ),
    run_judge: bool = typer.Option(
        True,
        "--judge/--no-judge",
        help="Run LLM-as-judge evaluation"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging"
    )
):
    """
    Evaluate multiple LLMs on a specific task.

    Example:
        taskbench evaluate tasks/lecture_analysis.yaml --models claude-sonnet-4.5,gpt-4o --input-file data/transcript.txt
    """
    asyncio.run(_evaluate_async(task_yaml, models, input_file, output, run_judge, verbose))


async def _evaluate_async(
    task_yaml: str,
    models_str: str,
    input_file: Optional[str],
    output: str,
    run_judge: bool,
    verbose: bool
):
    """Async implementation of evaluate command."""
    try:
        # Get API key
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            console.print("[red]Error: OPENROUTER_API_KEY not set in environment[/red]")
            console.print("Please set it in .env file or export it as an environment variable")
            raise typer.Exit(1)

        # Load task definition
        console.print(f"[cyan]Loading task definition from {task_yaml}...[/cyan]")
        parser = TaskParser()
        task = parser.load_from_yaml(task_yaml)

        # Validate task
        is_valid, errors = parser.validate_task(task)
        if not is_valid:
            console.print("[red]Task validation failed:[/red]")
            for error in errors:
                console.print(f"  - {error}")
            raise typer.Exit(1)

        console.print(f"[green] Task '{task.name}' loaded successfully[/green]")

        # Load input data
        if input_file:
            with open(input_file, 'r', encoding='utf-8') as f:
                input_data = f.read()
            console.print(f"[green] Loaded input data from {input_file}[/green]")
        else:
            # Use example data if available
            if task.examples and len(task.examples) > 0:
                input_data = task.examples[0].get("input", "Sample input data")
                console.print("[yellow]No input file provided, using example from task definition[/yellow]")
            else:
                console.print("[red]Error: No input file provided and no examples in task definition[/red]")
                raise typer.Exit(1)

        # Parse model list
        model_ids = [m.strip() for m in models_str.split(",")]
        console.print(f"[cyan]Evaluating {len(model_ids)} models: {', '.join(model_ids)}[/cyan]\n")

        # Initialize components
        async with OpenRouterClient(api_key) as client:
            cost_tracker = CostTracker()
            executor = ModelExecutor(client, cost_tracker)

            # Run evaluations
            results = await executor.evaluate_multiple(
                model_ids=model_ids,
                task=task,
                input_data=input_data
            )

            # Run judge if requested
            if run_judge:
                console.print("\n[bold cyan]Running LLM-as-judge evaluation...[/bold cyan]\n")
                judge = LLMJudge(client)
                scores = []

                for result in results:
                    if result.status == "success":
                        try:
                            score = await judge.evaluate(task, result, input_data)
                            scores.append(score)
                            console.print(
                                f"[green] {result.model_name}[/green]: "
                                f"Score {score.overall_score}/100, "
                                f"{len(score.violations)} violations"
                            )
                        except Exception as e:
                            console.print(f"[red] Failed to judge {result.model_name}: {str(e)}[/red]")
                            scores.append(None)
                    else:
                        scores.append(None)

                # Show comparison if we have scores
                valid_results = [r for r, s in zip(results, scores) if s is not None]
                valid_scores = [s for s in scores if s is not None]

                if valid_scores:
                    console.print("\n")
                    comparison = ModelComparison.compare_results(valid_results, valid_scores)
                    table = ModelComparison.generate_comparison_table(comparison)
                    console.print(table)

                    # Show recommendations
                    console.print("\n[bold cyan]=Ê RECOMMENDATIONS[/bold cyan]\n")

                    best_model = ModelComparison.identify_best(comparison)
                    best_value = ModelComparison.identify_best_value(comparison)

                    console.print(f"<Æ [bold green]Best Overall[/bold green]: {best_model}")
                    best_item = next(c for c in comparison if c["model"] == best_model)
                    console.print(f"   Score: {best_item['overall_score']}/100, Cost: ${best_item['cost_usd']:.4f}")

                    if best_value != best_model:
                        console.print(f"\n=Ž [bold yellow]Best Value[/bold yellow]: {best_value}")
                        value_item = next(c for c in comparison if c["model"] == best_value)
                        console.print(f"   Score: {value_item['overall_score']}/100, Cost: ${value_item['cost_usd']:.4f}")

            # Save results
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            output_data = {
                "task": task.model_dump(),
                "results": [r.model_dump() for r in results],
                "scores": [s.model_dump() if s else None for s in scores] if run_judge else [],
                "statistics": cost_tracker.get_statistics()
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, default=str)

            console.print(f"\n[green] Results saved to {output}[/green]")

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)


@app.command()
def models(
    list_models: bool = typer.Option(
        False,
        "--list",
        "-l",
        help="List all available models with pricing"
    ),
    info: Optional[str] = typer.Option(
        None,
        "--info",
        "-i",
        help="Show detailed info for specific model"
    )
):
    """
    Show available models and pricing information.

    Example:
        taskbench models --list
        taskbench models --info anthropic/claude-sonnet-4.5
    """
    try:
        cost_tracker = CostTracker()

        if info:
            # Show detailed info for one model
            model = cost_tracker.get_model_config(info)
            if not model:
                console.print(f"[red]Model '{info}' not found[/red]")
                raise typer.Exit(1)

            console.print(f"\n[bold cyan]{model.display_name}[/bold cyan]")
            console.print(f"ID: {model.model_id}")
            console.print(f"Provider: {model.provider}")
            console.print(f"Context Window: {model.context_window:,} tokens")
            console.print(f"Input Price: ${model.input_price_per_1m:.2f} per 1M tokens")
            console.print(f"Output Price: ${model.output_price_per_1m:.2f} per 1M tokens")
            console.print()

        elif list_models:
            # Show all models in table
            models_list = cost_tracker.list_models()

            table = Table(title="Available Models", show_header=True, header_style="bold magenta")
            table.add_column("Model ID", style="cyan")
            table.add_column("Display Name")
            table.add_column("Provider", style="green")
            table.add_column("Input $/1M", justify="right")
            table.add_column("Output $/1M", justify="right")

            for model in models_list:
                table.add_row(
                    model.model_id,
                    model.display_name,
                    model.provider,
                    f"${model.input_price_per_1m:.2f}",
                    f"${model.output_price_per_1m:.2f}"
                )

            console.print()
            console.print(table)
            console.print()

        else:
            console.print("[yellow]Use --list to show all models or --info <model_id> for details[/yellow]")

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def validate(
    task_yaml: str = typer.Argument(..., help="Path to task definition YAML file")
):
    """
    Validate a task definition file.

    Example:
        taskbench validate tasks/my_task.yaml
    """
    try:
        parser = TaskParser()

        console.print(f"[cyan]Validating {task_yaml}...[/cyan]")

        # Load task
        task = parser.load_from_yaml(task_yaml)
        console.print(f"[green] YAML is valid and parseable[/green]")

        # Validate task
        is_valid, errors = parser.validate_task(task)

        if is_valid:
            console.print(f"[green] Task '{task.name}' passed all validation checks[/green]")
            console.print(f"\nTask: {task.name}")
            console.print(f"Input type: {task.input_type}")
            console.print(f"Output format: {task.output_format}")
            console.print(f"Evaluation criteria: {len(task.evaluation_criteria)}")
            console.print(f"Constraints: {len(task.constraints)}")
            raise typer.Exit(0)
        else:
            console.print(f"[red] Task validation failed with {len(errors)} error(s):[/red]")
            for error in errors:
                console.print(f"  - {error}")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
