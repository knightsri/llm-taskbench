"""
CLI interface for LLM TaskBench.

Provides command-line interface for running evaluations, viewing results,
and getting recommendations.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from taskbench.api.client import OpenRouterClient
from taskbench.core.task import TaskParser
from taskbench.core.models import EvaluationResult, JudgeScore
from taskbench.evaluation.cost import CostTracker
from taskbench.evaluation.executor import ModelExecutor
from taskbench.evaluation.judge import LLMJudge
from taskbench.evaluation.comparison import ModelComparison
from taskbench.evaluation.recommender import RecommendationEngine
from taskbench.usecase import UseCase, list_usecases, UseCaseRun
from taskbench.usecase_parser import load_usecase_from_folder, list_sample_usecases

# Load environment variables
load_dotenv()

app = typer.Typer(
    name="taskbench",
    help="LLM TaskBench - Task-specific LLM evaluation framework",
    add_completion=False
)

# Configure console for Windows compatibility
# Use legacy_windows=True to avoid ANSI escape code issues in PowerShell
if sys.platform == "win32":
    console = Console(legacy_windows=True, force_terminal=True)
else:
    console = Console()


@app.command()
def evaluate(
    task_yaml: str = typer.Argument(..., help="Path to task definition YAML file"),
    usecase_yaml: str = typer.Option(
        "usecases/concepts_extraction.yaml",
        "--usecase",
        "-u",
        help="Path to use-case YAML file"
    ),
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
    skip_judge: bool = typer.Option(
        False,
        "--skip-judge",
        help="Skip judge evaluation (overrides --judge)"
    ),
    chunked: bool = typer.Option(
        False,
        "--chunked/--no-chunked",
        help="Enable chunked processing for long inputs"
    ),
    chunk_chars: int = typer.Option(
        20000,
        "--chunk-chars",
        help="Max characters per chunk when chunked mode is enabled"
    ),
    chunk_overlap: int = typer.Option(
        500,
        "--chunk-overlap",
        help="Overlap characters between chunks when chunked mode is enabled"
    ),
    dynamic_chunk: bool = typer.Option(
        True,
        "--dynamic-chunk/--no-dynamic-chunk",
        help="Derive chunk size from selected models' context windows (default: on)"
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
    asyncio.run(_evaluate_async(
        task_yaml,
        usecase_yaml,
        models,
        input_file,
        output,
        run_judge,
        skip_judge,
        chunked,
        chunk_chars,
        chunk_overlap,
        dynamic_chunk,
        verbose
    ))


async def _evaluate_async(
    task_yaml: str,
    usecase_yaml: str,
    models_str: str,
    input_file: Optional[str],
    output: str,
    run_judge: bool,
    skip_judge: bool,
    chunked: bool,
    chunk_chars: int,
    chunk_overlap: int,
    dynamic_chunk: bool,
    verbose: bool
):
    """Async implementation of evaluate command."""
    try:
        if skip_judge:
            run_judge = False
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
        usecase = None
        if usecase_yaml and Path(usecase_yaml).exists():
            try:
                from taskbench.usecase import UseCase
                usecase = UseCase.load(usecase_yaml)
                console.print(f"[green]Use case loaded: {usecase.name}[/green]")
            except Exception as uc_err:
                console.print(f"[yellow]Warning: failed to load use case {usecase_yaml}: {uc_err}[/yellow]")

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

        # Parse model list (allow "auto" to use orchestrator with usecase)
        model_ids = [m.strip() for m in models_str.split(",")]
        if len(model_ids) == 1 and model_ids[0].lower() == "auto":
            from taskbench.evaluation.orchestrator import LLMOrchestrator
            async with OpenRouterClient(api_key) as client_tmp:
                orch = LLMOrchestrator(client_tmp)
                model_ids = orch.recommend_for_usecase(
                    usecase_goal=usecase.goal if usecase else task.description,
                    require_large_context=True,
                    prioritize_cost=(usecase.cost_priority == "high") if usecase else False
                )
            console.print(f"[cyan]Recommended models: {', '.join(model_ids)}[/cyan]")
        console.print(f"[cyan]Evaluating {len(model_ids)} models: {', '.join(model_ids)}[/cyan]\n")

        # Initialize components
        async with OpenRouterClient(api_key) as client:
            cost_tracker = CostTracker()
            executor = ModelExecutor(client, cost_tracker)

            # Run evaluations
            results = await executor.evaluate_multiple(
                model_ids=model_ids,
                task=task,
                input_data=input_data,
                usecase=usecase,
                chunk_mode=chunked,
                chunk_chars=chunk_chars,
                chunk_overlap=chunk_overlap,
                dynamic_chunk=dynamic_chunk
            )

            scores: List[Optional[JudgeScore]] = []

            # Run judge if requested
            if run_judge:
                console.print("\n[bold cyan]Running LLM-as-judge evaluation...[/bold cyan]\n")
                judge = LLMJudge(client)

                for result in results:
                    if result.status == "success":
                        try:
                            score = await judge.evaluate(task, result, input_data, usecase=usecase)
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
                    comp = ModelComparison()
                    comparison = comp.compare_results(valid_results, valid_scores)
                    table = comp.generate_comparison_table(comparison)
                    console.print(table)
                    # Show recommendations
                    console.print("\n[bold cyan]Recommendations[/bold cyan]\n")
                    engine = RecommendationEngine()
                    recs = engine.generate_recommendations(comparison)
                    console.print(engine.format_recommendations(recs))









            # Save results
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            output_data = {
                "task": task.model_dump(),
                "results": [r.model_dump() for r in results],
                "scores": [s.model_dump() if s else None for s in scores] if run_judge else [],
                "statistics": cost_tracker.get_statistics(),
                "run_judge": run_judge
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


@app.command()
def recommend(
    results_file: str = typer.Option(
        "results/evaluation_results.json",
        "--results",
        "-r",
        help="Path to a saved evaluation results JSON file"
    )
):
    """
    Generate recommendations from a saved evaluation run.
    """
    try:
        path = Path(results_file)
        if not path.exists():
            console.print(f"[red]Results file not found: {results_file}[/red]")
            raise typer.Exit(1)

        data = json.loads(path.read_text(encoding="utf-8"))
        raw_results = data.get("results", [])
        raw_scores = data.get("scores", [])

        results = []
        scores = []

        for item in raw_results:
            results.append(EvaluationResult(**item))

        for s in raw_scores:
            if s is not None:
                scores.append(JudgeScore(**s))

        if not results or not scores:
            console.print("[red]No results or scores found in the file.[/red]")
            raise typer.Exit(1)

        comp = ModelComparison()
        comparison = comp.compare_results(results, scores)

        try:
            table = comp.generate_comparison_table(comparison)
            console.print(table)

            engine = RecommendationEngine()
            recs = engine.generate_recommendations(comparison)
            console.print("\n[bold cyan]Recommendations[/bold cyan]\n")
            console.print(engine.format_recommendations(recs))
        except UnicodeEncodeError:
            # Windows console encoding issue - print simplified results
            console.print("[yellow]Note: Rich table display failed. Showing simplified results:[/yellow]\n")
            console.print("[bold]Model Comparison:[/bold]")
            for model_data in comparison.get("models", []):
                name = model_data.get("name", "Unknown")
                score = model_data.get("overall_score", "N/A")
                cost = model_data.get("cost", 0)
                violations = model_data.get("violations", 0)
                console.print(f"  {name}: Score {score}/100, Cost ${cost:.4f}, Violations: {violations}")

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def sample(
    models: str = typer.Option(
        "anthropic/claude-sonnet-4.5,openai/gpt-4o,qwen/qwen-2.5-72b-instruct",
        "--models",
        "-m",
        help="Comma-separated list of model IDs to evaluate for the sample run"
    ),
    judge: bool = typer.Option(
        False,
        "--judge/--no-judge",
        help="Run judge evaluation for the sample run (default: no judge)"
    ),
    output: str = typer.Option(
        "results/sample_run.json",
        "--output",
        "-o",
        help="Output file for the sample run results"
    )
):
    """
    Run a sample evaluation using the bundled lecture analysis task and sample transcript.
    """
    sample_task = Path("tasks/lecture_analysis.yaml")
    sample_input = Path("tests/fixtures/sample_transcript.txt")

    if not sample_task.exists():
        console.print(f"[red]Sample task not found at {sample_task}[/red]")
        raise typer.Exit(1)
    if not sample_input.exists():
        console.print(f"[red]Sample input not found at {sample_input}[/red]")
        raise typer.Exit(1)

    console.print("[cyan]Running sample evaluation...[/cyan]")
    evaluate.callback(  # type: ignore
        task_yaml=str(sample_task),
        models=models,
        input_file=str(sample_input),
        output=output,
        run_judge=judge,
        skip_judge=not judge,
        verbose=False
    )


@app.command()
def status(
    usecase: Optional[str] = typer.Option(
        None,
        "--usecase",
        "-u",
        help="Show status for a specific use case"
    ),
    detail: bool = typer.Option(
        False,
        "--detail",
        "-d",
        help="Show detailed run information"
    )
):
    """
    Show status of use cases and their evaluation runs.

    Examples:
        taskbench status                    # Show all use cases
        taskbench status -u concepts_extraction  # Show specific use case
        taskbench status -d                 # Show with run details
    """
    try:
        if usecase:
            # Show specific use case
            uc_path = f"usecases/{usecase}.yaml"
            if not Path(uc_path).exists():
                # Try direct path
                uc_path = usecase if usecase.endswith(".yaml") else f"{usecase}.yaml"

            if not Path(uc_path).exists():
                console.print(f"[red]Use case not found: {usecase}[/red]")
                raise typer.Exit(1)

            uc = UseCase.load(uc_path)
            summary = uc.get_summary()

            console.print(f"\n[bold cyan]{summary['name']}[/bold cyan]")
            console.print(f"Goal: {summary['goal'][:100]}...")
            console.print(f"Total runs: {summary['total_runs']}")
            console.print(f"Judged runs: {summary['judged_runs']}")
            console.print(f"Total cost: ${summary['total_cost']:.4f}")

            if summary['best_model']:
                console.print(f"[green]Best model: {summary['best_model']} ({summary['best_score']}/100)[/green]")

            if summary['last_recommendation']:
                console.print(f"\n[cyan]Last recommendation:[/cyan]\n{summary['last_recommendation']}")

            if detail and uc.runs:
                console.print("\n[bold]Runs:[/bold]")
                for run in uc.runs[-5:]:  # Show last 5 runs
                    status_color = "green" if run.status == "completed" else "yellow"
                    judged_icon = "[J]" if run.judged else "[ ]"
                    console.print(
                        f"  {judged_icon} [{status_color}]{run.run_id[:8]}[/{status_color}] "
                        f"| {', '.join(run.models[:3])}{'...' if len(run.models) > 3 else ''} "
                        f"| ${run.total_cost:.4f}"
                    )

        else:
            # Show all use cases - check both legacy YAML and new folder-based results
            statuses = []

            # Check new folder-based results in results/ directory
            results_dir = Path("results")
            if results_dir.exists():
                for usecase_dir in sorted(results_dir.iterdir()):
                    if not usecase_dir.is_dir() or usecase_dir.name.startswith("_"):
                        continue

                    result_files = list(usecase_dir.glob("*.json"))
                    if not result_files:
                        continue

                    # Aggregate stats from all runs
                    total_runs = len(result_files)
                    judged_runs = 0
                    total_cost = 0.0
                    best_model = None
                    best_score = 0

                    for result_file in result_files:
                        try:
                            with open(result_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)

                            scores = data.get("scores", [])
                            if scores and any(s for s in scores if s is not None):
                                judged_runs += 1

                                # Find best model in this run
                                for i, score in enumerate(scores):
                                    if score and score.get("overall_score", 0) > best_score:
                                        best_score = score["overall_score"]
                                        results = data.get("results", [])
                                        if i < len(results):
                                            best_model = results[i].get("model_name", "Unknown")

                            stats = data.get("statistics", {})
                            total_cost += stats.get("total_cost_usd", 0)
                        except Exception:
                            continue

                    statuses.append({
                        "name": usecase_dir.name,
                        "total_runs": total_runs,
                        "judged_runs": judged_runs,
                        "total_cost": total_cost,
                        "best_model": best_model,
                        "best_score": best_score
                    })

            if not statuses:
                console.print("[yellow]No use cases found[/yellow]")
                console.print("Run evaluations with: taskbench run <usecase-folder>")
                raise typer.Exit(0)

            table = Table(title="Use Cases", show_header=True, header_style="bold magenta")
            table.add_column("Name", style="cyan")
            table.add_column("Runs", justify="right")
            table.add_column("Judged", justify="right")
            table.add_column("Cost", justify="right")
            table.add_column("Best Model")
            table.add_column("Score", justify="right")

            for s in statuses:
                if "error" in s:
                    table.add_row(s.get("path", s.get("name", "?")), "[red]Error[/red]", "-", "-", "-", "-")
                else:
                    table.add_row(
                        s["name"],
                        str(s["total_runs"]),
                        str(s["judged_runs"]),
                        f"${s['total_cost']:.2f}",
                        s["best_model"] or "-",
                        f"{s['best_score']}/100" if s["best_score"] else "-"
                    )

            console.print()
            console.print(table)
            console.print()

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def costs(
    openrouter: bool = typer.Option(
        False,
        "--openrouter",
        "-o",
        help="Check OpenRouter account balance"
    )
):
    """
    Show cost tracking information across all use cases.

    Examples:
        taskbench costs            # Show local cost tracking
        taskbench costs --openrouter  # Also check OpenRouter balance
    """
    try:
        global_costs = CostTracker.get_global_costs()

        console.print("\n[bold cyan]TaskBench Cost Summary[/bold cyan]\n")
        console.print(f"Total tracked cost: [green]${global_costs.get('total_cost', 0.0):.4f}[/green]")

        usecases = global_costs.get("usecases", {})
        if usecases:
            console.print("\n[bold]By Use Case:[/bold]")
            for name, data in usecases.items():
                console.print(f"  {name}: ${data.get('total_cost', 0.0):.4f} ({len(data.get('runs', []))} runs)")

        if openrouter:
            console.print("\n[cyan]Checking OpenRouter balance...[/cyan]")
            balance_info = asyncio.run(_check_openrouter_balance())
            if balance_info:
                console.print(f"\n[bold]OpenRouter Account:[/bold]")
                console.print(f"  Label: {balance_info.get('label', 'N/A')}")
                if balance_info.get('limit'):
                    console.print(f"  Limit: ${balance_info['limit']:.2f}")
                if balance_info.get('usage') is not None:
                    console.print(f"  Usage: ${balance_info['usage']:.4f}")
                if balance_info.get('balance') is not None:
                    console.print(f"  [green]Remaining: ${balance_info['balance']:.2f}[/green]")
            else:
                console.print("[yellow]Could not retrieve OpenRouter balance[/yellow]")

        console.print()

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


async def _check_openrouter_balance():
    """Helper to check OpenRouter balance."""
    return await CostTracker.get_openrouter_balance()


async def _convert_text_to_usecase(name: str, free_text: str) -> Optional[str]:
    """Use LLM to convert free-form text to structured use case YAML."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        console.print("[red]OPENROUTER_API_KEY not set[/red]")
        return None

    # Use model selector to get recommended models for this task
    console.print("[cyan]Analyzing task and selecting models...[/cyan]")
    from taskbench.evaluation.model_selector import select_models_for_task

    try:
        model_recommendation = await select_models_for_task(free_text)
        recommended_models = model_recommendation.get("suggested_test_order", [])
        task_analysis = model_recommendation.get("task_analysis", {})

        if recommended_models:
            console.print(f"[green]Recommended models: {', '.join(recommended_models[:5])}[/green]")
            models_yaml = "\n".join([f"  - {m}" for m in recommended_models[:5]])
        else:
            models_yaml = """  - google/gemini-2.5-flash
  - anthropic/claude-sonnet-4.5
  - openai/gpt-4o"""
    except Exception as e:
        console.print(f"[yellow]Model selection failed, using defaults: {e}[/yellow]")
        models_yaml = """  - google/gemini-2.5-flash
  - anthropic/claude-sonnet-4.5
  - openai/gpt-4o"""
        task_analysis = {}

    prompt = f"""Convert the following free-form use case description into a structured YAML format.

## User's Input:
{free_text}

## Task Analysis (from model selector):
{json.dumps(task_analysis, indent=2) if task_analysis else "Not available"}

## Pre-Selected Models (already analyzed for this task):
{models_yaml}

## Required YAML Structure:
Generate a complete use case YAML with these fields:
- name: The use case name (use "{name}" as the identifier)
- goal: Multi-line description of the goal (copy user's goal verbatim if provided)
- llm_notes: Notes/hints for LLMs (copy user's LLM-Notes verbatim if provided)
- output_format: Infer expected output structure from the goal
  - format_type: "json", "csv", or "markdown"
  - fields: List of field definitions with name, type, description
  - example: A realistic example matching the fields
  - notes: Format constraints
- chunk_min_minutes/chunk_max_minutes: If timing constraints mentioned
- coverage_required: true if "no gaps" or full coverage is mentioned
- cost_priority: "high" if low-cost is emphasized, else "balanced"
- quality_priority: "high" if quality is emphasized, else "balanced"
- notes: Additional constraints as bullet points
- default_candidate_models: Use the pre-selected models above

## Output:
Respond with ONLY the YAML content, no explanations or markdown code blocks.
Start directly with "name:" - do not include ```yaml or any other markers."""

    try:
        async with OpenRouterClient(api_key) as client:
            model = os.getenv("GENERAL_TASK_LLM", "anthropic/claude-sonnet-4.5")
            response = await client.complete(
                model=model,
                prompt=prompt,
                max_tokens=2000,
                temperature=0.3
            )

            content = response.content.strip()
            # Remove any markdown code blocks if present
            if content.startswith("```yaml"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]

            return content.strip()

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return None


@app.command()
def wizard():
    """
    Interactive wizard for creating and running evaluations.

    Guides you through:
    1. Selecting or creating a use case
    2. Choosing models to evaluate
    3. Providing input data
    4. Running the evaluation
    5. Viewing results
    """
    console.print("\n[bold cyan]LLM TaskBench - Interactive Wizard[/bold cyan]\n")

    # Step 1: Select or create use case
    console.print("[bold]Step 1: Select Use Case[/bold]")
    uc_files = list_usecases()

    if uc_files:
        console.print("Available use cases:")
        for i, uc_path in enumerate(uc_files, 1):
            try:
                uc = UseCase.load(uc_path)
                console.print(f"  {i}. {uc.name} - {uc.goal[:60]}...")
            except Exception:
                console.print(f"  {i}. {uc_path} [red](error loading)[/red]")
        console.print(f"  {len(uc_files) + 1}. Create new use case")

        choice = typer.prompt("Select option", default="1")
        try:
            idx = int(choice)
            if 1 <= idx <= len(uc_files):
                selected_uc_path = uc_files[idx - 1]
                usecase = UseCase.load(selected_uc_path)
                console.print(f"[green]Selected: {usecase.name}[/green]")
            else:
                console.print("[yellow]Creating new use case not yet implemented in wizard[/yellow]")
                console.print("Create a YAML file in usecases/ directory manually.")
                raise typer.Exit(0)
        except ValueError:
            console.print("[red]Invalid choice[/red]")
            raise typer.Exit(1)
    else:
        console.print("[yellow]No use cases found. Create one in usecases/ directory.[/yellow]")
        raise typer.Exit(1)

    # Step 2: Select models
    console.print("\n[bold]Step 2: Select Models[/bold]")
    if usecase.default_candidate_models:
        console.print(f"Default models: {', '.join(usecase.default_candidate_models)}")
        use_defaults = typer.confirm("Use default models?", default=True)
        if use_defaults:
            model_ids = usecase.default_candidate_models
        else:
            models_input = typer.prompt("Enter model IDs (comma-separated)")
            model_ids = [m.strip() for m in models_input.split(",")]
    else:
        models_input = typer.prompt(
            "Enter model IDs (comma-separated)",
            default="anthropic/claude-sonnet-4.5,openai/gpt-4o"
        )
        model_ids = [m.strip() for m in models_input.split(",")]

    console.print(f"[green]Models: {', '.join(model_ids)}[/green]")

    # Step 3: Input file
    console.print("\n[bold]Step 3: Input Data[/bold]")
    input_file = typer.prompt("Path to input file", default="tasks/lecture_transcript.txt")

    if not Path(input_file).exists():
        console.print(f"[red]File not found: {input_file}[/red]")
        raise typer.Exit(1)

    console.print(f"[green]Input: {input_file}[/green]")

    # Step 4: Confirm and run
    console.print("\n[bold]Step 4: Run Evaluation[/bold]")
    console.print(f"Use case: {usecase.name}")
    console.print(f"Models: {', '.join(model_ids)}")
    console.print(f"Input: {input_file}")

    run_now = typer.confirm("Run evaluation now?", default=True)
    if not run_now:
        console.print("[yellow]Cancelled[/yellow]")
        raise typer.Exit(0)

    # Run the evaluation
    import uuid
    run_id = str(uuid.uuid4())[:8]
    output_file = f"results/{usecase.name}/{run_id}.json"

    console.print(f"\n[cyan]Starting evaluation (Run ID: {run_id})...[/cyan]")

    # Find task file
    task_file = "tasks/lecture_analysis.yaml"  # Default
    if not Path(task_file).exists():
        task_files = list(Path("tasks").glob("*.yaml"))
        if task_files:
            task_file = str(task_files[0])

    evaluate.callback(
        task_yaml=task_file,
        usecase_yaml=selected_uc_path,
        models=",".join(model_ids),
        input_file=input_file,
        output=output_file,
        run_judge=True,
        skip_judge=False,
        chunked=True,
        chunk_chars=20000,
        chunk_overlap=500,
        dynamic_chunk=True,
        verbose=False
    )

    console.print(f"\n[bold green]Evaluation complete![/bold green]")
    console.print(f"Results saved to: {output_file}")
    console.print(f"\nView recommendations with: taskbench recommend --results {output_file}")


@app.command()
def usecases(
    create: Optional[str] = typer.Option(
        None,
        "--create",
        "-c",
        help="Create a new use case with the given name"
    ),
    from_text: bool = typer.Option(
        False,
        "--from-text",
        "-t",
        help="Create use case from free-form text using LLM"
    )
):
    """
    List or create use cases.

    Examples:
        taskbench usecases                    # List all use cases
        taskbench usecases -c mycase          # Create template
        taskbench usecases -c mycase --from-text  # Create from free-form text
    """
    try:
        if create:
            # Create new use case
            uc_path = Path(f"usecases/{create}.yaml")
            if uc_path.exists():
                console.print(f"[red]Use case already exists: {uc_path}[/red]")
                raise typer.Exit(1)

            if from_text:
                # Create from free-form text using LLM
                console.print("\n[bold cyan]Create Use Case from Free-Form Text[/bold cyan]\n")
                console.print("Enter your use case information (Name, Goal, LLM-Notes, etc.)")
                console.print("Press Enter twice when done:\n")

                lines = []
                while True:
                    line = input()
                    if line == "" and lines and lines[-1] == "":
                        break
                    lines.append(line)

                free_text = "\n".join(lines).strip()

                if not free_text:
                    console.print("[red]No input provided[/red]")
                    raise typer.Exit(1)

                console.print("\n[cyan]Converting to structured use case...[/cyan]")

                # Use LLM to convert free-form text to structured YAML
                yaml_content = asyncio.run(_convert_text_to_usecase(create, free_text))

                if yaml_content:
                    console.print("\n[bold]Generated Use Case:[/bold]\n")
                    console.print(yaml_content)

                    confirm = typer.confirm("\nSave this use case?", default=True)
                    if confirm:
                        uc_path.parent.mkdir(parents=True, exist_ok=True)
                        uc_path.write_text(yaml_content, encoding="utf-8")
                        console.print(f"\n[green]Saved: {uc_path}[/green]")
                    else:
                        console.print("[yellow]Cancelled[/yellow]")
                else:
                    console.print("[red]Failed to generate use case[/red]")
                    raise typer.Exit(1)
            else:
                # Create blank template
                template = f"""name: "{create}"
goal: "Describe what you want to achieve with this use case"

llm_notes: |
  Add notes and hints for LLMs processing this use case.
  What priorities matter? Cost vs quality trade-offs?

output_format:
  format_type: "json"
  fields:
    - name: "content"
      type: "string"
      description: "Main output content"
  example: |
    {{"content": "Example output here"}}
  notes: "Describe any format requirements"

cost_priority: "balanced"  # high/low/balanced
quality_priority: "balanced"  # high/low/balanced

notes: |
  - Add specific notes about constraints
  - What should the output include?
  - What should it avoid?

default_judge_model: "${{TASKBENCH_DEFAULT_JUDGE_MODEL:-anthropic/claude-sonnet-4.5}}"
default_candidate_models:
  - anthropic/claude-sonnet-4.5
  - openai/gpt-4o
  - google/gemini-2.5-flash
"""
                uc_path.parent.mkdir(parents=True, exist_ok=True)
                uc_path.write_text(template, encoding="utf-8")
                console.print(f"[green]Created use case template: {uc_path}[/green]")
                console.print("Edit the file to customize your use case.")

        else:
            # List use cases
            uc_files = list_usecases()

            if not uc_files:
                console.print("[yellow]No use cases found[/yellow]")
                console.print("Create one with: taskbench usecases --create mycase")
                raise typer.Exit(0)

            console.print("\n[bold cyan]Use Cases[/bold cyan]\n")
            for uc_path in uc_files:
                try:
                    uc = UseCase.load(uc_path)
                    summary = uc.get_summary()
                    runs_info = f"{summary['total_runs']} runs" if summary['total_runs'] > 0 else "no runs"
                    console.print(f"[cyan]{uc.name}[/cyan]")
                    console.print(f"  Path: {uc_path}")
                    console.print(f"  Goal: {uc.goal[:80]}...")
                    console.print(f"  Status: {runs_info}, ${summary['total_cost']:.2f} spent")
                    if summary['best_model']:
                        console.print(f"  Best: {summary['best_model']} ({summary['best_score']}/100)")
                    console.print()
                except Exception as e:
                    console.print(f"[red]{uc_path}: Error loading - {e}[/red]")

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def run(
    usecase_folder: str = typer.Argument(
        ...,
        help="Path to use case folder (e.g., sample-usecases/00-lecture-concept-extraction)"
    ),
    data_file: Optional[str] = typer.Option(
        None,
        "--data",
        "-d",
        help="Specific data file to use (defaults to first in data/ folder)"
    ),
    models: Optional[str] = typer.Option(
        None,
        "--models",
        "-m",
        help="Comma-separated list of model IDs (auto-selects if not specified)"
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file for results (defaults to use case folder)"
    ),
    regenerate_prompts: bool = typer.Option(
        False,
        "--regenerate-prompts",
        help="Force regeneration of prompts even if they exist"
    ),
    skip_judge: bool = typer.Option(
        False,
        "--skip-judge",
        help="Skip judge evaluation"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging"
    )
):
    """
    Run evaluation on a folder-based use case.

    This command uses the new architecture where use cases are defined as folders
    containing USE-CASE.md, data/, and ground-truth/ subdirectories.

    Examples:
        taskbench run sample-usecases/00-lecture-concept-extraction
        taskbench run sample-usecases/01-meeting-action-items --models gpt-4o,claude-sonnet-4.5
    """
    asyncio.run(_run_usecase_async(
        usecase_folder,
        data_file,
        models,
        output,
        regenerate_prompts,
        skip_judge,
        verbose
    ))


import re


def _extract_timestamp_range(text: str) -> tuple[Optional[str], Optional[str]]:
    """Extract the first and last timestamps from text.

    Looks for timestamps in formats like [00:00:00], 00:00:00, HH:MM:SS, MM:SS
    Returns (first_timestamp, last_timestamp) or (None, None) if none found.
    """
    # Match various timestamp formats
    patterns = [
        r'\[(\d{1,2}:\d{2}:\d{2})\]',  # [00:00:00]
        r'\[(\d{1,2}:\d{2})\]',         # [00:00]
        r'(?:^|\s)(\d{1,2}:\d{2}:\d{2})(?:\s|$)',  # 00:00:00
        r'(?:^|\s)(\d{1,2}:\d{2})(?:\s|$)',         # 00:00
    ]

    all_timestamps = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.MULTILINE)
        all_timestamps.extend(matches)

    if not all_timestamps:
        return None, None

    # Normalize to comparable format and find first/last
    def normalize_ts(ts: str) -> tuple[int, str]:
        """Convert timestamp to seconds for comparison, return (seconds, original)."""
        parts = ts.split(':')
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1]), ts
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2]), ts
        return 0, ts

    normalized = [normalize_ts(ts) for ts in all_timestamps]
    normalized.sort(key=lambda x: x[0])

    return normalized[0][1], normalized[-1][1]


async def _run_usecase_async(
    usecase_folder: str,
    data_file: Optional[str],
    models_str: Optional[str],
    output: Optional[str],
    regenerate_prompts: bool,
    skip_judge: bool,
    verbose: bool
):
    """Async implementation of run command."""
    try:
        # Get API key
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            console.print("[red]Error: OPENROUTER_API_KEY not set in environment[/red]")
            raise typer.Exit(1)

        # Load and parse use case
        console.print(f"[cyan]Loading use case from {usecase_folder}...[/cyan]")
        parsed = load_usecase_from_folder(usecase_folder)
        console.print(f"[green]Use case: {parsed.name}[/green]")
        console.print(f"  Difficulty: {parsed.difficulty}")
        console.print(f"  Output format: {parsed.output_format}")
        console.print(f"  Data files: {len(parsed.data_files)}")
        console.print(f"  Ground truth files: {len(parsed.ground_truth_files)}")
        console.print(f"  Matched pairs: {len(parsed.matched_pairs)}")

        if not parsed.matched_pairs:
            console.print("[red]No matched data/ground-truth pairs found[/red]")
            raise typer.Exit(1)

        # Generate or load prompts
        console.print("\n[cyan]Loading/generating prompts...[/cyan]")
        from taskbench.prompt_generator import generate_prompts_for_usecase
        prompts = await generate_prompts_for_usecase(
            usecase_folder,
            api_key=api_key,
            force_regenerate=regenerate_prompts
        )

        console.print(f"[green]Prompts ready[/green]")
        console.print(f"  Transformation: {prompts['analysis'].get('transformation_type', 'unknown')}")
        console.print(f"  Key fields: {', '.join(prompts['analysis'].get('key_fields', [])[:5])}")

        # Select data file
        if data_file:
            selected_pair = None
            for pair in parsed.matched_pairs:
                if pair.data_file.name in data_file or data_file in str(pair.data_file.path):
                    selected_pair = pair
                    break
            if not selected_pair:
                console.print(f"[red]Data file not found: {data_file}[/red]")
                raise typer.Exit(1)
        else:
            selected_pair = parsed.matched_pairs[0]

        console.print(f"\n[cyan]Using data file: {selected_pair.data_file.name}[/cyan]")

        # Load input data
        input_data = selected_pair.data_file.path.read_text(encoding="utf-8")
        console.print(f"  Size: {len(input_data):,} characters, {selected_pair.data_file.line_count} lines")

        # Select models
        if models_str:
            model_ids = [m.strip() for m in models_str.split(",")]
        else:
            # Auto-select models based on use case
            console.print("\n[cyan]Auto-selecting models for use case...[/cyan]")
            from taskbench.evaluation.model_selector import select_models_for_task
            try:
                model_result = await select_models_for_task(parsed.goal)
                model_ids = model_result.get("suggested_test_order", [])[:3]
                if not model_ids:
                    model_ids = ["google/gemini-2.5-flash", "openai/gpt-4o", "anthropic/claude-sonnet-4.5"]
            except Exception as e:
                console.print(f"[yellow]Model selection failed, using defaults: {e}[/yellow]")
                model_ids = ["google/gemini-2.5-flash", "openai/gpt-4o", "anthropic/claude-sonnet-4.5"]

        console.print(f"[green]Models: {', '.join(model_ids)}[/green]")

        # Create a task definition with generated prompt as description
        from taskbench.core.models import TaskDefinition
        task = TaskDefinition(
            name=parsed.name.replace(" ", "_").lower(),
            description=prompts['task_prompt'],  # Use generated task prompt
            input_type="text",
            output_format=parsed.output_format,
            evaluation_criteria=prompts['analysis'].get('quality_indicators', []),
            constraints={},
            examples=[],
            judge_instructions=prompts['judge_prompt']
        )

        # Initialize components
        async with OpenRouterClient(api_key) as client:
            cost_tracker = CostTracker()
            executor = ModelExecutor(client, cost_tracker)

            # Run evaluation using evaluate_multiple
            results = await executor.evaluate_multiple(
                model_ids=model_ids,
                task=task,
                input_data=input_data,
                max_tokens=4000,
                temperature=0.7
            )

            # Run judge if requested
            scores = []
            if not skip_judge and results:
                console.print("\n[bold cyan]Running LLM-as-judge evaluation...[/bold cyan]\n")
                judge = LLMJudge(client)

                # Load ground truth for comparison
                gt_content = selected_pair.ground_truth_file.content
                if isinstance(gt_content, (dict, list)):
                    gt_str = json.dumps(gt_content, indent=2)
                else:
                    gt_str = str(gt_content)

                # Extract timestamp range from input for judge context
                first_ts, last_ts = _extract_timestamp_range(input_data)
                timestamp_info = ""
                if first_ts and last_ts:
                    timestamp_info = f"\n**Input timestamp range: {first_ts} to {last_ts}**\n"

                for result in results:
                    if result.status == "success":
                        try:
                            # Create enhanced judge prompt with ground truth
                            # Use full ground truth (usually small) and preview of input
                            # Show both start AND end of input to give judge full timestamp range context
                            input_len = len(input_data)
                            preview_start_len = min(input_len, 5000)
                            preview_end_len = min(input_len, 3000)

                            # Build input preview section
                            if input_len <= 8000:
                                # Input is small enough to show entirely
                                input_preview = input_data
                            else:
                                # Show first 5000 chars and last 3000 chars
                                input_preview = f"""{input_data[:preview_start_len]}

[... MIDDLE CONTENT OMITTED FOR BREVITY ...]

{input_data[-preview_end_len:]}"""

                            enhanced_input = f"""# EXPECTED OUTPUT (GROUND TRUTH)
```{selected_pair.ground_truth_file.format_type}
{gt_str}
```

# MODEL'S ACTUAL OUTPUT
{result.output}

# ORIGINAL INPUT DATA CONTEXT
{timestamp_info}
**IMPORTANT: The input spans the FULL timestamp range shown above. Trust the timestamp range, not just the preview below.**

Input length: {input_len:,} characters

**Input content (first and last portions shown):**
{input_preview}
"""
                            score = await judge.evaluate(task, result, enhanced_input)
                            scores.append(score)
                            console.print(
                                f"[green]{result.model_name}[/green]: "
                                f"Score {score.overall_score}/100, "
                                f"{len(score.violations)} violations"
                            )
                        except Exception as e:
                            console.print(f"[red]Failed to judge {result.model_name}: {e}[/red]")
                            scores.append(None)
                    else:
                        scores.append(None)

                # Show comparison
                valid_results = [r for r, s in zip(results, scores) if s is not None]
                valid_scores = [s for s in scores if s is not None]

                if valid_scores:
                    try:
                        console.print("\n")
                        comp = ModelComparison()
                        comparison = comp.compare_results(valid_results, valid_scores)
                        table = comp.generate_comparison_table(comparison)
                        console.print(table)

                        console.print("\n[bold cyan]Recommendations[/bold cyan]\n")
                        engine = RecommendationEngine()
                        recs = engine.generate_recommendations(comparison)
                        console.print(engine.format_recommendations(recs))
                    except UnicodeEncodeError:
                        # Windows console encoding issue - print simplified results
                        console.print("[yellow]Note: Table display failed due to encoding. Showing simplified results:[/yellow]\n")
                        for model_data in comparison.get("models", []):
                            name = model_data.get("name", "Unknown")
                            score = model_data.get("overall_score", "N/A")
                            cost = model_data.get("cost", 0)
                            console.print(f"  {name}: Score {score}/100, Cost ${cost:.4f}")


            # Save results - organize by use case folder
            if output:
                output_path = Path(output)
            else:
                # Extract use case folder name (e.g., "00-lecture-concept-extraction")
                usecase_name = Path(usecase_folder).name

                # Create timestamp for unique filename
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

                # Build path: results/{usecase-name}/{timestamp}_{data-file}.json
                results_dir = Path("results") / usecase_name
                output_path = results_dir / f"{timestamp}_{selected_pair.data_file.name}.json"

            output_path.parent.mkdir(parents=True, exist_ok=True)

            output_data = {
                "usecase": {
                    "name": parsed.name,
                    "folder": usecase_folder,
                    "data_file": str(selected_pair.data_file.path),
                    "ground_truth_file": str(selected_pair.ground_truth_file.path)
                },
                "prompts": prompts,
                "results": [r.model_dump() for r in results],
                "scores": [s.model_dump() if s else None for s in scores] if scores else [],
                "statistics": cost_tracker.get_statistics()
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, default=str)

            console.print(f"\n[green]Results saved to {output_path}[/green]")

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)


@app.command("list-usecases")
def list_usecases_cmd(
    folder: str = typer.Option(
        "sample-usecases",
        "--folder",
        "-f",
        help="Folder containing use cases"
    )
):
    """
    List available use cases from the sample-usecases folder.

    Example:
        taskbench list-usecases
        taskbench list-usecases --folder my-usecases
    """
    try:
        usecases = list_sample_usecases(folder)

        if not usecases:
            console.print(f"[yellow]No use cases found in {folder}[/yellow]")
            raise typer.Exit(0)

        console.print(f"\n[bold cyan]Available Use Cases ({folder})[/bold cyan]\n")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("#", style="dim")
        table.add_column("Name", style="cyan")
        table.add_column("Difficulty")
        table.add_column("Format")
        table.add_column("Data", justify="right")
        table.add_column("GT", justify="right")

        for i, uc in enumerate(usecases, 1):
            if "error" in uc:
                table.add_row(
                    str(i),
                    uc["folder_name"],
                    "[red]Error[/red]",
                    "-",
                    "-",
                    "-"
                )
            else:
                table.add_row(
                    str(i),
                    uc["name"][:40] + "..." if len(uc.get("name", "")) > 40 else uc.get("name", ""),
                    uc.get("difficulty", "-"),
                    uc.get("output_format", "-"),
                    str(uc.get("data_files", 0)),
                    str(uc.get("ground_truth_files", 0))
                )

        console.print(table)
        console.print()
        console.print("[dim]Run a use case with: taskbench run <folder-path>[/dim]")
        console.print()

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command("generate-prompts")
def generate_prompts_cmd(
    usecase_folder: str = typer.Argument(
        ...,
        help="Path to use case folder"
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force regeneration even if prompts exist"
    )
):
    """
    Generate prompts for a use case by analyzing its data and ground truth.

    This command uses an LLM to analyze the use case and generate:
    - Task prompt (what to send to models being evaluated)
    - Judge prompt (how to evaluate outputs)
    - Rubric (scoring criteria)

    Example:
        taskbench generate-prompts sample-usecases/00-lecture-concept-extraction
    """
    asyncio.run(_generate_prompts_async(usecase_folder, force))


async def _generate_prompts_async(usecase_folder: str, force: bool):
    """Async implementation of generate-prompts command."""
    try:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            console.print("[red]Error: OPENROUTER_API_KEY not set[/red]")
            raise typer.Exit(1)

        console.print(f"[cyan]Loading use case from {usecase_folder}...[/cyan]")
        parsed = load_usecase_from_folder(usecase_folder)
        console.print(f"[green]Use case: {parsed.name}[/green]")

        console.print("\n[cyan]Generating prompts...[/cyan]")
        from taskbench.prompt_generator import generate_prompts_for_usecase

        prompts = await generate_prompts_for_usecase(
            usecase_folder,
            api_key=api_key,
            force_regenerate=force
        )

        console.print("\n[bold green]Prompts generated successfully![/bold green]")
        console.print(f"\n[bold]Analysis:[/bold]")
        console.print(f"  Transformation type: {prompts['analysis'].get('transformation_type')}")
        console.print(f"  Key fields: {', '.join(prompts['analysis'].get('key_fields', []))}")
        console.print(f"  Comparison strategy: {prompts['analysis'].get('comparison_strategy')}")

        console.print(f"\n[bold]Files saved to {usecase_folder}/prompts/:[/bold]")
        console.print("  - task-prompt.txt")
        console.print("  - judge-prompt.txt")
        console.print("  - rubric.json")
        console.print("  - analysis.json")
        console.print(f"\n  Full prompts: {usecase_folder}/generated-prompts.json")

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
