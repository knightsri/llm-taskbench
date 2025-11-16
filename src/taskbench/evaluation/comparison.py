"""
Model comparison logic for LLM TaskBench.

This module provides the ModelComparison class for comparing evaluation results
across multiple models, calculating rankings, and generating comparison tables.
"""

import logging
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.table import Table
from rich.text import Text

from taskbench.core.models import EvaluationResult, JudgeScore

logger = logging.getLogger(__name__)


class ModelComparison:
    """
    Compare and analyze evaluation results across multiple models.

    This class provides methods to:
    - Combine evaluation results with judge scores
    - Rank models by performance
    - Identify best overall and best value models
    - Generate formatted comparison tables

    Example:
        >>> comparison = ModelComparison()
        >>> results = [result1, result2, result3]
        >>> scores = [score1, score2, score3]
        >>>
        >>> compared = comparison.compare_results(results, scores)
        >>> best = comparison.identify_best(compared)
        >>> print(f"Best model: {best}")
        >>>
        >>> table = comparison.generate_comparison_table(compared)
        >>> print(table)
    """

    def __init__(self):
        """
        Initialize the ModelComparison.

        Example:
            >>> comparison = ModelComparison()
        """
        self.console = Console()
        logger.info("ModelComparison initialized")

    def compare_results(
        self,
        results: List[EvaluationResult],
        scores: List[JudgeScore]
    ) -> Dict[str, Any]:
        """
        Combine evaluation results with judge scores and calculate rankings.

        This method merges EvaluationResult and JudgeScore data for each model,
        then sorts by overall_score (descending) to create a ranked comparison.

        Args:
            results: List of EvaluationResult objects from model executions
            scores: List of JudgeScore objects from judge evaluations

        Returns:
            Dictionary with:
            - models: List of combined model data, sorted by score (best first)
            - total_models: Number of models compared
            - successful_models: Number of models that succeeded
            - failed_models: Number of models that failed

        Example:
            >>> comparison = ModelComparison()
            >>> compared = comparison.compare_results(results, scores)
            >>> for model in compared['models']:
            ...     print(f"{model['name']}: {model['overall_score']}")
        """
        logger.info(f"Comparing {len(results)} results with {len(scores)} scores")

        # Create a mapping of model names to scores
        score_map = {score.model_evaluated: score for score in scores}

        # Combine results with scores
        combined = []
        for result in results:
            # Get corresponding judge score if available
            judge_score = score_map.get(result.model_name)

            # Build combined model data
            model_data = {
                "name": result.model_name,
                "status": result.status,
                "output": result.output,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "total_tokens": result.total_tokens,
                "cost": result.cost_usd,
                "latency_ms": result.latency_ms,
                "error": result.error,
            }

            # Add judge scores if available
            if judge_score:
                model_data.update({
                    "accuracy_score": judge_score.accuracy_score,
                    "format_score": judge_score.format_score,
                    "compliance_score": judge_score.compliance_score,
                    "overall_score": judge_score.overall_score,
                    "violations": judge_score.violations,
                    "reasoning": judge_score.reasoning,
                })
            else:
                # No judge score available (likely failed evaluation)
                model_data.update({
                    "accuracy_score": 0,
                    "format_score": 0,
                    "compliance_score": 0,
                    "overall_score": 0,
                    "violations": [],
                    "reasoning": "Evaluation failed or not judged",
                })

            # Calculate value rating (score per dollar spent)
            if model_data["cost"] > 0 and model_data["overall_score"] > 0:
                model_data["value_rating"] = model_data["overall_score"] / model_data["cost"]
            else:
                model_data["value_rating"] = 0.0

            combined.append(model_data)

        # Sort by overall_score (descending - best first)
        combined.sort(key=lambda x: x["overall_score"], reverse=True)

        # Add rank to each model
        for rank, model in enumerate(combined, 1):
            model["rank"] = rank

        # Calculate statistics
        successful = sum(1 for m in combined if m["status"] == "success")
        failed = len(combined) - successful

        comparison = {
            "models": combined,
            "total_models": len(combined),
            "successful_models": successful,
            "failed_models": failed,
        }

        logger.info(
            f"Comparison complete: {successful} successful, {failed} failed, "
            f"best score: {combined[0]['overall_score'] if combined else 0}"
        )

        return comparison

    def identify_best(self, comparison: Dict[str, Any]) -> str:
        """
        Identify the model with the highest overall score.

        Args:
            comparison: Dictionary from compare_results()

        Returns:
            Name of the best-performing model

        Raises:
            ValueError: If no models are available

        Example:
            >>> comparison = ModelComparison()
            >>> compared = comparison.compare_results(results, scores)
            >>> best = comparison.identify_best(compared)
            >>> print(f"Best model: {best}")
        """
        models = comparison.get("models", [])

        if not models:
            raise ValueError("No models to compare")

        # Models are already sorted by overall_score (descending)
        best_model = models[0]

        logger.info(
            f"Best model: {best_model['name']} with score {best_model['overall_score']}"
        )

        return best_model["name"]

    def identify_best_value(
        self,
        comparison: Dict[str, Any],
        max_cost: Optional[float] = None
    ) -> str:
        """
        Identify the model with the best score-to-cost ratio.

        This method finds the model that provides the best value by calculating
        the score per dollar spent. Optionally filters by maximum cost.

        Args:
            comparison: Dictionary from compare_results()
            max_cost: Optional maximum cost filter (e.g., 0.10 for 10 cents)

        Returns:
            Name of the best value model

        Raises:
            ValueError: If no models meet the criteria

        Example:
            >>> comparison = ModelComparison()
            >>> compared = comparison.compare_results(results, scores)
            >>> best_value = comparison.identify_best_value(compared, max_cost=0.50)
            >>> print(f"Best value: {best_value}")
        """
        models = comparison.get("models", [])

        if not models:
            raise ValueError("No models to compare")

        # Filter by max_cost if specified
        eligible_models = models
        if max_cost is not None:
            eligible_models = [m for m in models if m["cost"] <= max_cost]

        if not eligible_models:
            raise ValueError(
                f"No models found within budget of ${max_cost:.2f}"
            )

        # Filter successful models only
        successful_models = [
            m for m in eligible_models
            if m["status"] == "success" and m["overall_score"] > 0
        ]

        if not successful_models:
            raise ValueError("No successful models to compare")

        # Find model with highest value_rating
        best_value_model = max(successful_models, key=lambda m: m["value_rating"])

        logger.info(
            f"Best value model: {best_value_model['name']} with "
            f"value rating {best_value_model['value_rating']:.2f} "
            f"(score {best_value_model['overall_score']}, cost ${best_value_model['cost']:.4f})"
        )

        return best_value_model["name"]

    def generate_comparison_table(self, comparison: Dict[str, Any]) -> str:
        """
        Generate a beautiful comparison table using Rich.

        Creates a formatted table showing:
        - Rank
        - Model name
        - Overall score
        - Violations count
        - Cost
        - Value rating (score/cost)

        Models are color-coded based on performance tier:
        - Green: Excellent (90+)
        - Cyan: Good (80-89)
        - Yellow: Acceptable (70-79)
        - Red: Poor (<70)

        Args:
            comparison: Dictionary from compare_results()

        Returns:
            Formatted table as a string

        Example:
            >>> comparison = ModelComparison()
            >>> compared = comparison.compare_results(results, scores)
            >>> table = comparison.generate_comparison_table(compared)
            >>> print(table)
        """
        models = comparison.get("models", [])

        if not models:
            return "No models to display."

        # Create Rich table
        table = Table(
            title="Model Comparison Results",
            show_header=True,
            header_style="bold magenta",
            show_lines=True,
            title_style="bold blue"
        )

        # Add columns
        table.add_column("Rank", justify="center", style="cyan", width=6)
        table.add_column("Model", justify="left", style="white", width=35)
        table.add_column("Score", justify="center", width=8)
        table.add_column("Violations", justify="center", width=11)
        table.add_column("Cost", justify="right", style="yellow", width=10)
        table.add_column("Value", justify="center", width=10)
        table.add_column("Status", justify="center", width=10)

        # Add rows
        for model in models:
            # Determine score color based on performance tier
            score = model["overall_score"]
            if score >= 90:
                score_color = "green"
                tier = "Excellent"
            elif score >= 80:
                score_color = "cyan"
                tier = "Good"
            elif score >= 70:
                score_color = "yellow"
                tier = "Acceptable"
            else:
                score_color = "red"
                tier = "Poor"

            # Format score with color
            score_text = Text(str(score), style=f"bold {score_color}")

            # Format violations
            violation_count = len(model["violations"])
            if violation_count == 0:
                violations_text = Text("None", style="green")
            elif violation_count <= 2:
                violations_text = Text(str(violation_count), style="yellow")
            else:
                violations_text = Text(str(violation_count), style="red")

            # Format cost
            cost_text = f"${model['cost']:.4f}"

            # Format value rating
            value = model["value_rating"]
            if value > 0:
                value_text = Text(f"{value:.1f}", style="cyan")
            else:
                value_text = Text("N/A", style="dim")

            # Format status
            if model["status"] == "success":
                status_text = Text("✓ Success", style="green")
            else:
                status_text = Text("✗ Failed", style="red")

            # Format model name (truncate if too long)
            model_name = model["name"]
            if len(model_name) > 33:
                model_name = model_name[:30] + "..."

            # Add row
            table.add_row(
                f"#{model['rank']}",
                model_name,
                score_text,
                violations_text,
                cost_text,
                value_text,
                status_text
            )

        # Render table to string
        from io import StringIO
        string_io = StringIO()
        temp_console = Console(file=string_io, force_terminal=True)
        temp_console.print(table)
        table_str = string_io.getvalue()

        logger.debug("Generated comparison table")
        return table_str

    def get_summary_statistics(self, comparison: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate summary statistics from comparison results.

        Args:
            comparison: Dictionary from compare_results()

        Returns:
            Dictionary with summary statistics:
            - total_models: Total number of models
            - successful_models: Number of successful models
            - failed_models: Number of failed models
            - average_score: Average overall score
            - average_cost: Average cost
            - total_cost: Total cost across all models
            - best_score: Highest score achieved
            - worst_score: Lowest score achieved
            - best_value: Highest value rating

        Example:
            >>> comparison = ModelComparison()
            >>> compared = comparison.compare_results(results, scores)
            >>> stats = comparison.get_summary_statistics(compared)
            >>> print(f"Average score: {stats['average_score']:.1f}")
        """
        models = comparison.get("models", [])
        successful = [m for m in models if m["status"] == "success"]

        if not successful:
            return {
                "total_models": len(models),
                "successful_models": 0,
                "failed_models": len(models),
                "average_score": 0.0,
                "average_cost": 0.0,
                "total_cost": 0.0,
                "best_score": 0,
                "worst_score": 0,
                "best_value": 0.0,
            }

        scores = [m["overall_score"] for m in successful]
        costs = [m["cost"] for m in successful]
        values = [m["value_rating"] for m in successful if m["value_rating"] > 0]

        stats = {
            "total_models": len(models),
            "successful_models": len(successful),
            "failed_models": len(models) - len(successful),
            "average_score": sum(scores) / len(scores) if scores else 0.0,
            "average_cost": sum(costs) / len(costs) if costs else 0.0,
            "total_cost": sum(costs),
            "best_score": max(scores) if scores else 0,
            "worst_score": min(scores) if scores else 0,
            "best_value": max(values) if values else 0.0,
        }

        logger.debug(f"Summary statistics calculated: {stats}")
        return stats

    def __str__(self) -> str:
        """String representation of ModelComparison."""
        return "ModelComparison()"

    def __repr__(self) -> str:
        """Repr of ModelComparison."""
        return self.__str__()
