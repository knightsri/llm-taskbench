"""
Recommendation engine for LLM TaskBench.

This module provides the RecommendationEngine class for generating actionable
recommendations about which models to use based on evaluation results.
"""

import logging
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """
    Generate intelligent recommendations from model evaluation results.

    This class analyzes comparison results and provides actionable recommendations
    for different use cases:
    - Best overall performance
    - Best value for money
    - Budget-friendly option
    - Premium option

    Example:
        >>> engine = RecommendationEngine()
        >>> comparison = {...}  # from ModelComparison.compare_results()
        >>>
        >>> recommendations = engine.generate_recommendations(comparison)
        >>> formatted = engine.format_recommendations(recommendations)
        >>> print(formatted)
    """

    # Performance tier thresholds
    EXCELLENT_THRESHOLD = 90
    GOOD_THRESHOLD = 80
    ACCEPTABLE_THRESHOLD = 70

    def __init__(self):
        """
        Initialize the RecommendationEngine.

        Example:
            >>> engine = RecommendationEngine()
        """
        self.console = Console()
        logger.info("RecommendationEngine initialized")

    def _classify_tier(self, score: int) -> str:
        """
        Classify a score into a performance tier.

        Args:
            score: Overall score (0-100)

        Returns:
            Tier name: 'Excellent', 'Good', 'Acceptable', or 'Poor'
        """
        if score >= self.EXCELLENT_THRESHOLD:
            return "Excellent"
        elif score >= self.GOOD_THRESHOLD:
            return "Good"
        elif score >= self.ACCEPTABLE_THRESHOLD:
            return "Acceptable"
        else:
            return "Poor"

    def generate_recommendations(
        self,
        comparison: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive recommendations from comparison results.

        This method analyzes all models and identifies:
        - Performance tiers (Excellent, Good, Acceptable, Poor)
        - Best overall performer
        - Best value (score/cost ratio)
        - Budget option (cheapest acceptable model)
        - Premium option (highest score regardless of cost)
        - Use case recommendations

        Args:
            comparison: Dictionary from ModelComparison.compare_results()

        Returns:
            Dictionary containing:
            - tiers: Models grouped by performance tier
            - best_overall: Best performing model details
            - best_value: Best value model details
            - budget_option: Most cost-effective acceptable model
            - premium_option: Highest scoring model
            - recommendations: Specific use case recommendations

        Example:
            >>> engine = RecommendationEngine()
            >>> recs = engine.generate_recommendations(comparison)
            >>> print(recs['best_overall']['name'])
        """
        logger.info("Generating recommendations from comparison data")

        models = comparison.get("models", [])
        successful_models = [m for m in models if m["status"] == "success"]

        if not successful_models:
            logger.warning("No successful models to generate recommendations")
            return {
                "tiers": {"Excellent": [], "Good": [], "Acceptable": [], "Poor": []},
                "best_overall": None,
                "best_value": None,
                "budget_option": None,
                "premium_option": None,
                "recommendations": {
                    "general": "No successful evaluations completed. Please check for errors."
                }
            }

        # Classify models into tiers
        tiers = {
            "Excellent": [],
            "Good": [],
            "Acceptable": [],
            "Poor": []
        }

        for model in successful_models:
            tier = self._classify_tier(model["overall_score"])
            tiers[tier].append(model)

        # Identify best overall (highest score)
        best_overall = max(successful_models, key=lambda m: m["overall_score"])

        # Identify best value (highest value_rating)
        models_with_value = [m for m in successful_models if m["value_rating"] > 0]
        best_value = (
            max(models_with_value, key=lambda m: m["value_rating"])
            if models_with_value else None
        )

        # Identify budget option (cheapest model with acceptable score >= 70)
        acceptable_models = [
            m for m in successful_models
            if m["overall_score"] >= self.ACCEPTABLE_THRESHOLD
        ]
        budget_option = (
            min(acceptable_models, key=lambda m: m["cost"])
            if acceptable_models else None
        )

        # Premium option is same as best overall
        premium_option = best_overall

        # Generate use case recommendations
        recommendations = self._generate_use_case_recommendations(
            best_overall=best_overall,
            best_value=best_value,
            budget_option=budget_option,
            tiers=tiers
        )

        result = {
            "tiers": tiers,
            "best_overall": best_overall,
            "best_value": best_value,
            "budget_option": budget_option,
            "premium_option": premium_option,
            "recommendations": recommendations,
        }

        logger.info(
            f"Recommendations generated: {len(tiers['Excellent'])} excellent, "
            f"{len(tiers['Good'])} good, {len(tiers['Acceptable'])} acceptable, "
            f"{len(tiers['Poor'])} poor"
        )

        return result

    def _generate_use_case_recommendations(
        self,
        best_overall: Dict[str, Any],
        best_value: Optional[Dict[str, Any]],
        budget_option: Optional[Dict[str, Any]],
        tiers: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, str]:
        """
        Generate specific recommendations for different use cases.

        Args:
            best_overall: Best overall model
            best_value: Best value model
            budget_option: Budget-friendly model
            tiers: Models grouped by tier

        Returns:
            Dictionary with recommendations for different use cases
        """
        recommendations = {}

        # General recommendation
        if best_overall["overall_score"] >= self.EXCELLENT_THRESHOLD:
            recommendations["general"] = (
                f"Use {best_overall['name']} for best results "
                f"(score: {best_overall['overall_score']}, cost: ${best_overall['cost']:.4f})"
            )
        elif tiers["Excellent"]:
            top_model = tiers["Excellent"][0]
            recommendations["general"] = (
                f"Use {top_model['name']} for excellent performance "
                f"(score: {top_model['overall_score']}, cost: ${top_model['cost']:.4f})"
            )
        else:
            recommendations["general"] = (
                "No models achieved excellent scores. Consider task refinement."
            )

        # Production use
        if best_overall["overall_score"] >= self.GOOD_THRESHOLD:
            recommendations["production"] = (
                f"{best_overall['name']} - Highest quality for production workloads"
            )
        else:
            recommendations["production"] = (
                "No models meet recommended threshold (80+) for production use"
            )

        # Cost-sensitive use
        if best_value:
            recommendations["cost_sensitive"] = (
                f"{best_value['name']} - Best value "
                f"(score: {best_value['overall_score']}, "
                f"cost: ${best_value['cost']:.4f}, "
                f"value: {best_value['value_rating']:.1f} points/$)"
            )
        else:
            recommendations["cost_sensitive"] = "No cost-effective options available"

        # Budget-constrained use
        if budget_option:
            recommendations["budget"] = (
                f"{budget_option['name']} - Cheapest acceptable option "
                f"(score: {budget_option['overall_score']}, cost: ${budget_option['cost']:.4f})"
            )
        else:
            recommendations["budget"] = (
                "No models meet minimum quality threshold (70+)"
            )

        # Development/testing
        if budget_option and budget_option["cost"] < best_overall["cost"] * 0.5:
            recommendations["development"] = (
                f"{budget_option['name']} - Good for development/testing "
                f"(low cost: ${budget_option['cost']:.4f})"
            )
        elif tiers["Good"] or tiers["Acceptable"]:
            dev_models = tiers["Good"] + tiers["Acceptable"]
            cheapest = min(dev_models, key=lambda m: m["cost"])
            recommendations["development"] = (
                f"{cheapest['name']} - Suitable for development "
                f"(cost: ${cheapest['cost']:.4f})"
            )
        else:
            recommendations["development"] = (
                f"{best_overall['name']} - Use best model for all environments"
            )

        return recommendations

    def format_recommendations(self, recs: Dict[str, Any]) -> str:
        """
        Format recommendations into a beautiful Rich display.

        Creates a formatted output with:
        - Performance tier breakdown table
        - Highlighted recommendations in panels
        - Specific numbers and actionable advice
        - Visual indicators (icons/colors)

        Args:
            recs: Dictionary from generate_recommendations()

        Returns:
            Formatted string with recommendations

        Example:
            >>> engine = RecommendationEngine()
            >>> recs = engine.generate_recommendations(comparison)
            >>> formatted = engine.format_recommendations(recs)
            >>> print(formatted)
        """
        from io import StringIO

        # Create string buffer to capture Rich output
        string_io = StringIO()
        temp_console = Console(file=string_io, force_terminal=True, width=100)

        # Title
        temp_console.print("\n")
        temp_console.print(
            Panel.fit(
                "[bold cyan]Model Recommendations[/bold cyan]",
                border_style="cyan"
            )
        )
        temp_console.print("\n")

        # Performance Tier Summary
        tiers = recs.get("tiers", {})
        tier_table = Table(
            title="Performance Tier Summary",
            show_header=True,
            header_style="bold magenta"
        )
        tier_table.add_column("Tier", style="cyan", width=15)
        tier_table.add_column("Models", justify="center", width=10)
        tier_table.add_column("Score Range", justify="center", width=15)

        tier_table.add_row(
            "🏆 Excellent",
            str(len(tiers.get("Excellent", []))),
            "90-100",
            style="green"
        )
        tier_table.add_row(
            "⭐ Good",
            str(len(tiers.get("Good", []))),
            "80-89",
            style="cyan"
        )
        tier_table.add_row(
            "✓ Acceptable",
            str(len(tiers.get("Acceptable", []))),
            "70-79",
            style="yellow"
        )
        tier_table.add_row(
            "✗ Poor",
            str(len(tiers.get("Poor", []))),
            "0-69",
            style="red"
        )

        temp_console.print(tier_table)
        temp_console.print("\n")

        # Key Recommendations
        best_overall = recs.get("best_overall")
        best_value = recs.get("best_value")
        budget_option = recs.get("budget_option")
        premium_option = recs.get("premium_option")

        # Best Overall Panel
        if best_overall:
            overall_text = (
                f"[bold white]{best_overall['name']}[/bold white]\n\n"
                f"[cyan]Overall Score:[/cyan] {best_overall['overall_score']}/100\n"
                f"[cyan]Accuracy:[/cyan] {best_overall['accuracy_score']}/100 | "
                f"[cyan]Format:[/cyan] {best_overall['format_score']}/100 | "
                f"[cyan]Compliance:[/cyan] {best_overall['compliance_score']}/100\n"
                f"[yellow]Cost:[/yellow] ${best_overall['cost']:.4f}\n"
                f"[magenta]Violations:[/magenta] {len(best_overall['violations'])}"
            )
            temp_console.print(
                Panel(
                    overall_text,
                    title="[bold green]🏆 Best Overall Performance[/bold green]",
                    border_style="green",
                    padding=(1, 2)
                )
            )
            temp_console.print("\n")

        # Best Value Panel
        if best_value and best_value != best_overall:
            value_text = (
                f"[bold white]{best_value['name']}[/bold white]\n\n"
                f"[cyan]Overall Score:[/cyan] {best_value['overall_score']}/100\n"
                f"[yellow]Cost:[/yellow] ${best_value['cost']:.4f}\n"
                f"[green]Value Rating:[/green] {best_value['value_rating']:.1f} points per dollar\n"
                f"[magenta]Violations:[/magenta] {len(best_value['violations'])}"
            )
            temp_console.print(
                Panel(
                    value_text,
                    title="[bold cyan]💰 Best Value for Money[/bold cyan]",
                    border_style="cyan",
                    padding=(1, 2)
                )
            )
            temp_console.print("\n")

        # Budget Option Panel
        if budget_option and budget_option not in [best_overall, best_value]:
            budget_text = (
                f"[bold white]{budget_option['name']}[/bold white]\n\n"
                f"[cyan]Overall Score:[/cyan] {budget_option['overall_score']}/100\n"
                f"[yellow]Cost:[/yellow] ${budget_option['cost']:.4f}\n"
                f"[green]Savings:[/green] "
                f"${(best_overall['cost'] - budget_option['cost']):.4f} "
                f"vs. best overall\n"
                f"[magenta]Violations:[/magenta] {len(budget_option['violations'])}"
            )
            temp_console.print(
                Panel(
                    budget_text,
                    title="[bold yellow]💵 Budget-Friendly Option[/bold yellow]",
                    border_style="yellow",
                    padding=(1, 2)
                )
            )
            temp_console.print("\n")

        # Use Case Recommendations
        recommendations = recs.get("recommendations", {})
        if recommendations:
            rec_table = Table(
                title="Use Case Recommendations",
                show_header=True,
                header_style="bold magenta",
                show_lines=True
            )
            rec_table.add_column("Use Case", style="cyan", width=20)
            rec_table.add_column("Recommendation", style="white", width=70)

            # Map use cases to friendly names
            use_case_names = {
                "general": "🎯 General Purpose",
                "production": "🚀 Production",
                "cost_sensitive": "💰 Cost-Sensitive",
                "budget": "💵 Budget-Constrained",
                "development": "🔧 Development/Testing"
            }

            for key, name in use_case_names.items():
                if key in recommendations:
                    rec_table.add_row(name, recommendations[key])

            temp_console.print(rec_table)
            temp_console.print("\n")

        # Additional Insights
        if tiers:
            insights = []

            excellent_count = len(tiers.get("Excellent", []))
            if excellent_count > 1:
                insights.append(
                    f"✨ {excellent_count} models achieved excellent scores (90+). "
                    "You have multiple high-quality options."
                )

            poor_count = len(tiers.get("Poor", []))
            total_models = sum(len(models) for models in tiers.values())
            if poor_count > 0 and poor_count == total_models:
                insights.append(
                    "⚠️  All models scored below 70. Consider refining your task "
                    "definition or trying different models."
                )

            if best_value and best_overall and best_value != best_overall:
                savings = best_overall["cost"] - best_value["cost"]
                score_diff = best_overall["overall_score"] - best_value["overall_score"]
                if savings > 0:
                    insights.append(
                        f"💡 You can save ${savings:.4f} per request by using "
                        f"{best_value['name']} with only {score_diff} point score reduction."
                    )

            if insights:
                insights_text = "\n\n".join(insights)
                temp_console.print(
                    Panel(
                        insights_text,
                        title="[bold blue]💡 Insights[/bold blue]",
                        border_style="blue",
                        padding=(1, 2)
                    )
                )
                temp_console.print("\n")

        # Get the formatted string
        result = string_io.getvalue()
        logger.debug("Formatted recommendations")
        return result

    def export_recommendations_json(self, recs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Export recommendations in a JSON-friendly format.

        Args:
            recs: Dictionary from generate_recommendations()

        Returns:
            JSON-serializable dictionary with recommendations

        Example:
            >>> engine = RecommendationEngine()
            >>> recs = engine.generate_recommendations(comparison)
            >>> json_data = engine.export_recommendations_json(recs)
            >>> import json
            >>> print(json.dumps(json_data, indent=2))
        """
        # Create simplified version for JSON export
        export = {
            "tier_summary": {
                tier: len(models) for tier, models in recs["tiers"].items()
            },
            "recommendations": recs["recommendations"],
        }

        # Add model details (simplified)
        if recs["best_overall"]:
            export["best_overall"] = {
                "name": recs["best_overall"]["name"],
                "score": recs["best_overall"]["overall_score"],
                "cost": recs["best_overall"]["cost"],
                "violations": len(recs["best_overall"]["violations"])
            }

        if recs["best_value"]:
            export["best_value"] = {
                "name": recs["best_value"]["name"],
                "score": recs["best_value"]["overall_score"],
                "cost": recs["best_value"]["cost"],
                "value_rating": recs["best_value"]["value_rating"]
            }

        if recs["budget_option"]:
            export["budget_option"] = {
                "name": recs["budget_option"]["name"],
                "score": recs["budget_option"]["overall_score"],
                "cost": recs["budget_option"]["cost"]
            }

        logger.debug("Exported recommendations to JSON format")
        return export

    def __str__(self) -> str:
        """String representation of RecommendationEngine."""
        return "RecommendationEngine()"

    def __repr__(self) -> str:
        """Repr of RecommendationEngine."""
        return self.__str__()
