"""
Dynamic rubric generator for use-case-aware evaluation.

This module uses an LLM to analyze use cases and generate
evaluation rubrics dynamically, ensuring the Judge properly
weights what matters for each specific use case.
"""

import json
import logging
import os
from typing import Any, Dict, Optional

from taskbench.api.client import OpenRouterClient
from taskbench.usecase import UseCase

logger = logging.getLogger(__name__)

# Default LLM for rubric generation
RUBRIC_GEN_MODEL = os.getenv("TASKBENCH_RUBRIC_MODEL", "anthropic/claude-sonnet-4.5")

# Prompt template for rubric generation
RUBRIC_GENERATION_PROMPT = """You are an expert evaluation designer. Analyze the following use case and generate a detailed evaluation rubric that will be used by an LLM Judge to score model outputs.

# USE CASE TO ANALYZE

**Name**: {name}

**Goal**:
{goal}

**LLM Notes** (instructions for evaluation):
{llm_notes}

**Output Format**:
- Type: {output_format_type}
- Required Fields: {output_fields}

**Constraints**:
{constraints}

**Additional Notes**:
{notes}

# YOUR TASK

Generate an evaluation rubric with:

1. **Critical Requirements**: What are the MOST important requirements that outputs MUST meet? These should be derived from the goal and constraints. For each:
   - What is the requirement?
   - Why is it critical for this use case?
   - What penalty should apply per violation? (5-15 points)

2. **Compliance Checks**: Specific, measurable checks that can be performed on outputs:
   - What exactly should be checked?
   - How severe is a violation? (CRITICAL/HIGH/MEDIUM)
   - What penalty per occurrence?

3. **Scoring Weights**: Based on this use case, what should the weight distribution be?
   - Accuracy (content correctness): X%
   - Format (structural compliance): Y%
   - Constraint Compliance (meeting requirements): Z%
   - Must sum to 100%

4. **Scoring Instructions**: Write specific instructions for the Judge LLM explaining:
   - How to calculate each score
   - What constitutes a violation
   - How violations should impact the overall score
   - Clear examples of acceptable vs unacceptable outputs

# RESPONSE FORMAT

You MUST respond with ONLY valid JSON in this exact format:
```json
{{
  "critical_requirements": [
    {{
      "name": "requirement_name",
      "description": "What must be met",
      "rationale": "Why this matters for the use case",
      "penalty_per_violation": 8
    }}
  ],
  "compliance_checks": [
    {{
      "check": "check_name",
      "condition": "When this check fails",
      "severity": "CRITICAL|HIGH|MEDIUM",
      "penalty": 8
    }}
  ],
  "weights": {{
    "accuracy": 30,
    "format": 20,
    "compliance": 50
  }},
  "scoring_instructions": "Detailed instructions for the Judge..."
}}
```

Provide ONLY the JSON, no other text.
"""


class RubricGenerator:
    """
    Generate dynamic evaluation rubrics by analyzing use cases with LLM.

    Uses an LLM to analyze a use case's goals, constraints, and requirements
    to generate weighted scoring criteria that properly evaluate what matters
    for that specific use case.
    """

    def __init__(self, api_client: Optional[OpenRouterClient] = None):
        """
        Initialize the rubric generator.

        Args:
            api_client: OpenRouterClient for LLM-based analysis.
        """
        self.api_client = api_client
        self._cache: Dict[str, Dict[str, Any]] = {}

    def _format_usecase_for_prompt(self, usecase: UseCase) -> Dict[str, str]:
        """Format use case fields for the prompt template."""
        # Format output fields
        output_fields = "None specified"
        output_format_type = "Not specified"
        if usecase.output_format:
            output_format_type = usecase.output_format.format_type
            if usecase.output_format.fields:
                output_fields = ", ".join(
                    f"{f.get('name')} ({f.get('type')})"
                    for f in usecase.output_format.fields
                )

        # Format constraints
        constraints_parts = []
        if usecase.chunk_min_minutes:
            constraints_parts.append(f"- Minimum chunk duration: {usecase.chunk_min_minutes} minutes")
        if usecase.chunk_max_minutes:
            constraints_parts.append(f"- Maximum chunk duration: {usecase.chunk_max_minutes} minutes")
        if usecase.coverage_required:
            constraints_parts.append("- Full coverage required (no gaps)")
        if usecase.allow_shorter_if_justified:
            constraints_parts.append("- Shorter segments allowed if justified by content")
        if usecase.cost_priority:
            constraints_parts.append(f"- Cost priority: {usecase.cost_priority}")
        if usecase.quality_priority:
            constraints_parts.append(f"- Quality priority: {usecase.quality_priority}")

        constraints = "\n".join(constraints_parts) if constraints_parts else "None specified"

        return {
            "name": usecase.name,
            "goal": usecase.goal or "Not specified",
            "llm_notes": usecase.llm_notes or "None",
            "output_format_type": output_format_type,
            "output_fields": output_fields,
            "constraints": constraints,
            "notes": usecase.notes or "None"
        }

    async def generate_rubric_async(self, usecase: UseCase) -> Dict[str, Any]:
        """
        Generate a weighted evaluation rubric from use case using LLM.

        Args:
            usecase: UseCase with goals, constraints, and priorities

        Returns:
            Dict containing:
            - critical_requirements: List of requirements with weights
            - compliance_checks: Specific checks with per-violation penalties
            - weights: Score weighting (accuracy, format, compliance)
            - scoring_instructions: Dynamic scoring guidance for Judge
        """
        # Check cache
        cache_key = usecase.name
        if cache_key in self._cache:
            logger.info(f"Using cached rubric for {cache_key}")
            return self._cache[cache_key]

        if not self.api_client:
            logger.warning("No API client provided, using fallback rubric")
            return self._generate_fallback_rubric(usecase)

        # Format use case for prompt
        params = self._format_usecase_for_prompt(usecase)
        prompt = RUBRIC_GENERATION_PROMPT.format(**params)

        # Call LLM to generate rubric
        try:
            logger.info(f"Generating rubric for use case: {usecase.name}")
            response = await self.api_client.complete_with_json(
                model=RUBRIC_GEN_MODEL,
                prompt=prompt,
                max_tokens=4000,  # Increased to avoid truncation
                temperature=0.3
            )

            rubric = json.loads(response.content)

            # Validate required fields
            required_keys = ["critical_requirements", "compliance_checks", "weights", "scoring_instructions"]
            for key in required_keys:
                if key not in rubric:
                    raise ValueError(f"Missing required key: {key}")

            # Cache the result
            self._cache[cache_key] = rubric
            logger.info(f"Generated rubric with {len(rubric['critical_requirements'])} critical requirements")

            return rubric

        except Exception as e:
            logger.error(f"Failed to generate rubric via LLM: {e}")
            return self._generate_fallback_rubric(usecase)

    def generate_rubric(self, usecase: UseCase) -> Dict[str, Any]:
        """
        Synchronous wrapper for generate_rubric_async.
        Uses fallback if no async context available.
        """
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            # If we're in an async context, we can't run nested event loops
            logger.warning("Called sync generate_rubric from async context, using fallback")
            return self._generate_fallback_rubric(usecase)
        except RuntimeError:
            # No running loop, safe to run async
            if self.api_client:
                return asyncio.run(self.generate_rubric_async(usecase))
            return self._generate_fallback_rubric(usecase)

    def _generate_fallback_rubric(self, usecase: UseCase) -> Dict[str, Any]:
        """
        Generate a basic rubric without LLM when API is unavailable.

        This is a rule-based fallback that extracts what it can from
        the use case structure.
        """
        critical_requirements = []
        compliance_checks = []

        # Extract duration requirements if present
        if usecase.chunk_min_minutes or usecase.chunk_max_minutes:
            min_dur = usecase.chunk_min_minutes or 0
            max_dur = usecase.chunk_max_minutes or 60

            critical_requirements.append({
                "name": "duration_compliance",
                "description": f"Each segment must be {min_dur}-{max_dur} minutes",
                "rationale": "Derived from use case chunk constraints",
                "penalty_per_violation": 8
            })

            if usecase.chunk_min_minutes:
                compliance_checks.append({
                    "check": "segment_under_minimum",
                    "condition": f"Duration < {usecase.chunk_min_minutes} minutes without justification",
                    "severity": "HIGH",
                    "penalty": 5
                })

            if usecase.chunk_max_minutes:
                compliance_checks.append({
                    "check": "segment_over_maximum",
                    "condition": f"Duration > {usecase.chunk_max_minutes} minutes",
                    "severity": "CRITICAL",
                    "penalty": 8
                })

        if usecase.coverage_required:
            critical_requirements.append({
                "name": "full_coverage",
                "description": "No time gaps between segments",
                "rationale": "Coverage is required by use case",
                "penalty_per_violation": 5
            })
            compliance_checks.append({
                "check": "coverage_gap",
                "condition": "Gap between consecutive segment end/start times",
                "severity": "HIGH",
                "penalty": 5
            })

        # Always check for overlaps
        compliance_checks.append({
            "check": "timestamp_overlap",
            "condition": "Segment end_time > next segment start_time",
            "severity": "CRITICAL",
            "penalty": 10
        })

        # Determine weights based on priorities
        weights = {"accuracy": 30, "format": 20, "compliance": 50}
        if usecase.quality_priority == "high":
            weights = {"accuracy": 35, "format": 15, "compliance": 50}

        # Generate basic scoring instructions
        scoring_instructions = self._generate_fallback_scoring_instructions(
            usecase, critical_requirements, compliance_checks
        )

        return {
            "critical_requirements": critical_requirements,
            "compliance_checks": compliance_checks,
            "weights": weights,
            "scoring_instructions": scoring_instructions
        }

    def _generate_fallback_scoring_instructions(
        self,
        usecase: UseCase,
        critical_requirements: list,
        compliance_checks: list
    ) -> str:
        """Generate scoring instructions without LLM."""
        instructions = []

        instructions.append("## Scoring Instructions for This Use Case")
        instructions.append("")

        if usecase.goal:
            instructions.append(f"**Primary Goal**: {usecase.goal.strip()}")
            instructions.append("")

        if critical_requirements:
            instructions.append("### Critical Requirements")
            for req in critical_requirements:
                instructions.append(f"- **{req['name']}**: {req['description']} (-{req['penalty_per_violation']} pts per violation)")
            instructions.append("")

        if usecase.chunk_min_minutes and usecase.chunk_max_minutes:
            instructions.append("### Duration Compliance Calculation")
            instructions.append(f"Required range: {usecase.chunk_min_minutes}-{usecase.chunk_max_minutes} minutes per segment")
            instructions.append("- Count segments exceeding maximum: each is a violation")
            instructions.append("- Count segments under minimum (without justification): each is a violation")
            instructions.append("- compliance_score = 100 - (sum of violation penalties)")
            instructions.append("")

        instructions.append("### Overall Score Calculation")
        instructions.append("1. Calculate accuracy_score based on content correctness")
        instructions.append("2. Calculate format_score based on structural compliance")
        instructions.append("3. Calculate compliance_score based on ACTUAL violation count")
        instructions.append("4. overall_score = accuracy*0.35 + format*0.15 + compliance*0.50")
        instructions.append("")
        instructions.append("**IMPORTANT**: Models with many compliance violations CANNOT score above 60 overall.")

        return "\n".join(instructions)

    async def generate_judge_prompt_section_async(self, usecase: UseCase) -> str:
        """
        Generate the complete judge prompt section for use-case-aware evaluation.

        Args:
            usecase: UseCase to generate prompt for

        Returns:
            String containing the dynamic evaluation section for Judge prompt
        """
        rubric = await self.generate_rubric_async(usecase)
        return self._format_rubric_as_prompt(rubric, usecase)

    def generate_judge_prompt_section(self, usecase: UseCase) -> str:
        """
        Synchronous version of generate_judge_prompt_section_async.
        """
        rubric = self.generate_rubric(usecase)
        return self._format_rubric_as_prompt(rubric, usecase)

    def _format_rubric_as_prompt(self, rubric: Dict[str, Any], usecase: UseCase) -> str:
        """Format a rubric dict as a prompt section."""
        sections = []

        # Add the LLM-generated scoring instructions
        sections.append(rubric["scoring_instructions"])
        sections.append("")

        # Add critical requirements
        if rubric["critical_requirements"]:
            sections.append("### Critical Requirements (Per-Violation Penalties)")
            sections.append("")
            for req in rubric["critical_requirements"]:
                sections.append(f"- **{req['name']}**: {req['description']}")
                sections.append(f"  Penalty: -{req['penalty_per_violation']} points per violation")
                if req.get('rationale'):
                    sections.append(f"  Rationale: {req['rationale']}")
            sections.append("")

        # Add compliance checks
        if rubric["compliance_checks"]:
            sections.append("### Compliance Checks to Perform")
            sections.append("")
            for check in rubric["compliance_checks"]:
                sections.append(f"- **{check['check']}** ({check['severity']}): {check['condition']}")
                sections.append(f"  Penalty: -{check['penalty']} points per occurrence")
            sections.append("")

        # Add scoring weights
        weights = rubric["weights"]
        sections.append("### Score Weighting")
        sections.append("")
        sections.append(f"- Accuracy: {weights.get('accuracy', 30)}%")
        sections.append(f"- Format: {weights.get('format', 20)}%")
        sections.append(f"- Compliance: {weights.get('compliance', 50)}%")
        sections.append("")
        sections.append("Calculate overall_score as weighted average AFTER applying compliance penalties.")
        sections.append("A model with many compliance violations CANNOT score above 60 overall.")

        return "\n".join(sections)
