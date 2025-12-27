"""
LLM Assistant for intelligent task analysis and prompt generation.

This module uses an LLM (configurable via GENERAL_TASK_LLM) to:
1. Analyze use cases and generate clarifying questions
2. Recommend appropriate models based on task requirements
3. Generate optimized prompts for evaluation
4. Determine optimal output format specifications
5. Generate context-aware judge prompts
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple

from taskbench.api.client import OpenRouterClient
from taskbench.core.models import TaskDefinition
from taskbench.usecase import UseCase, OutputFormatSpec
from taskbench.evaluation.cost import CostTracker

logger = logging.getLogger(__name__)


class LLMAssistant:
    """
    Intelligent assistant for LLM-powered task analysis and prompt generation.

    Uses GENERAL_TASK_LLM (default: anthropic/claude-sonnet-4.5) for all
    meta-level operations like analyzing tasks, generating prompts, and
    recommending models.
    """

    def __init__(self, api_client: OpenRouterClient):
        self.api_client = api_client
        self.assistant_model = os.getenv("GENERAL_TASK_LLM", "anthropic/claude-sonnet-4.5")
        self.cost_tracker = CostTracker()
        logger.info(f"LLMAssistant initialized with model: {self.assistant_model}")

    async def analyze_usecase(
        self,
        usecase: UseCase,
        input_preview: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze a use case and return insights, potential issues, and suggestions.

        Args:
            usecase: The UseCase to analyze
            input_preview: Optional preview of input data (first N chars)

        Returns:
            Dict with analysis results including:
            - clarifying_questions: Questions to ask the user
            - suggested_output_format: Recommended output structure
            - recommended_models: List of suitable models
            - prompt_hints: Hints to include in the evaluation prompt
        """
        prompt = f"""You are an expert at designing LLM evaluation tasks. Analyze this use case and provide recommendations.

## Use Case
Name: {usecase.name}
Goal: {usecase.goal}
{f"LLM Notes: {usecase.llm_notes}" if usecase.llm_notes else ""}
Cost Priority: {usecase.cost_priority}
Quality Priority: {usecase.quality_priority}
{f"Notes: {chr(10).join('- ' + n for n in usecase.notes)}" if usecase.notes else ""}

{f"## Input Preview (first 2000 chars){chr(10)}{input_preview[:2000]}" if input_preview else ""}

## Your Task
Analyze this use case and provide:

1. **clarifying_questions**: 2-3 questions to clarify requirements (if any ambiguity exists)
2. **suggested_output_format**: Recommend the output structure including:
   - format_type: "json", "csv", or "markdown"
   - fields: List of fields with name, type, and description
   - example: A brief example of expected output
3. **recommended_model_types**: Categories of models suitable for this task (e.g., "large context", "analytical", "cost-effective")
4. **prompt_hints**: Key hints to include in the evaluation prompt to guide LLMs
5. **potential_issues**: Any concerns or challenges with this use case

Respond in JSON format only."""

        try:
            response = await self.api_client.complete(
                model=self.assistant_model,
                prompt=prompt,
                max_tokens=2000,
                temperature=0.3
            )

            # Parse response
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]

            return json.loads(content.strip())

        except Exception as e:
            logger.error(f"Failed to analyze use case: {e}")
            return {
                "clarifying_questions": [],
                "suggested_output_format": None,
                "recommended_model_types": ["general-purpose"],
                "prompt_hints": [],
                "potential_issues": [str(e)]
            }

    async def recommend_models(
        self,
        usecase: UseCase,
        available_models: List[Dict[str, Any]],
        max_models: int = 5
    ) -> List[str]:
        """
        Use LLM to recommend the best models for a use case.

        Args:
            usecase: The UseCase to recommend models for
            available_models: List of available model configs
            max_models: Maximum number of models to recommend

        Returns:
            List of recommended model IDs in priority order
        """
        models_info = "\n".join([
            f"- {m['id']}: {m.get('display_name', m['id'])} | "
            f"Context: {m.get('context_window', 'unknown')} | "
            f"Input: ${m.get('input_price_per_1m', 0):.2f}/1M | "
            f"Output: ${m.get('output_price_per_1m', 0):.2f}/1M"
            for m in available_models
        ])

        prompt = f"""You are an expert at selecting LLMs for specific tasks.

## Use Case
Name: {usecase.name}
Goal: {usecase.goal}
Cost Priority: {usecase.cost_priority} (high = prefer cheaper models)
Quality Priority: {usecase.quality_priority} (high = prefer best quality)
{f"Notes: {usecase.llm_notes}" if usecase.llm_notes else ""}

## Available Models
{models_info}

## Task
Select the {max_models} best models for this use case, considering:
1. Task requirements (analytical ability, context length needs)
2. Cost vs quality trade-off based on priorities
3. Known model strengths (Claude for analysis, GPT for structured output, Gemini for long context)
4. Avoid models known to have issues with similar tasks

Respond with a JSON array of model IDs in priority order (best first):
["model-id-1", "model-id-2", ...]

Respond with ONLY the JSON array, no explanation."""

        try:
            response = await self.api_client.complete(
                model=self.assistant_model,
                prompt=prompt,
                max_tokens=500,
                temperature=0.2
            )

            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            if content.endswith("```"):
                content = content[:-3]

            recommended = json.loads(content.strip())

            # Validate all returned models exist
            valid_ids = {m["id"] for m in available_models}
            return [m for m in recommended if m in valid_ids][:max_models]

        except Exception as e:
            logger.error(f"Failed to recommend models: {e}")
            # Fallback to default selection
            return [m["id"] for m in available_models[:max_models]]

    async def generate_evaluation_prompt(
        self,
        task: TaskDefinition,
        usecase: UseCase,
        input_preview: Optional[str] = None
    ) -> str:
        """
        Generate an optimized evaluation prompt based on task and use case.

        This creates a prompt that will be used for ALL models in the evaluation,
        ensuring fair comparison.

        Args:
            task: The TaskDefinition
            usecase: The UseCase with goals and constraints
            input_preview: Preview of input to understand format

        Returns:
            Optimized prompt template (input data will be appended)
        """
        output_format_desc = ""
        if usecase.output_format:
            fields_desc = "\n".join([
                f"  - {f.get('name')}: {f.get('type')} - {f.get('description', '')}"
                for f in usecase.output_format.fields
            ])
            output_format_desc = f"""
Output Format: {usecase.output_format.format_type.upper()}
Required Fields:
{fields_desc}
{f"Example: {usecase.output_format.example}" if usecase.output_format.example else ""}
"""

        prompt = f"""You are an expert prompt engineer. Create an optimized evaluation prompt for this task.

## Task Definition
Name: {task.name}
Description: {task.description}
Input Type: {task.input_type}
Expected Output: {task.output_format}

## Use Case Context
Goal: {usecase.goal}
{f"LLM Notes: {usecase.llm_notes}" if usecase.llm_notes else ""}
{output_format_desc}

## Constraints
{json.dumps(task.constraints, indent=2) if task.constraints else "None specified"}

## Evaluation Criteria
{chr(10).join('- ' + c for c in task.evaluation_criteria)}

## Additional Notes
{chr(10).join('- ' + n for n in usecase.notes) if usecase.notes else "None"}

{f"## Input Preview{chr(10)}{input_preview[:1500]}" if input_preview else ""}

## Your Task
Create a comprehensive prompt that:
1. Clearly explains the task to any LLM
2. Specifies exact output format requirements
3. Includes anti-hallucination instructions
4. Addresses the evaluation criteria
5. Is fair and unbiased for comparing different LLMs

The prompt should end with a placeholder for input data: {{INPUT_DATA}}

Write ONLY the prompt text, no meta-commentary."""

        try:
            response = await self.api_client.complete(
                model=self.assistant_model,
                prompt=prompt,
                max_tokens=2000,
                temperature=0.4
            )

            return response.content.strip()

        except Exception as e:
            logger.error(f"Failed to generate prompt: {e}")
            # Return a basic fallback prompt
            return self._build_fallback_prompt(task, usecase)

    def _build_fallback_prompt(self, task: TaskDefinition, usecase: UseCase) -> str:
        """Build a basic prompt as fallback."""
        return f"""# Task: {task.name}

{task.description}

## Goal
{usecase.goal}

## Output Format
Provide your output in {task.output_format.upper()} format.

## Constraints
{chr(10).join(f'- {k}: {v}' for k, v in (task.constraints or {}).items())}

## Important
- Only use information from the provided input
- Do not fabricate or hallucinate any content
- Follow the output format exactly

## Input Data
{{INPUT_DATA}}

Process the input and provide your response."""

    async def generate_output_format(
        self,
        usecase: UseCase,
        input_preview: Optional[str] = None
    ) -> OutputFormatSpec:
        """
        Use LLM to determine the optimal output format for a use case.

        Args:
            usecase: The UseCase to analyze
            input_preview: Preview of input data

        Returns:
            OutputFormatSpec with recommended format
        """
        prompt = f"""Analyze this use case and determine the optimal output format.

## Use Case
Name: {usecase.name}
Goal: {usecase.goal}
{f"Notes: {usecase.llm_notes}" if usecase.llm_notes else ""}

{f"## Input Preview{chr(10)}{input_preview[:2000]}" if input_preview else ""}

## Task
Determine the best output format for evaluating LLMs on this task.
Consider what fields would be needed for:
1. Validating correctness
2. Comparing across models
3. Easy analysis by the Judge

Respond in this exact JSON format:
{{
  "format_type": "json|csv|markdown",
  "fields": [
    {{"name": "field_name", "type": "string|number|timestamp", "description": "what this field contains"}}
  ],
  "example": "brief example of one output item",
  "notes": "any important notes about the format"
}}

Respond with ONLY the JSON, no explanation."""

        try:
            response = await self.api_client.complete(
                model=self.assistant_model,
                prompt=prompt,
                max_tokens=1000,
                temperature=0.3
            )

            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            if content.endswith("```"):
                content = content[:-3]

            data = json.loads(content.strip())
            return OutputFormatSpec(**data)

        except Exception as e:
            logger.error(f"Failed to generate output format: {e}")
            # Return a default format
            return OutputFormatSpec(
                format_type="json",
                fields=[
                    {"name": "content", "type": "string", "description": "Main output content"}
                ],
                notes="Default format - customize as needed"
            )

    async def generate_judge_prompt(
        self,
        task: TaskDefinition,
        usecase: UseCase,
        evaluation_prompt: str,
        input_preview: Optional[str] = None
    ) -> str:
        """
        Generate a context-aware judge prompt that understands the full evaluation context.

        The judge needs to know:
        1. The original use case goals
        2. The prompt given to models
        3. Expected output format
        4. How to score fairly

        Args:
            task: The TaskDefinition
            usecase: The UseCase
            evaluation_prompt: The prompt used for evaluation
            input_preview: Preview of input data

        Returns:
            Judge-specific prompt for scoring outputs
        """
        prompt = f"""You are creating a scoring rubric for an LLM-as-Judge evaluation.

## Original Use Case
Name: {usecase.name}
Goal: {usecase.goal}
{f"LLM Notes: {usecase.llm_notes}" if usecase.llm_notes else ""}

## Evaluation Prompt Given to Models
{evaluation_prompt[:3000]}

## Expected Output Format
{json.dumps(usecase.output_format.model_dump(), indent=2) if usecase.output_format else task.output_format}

## Evaluation Criteria
{chr(10).join('- ' + c for c in task.evaluation_criteria)}

{f"## Input Preview{chr(10)}{input_preview[:1500]}" if input_preview else ""}

## Your Task
Create a judge prompt that will:
1. Evaluate model outputs against the original use case goals
2. Check for format compliance
3. Detect hallucinations or fabricated content
4. Score on multiple dimensions (accuracy, format, completeness, compliance)
5. Provide specific reasoning for scores

The judge prompt should expect:
- The original input data
- The model's output
- And produce scores from 0-100 for each dimension

Write the complete judge prompt. Include placeholders:
- {{INPUT_DATA}} for the original input
- {{MODEL_OUTPUT}} for the model's response

Respond with ONLY the judge prompt, no meta-commentary."""

        try:
            response = await self.api_client.complete(
                model=self.assistant_model,
                prompt=prompt,
                max_tokens=2500,
                temperature=0.3
            )

            return response.content.strip()

        except Exception as e:
            logger.error(f"Failed to generate judge prompt: {e}")
            return self._build_fallback_judge_prompt(task, usecase)

    def _build_fallback_judge_prompt(self, task: TaskDefinition, usecase: UseCase) -> str:
        """Build a basic judge prompt as fallback."""
        return f"""You are an expert evaluator assessing an LLM's output for this task:

## Task: {task.name}
Goal: {usecase.goal}

## Original Input
{{INPUT_DATA}}

## Model Output to Evaluate
{{MODEL_OUTPUT}}

## Scoring Criteria
Evaluate on these dimensions (0-100 each):
1. **Accuracy**: Does the output correctly address the task?
2. **Format**: Does it follow the required format?
3. **Completeness**: Is the output comprehensive?
4. **Compliance**: Are all constraints satisfied?

## Response Format
Respond with JSON:
{{
  "accuracy_score": <0-100>,
  "format_score": <0-100>,
  "compliance_score": <0-100>,
  "overall_score": <0-100>,
  "violations": ["list of specific issues"],
  "reasoning": "detailed explanation"
}}"""

    async def summarize_results(
        self,
        usecase: UseCase,
        results_data: List[Dict[str, Any]],
        scores_data: List[Dict[str, Any]]
    ) -> str:
        """
        Generate a natural language summary of evaluation results.

        Args:
            usecase: The UseCase evaluated
            results_data: List of model results
            scores_data: List of judge scores

        Returns:
            Human-readable summary with recommendations
        """
        results_summary = []
        for r, s in zip(results_data, scores_data):
            if s:
                results_summary.append({
                    "model": r.get("model_name"),
                    "score": s.get("overall_score"),
                    "cost": r.get("cost_usd"),
                    "violations": s.get("violations", [])
                })

        prompt = f"""Summarize these LLM evaluation results for the user.

## Use Case
Name: {usecase.name}
Goal: {usecase.goal}

## Results
{json.dumps(results_summary, indent=2)}

## Your Task
Write a concise summary (2-3 paragraphs) that:
1. States which model performed best and why
2. Highlights any significant issues found
3. Provides a clear recommendation
4. Notes cost-effectiveness if relevant

Be direct and actionable. Write in plain language."""

        try:
            response = await self.api_client.complete(
                model=self.assistant_model,
                prompt=prompt,
                max_tokens=1000,
                temperature=0.4
            )

            return response.content.strip()

        except Exception as e:
            logger.error(f"Failed to summarize results: {e}")
            # Basic fallback summary
            if results_summary:
                best = max(results_summary, key=lambda x: x.get("score", 0))
                return f"Best performing model: {best['model']} with score {best['score']}/100."
            return "No valid results to summarize."
