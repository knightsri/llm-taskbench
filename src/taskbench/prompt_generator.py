"""
Prompt Generator for Use Cases.

This module uses LLM to analyze use cases and generate all prompts needed
for the evaluation framework:
- Model execution prompt (task prompt)
- Judge evaluation prompt
- Rubric/scoring criteria
"""

import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from taskbench.api.client import OpenRouterClient
from taskbench.usecase_parser import ParsedUseCase, load_usecase_from_folder

logger = logging.getLogger(__name__)

# LLM for prompt generation
PROMPT_GEN_MODEL = os.getenv("TASKBENCH_PROMPT_GEN_MODEL", "anthropic/claude-sonnet-4.5")


# Prompt to analyze use case and generate all needed prompts
PROMPT_GENERATION_TEMPLATE = """You are an expert at designing LLM evaluation tasks. Analyze this use case and generate all the prompts needed for evaluation.

# USE CASE INFORMATION

**Name**: {name}
**Goal**: {goal}

**LLM Evaluation Notes**:
{llm_notes}

**Expected Output Format**: {output_format}

**Expected Output Schema**:
{expected_output_schema}

**Quality Criteria** (if specified):
{quality_criteria}

**Edge Cases to Watch**:
{edge_cases}

# INPUT DATA SAMPLE

File: {data_file_name}
Preview (first 1000 chars):
```
{data_preview}
```

# EXPECTED OUTPUT (GROUND TRUTH)

File: {gt_file_name}
Content:
```{gt_format}
{gt_content}
```

# YOUR TASK

Analyze the input data and ground truth to understand:
1. What transformation is the LLM expected to perform?
2. What are the key quality criteria based on the ground truth?
3. How should outputs be compared to ground truth?

Generate THREE prompts:

1. **task_prompt**: The prompt to send to LLMs being evaluated. This should clearly instruct the model what to do with the input data to produce output matching the ground truth format.

2. **judge_prompt**: Instructions for the Judge LLM to evaluate outputs. Include:
   - How to compare output against ground truth
   - What constitutes a violation
   - Scoring criteria derived from the ground truth structure
   - Per-violation penalties

3. **rubric**: Structured evaluation criteria including:
   - Critical requirements (derived from ground truth analysis)
   - Compliance checks with severity and penalties
   - Scoring weights (accuracy, format, compliance)

# RESPONSE FORMAT

Respond with ONLY valid JSON:
```json
{{
  "analysis": {{
    "transformation_type": "What the LLM needs to do (e.g., extraction, classification, generation)",
    "key_fields": ["List of key output fields derived from ground truth"],
    "quality_indicators": ["What makes a good output based on ground truth"],
    "comparison_strategy": "How to compare output vs ground truth"
  }},
  "task_prompt": "The complete prompt to send to LLMs...",
  "judge_prompt": "Complete instructions for the Judge LLM...",
  "rubric": {{
    "critical_requirements": [
      {{
        "name": "requirement_name",
        "description": "What must be met",
        "derived_from": "How this was derived from ground truth",
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
      "accuracy": 40,
      "format": 20,
      "compliance": 40
    }},
    "scoring_instructions": "How to calculate scores..."
  }}
}}
```

Provide ONLY the JSON, no other text.
"""


class PromptGenerator:
    """
    Generates all prompts needed for use case evaluation by analyzing
    the use case, input data, and ground truth with an LLM.
    """

    def __init__(self, api_client: Optional[OpenRouterClient] = None):
        """
        Initialize the prompt generator.

        Args:
            api_client: OpenRouterClient for LLM calls
        """
        self.api_client = api_client

    async def generate_prompts(self, parsed_usecase: ParsedUseCase) -> Dict[str, Any]:
        """
        Generate all prompts for a use case.

        Args:
            parsed_usecase: Parsed use case with data and ground truth

        Returns:
            Dict with analysis, task_prompt, judge_prompt, rubric
        """
        if not parsed_usecase.matched_pairs:
            raise ValueError("No matched data/ground-truth pairs found")

        # Use the first matched pair for analysis
        pair = parsed_usecase.matched_pairs[0]

        # Prepare ground truth content for prompt
        gt_content = pair.ground_truth_file.content
        if isinstance(gt_content, dict) or isinstance(gt_content, list):
            gt_content_str = json.dumps(gt_content, indent=2)[:3000]
        else:
            gt_content_str = str(gt_content)[:3000]

        # Format edge cases
        edge_cases_str = "\n".join(f"- {ec}" for ec in parsed_usecase.edge_cases) or "None specified"

        # Build the prompt
        prompt = PROMPT_GENERATION_TEMPLATE.format(
            name=parsed_usecase.name,
            goal=parsed_usecase.goal,
            llm_notes=parsed_usecase.llm_notes or "None",
            output_format=parsed_usecase.output_format,
            expected_output_schema=parsed_usecase.expected_output_schema or "See ground truth",
            quality_criteria=parsed_usecase.quality_criteria or "Not explicitly specified - derive from ground truth",
            edge_cases=edge_cases_str,
            data_file_name=pair.data_file.name + pair.data_file.extension,
            data_preview=pair.data_file.content_preview,
            gt_file_name=pair.ground_truth_file.name + pair.ground_truth_file.extension,
            gt_format=pair.ground_truth_file.format_type,
            gt_content=gt_content_str
        )

        if not self.api_client:
            logger.warning("No API client provided, cannot generate prompts")
            return self._generate_fallback_prompts(parsed_usecase)

        try:
            logger.info(f"Generating prompts for use case: {parsed_usecase.name}")

            response = await self.api_client.complete_with_json(
                model=PROMPT_GEN_MODEL,
                prompt=prompt,
                max_tokens=6000,
                temperature=0.3
            )

            result = json.loads(response.content)

            # Validate required fields
            required_keys = ["analysis", "task_prompt", "judge_prompt", "rubric"]
            for key in required_keys:
                if key not in result:
                    raise ValueError(f"Missing required key: {key}")

            logger.info(f"Generated prompts successfully for {parsed_usecase.name}")
            return result

        except Exception as e:
            logger.error(f"Failed to generate prompts: {e}")
            return self._generate_fallback_prompts(parsed_usecase)

    def _generate_fallback_prompts(self, parsed_usecase: ParsedUseCase) -> Dict[str, Any]:
        """Generate basic prompts without LLM when API is unavailable."""
        return {
            "analysis": {
                "transformation_type": "extraction",
                "key_fields": [],
                "quality_indicators": ["Matches ground truth format"],
                "comparison_strategy": "direct_comparison"
            },
            "task_prompt": f"""Perform the following task:

{parsed_usecase.goal}

Output your response in {parsed_usecase.output_format} format.

{parsed_usecase.expected_output_schema}
""",
            "judge_prompt": f"""Evaluate the LLM's output against the expected ground truth.

Task: {parsed_usecase.goal}

Compare the output to the expected result and score based on:
1. Accuracy - Does the output match the ground truth content?
2. Format - Is the output in the correct {parsed_usecase.output_format} format?
3. Compliance - Does it meet all specified requirements?

Provide scores from 0-100 for each category.
""",
            "rubric": {
                "critical_requirements": [],
                "compliance_checks": [],
                "weights": {"accuracy": 40, "format": 30, "compliance": 30},
                "scoring_instructions": "Compare output to ground truth and score based on match quality."
            }
        }

    async def generate_and_save_prompts(self, usecase_folder: str) -> Dict[str, Any]:
        """
        Generate prompts for a use case and save them to the folder.

        Args:
            usecase_folder: Path to the use case folder

        Returns:
            Generated prompts dict
        """
        # Load and parse the use case
        parsed = load_usecase_from_folder(usecase_folder)

        # Generate prompts
        prompts = await self.generate_prompts(parsed)

        # Save to use case folder
        prompts_file = Path(usecase_folder) / "generated-prompts.json"
        prompts_file.write_text(json.dumps(prompts, indent=2), encoding="utf-8")
        logger.info(f"Saved generated prompts to {prompts_file}")

        # Also save individual prompt files for easy review
        prompts_folder = Path(usecase_folder) / "prompts"
        prompts_folder.mkdir(exist_ok=True)

        # Task prompt
        (prompts_folder / "task-prompt.txt").write_text(
            prompts["task_prompt"],
            encoding="utf-8"
        )

        # Judge prompt
        (prompts_folder / "judge-prompt.txt").write_text(
            prompts["judge_prompt"],
            encoding="utf-8"
        )

        # Rubric as JSON
        (prompts_folder / "rubric.json").write_text(
            json.dumps(prompts["rubric"], indent=2),
            encoding="utf-8"
        )

        # Analysis
        (prompts_folder / "analysis.json").write_text(
            json.dumps(prompts["analysis"], indent=2),
            encoding="utf-8"
        )

        logger.info(f"Saved individual prompt files to {prompts_folder}")

        return prompts

    def load_saved_prompts(self, usecase_folder: str) -> Optional[Dict[str, Any]]:
        """
        Load previously generated prompts from a use case folder.

        Args:
            usecase_folder: Path to the use case folder

        Returns:
            Prompts dict if found, None otherwise
        """
        prompts_file = Path(usecase_folder) / "generated-prompts.json"
        if prompts_file.exists():
            try:
                return json.loads(prompts_file.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"Could not load saved prompts: {e}")
        return None


async def generate_prompts_for_usecase(
    usecase_folder: str,
    api_key: Optional[str] = None,
    force_regenerate: bool = False
) -> Dict[str, Any]:
    """
    Generate or load prompts for a use case.

    Args:
        usecase_folder: Path to the use case folder
        api_key: OpenRouter API key (uses env var if not provided)
        force_regenerate: If True, regenerate even if prompts exist

    Returns:
        Dict with all generated prompts
    """
    api_key = api_key or os.getenv("OPENROUTER_API_KEY")

    generator = PromptGenerator()

    # Check for existing prompts
    if not force_regenerate:
        existing = generator.load_saved_prompts(usecase_folder)
        if existing:
            logger.info(f"Using existing prompts from {usecase_folder}")
            return existing

    # Generate new prompts
    if api_key:
        generator.api_client = OpenRouterClient(api_key=api_key)

    return await generator.generate_and_save_prompts(usecase_folder)
