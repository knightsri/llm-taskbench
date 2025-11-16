"""
Task definition parser and validator.

This module provides functionality to load, validate, and save task definitions
from YAML files.
"""

import logging
from pathlib import Path
from typing import List, Tuple

import yaml
from pydantic import ValidationError

from taskbench.core.models import TaskDefinition

logger = logging.getLogger(__name__)


class TaskParser:
    """
    Parser and validator for task definitions.

    Handles loading task definitions from YAML files, validating their structure,
    and saving them back to YAML format.

    Example:
        ```python
        parser = TaskParser()
        task = parser.load_from_yaml("tasks/lecture_analysis.yaml")
        is_valid, errors = parser.validate_task(task)
        if is_valid:
            parser.save_to_yaml(task, "tasks/modified_task.yaml")
        ```
    """

    @staticmethod
    def load_from_yaml(yaml_path: str) -> TaskDefinition:
        """
        Load a task definition from a YAML file.

        Args:
            yaml_path: Path to the YAML file containing the task definition

        Returns:
            TaskDefinition object parsed from the YAML file

        Raises:
            FileNotFoundError: If the YAML file doesn't exist
            yaml.YAMLError: If the YAML is malformed
            ValidationError: If the YAML doesn't match TaskDefinition schema

        Example:
            ```python
            parser = TaskParser()
            task = parser.load_from_yaml("tasks/my_task.yaml")
            ```
        """
        path = Path(yaml_path)

        # Check if file exists
        if not path.exists():
            raise FileNotFoundError(
                f"Task definition file not found: {yaml_path}\n"
                f"Please ensure the file exists at the specified path."
            )

        # Load and parse YAML
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(
                f"Failed to parse YAML file '{yaml_path}': {str(e)}\n"
                f"Please check the YAML syntax."
            ) from e

        # Validate and create TaskDefinition
        try:
            task = TaskDefinition(**data)
            logger.info(f"Successfully loaded task definition: {task.name}")
            return task
        except ValidationError as e:
            raise ValidationError(
                f"Invalid task definition in '{yaml_path}':\n{str(e)}"
            ) from e

    @staticmethod
    def validate_task(task: TaskDefinition) -> Tuple[bool, List[str]]:
        """
        Validate a task definition for logical consistency.

        Performs additional validation beyond Pydantic's schema validation,
        checking for logical errors in constraints and criteria.

        Args:
            task: TaskDefinition to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
            - is_valid: True if task is valid, False otherwise
            - list_of_errors: List of error messages (empty if valid)

        Example:
            ```python
            parser = TaskParser()
            task = parser.load_from_yaml("task.yaml")
            is_valid, errors = parser.validate_task(task)
            if not is_valid:
                for error in errors:
                    print(f"Error: {error}")
            ```
        """
        errors = []

        # Check that evaluation_criteria is non-empty
        if not task.evaluation_criteria:
            errors.append("evaluation_criteria cannot be empty")

        # Check that judge_instructions is not empty
        if not task.judge_instructions.strip():
            errors.append("judge_instructions cannot be empty")

        # Validate constraints make sense
        constraints = task.constraints

        # Check min/max duration constraints
        if "min_duration_minutes" in constraints and "max_duration_minutes" in constraints:
            min_dur = constraints["min_duration_minutes"]
            max_dur = constraints["max_duration_minutes"]

            # Check types
            if not isinstance(min_dur, (int, float)):
                errors.append(f"min_duration_minutes must be a number, got {type(min_dur)}")
            if not isinstance(max_dur, (int, float)):
                errors.append(f"max_duration_minutes must be a number, got {type(max_dur)}")

            # Check min < max
            if isinstance(min_dur, (int, float)) and isinstance(max_dur, (int, float)):
                if min_dur >= max_dur:
                    errors.append(
                        f"min_duration_minutes ({min_dur}) must be less than "
                        f"max_duration_minutes ({max_dur})"
                    )

        # Check for other common constraint pairs
        for min_key, max_key in [
            ("min_count", "max_count"),
            ("min_length", "max_length"),
            ("min_tokens", "max_tokens")
        ]:
            if min_key in constraints and max_key in constraints:
                min_val = constraints[min_key]
                max_val = constraints[max_key]

                if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                    if min_val >= max_val:
                        errors.append(
                            f"{min_key} ({min_val}) must be less than {max_key} ({max_val})"
                        )

        # Validate required_csv_columns if output_format is csv
        if task.output_format == "csv" and "required_csv_columns" in constraints:
            required_columns = constraints["required_csv_columns"]
            if not isinstance(required_columns, list):
                errors.append("required_csv_columns must be a list")
            elif len(required_columns) == 0:
                errors.append("required_csv_columns cannot be empty for CSV output")

        # Log validation results
        if errors:
            logger.warning(f"Task '{task.name}' has {len(errors)} validation error(s)")
            for error in errors:
                logger.warning(f"  - {error}")
        else:
            logger.info(f"Task '{task.name}' passed validation")

        return len(errors) == 0, errors

    @staticmethod
    def save_to_yaml(task: TaskDefinition, yaml_path: str) -> None:
        """
        Save a task definition to a YAML file.

        Args:
            task: TaskDefinition to save
            yaml_path: Path where the YAML file should be saved

        Raises:
            IOError: If the file cannot be written

        Example:
            ```python
            parser = TaskParser()
            task = TaskDefinition(...)
            parser.save_to_yaml(task, "tasks/new_task.yaml")
            ```
        """
        path = Path(yaml_path)

        # Create parent directories if needed
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert task to dict
        task_dict = task.model_dump()

        # Save to YAML
        try:
            with open(path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(
                    task_dict,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False,
                    indent=2
                )
            logger.info(f"Successfully saved task definition to: {yaml_path}")
        except IOError as e:
            raise IOError(
                f"Failed to write task definition to '{yaml_path}': {str(e)}"
            ) from e
