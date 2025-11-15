"""
Task definition parser for LLM TaskBench.

This module handles loading, validation, and saving of task definitions
from/to YAML files.
"""

import logging
from pathlib import Path
from typing import List, Tuple

import yaml

from taskbench.core.models import TaskDefinition

logger = logging.getLogger(__name__)


class TaskParser:
    """
    Parser for task definition YAML files.

    This class provides methods to load, validate, and save task definitions.

    Example:
        >>> parser = TaskParser()
        >>> task = parser.load_from_yaml("tasks/lecture_analysis.yaml")
        >>> is_valid, errors = parser.validate_task(task)
        >>> if is_valid:
        ...     print(f"Task {task.name} is valid!")
    """

    @staticmethod
    def load_from_yaml(yaml_path: str) -> TaskDefinition:
        """
        Load a task definition from a YAML file.

        Args:
            yaml_path: Path to the YAML file

        Returns:
            TaskDefinition object

        Raises:
            FileNotFoundError: If the YAML file doesn't exist
            ValueError: If the YAML is malformed or missing required fields
            yaml.YAMLError: If the YAML cannot be parsed

        Example:
            >>> parser = TaskParser()
            >>> task = parser.load_from_yaml("tasks/my_task.yaml")
        """
        path = Path(yaml_path)

        # Check if file exists
        if not path.exists():
            raise FileNotFoundError(f"Task file not found: {yaml_path}")

        # Load YAML content
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML file {yaml_path}: {e}") from e
        except Exception as e:
            raise ValueError(f"Error reading file {yaml_path}: {e}") from e

        # Check if data was loaded
        if data is None:
            raise ValueError(f"YAML file {yaml_path} is empty")

        # Validate YAML structure
        if not isinstance(data, dict):
            raise ValueError(f"YAML file {yaml_path} must contain a dictionary at the root")

        # Try to create TaskDefinition
        try:
            task = TaskDefinition(**data)
            logger.info(f"Successfully loaded task '{task.name}' from {yaml_path}")
            return task
        except Exception as e:
            raise ValueError(
                f"Failed to create TaskDefinition from {yaml_path}: {e}"
            ) from e

    @staticmethod
    def validate_task(task: TaskDefinition) -> Tuple[bool, List[str]]:
        """
        Validate a task definition.

        Checks for:
        - Required fields are present
        - Constraints are logically consistent
        - Evaluation criteria is non-empty

        Args:
            task: TaskDefinition to validate

        Returns:
            Tuple of (is_valid, list_of_errors)

        Example:
            >>> task = TaskDefinition(...)
            >>> is_valid, errors = TaskParser.validate_task(task)
            >>> if not is_valid:
            ...     for error in errors:
            ...         print(f"Error: {error}")
        """
        errors = []

        # Check required fields
        if not task.name:
            errors.append("Task name is required")
        if not task.description:
            errors.append("Task description is required")
        if not task.evaluation_criteria:
            errors.append("At least one evaluation criterion is required")
        if not task.judge_instructions:
            errors.append("Judge instructions are required")

        # Validate constraints
        if task.constraints:
            # Check for min/max consistency
            if "min_duration_minutes" in task.constraints and "max_duration_minutes" in task.constraints:
                min_dur = task.constraints["min_duration_minutes"]
                max_dur = task.constraints["max_duration_minutes"]
                if isinstance(min_dur, (int, float)) and isinstance(max_dur, (int, float)):
                    if min_dur > max_dur:
                        errors.append(
                            f"min_duration_minutes ({min_dur}) cannot be greater than "
                            f"max_duration_minutes ({max_dur})"
                        )

            # Check for other min/max pairs
            for key in task.constraints:
                if key.startswith("min_"):
                    max_key = key.replace("min_", "max_", 1)
                    if max_key in task.constraints:
                        min_val = task.constraints[key]
                        max_val = task.constraints[max_key]
                        if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                            if min_val > max_val:
                                errors.append(
                                    f"{key} ({min_val}) cannot be greater than {max_key} ({max_val})"
                                )

        is_valid = len(errors) == 0

        if is_valid:
            logger.info(f"Task '{task.name}' validation passed")
        else:
            logger.warning(f"Task '{task.name}' validation failed: {errors}")

        return is_valid, errors

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
            >>> task = TaskDefinition(...)
            >>> TaskParser.save_to_yaml(task, "tasks/my_task.yaml")
        """
        path = Path(yaml_path)

        # Create parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert task to dictionary
        data = task.model_dump()

        # Write to YAML file
        try:
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(
                    data,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                    indent=2
                )
            logger.info(f"Successfully saved task '{task.name}' to {yaml_path}")
        except Exception as e:
            raise IOError(f"Failed to save task to {yaml_path}: {e}") from e

    @staticmethod
    def load_all_from_directory(directory: str) -> List[TaskDefinition]:
        """
        Load all task definitions from a directory.

        Args:
            directory: Path to directory containing YAML task files

        Returns:
            List of TaskDefinition objects

        Example:
            >>> parser = TaskParser()
            >>> tasks = parser.load_all_from_directory("tasks/")
            >>> print(f"Loaded {len(tasks)} tasks")
        """
        dir_path = Path(directory)

        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        if not dir_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory}")

        tasks = []
        yaml_files = list(dir_path.glob("*.yaml")) + list(dir_path.glob("*.yml"))

        for yaml_file in yaml_files:
            try:
                task = TaskParser.load_from_yaml(str(yaml_file))
                tasks.append(task)
            except Exception as e:
                logger.warning(f"Failed to load task from {yaml_file}: {e}")
                continue

        logger.info(f"Loaded {len(tasks)} tasks from {directory}")
        return tasks
