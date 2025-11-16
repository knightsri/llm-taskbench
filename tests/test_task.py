"""
Tests for task definition parser and validator.

This module tests the TaskParser class for loading, validating,
and saving task definitions.
"""

import tempfile
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from taskbench.core.models import TaskDefinition
from taskbench.core.task import TaskParser


class TestTaskParser:
    """Tests for TaskParser class."""

    @pytest.fixture
    def parser(self):
        """Create a TaskParser instance."""
        return TaskParser()

    @pytest.fixture
    def valid_task_path(self):
        """Path to valid task fixture."""
        return "tests/fixtures/valid_task.yaml"

    @pytest.fixture
    def invalid_task_path(self):
        """Path to invalid task fixture."""
        return "tests/fixtures/invalid_task.yaml"

    @pytest.fixture
    def sample_task(self):
        """Create a sample TaskDefinition."""
        return TaskDefinition(
            name="sample_task",
            description="A sample task for testing",
            input_type="text",
            output_format="json",
            evaluation_criteria=["Accuracy", "Completeness"],
            constraints={"min_length": 10, "max_length": 100},
            examples=[],
            judge_instructions="Evaluate carefully"
        )

    def test_load_valid_yaml(self, parser, valid_task_path):
        """Test loading a valid YAML file."""
        task = parser.load_from_yaml(valid_task_path)
        assert isinstance(task, TaskDefinition)
        assert task.name == "test_lecture_analysis"
        assert task.input_type == "transcript"
        assert task.output_format == "csv"
        assert len(task.evaluation_criteria) > 0
        assert task.constraints["min_duration_minutes"] == 2
        assert task.constraints["max_duration_minutes"] == 7

    def test_load_nonexistent_file(self, parser):
        """Test loading a file that doesn't exist."""
        with pytest.raises(FileNotFoundError) as exc_info:
            parser.load_from_yaml("nonexistent/file.yaml")
        assert "not found" in str(exc_info.value)

    def test_load_invalid_yaml_syntax(self, parser, tmp_path):
        """Test loading a file with invalid YAML syntax."""
        invalid_yaml = tmp_path / "invalid.yaml"
        invalid_yaml.write_text("name: [invalid yaml\nno closing bracket")

        with pytest.raises(yaml.YAMLError):
            parser.load_from_yaml(str(invalid_yaml))

    def test_load_invalid_task_structure(self, parser, invalid_task_path):
        """Test loading a YAML file with invalid task structure."""
        with pytest.raises(ValidationError):
            parser.load_from_yaml(invalid_task_path)

    def test_validate_valid_task(self, parser, sample_task):
        """Test validating a valid task."""
        is_valid, errors = parser.validate_task(sample_task)
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_empty_criteria(self, parser):
        """Test validation catches empty evaluation_criteria."""
        task = TaskDefinition(
            name="test",
            description="Test",
            input_type="text",
            output_format="json",
            evaluation_criteria=[],  # Empty!
            judge_instructions="Test"
        )
        is_valid, errors = parser.validate_task(task)
        assert is_valid is False
        assert any("evaluation_criteria" in err for err in errors)

    def test_validate_empty_judge_instructions(self, parser):
        """Test validation catches empty judge_instructions."""
        task = TaskDefinition(
            name="test",
            description="Test",
            input_type="text",
            output_format="json",
            evaluation_criteria=["Accuracy"],
            judge_instructions=""  # Empty!
        )
        is_valid, errors = parser.validate_task(task)
        assert is_valid is False
        assert any("judge_instructions" in err for err in errors)

    def test_validate_min_max_constraints(self, parser):
        """Test validation catches invalid min/max constraints."""
        task = TaskDefinition(
            name="test",
            description="Test",
            input_type="text",
            output_format="json",
            evaluation_criteria=["Accuracy"],
            constraints={"min_duration_minutes": 10, "max_duration_minutes": 5},  # min > max!
            judge_instructions="Test"
        )
        is_valid, errors = parser.validate_task(task)
        assert is_valid is False
        assert any("min_duration_minutes" in err and "max_duration_minutes" in err for err in errors)

    def test_validate_csv_columns_constraint(self, parser):
        """Test validation of required_csv_columns for CSV output."""
        task = TaskDefinition(
            name="test",
            description="Test",
            input_type="text",
            output_format="csv",
            evaluation_criteria=["Accuracy"],
            constraints={"required_csv_columns": []},  # Empty list!
            judge_instructions="Test"
        )
        is_valid, errors = parser.validate_task(task)
        assert is_valid is False
        assert any("required_csv_columns" in err for err in errors)

    def test_save_to_yaml(self, parser, sample_task, tmp_path):
        """Test saving a task to YAML."""
        output_path = tmp_path / "output" / "saved_task.yaml"
        parser.save_to_yaml(sample_task, str(output_path))

        # Verify file was created
        assert output_path.exists()

        # Verify content can be loaded back
        loaded_task = parser.load_from_yaml(str(output_path))
        assert loaded_task.name == sample_task.name
        assert loaded_task.description == sample_task.description
        assert loaded_task.input_type == sample_task.input_type
        assert loaded_task.output_format == sample_task.output_format

    def test_round_trip_preservation(self, parser, valid_task_path, tmp_path):
        """Test that load ’ save ’ load preserves data."""
        # Load original task
        original_task = parser.load_from_yaml(valid_task_path)

        # Save to temporary file
        temp_path = tmp_path / "round_trip.yaml"
        parser.save_to_yaml(original_task, str(temp_path))

        # Load again
        reloaded_task = parser.load_from_yaml(str(temp_path))

        # Compare
        assert reloaded_task.name == original_task.name
        assert reloaded_task.description == original_task.description
        assert reloaded_task.input_type == original_task.input_type
        assert reloaded_task.output_format == original_task.output_format
        assert reloaded_task.evaluation_criteria == original_task.evaluation_criteria
        assert reloaded_task.constraints == original_task.constraints

    def test_save_creates_parent_directories(self, parser, sample_task, tmp_path):
        """Test that save_to_yaml creates parent directories if needed."""
        output_path = tmp_path / "nested" / "deep" / "path" / "task.yaml"
        parser.save_to_yaml(sample_task, str(output_path))

        assert output_path.exists()
        assert output_path.parent.exists()
