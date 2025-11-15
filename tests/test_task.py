"""
Tests for Task Parser.
"""

import tempfile
from pathlib import Path

import pytest

from taskbench.core.models import TaskDefinition
from taskbench.core.task import TaskParser


class TestTaskParser:
    """Tests for TaskParser class."""

    def test_load_valid_yaml(self):
        """Test loading a valid YAML file."""
        parser = TaskParser()
        fixture_path = Path(__file__).parent / "fixtures" / "valid_task.yaml"
        task = parser.load_from_yaml(str(fixture_path))

        assert isinstance(task, TaskDefinition)
        assert task.name == "test_lecture_analysis"
        assert task.input_type == "transcript"
        assert task.output_format == "csv"
        assert len(task.evaluation_criteria) > 0

    def test_load_nonexistent_file(self):
        """Test that loading a non-existent file raises FileNotFoundError."""
        parser = TaskParser()
        with pytest.raises(FileNotFoundError):
            parser.load_from_yaml("nonexistent_file.yaml")

    def test_load_invalid_yaml(self):
        """Test that loading invalid YAML raises ValueError."""
        parser = TaskParser()

        # Create a temporary file with invalid YAML
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content:\n  - malformed")
            temp_path = f.name

        try:
            with pytest.raises(ValueError) as exc_info:
                parser.load_from_yaml(temp_path)
            assert "Failed to parse YAML" in str(exc_info.value) or "Failed to create TaskDefinition" in str(exc_info.value)
        finally:
            Path(temp_path).unlink()

    def test_load_empty_yaml(self):
        """Test that loading an empty YAML file raises ValueError."""
        parser = TaskParser()

        # Create a temporary empty file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")
            temp_path = f.name

        try:
            with pytest.raises(ValueError) as exc_info:
                parser.load_from_yaml(temp_path)
            assert "empty" in str(exc_info.value).lower()
        finally:
            Path(temp_path).unlink()

    def test_validate_valid_task(self):
        """Test validation of a valid task."""
        task = TaskDefinition(
            name="test_task",
            description="A test task",
            input_type="transcript",
            output_format="csv",
            evaluation_criteria=["Accuracy", "Format"],
            constraints={"min_duration_minutes": 2, "max_duration_minutes": 7},
            judge_instructions="Evaluate based on criteria"
        )

        is_valid, errors = TaskParser.validate_task(task)
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_missing_fields(self):
        """Test validation catches missing fields."""
        task = TaskDefinition(
            name="",  # Empty name
            description="A test task",
            input_type="transcript",
            output_format="csv",
            evaluation_criteria=[],  # Empty criteria
            judge_instructions=""  # Empty instructions
        )

        is_valid, errors = TaskParser.validate_task(task)
        assert is_valid is False
        assert len(errors) > 0
        assert any("name" in error.lower() for error in errors)
        assert any("criteria" in error.lower() for error in errors)

    def test_validate_invalid_constraints(self):
        """Test validation catches invalid constraints."""
        task = TaskDefinition(
            name="test_task",
            description="A test task",
            input_type="transcript",
            output_format="csv",
            evaluation_criteria=["Accuracy"],
            constraints={
                "min_duration_minutes": 10,
                "max_duration_minutes": 5  # Max < Min, should fail
            },
            judge_instructions="Test"
        )

        is_valid, errors = TaskParser.validate_task(task)
        assert is_valid is False
        assert len(errors) > 0
        assert any("min" in error.lower() and "max" in error.lower() for error in errors)

    def test_save_and_load_round_trip(self):
        """Test that saving and loading preserves data."""
        original_task = TaskDefinition(
            name="test_task",
            description="A test task",
            input_type="transcript",
            output_format="csv",
            evaluation_criteria=["Accuracy", "Format"],
            constraints={"min_duration_minutes": 2, "max_duration_minutes": 7},
            examples=[{"input": "test", "output": "result"}],
            judge_instructions="Evaluate based on criteria"
        )

        # Save to temporary file
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "test_task.yaml"
            TaskParser.save_to_yaml(original_task, str(temp_path))

            # Load it back
            loaded_task = TaskParser.load_from_yaml(str(temp_path))

            # Verify all fields match
            assert loaded_task.name == original_task.name
            assert loaded_task.description == original_task.description
            assert loaded_task.input_type == original_task.input_type
            assert loaded_task.output_format == original_task.output_format
            assert loaded_task.evaluation_criteria == original_task.evaluation_criteria
            assert loaded_task.constraints == original_task.constraints
            assert loaded_task.judge_instructions == original_task.judge_instructions

    def test_save_creates_directories(self):
        """Test that save_to_yaml creates parent directories."""
        task = TaskDefinition(
            name="test_task",
            description="A test task",
            input_type="text",
            output_format="json",
            evaluation_criteria=["Accuracy"],
            judge_instructions="Test"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            # Path with non-existent subdirectories
            temp_path = Path(temp_dir) / "subdir1" / "subdir2" / "test_task.yaml"

            TaskParser.save_to_yaml(task, str(temp_path))

            # Verify file was created
            assert temp_path.exists()
            assert temp_path.is_file()

    def test_load_all_from_directory(self):
        """Test loading all tasks from a directory."""
        # Use the fixtures directory
        fixtures_dir = Path(__file__).parent / "fixtures"

        parser = TaskParser()
        tasks = parser.load_all_from_directory(str(fixtures_dir))

        # Should load at least the valid_task.yaml
        assert len(tasks) >= 1
        assert any(task.name == "test_lecture_analysis" for task in tasks)

    def test_load_all_from_nonexistent_directory(self):
        """Test that loading from non-existent directory raises FileNotFoundError."""
        parser = TaskParser()
        with pytest.raises(FileNotFoundError):
            parser.load_all_from_directory("/nonexistent/directory")
