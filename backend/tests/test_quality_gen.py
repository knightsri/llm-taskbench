"""Tests for quality check generation."""

import pytest
from unittest.mock import patch, AsyncMock
from app.core.quality_gen import generate_quality_checks, _default_quality_checks


@pytest.mark.asyncio
async def test_generate_quality_checks_success():
    """Test successful quality check generation."""

    mock_response = {
        "content": [{
            "text": '''[
                {
                    "name": "no_empty_output",
                    "description": "Output must not be empty",
                    "validation_function": "len(output) > 0",
                    "severity": "critical"
                },
                {
                    "name": "valid_json",
                    "description": "Output must be valid JSON",
                    "validation_function": "validate_json(output)",
                    "severity": "critical"
                }
            ]'''
        }]
    }

    with patch('httpx.AsyncClient') as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(
            return_value=AsyncMock(
                status_code=200,
                json=lambda: mock_response,
                raise_for_status=lambda: None
            )
        )

        result = await generate_quality_checks(
            "Extract concepts from text",
            domain="education",
            output_format="json"
        )

        assert len(result) == 2
        assert result[0]["name"] == "no_empty_output"
        assert result[0]["severity"] == "critical"


def test_default_quality_checks():
    """Test default quality checks fallback."""

    checks = _default_quality_checks("json")

    assert len(checks) >= 2
    assert all("name" in check for check in checks)
    assert all("severity" in check for check in checks)
    assert all("validation_function" in check for check in checks)
