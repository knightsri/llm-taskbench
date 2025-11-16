"""
Tests for OpenRouter API client.

This module tests the OpenRouterClient class including:
- Initialization and configuration
- API completion requests
- JSON mode requests
- Error handling
- Token usage tracking
- Latency measurement
"""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from taskbench.api.client import OpenRouterClient, OpenRouterAPIError
from taskbench.core.models import CompletionResponse


class TestOpenRouterClient:
    """Test suite for OpenRouterClient."""

    def test_init_with_api_key(self):
        """Test initialization with API key provided."""
        client = OpenRouterClient(api_key="test-key-123")
        assert client.api_key == "test-key-123"
        assert client.timeout == OpenRouterClient.DEFAULT_TIMEOUT

    def test_init_with_env_var(self, monkeypatch):
        """Test initialization with API key from environment variable."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "env-key-456")
        client = OpenRouterClient()
        assert client.api_key == "env-key-456"

    def test_init_without_api_key(self, monkeypatch):
        """Test initialization fails without API key."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OpenRouter API key not found"):
            OpenRouterClient()

    def test_init_custom_timeout(self):
        """Test initialization with custom timeout."""
        client = OpenRouterClient(api_key="test-key", timeout=60.0)
        assert client.timeout == 60.0

    @pytest.mark.asyncio
    async def test_complete_success(self):
        """Test successful completion request."""
        # Mock response data
        mock_response_data = {
            "choices": [
                {
                    "message": {
                        "content": "This is a test response"
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            }
        }

        # Create mock HTTP response
        mock_http_response = MagicMock(spec=httpx.Response)
        mock_http_response.status_code = 200
        mock_http_response.json.return_value = mock_response_data

        # Create client with mocked HTTP client
        client = OpenRouterClient(api_key="test-key")

        # Mock the async HTTP client
        mock_http_client = AsyncMock()
        mock_http_client.post.return_value = mock_http_response
        client._client = mock_http_client

        # Execute completion
        response = await client.complete(
            model="test-model",
            prompt="Test prompt",
            temperature=0.7
        )

        # Verify response
        assert isinstance(response, CompletionResponse)
        assert response.content == "This is a test response"
        assert response.model == "test-model"
        assert response.input_tokens == 100
        assert response.output_tokens == 50
        assert response.total_tokens == 150
        assert response.latency_ms > 0

        # Verify API call was made correctly
        mock_http_client.post.assert_called_once()
        call_args = mock_http_client.post.call_args
        assert call_args[0][0].endswith("/chat/completions")
        payload = call_args[1]["json"]
        assert payload["model"] == "test-model"
        assert payload["messages"][0]["content"] == "Test prompt"
        assert payload["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_complete_with_max_tokens(self):
        """Test completion request with max_tokens parameter."""
        mock_response_data = {
            "choices": [{"message": {"content": "Response"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }

        mock_http_response = MagicMock(spec=httpx.Response)
        mock_http_response.status_code = 200
        mock_http_response.json.return_value = mock_response_data

        client = OpenRouterClient(api_key="test-key")
        mock_http_client = AsyncMock()
        mock_http_client.post.return_value = mock_http_response
        client._client = mock_http_client

        await client.complete(
            model="test-model",
            prompt="Test",
            max_tokens=100
        )

        # Verify max_tokens was included in payload
        call_args = mock_http_client.post.call_args
        payload = call_args[1]["json"]
        assert payload["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_complete_with_json_mode(self):
        """Test completion request with JSON mode."""
        mock_response_data = {
            "choices": [{"message": {"content": '{"key": "value"}'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }

        mock_http_response = MagicMock(spec=httpx.Response)
        mock_http_response.status_code = 200
        mock_http_response.json.return_value = mock_response_data

        client = OpenRouterClient(api_key="test-key")
        mock_http_client = AsyncMock()
        mock_http_client.post.return_value = mock_http_response
        client._client = mock_http_client

        response = await client.complete_with_json(
            model="test-model",
            prompt="Return JSON"
        )

        # Verify response format parameter was set
        call_args = mock_http_client.post.call_args
        payload = call_args[1]["json"]
        assert "response_format" in payload
        assert payload["response_format"]["type"] == "json_object"

        # Verify response content is JSON-like
        assert response.content == '{"key": "value"}'

    @pytest.mark.asyncio
    async def test_complete_401_authentication_error(self):
        """Test handling of 401 authentication error."""
        mock_http_response = MagicMock(spec=httpx.Response)
        mock_http_response.status_code = 401
        mock_http_response.text = "Invalid API key"
        mock_http_response.json.return_value = {"error": {"message": "Invalid API key"}}

        client = OpenRouterClient(api_key="invalid-key")
        mock_http_client = AsyncMock()
        mock_http_client.post.return_value = mock_http_response
        client._client = mock_http_client

        with pytest.raises(OpenRouterAPIError, match="Authentication failed"):
            await client.complete(model="test-model", prompt="Test")

    @pytest.mark.asyncio
    async def test_complete_429_rate_limit_error(self):
        """Test handling of 429 rate limit error."""
        mock_http_response = MagicMock(spec=httpx.Response)
        mock_http_response.status_code = 429
        mock_http_response.text = "Rate limit exceeded"
        mock_http_response.json.return_value = {"error": {"message": "Rate limit exceeded"}}

        client = OpenRouterClient(api_key="test-key")
        mock_http_client = AsyncMock()
        mock_http_client.post.return_value = mock_http_response
        client._client = mock_http_client

        with pytest.raises(OpenRouterAPIError, match="Rate limit exceeded"):
            await client.complete(model="test-model", prompt="Test")

    @pytest.mark.asyncio
    async def test_complete_500_server_error(self):
        """Test handling of 500 server error."""
        mock_http_response = MagicMock(spec=httpx.Response)
        mock_http_response.status_code = 500
        mock_http_response.text = "Internal server error"
        mock_http_response.json.return_value = {"error": {"message": "Internal server error"}}

        client = OpenRouterClient(api_key="test-key")
        mock_http_client = AsyncMock()
        mock_http_client.post.return_value = mock_http_response
        client._client = mock_http_client

        with pytest.raises(OpenRouterAPIError, match="Server error.*500"):
            await client.complete(model="test-model", prompt="Test")

    @pytest.mark.asyncio
    async def test_complete_400_bad_request(self):
        """Test handling of 400 bad request error."""
        mock_http_response = MagicMock(spec=httpx.Response)
        mock_http_response.status_code = 400
        mock_http_response.text = "Invalid model"
        mock_http_response.json.return_value = {"error": {"message": "Invalid model"}}

        client = OpenRouterClient(api_key="test-key")
        mock_http_client = AsyncMock()
        mock_http_client.post.return_value = mock_http_response
        client._client = mock_http_client

        with pytest.raises(OpenRouterAPIError, match="Bad request.*Invalid model"):
            await client.complete(model="invalid-model", prompt="Test")

    @pytest.mark.asyncio
    async def test_complete_timeout_error(self):
        """Test handling of timeout error."""
        client = OpenRouterClient(api_key="test-key", timeout=1.0)
        mock_http_client = AsyncMock()
        mock_http_client.post.side_effect = httpx.TimeoutException("Request timed out")
        client._client = mock_http_client

        with pytest.raises(OpenRouterAPIError, match="timed out"):
            await client.complete(model="test-model", prompt="Test")

    @pytest.mark.asyncio
    async def test_complete_network_error(self):
        """Test handling of network error."""
        client = OpenRouterClient(api_key="test-key")
        mock_http_client = AsyncMock()
        mock_http_client.post.side_effect = httpx.ConnectError("Connection failed")
        client._client = mock_http_client

        with pytest.raises(OpenRouterAPIError, match="HTTP error"):
            await client.complete(model="test-model", prompt="Test")

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager usage."""
        async with OpenRouterClient(api_key="test-key") as client:
            assert client._client is not None

        # Client should be closed after context
        assert client._client is None

    @pytest.mark.asyncio
    async def test_close(self):
        """Test explicit close method."""
        client = OpenRouterClient(api_key="test-key")
        await client._ensure_client()
        assert client._client is not None

        await client.close()
        assert client._client is None

    def test_extract_token_usage_complete_data(self):
        """Test token usage extraction with complete data."""
        client = OpenRouterClient(api_key="test-key")
        response_data = {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            }
        }

        usage = client._extract_token_usage(response_data)
        assert usage["input_tokens"] == 100
        assert usage["output_tokens"] == 50
        assert usage["total_tokens"] == 150

    def test_extract_token_usage_missing_total(self):
        """Test token usage extraction when total is missing."""
        client = OpenRouterClient(api_key="test-key")
        response_data = {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50
            }
        }

        usage = client._extract_token_usage(response_data)
        assert usage["input_tokens"] == 100
        assert usage["output_tokens"] == 50
        assert usage["total_tokens"] == 150  # Should be calculated

    def test_extract_token_usage_missing_usage(self):
        """Test token usage extraction when usage is missing."""
        client = OpenRouterClient(api_key="test-key")
        response_data = {}

        usage = client._extract_token_usage(response_data)
        assert usage["input_tokens"] == 0
        assert usage["output_tokens"] == 0
        assert usage["total_tokens"] == 0


class TestOpenRouterAPIError:
    """Test suite for OpenRouterAPIError exception."""

    def test_error_with_status_code(self):
        """Test error creation with status code."""
        error = OpenRouterAPIError(
            "Test error",
            status_code=429,
            response_body="Rate limit exceeded"
        )
        assert str(error) == "Test error"
        assert error.status_code == 429
        assert error.response_body == "Rate limit exceeded"

    def test_error_without_status_code(self):
        """Test error creation without status code."""
        error = OpenRouterAPIError("Test error")
        assert str(error) == "Test error"
        assert error.status_code is None
        assert error.response_body is None
