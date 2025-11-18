"""Pytest configuration and fixtures."""

import pytest
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool

from app.models.database import Base
from app.core.config import settings


@pytest.fixture
async def test_db() -> AsyncGenerator[AsyncSession, None]:
    """Create test database session."""

    # Use in-memory SQLite for tests
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session() as session:
        yield session

    await engine.dispose()


@pytest.fixture
def sample_task_data():
    """Sample task data for testing."""
    return {
        "name": "test_task",
        "description": "Extract concepts from text",
        "domain": "education",
        "input_format": "text",
        "output_format": "json",
        "constraints": {"min_items": 3},
        "evaluation_criteria": ["accuracy", "completeness"]
    }


@pytest.fixture
def sample_quality_checks():
    """Sample quality checks for testing."""
    return [
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
    ]
