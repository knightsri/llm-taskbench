# CLAUDE.md

**Context file for AI-assisted development of LLM TaskBench**

This document provides essential context to help Claude (or any AI assistant) make decisions aligned with the project's goals, constraints, and architecture.

---

## Project Overview

**LLM TaskBench** is a task-specific LLM evaluation framework that shifts evaluation from generic metrics (BLEU, ROUGE) to real-world task performance. It enables developers to evaluate multiple LLM models on their actual use cases using LLM-as-judge methodology with cost tracking.

### Core Purpose

- Enable **task-first** evaluation instead of metric-first evaluation
- Help developers find the **best model for their specific use case**
- Provide **cost-aware recommendations** balancing quality and budget
- Reveal insights like "405B models don't always beat 72B models" for specific tasks

---

## Project Goals

### Primary Goals

1. **Task-Specific Evaluation**: Allow users to define custom evaluation tasks with YAML configs
2. **LLM-as-Judge**: Automated quality assessment using Claude Sonnet 4.5
3. **Multi-Model Comparison**: Evaluate 5+ models simultaneously via OpenRouter
4. **Cost Awareness**: Track tokens, costs, and provide value-based recommendations
5. **Production Ready**: Robust error handling, retry logic, rate limiting

### Non-Goals

- Generic benchmarking (we're task-specific)
- Model training or fine-tuning (evaluation only)
- Real-time inference (batch evaluation focus)
- Web scraping or data collection (users provide inputs)

---

## Tech Stack

### Core Technologies

- **Python 3.11+**: Modern Python features, type hints required
- **Pydantic 2.0+**: Data validation and settings management
- **Typer**: CLI framework with rich terminal output
- **HTTPX**: Async HTTP client for API calls
- **PyYAML**: Task definition parsing

### Development Tools

- **pytest**: Testing framework with async support
- **black**: Code formatting (100 char line length)
- **isort**: Import sorting (black profile)
- **mypy**: Type checking (strict mode)
- **pytest-cov**: Code coverage reporting

### External Services

- **OpenRouter API**: Unified access to 100+ LLM models
- **Claude Sonnet 4.5**: Default judge model (via OpenRouter)
- **Anthropic/OpenAI** (optional): Direct API access

---

## Architecture

### Project Structure

```
src/taskbench/
├── core/           # Data models (Pydantic) and task parsing
├── api/            # OpenRouter client with retry logic
├── evaluation/     # Task execution, judging, cost tracking
├── cli/            # Typer-based command-line interface
└── utils/          # Logging, validation helpers

tasks/              # Built-in YAML task definitions
config/             # Model pricing database (models.yaml)
tests/              # Pytest test suite
docs/               # Documentation (ARCHITECTURE.md, API.md, USAGE.md)
```

### Key Components

1. **Task Parser** (`core/task.py`): Loads and validates YAML task definitions
2. **API Client** (`api/client.py`): Handles OpenRouter API calls with retry logic
3. **Executor** (`evaluation/executor.py`): Runs tasks against multiple models
4. **Judge** (`evaluation/judge.py`): LLM-as-judge evaluation with Claude
5. **Cost Tracker** (`evaluation/cost.py`): Token and cost calculation
6. **Recommender** (`evaluation/recommender.py`): Provides best overall/value recommendations

### Data Flow

```
User Input (task.yaml + input.txt)
    ↓
Task Parser → validates task definition
    ↓
Executor → runs task on N models in parallel
    ↓
Judge → evaluates each output with Claude
    ↓
Recommender → analyzes scores + costs
    ↓
CLI → displays comparison table + recommendations
```

---

## Design Principles

### Code Quality Standards

1. **Type Safety**: All functions must have type hints (`mypy --strict`)
2. **Pydantic Validation**: Use Pydantic models for all data structures
3. **Error Handling**: Comprehensive try/except with user-friendly messages
4. **Testing**: 80%+ code coverage, test edge cases
5. **Documentation**: Docstrings for all public functions

### API Design Principles

1. **Async-First**: Use async/await for all I/O operations
2. **Retry Logic**: Exponential backoff for API failures (3 retries max)
3. **Rate Limiting**: Respect OpenRouter rate limits
4. **Streaming**: Support streaming responses when available
5. **Timeouts**: 60s default, configurable per model

### CLI Design Principles

1. **Rich Output**: Use `rich` library for tables, progress bars
2. **Clear Errors**: Show actionable error messages with suggestions
3. **Confirmations**: Ask before expensive operations (>$1 estimated cost)
4. **Verbose Mode**: `--verbose` flag for debugging
5. **JSON Output**: Support `--output json` for scripting

---

## Constraints

### Technical Constraints

- **Python 3.11+**: Cannot support older versions (uses new typing features)
- **OpenRouter Dependency**: Primary API (direct APIs are optional)
- **Sync CLI**: CLI is synchronous (uses `asyncio.run()` internally)
- **File-Based Config**: No database, YAML configs only
- **Unix-First**: Primary support for Linux/macOS (Windows best-effort)

### Evaluation Constraints

- **Judge Model**: Claude Sonnet 4.5 is default (configurable but recommended)
- **Output Size**: Max 100KB per model output (OpenRouter limit)
- **Concurrent Requests**: 5 models max in parallel (rate limiting)
- **Timeout**: 60s per model invocation (configurable)

### Cost Constraints

- **Budget Awareness**: Warn if evaluation exceeds $1 estimated cost
- **No Streaming for Judge**: Judge uses complete outputs (no streaming)
- **Token Tracking**: Track input + output tokens separately

---

## Development Guidelines

### Adding New Features

When adding features, consider:

1. **Is it task-specific?** (Aligns with core mission)
2. **Does it add complexity?** (Prefer simple solutions)
3. **Is it testable?** (Write tests first)
4. **Does it affect cost?** (Document cost implications)
5. **Is it documented?** (Update docs/ folder)

### Code Style

```python
# Good: Type hints, Pydantic validation, async
async def evaluate_model(
    task: Task,
    model: str,
    input_text: str,
) -> ModelResult:
    """Evaluate a single model on a task.

    Args:
        task: Pydantic Task model
        model: Model ID (e.g., "anthropic/claude-sonnet-4.5")
        input_text: Input text to evaluate

    Returns:
        ModelResult with output, tokens, cost, latency

    Raises:
        APIError: If API call fails after retries
    """
    async with APIClient() as client:
        result = await client.complete(model, input_text)
        return ModelResult(**result)
```

### Testing Guidelines

- **Unit Tests**: Test pure functions in isolation
- **Integration Tests**: Test API client with mocked responses
- **Fixtures**: Use `tests/fixtures/` for sample data
- **Async Tests**: Use `pytest-asyncio` for async functions
- **Mocking**: Mock API calls to avoid costs

### Git Workflow

- **Branch Naming**: `claude/feature-name-<session-id>` for Claude-created branches
- **Commits**: Clear, descriptive messages ("Add retry logic to API client")
- **Push**: Use `git push -u origin <branch>` (required for Claude sessions)
- **PRs**: Include test results and coverage changes

---

## Key Design Decisions

### Why OpenRouter?

- **Unified API**: Access 100+ models without managing multiple API keys
- **Cost Tracking**: Built-in token and cost tracking
- **Rate Limiting**: Centralized rate limiting across providers
- **Model Discovery**: Automatically sync available models

### Why Claude Sonnet 4.5 as Judge?

- **Reliability**: Consistent scoring across evaluations
- **Reasoning**: Strong analytical capabilities for complex criteria
- **Structured Output**: Reliable JSON output for scores/violations
- **Cost**: Balanced cost vs. quality ($3/million tokens)

### Why YAML for Tasks?

- **Human-Readable**: Non-developers can define tasks
- **Version Control**: Easy to diff and review
- **Validation**: Pydantic validates schema automatically
- **Extensible**: Easy to add new fields

### Why CLI-First?

- **Developer Audience**: Primary users are developers
- **CI/CD Integration**: Easy to integrate into pipelines
- **Scriptable**: JSON output for automation
- **Lightweight**: No web server needed

---

## Common Patterns

### Error Handling

```python
# Pattern: Catch specific errors, provide context
try:
    result = await api_client.complete(model, prompt)
except APIError as e:
    logger.error(f"API error for {model}: {e}")
    raise TaskBenchError(f"Failed to evaluate {model}. Check API key and rate limits.")
except TimeoutError:
    raise TaskBenchError(f"{model} timed out after 60s. Try a smaller input or increase timeout.")
```

### Pydantic Models

```python
# Pattern: Strict validation, helpful errors
class Task(BaseModel):
    name: str = Field(..., min_length=1, description="Task identifier")
    description: str = Field(..., min_length=10, description="What this task evaluates")
    evaluation_criteria: list[str] = Field(..., min_items=1)

    model_config = ConfigDict(
        extra="forbid",  # Reject unknown fields
        str_strip_whitespace=True,
    )
```

### Async API Calls

```python
# Pattern: Parallel execution with timeout
async def evaluate_models(task: Task, models: list[str], input_text: str) -> list[ModelResult]:
    async with APIClient() as client:
        tasks = [
            asyncio.wait_for(
                client.complete(model, input_text),
                timeout=60.0
            )
            for model in models
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if not isinstance(r, Exception)]
```

---

## Future Roadmap

### Phase 2: Enhanced Features (Next)

- Batch evaluation (multiple inputs at once)
- Custom judge models (not just Claude)
- Results visualization (charts, graphs)
- Web interface (optional)

### Phase 3: Advanced Analytics

- Historical tracking (regression detection)
- A/B testing (compare model versions)
- Fine-tuning guidance (suggest when to fine-tune)

---

## Quick Reference

### Running Tests

```bash
pytest                          # All tests
pytest --cov                    # With coverage
pytest tests/test_models.py -v  # Specific module
```

### Code Quality

```bash
black src/ tests/               # Format
isort src/ tests/               # Sort imports
mypy src/                       # Type check
flake8 src/                     # Lint
```

### Common Tasks

```bash
# Evaluate models
taskbench evaluate tasks/lecture_analysis.yaml \
  --models anthropic/claude-sonnet-4.5,openai/gpt-4o \
  --input-file input.txt

# Validate task definition
taskbench validate tasks/my_task.yaml

# List available models
taskbench models --list
```

---

## Questions to Ask Before Making Changes

1. **Does this align with task-specific evaluation?**
2. **Will this increase costs for users?** (Consider default settings)
3. **Does this require a new dependency?** (Prefer stdlib or existing deps)
4. **Is this testable without API calls?** (Mock external APIs)
5. **Does this break backward compatibility?** (Avoid breaking changes in 0.x)
6. **Is there a simpler solution?** (Prefer simple over clever)

---

## When in Doubt

- **Check existing patterns**: Look at similar code in the codebase
- **Read the docs**: See `docs/ARCHITECTURE.md`, `docs/API.md`, `docs/USAGE.md`
- **Run tests**: `pytest --cov` before committing
- **Ask the user**: When unsure about requirements or tradeoffs

---

**Last Updated**: 2025-11-16
**Maintained By**: LLM TaskBench Contributors
**For Questions**: See GitHub Issues or Discussions
