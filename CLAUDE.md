# CLAUDE.md

**Context file for AI-assisted development of LLM TaskBench**

This document provides essential context to help Claude (or any AI assistant) make decisions aligned with the project's goals, constraints, and architecture.

---

## Project Overview

**LLM TaskBench** is a task-specific LLM evaluation framework that enables developers to benchmark multiple LLM models on their actual use cases with parallel execution, LLM-as-judge evaluation, and cost-aware recommendations.

### Core Purpose

- Enable **task-first** evaluation with declarative YAML task definitions
- Compare **multiple LLMs simultaneously** via OpenRouter API
- Provide **LLM-as-judge** evaluation using Claude Sonnet 4.5
- Deliver **cost-aware recommendations** with transparent cost breakdowns
- Support **use-case layer** for context-driven model selection and prompts
- Enable **chunked processing** for long inputs with dynamic sizing

### Current Architecture

The project is **CLI-first** with an optional Streamlit UI:

```
┌─────────────────────────────────────────────────────────────┐
│                    CLI Interface                            │
│        taskbench evaluate | models | validate | recommend   │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────┴───────────────────────────────┐
│                  Optional Streamlit UI                      │
│              FastAPI API + Streamlit Frontend               │
│           http://localhost:8000 + http://localhost:8501     │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────┴───────────────────────────────┐
│                    Core Evaluation                          │
├─────────────────────────────────────────────────────────────┤
│  Executor → Parallel execution, chunking, prompt building   │
│  Judge → LLM-as-judge scoring with JSON mode               │
│  Comparison → Ranking, value metrics, recommendations       │
│  CostTracker → Token/cost tracking, generation ID lookup    │
│  Orchestrator → Use-case aware model recommendations        │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────┴───────────────────────────────┐
│                    OpenRouter API                           │
│          100+ LLM models (Claude, GPT-4, Gemini, etc.)      │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
llm-taskbench/
├── src/taskbench/           # Main CLI application
│   ├── cli/main.py          # CLI commands (evaluate, models, validate, recommend, sample)
│   ├── api/
│   │   ├── client.py        # OpenRouter API client with retry/backoff
│   │   └── retry.py         # Retry logic and rate limiting
│   ├── core/
│   │   ├── models.py        # Pydantic models (TaskDefinition, EvaluationResult, JudgeScore)
│   │   └── task.py          # TaskParser for YAML loading/validation
│   ├── evaluation/
│   │   ├── executor.py      # ModelExecutor - parallel execution, chunking
│   │   ├── judge.py         # LLMJudge - LLM-as-judge evaluation
│   │   ├── comparison.py    # ModelComparison - results analysis
│   │   ├── recommender.py   # RecommendationEngine - model recommendations
│   │   ├── cost.py          # CostTracker - token/cost tracking
│   │   └── orchestrator.py  # LLMOrchestrator - use-case model selection
│   ├── ui_api.py            # FastAPI endpoints for Streamlit UI
│   ├── usecase.py           # UseCase model and loader
│   └── utils/               # Logging and validation utilities
│
├── tasks/                   # Task definitions (YAML)
│   ├── lecture_analysis.yaml
│   └── template.yaml
│
├── usecases/                # Use-case specifications (YAML)
│   └── concepts_extraction.yaml
│
├── config/
│   └── models.yaml          # Model pricing catalog (11 models)
│
├── results/                 # Evaluation outputs (JSON)
├── tests/                   # Test suite with fixtures
├── docs/                    # Documentation
│
├── ui_app.py                # Streamlit UI entry point
├── docker-compose.cli.yml   # CLI Docker config
├── docker-compose.ui.yml    # UI Docker config (API + Streamlit)
├── pyproject.toml           # Python package config
└── Dockerfile               # Container build
```

---

## Tech Stack

### Core Technologies (Working)

| Package | Version | Purpose |
|---------|---------|---------|
| **pydantic** | >=2.0.0 | Data validation and modeling |
| **pyyaml** | >=6.0 | YAML parsing for task definitions |
| **httpx** | >=0.25.0 | Async HTTP client for API calls |
| **typer** | >=0.9.0 | CLI framework |
| **rich** | >=13.0.0 | Terminal formatting and progress bars |
| **python-dotenv** | >=1.0.0 | Environment variable management |

### UI Stack (Optional)

| Package | Purpose |
|---------|---------|
| **fastapi** | REST API for UI backend |
| **uvicorn** | ASGI server |
| **streamlit** | Web UI for task selection, execution, judging |

### Development Tools

| Package | Purpose |
|---------|---------|
| **pytest** | Unit and integration testing |
| **pytest-asyncio** | Async test support |
| **pytest-cov** | Code coverage reporting |
| **black** | Code formatting (100 char line length) |
| **isort** | Import sorting (black profile) |
| **mypy** | Static type checking |

### Python Version

- **Required**: Python 3.11+
- Uses modern async features and type hints

---

## CLI Commands

### `taskbench evaluate`

Run multi-model evaluation on a task.

```bash
taskbench evaluate tasks/lecture_analysis.yaml \
  --usecase usecases/concepts_extraction.yaml \
  --models anthropic/claude-sonnet-4.5,openai/gpt-4o \
  --input-file tests/fixtures/sample_transcript.txt \
  --output results/my_run.json \
  --chunked --dynamic-chunk \
  --skip-judge  # or omit to judge automatically
```

**Key Options:**
- `--models`: Comma-separated model IDs, or `auto` for orchestrator recommendations
- `--usecase`: Use-case YAML for goal/notes that drive prompts
- `--chunked`: Enable chunked processing for long inputs
- `--dynamic-chunk`: Derive chunk size from model context windows
- `--skip-judge` / `--no-judge`: Skip judge evaluation
- `--chunk-chars`: Max characters per chunk (default: 20000)
- `--chunk-overlap`: Overlap between chunks (default: 500)

### `taskbench models`

List available models and pricing.

```bash
taskbench models --list
taskbench models --info anthropic/claude-sonnet-4.5
```

### `taskbench validate`

Validate a task definition YAML file.

```bash
taskbench validate tasks/my_task.yaml
```

### `taskbench recommend`

Generate recommendations from a saved evaluation run.

```bash
taskbench recommend --results results/my_run.json
```

### `taskbench sample`

Run bundled sample evaluation.

```bash
taskbench sample \
  --models anthropic/claude-sonnet-4.5,openai/gpt-4o \
  --no-judge
```

---

## Docker Deployment

### CLI Mode

```bash
docker compose -f docker-compose.cli.yml build
docker compose -f docker-compose.cli.yml run --rm taskbench-cli evaluate \
  tasks/lecture_analysis.yaml \
  --models anthropic/claude-sonnet-4.5 \
  --input-file tests/fixtures/sample_transcript.txt \
  --skip-judge
```

### UI Mode

```bash
cp .env.example .env
# Add your OPENROUTER_API_KEY to .env
docker compose -f docker-compose.ui.yml up --build
# API: http://localhost:8000
# UI: http://localhost:8501
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENROUTER_API_KEY` | Yes | - | OpenRouter API key |
| `TASKBENCH_MAX_CONCURRENCY` | No | 5 | Max parallel model calls |
| `TASKBENCH_USE_GENERATION_LOOKUP` | No | true | Fetch billed cost from generation endpoint |
| `TASKBENCH_DEFAULT_JUDGE_MODEL` | No | anthropic/claude-sonnet-4.5 | Default judge model |
| `GENERAL_TASK_LLM` | No | anthropic/claude-sonnet-4.5 | Default model for general tasks/fallback |

---

## Key Concepts

### Task Definition (YAML)

Tasks define what to evaluate:

```yaml
name: "Lecture Concept Extraction"
description: "Extract teaching concepts from lecture transcripts"
input_type: "text"
output_format: "json"
evaluation_criteria:
  - "Accurate timestamps"
  - "Complete coverage"
  - "Concise descriptions"
constraints:
  min_segment_duration_minutes: 4
  max_segment_duration_minutes: 8
examples:
  - input: "..."
    expected_output: "..."
judge_instructions: |
  Evaluate for accuracy, format compliance, and constraint adherence...
```

### Use-Case Layer (YAML)

Use-cases provide context for execution and judging:

```yaml
name: "concepts_extraction_from_lecture"
goal: "Extract teaching concepts from long lectures into 4-8 minute segments"
chunk_min_minutes: 4
chunk_max_minutes: 8
coverage_required: true
cost_priority: "high"
quality_priority: "high"
notes: |
  - No gaps from provided timestamps
  - Honor mid-session breaks
default_judge_model: "${TASKBENCH_DEFAULT_JUDGE_MODEL:-anthropic/claude-sonnet-4.5}"
default_candidate_models:
  - anthropic/claude-sonnet-4.5
  - openai/gpt-4o
  - google/gemini-2.5-flash
```

### LLM-as-Judge Evaluation

Judge model (Claude Sonnet 4.5 by default) evaluates outputs:

- **Accuracy Score** (0-100): Content correctness
- **Format Score** (0-100): Format compliance
- **Compliance Score** (0-100): Constraint adherence
- **Overall Score** (0-100): Weighted combination
- **Violations**: List of specific issues
- **Reasoning**: Detailed explanation

### Cost Tracking

- **Inline usage**: Requested on every API call
- **Generation lookup**: Fetches billed cost from OpenRouter `/generation` endpoint
- **Results store**: `generation_id`, `billed_cost_usd`, token counts, per-model totals

### Model Catalog

11 models configured in `config/models.yaml`:

| Model | Provider | Input $/1M | Output $/1M | Context |
|-------|----------|------------|-------------|---------|
| claude-sonnet-4.5 | Anthropic | $3.00 | $15.00 | 200K |
| gpt-4o | OpenAI | $5.00 | $15.00 | 128K |
| gemini-2.5-flash | Google | $0.40 | $0.80 | 1M |
| llama-3.1-405b-instruct | Meta | $2.70 | $2.70 | 128K |
| qwen-2.5-72b-instruct | Alibaba | $0.35 | $0.40 | 128K |
| glm-4.7 | Z.AI | $2.00 | $6.00 | 128K |
| claude-3.5-sonnet | Anthropic | $3.00 | $15.00 | 200K |
| gpt-4-turbo | OpenAI | $10.00 | $30.00 | 128K |
| gemini-pro-1.5 | Google | $1.25 | $5.00 | 2M |
| mistral-large | Mistral AI | $2.00 | $6.00 | 128K |
| command-r-plus | Cohere | $2.50 | $10.00 | 128K |

---

## Development Guidelines

### Code Style

```python
# Type hints required, Pydantic validation, async/await
async def evaluate_model(
    task: TaskDefinition,
    model_id: str,
    input_data: str,
) -> EvaluationResult:
    """Evaluate a single model on a task.

    Args:
        task: Pydantic TaskDefinition with criteria/constraints
        model_id: Model ID (e.g., "anthropic/claude-sonnet-4.5")
        input_data: Input text to evaluate

    Returns:
        EvaluationResult with output, tokens, cost, status
    """
    async with OpenRouterClient(api_key) as client:
        response = await client.complete(model_id, prompt, timeout=60)
        return EvaluationResult(
            model_name=model_id,
            output=response.content,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost=calculate_cost(response),
            status="success"
        )
```

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests with coverage
pytest --cov=taskbench --cov-report=term-missing

# Format code
black src/ tests/
isort src/ tests/

# Type check
mypy src/
```

### Adding New Tasks

1. Copy `tasks/template.yaml` to `tasks/my_task.yaml`
2. Fill in task details (name, description, criteria, constraints)
3. Validate: `taskbench validate tasks/my_task.yaml`
4. Run: `taskbench evaluate tasks/my_task.yaml --models ... --input-file ...`

### Adding New Models

1. Edit `config/models.yaml`
2. Add entry with `model_id`, `display_name`, `provider`, pricing, `context_window`
3. Models are automatically available for evaluation

---

## Design Principles

1. **CLI-First**: Primary interface is command-line; UI is optional
2. **Declarative Configuration**: Tasks and use-cases defined in YAML
3. **Type Safety**: Pydantic models throughout with strict validation
4. **Async-First**: Non-blocking I/O for efficient API interactions
5. **Fair Comparison**: Same prompt for all models by default
6. **Cost Transparency**: Always track and display costs
7. **Fail Fast, Continue Smart**: Retry 3x, abandon model after failures, continue with others
8. **User Interpretation**: Framework provides data; users decide value trade-offs

---

## Architecture Patterns

| Pattern | Usage |
|---------|-------|
| **Async I/O** | Non-blocking API calls via httpx |
| **Repository** | CostTracker for pricing data |
| **Strategy** | Different comparison metrics |
| **Template Method** | Prompt building in Executor |
| **Decorator** | Retry logic with exponential backoff |
| **DTOs** | Pydantic models for data transfer |
| **Command** | CLI commands via Typer |

---

## Error Handling

| Error Type | Handling |
|------------|----------|
| **AuthenticationError** | Invalid API key (401) - fail immediately |
| **RateLimitError** | Rate limit (429) - retry with backoff |
| **BadRequestError** | Malformed request (400) - fail immediately |
| **OpenRouterError** | Server errors (5xx) - retry with backoff |
| **Validation Errors** | Pydantic validation with clear messages |

Retry logic: Exponential backoff (2^attempt seconds), max 3 retries for transient errors.

---

## File Locations

| Purpose | Location |
|---------|----------|
| CLI entry point | `src/taskbench/cli/main.py` |
| Core models | `src/taskbench/core/models.py` |
| Task parser | `src/taskbench/core/task.py` |
| API client | `src/taskbench/api/client.py` |
| Executor | `src/taskbench/evaluation/executor.py` |
| Judge | `src/taskbench/evaluation/judge.py` |
| Cost tracker | `src/taskbench/evaluation/cost.py` |
| Orchestrator | `src/taskbench/evaluation/orchestrator.py` |
| UI API | `src/taskbench/ui_api.py` |
| Streamlit UI | `ui_app.py` |
| Task definitions | `tasks/*.yaml` |
| Use-cases | `usecases/*.yaml` |
| Model pricing | `config/models.yaml` |
| Test fixtures | `tests/fixtures/` |
| Results output | `results/` |

---

## Post-MVP Features

See `docs/TODO.md` for detailed roadmap including:

- Per-model prompt optimization
- Advanced parallel execution strategies
- Robustness testing (input corruption)
- Deep failure pattern analysis
- Cross-task comparative benchmarking
- Batch evaluation (multi-input processing)
- Custom judge models
- Multi-modal support

---

## Quick Reference

### Install & Run

```bash
# Install
pip install -e .

# Set API key
export OPENROUTER_API_KEY=sk-or-...

# Run sample evaluation
taskbench sample --models anthropic/claude-sonnet-4.5,openai/gpt-4o --no-judge

# Run with UI
docker compose -f docker-compose.ui.yml up --build
```

### Common Workflows

```bash
# Evaluate with judge
taskbench evaluate tasks/lecture_analysis.yaml \
  --models anthropic/claude-sonnet-4.5,openai/gpt-4o \
  --input-file data/transcript.txt

# Evaluate without judge, re-judge later
taskbench evaluate tasks/lecture_analysis.yaml \
  --models auto \
  --input-file data/transcript.txt \
  --skip-judge \
  --output results/run.json

# Judge saved results
taskbench recommend --results results/run.json
```

---

## Questions Before Making Changes

1. Does this align with CLI-first architecture?
2. Will this increase complexity significantly?
3. Is this testable without real API calls? (mock externals)
4. Does this affect user costs? (document clearly)
5. Should this be MVP or `docs/TODO.md`?
6. Is there a simpler solution?

---

**Last Updated**: 2025-12-26
**Version**: 0.1.0
**Maintained By**: LLM TaskBench Contributors
