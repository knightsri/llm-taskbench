# LLM TaskBench Architecture

## Table of Contents

- [System Overview](#system-overview)
- [Architecture Diagram](#architecture-diagram)
- [Component Descriptions](#component-descriptions)
- [Data Flow](#data-flow)
- [Design Decisions](#design-decisions)
- [Technology Choices](#technology-choices)

---

## System Overview

LLM TaskBench is a task-specific LLM evaluation framework that enables domain experts to compare multiple LLMs on their actual use cases. The system is designed with modularity, extensibility, and ease of use in mind.

### Key Features

- **Task-Centric**: Define custom evaluation tasks using simple YAML files
- **Multi-Model Evaluation**: Compare multiple LLMs simultaneously
- **LLM-as-Judge**: Automated quality assessment using powerful LLM judges
- **Cost Tracking**: Transparent token usage and cost calculation
- **Rich CLI**: Beautiful command-line interface with progress bars and tables
- **Intelligent Recommendations**: Data-driven model selection guidance

### Design Principles

1. **Separation of Concerns**: Each component has a single, well-defined responsibility
2. **Async-First**: Built on async/await for efficient API calls
3. **Type Safety**: Comprehensive Pydantic models for data validation
4. **Extensibility**: Easy to add new models, tasks, and evaluation criteria
5. **Developer Experience**: Rich feedback, clear error messages, comprehensive logging

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          CLI Interface                               │
│                     (taskbench.cli.main)                            │
│  Commands: evaluate | models | validate | results | recommend       │
└────────────────────┬────────────────────────────────────────────────┘
                     │
                     ├──────────────────────────────────────────┐
                     │                                          │
         ┌───────────▼─────────────┐              ┌────────────▼──────────┐
         │   Core Components       │              │  Evaluation Engine    │
         │  ┌──────────────────┐   │              │  ┌─────────────────┐  │
         │  │ TaskDefinition   │   │              │  │ ModelExecutor   │  │
         │  │  (models.py)     │   │              │  │  (executor.py)  │  │
         │  └──────────────────┘   │              │  └────────┬────────┘  │
         │  ┌──────────────────┐   │              │           │           │
         │  │ TaskParser       │   │              │  ┌────────▼────────┐  │
         │  │  (task.py)       │   │              │  │ LLMJudge        │  │
         │  └──────────────────┘   │              │  │  (judge.py)     │  │
         └────────────────────────┘              │  └─────────────────┘  │
                     │                            │  ┌─────────────────┐  │
                     │                            │  │ CostTracker     │  │
                     │                            │  │  (cost.py)      │  │
         ┌───────────▼─────────────┐              │  └─────────────────┘  │
         │   API Client Layer      │              │  ┌─────────────────┐  │
         │  ┌──────────────────┐   │              │  │ Orchestrator    │  │
         │  │ OpenRouterClient │◄──┼──────────────┼──│ (orchestrator)  │  │
         │  │  (client.py)     │   │              │  └─────────────────┘  │
         │  └────────┬─────────┘   │              └───────────────────────┘
         │           │              │                          │
         │  ┌────────▼─────────┐   │              ┌───────────▼───────────┐
         │  │ RetryLogic       │   │              │  Analysis & Reporting │
         │  │  (retry.py)      │   │              │  ┌─────────────────┐  │
         │  └──────────────────┘   │              │  │ ModelComparison │  │
         └────────────────────────┘              │  │  (comparison.py)│  │
                     │                            │  └─────────────────┘  │
                     │                            │  ┌─────────────────┐  │
         ┌───────────▼─────────────┐              │  │ Recommendation  │  │
         │  External Services      │              │  │ Engine          │  │
         │  ┌──────────────────┐   │              │  │ (recommender)   │  │
         │  │ OpenRouter API   │   │              │  └─────────────────┘  │
         │  │  (openrouter.ai) │   │              └───────────────────────┘
         │  └──────────────────┘   │
         │  ┌──────────────────┐   │
         │  │ Model Providers  │   │
         │  │ (Anthropic, OpenAI,  │
         │  │  Google, etc.)   │   │
         │  └──────────────────┘   │
         └────────────────────────┘

Legend:
  ┌─────┐
  │ Box │  = Component/Module
  └─────┘
    │      = Dependency/Data Flow
    ▼      = Directional Flow
```

---

## Component Descriptions

### Core Components (`taskbench.core`)

#### 1. Data Models (`models.py`)

Pydantic-based type-safe data structures for the entire system.

**TaskDefinition**
- Represents a user-defined evaluation task
- Validates input types, output formats, and constraints
- Contains evaluation criteria and judge instructions
- Enforces schema compliance with field validators

**CompletionResponse**
- Captures LLM API responses
- Tracks token usage (input/output/total)
- Records latency metrics
- Includes timestamp for tracking

**EvaluationResult**
- Complete evaluation outcome for one model on one task
- Includes output, costs, tokens, latency
- Tracks success/failure status and error details
- Timestamped for historical analysis

**JudgeScore**
- Detailed scoring from LLM-as-judge evaluation
- Multi-dimensional scores: accuracy, format, compliance
- Overall weighted score (0-100)
- Violation tracking with detailed reasoning

**ModelConfig**
- Model pricing and metadata
- Calculates costs from token counts
- Context window limits
- Provider information

#### 2. Task Parser (`task.py`)

Handles loading and validation of task definitions from YAML files.

**Responsibilities:**
- Parse YAML task definitions
- Validate task structure and constraints
- Load multiple tasks from directories
- Save tasks back to YAML format
- Ensure logical consistency (e.g., min < max)

**Key Features:**
- Comprehensive error handling
- Detailed validation error messages
- Support for batch loading
- Type-safe conversion to TaskDefinition objects

---

### API Layer (`taskbench.api`)

#### 1. OpenRouter Client (`client.py`)

Async HTTP client for OpenRouter API with comprehensive error handling.

**Responsibilities:**
- Manage HTTP connections to OpenRouter
- Handle authentication
- Track token usage and latency
- Support JSON mode for structured outputs
- Parse API responses into typed objects

**Key Features:**
- Async/await for non-blocking operations
- Automatic header management (API key, referer, title)
- Detailed error messages with status codes
- Context manager support for resource cleanup
- Token extraction from API responses

**Error Handling:**
- 401: Authentication errors
- 403: Access forbidden
- 429: Rate limit exceeded
- 400: Bad request validation
- 5xx: Server errors (retryable)

#### 2. Retry Logic (`retry.py`)

Exponential backoff and rate limiting for robust API interactions.

**retry_with_backoff Decorator:**
- Automatically retries transient failures
- Exponential backoff: `delay = initial_delay * (base ^ attempt)`
- Configurable retry attempts and delays
- Only retries retryable errors (429, 5xx)
- Logs all retry attempts

**RateLimiter Class:**
- Token bucket algorithm implementation
- Prevents burst requests
- Configurable requests per minute
- Thread-safe async operations
- Graceful degradation under load

---

### Evaluation Engine (`taskbench.evaluation`)

#### 1. Model Executor (`executor.py`)

Orchestrates model execution and result collection.

**Responsibilities:**
- Build comprehensive prompts from tasks
- Execute tasks on multiple models
- Track costs and performance metrics
- Handle failures gracefully
- Display progress with Rich progress bars

**Prompt Building:**
- Includes task description and objectives
- Emphasizes constraints and requirements
- Provides examples when available
- Formats input data clearly
- Specifies expected output format

**Parallel Execution:**
- Sequential execution with progress tracking
- Handles partial failures (some models can fail)
- Aggregates results across models
- Cost tracking for all evaluations

#### 2. LLM Judge (`judge.py`)

Uses a powerful LLM to evaluate other models' outputs.

**Responsibilities:**
- Build detailed evaluation prompts
- Request structured JSON scores
- Parse and validate judge responses
- Categorize violations by type
- Generate violation summaries

**Scoring Dimensions:**
- **Accuracy** (0-100): Content correctness
- **Format** (0-100): Output format compliance
- **Compliance** (0-100): Constraint adherence
- **Overall** (0-100): Weighted combination

**Violation Categories:**
- Under minimum constraints
- Over maximum constraints
- Format errors
- Missing required fields
- Other violations

#### 3. Cost Tracker (`cost.py`)

Calculates and tracks API costs across evaluations.

**Responsibilities:**
- Load model pricing from configuration
- Calculate costs from token usage
- Track cumulative costs
- Provide cost breakdowns by model
- Generate formatted cost summaries

**Features:**
- Per-model cost statistics
- Total cost aggregation
- Average cost per evaluation
- Breakdown by input/output tokens
- Export capabilities for reporting

#### 4. Orchestrator (`orchestrator.py`)

Intelligent model selection based on task characteristics.

**Responsibilities:**
- Analyze task requirements
- Suggest appropriate models
- Apply budget constraints
- Estimate cost ranges
- Categorize models by use case

**Selection Heuristics:**
- **Input Type**: Transcript → large context models
- **Output Format**: JSON → structured output models
- **Budget**: Filter by estimated costs
- **Quality**: Premium vs. budget models

**Model Categories:**
- Default: General-purpose models
- Large Context: For long documents
- Budget: Cost-effective options
- Premium: Highest quality models

#### 5. Model Comparison (`comparison.py`)

Compares and ranks models based on evaluation results.

**Responsibilities:**
- Merge results with judge scores
- Calculate value ratings (score/cost)
- Rank models by performance
- Identify best overall and best value
- Generate comparison tables

**Metrics:**
- Overall score ranking
- Cost efficiency (value rating)
- Violation counts
- Success/failure rates
- Performance tiers

#### 6. Recommendation Engine (`recommender.py`)

Generates actionable recommendations from evaluation data.

**Responsibilities:**
- Classify models into performance tiers
- Identify best options for different use cases
- Provide specific recommendations
- Generate insights and comparisons
- Export recommendations as JSON

**Use Cases:**
- General Purpose: Best overall model
- Production: Highest quality for critical workloads
- Cost-Sensitive: Best value for money
- Budget: Cheapest acceptable option
- Development: Good for testing

**Performance Tiers:**
- Excellent: 90-100
- Good: 80-89
- Acceptable: 70-79
- Poor: 0-69

---

### CLI Interface (`taskbench.cli`)

Rich command-line interface built with Typer.

**Commands:**

1. **evaluate**: Run evaluations on models
   - Load task definitions
   - Execute on multiple models
   - Display results in tables
   - Save to JSON files

2. **models**: Show available models and pricing
   - List all configured models
   - Show detailed model information
   - Display pricing and context windows

3. **validate**: Validate task definition files
   - Check YAML syntax
   - Validate required fields
   - Check constraint consistency
   - Show task details

4. **results**: Display saved evaluation results
   - Load from JSON files
   - Format as table, JSON, or CSV
   - Export to files

5. **recommend**: Generate recommendations
   - Load results and scores
   - Apply budget filters
   - Generate comparison tables
   - Show use case recommendations

**Features:**
- Rich console output with colors
- Progress bars for long operations
- Beautiful tables for data display
- Comprehensive error messages
- Verbose logging option

---

### Utility Components (`taskbench.utils`)

#### Logging (`logging.py`)

Configurable logging setup for the entire application.

**Features:**
- Console and file logging
- Configurable log levels
- Formatted log messages
- Component-specific loggers

#### Validation (`validation.py`)

Additional validation utilities for complex checks.

**Features:**
- Task constraint validation
- Data format validation
- Custom validation rules

---

## Data Flow

### 1. Task Evaluation Flow

```
User Command
    │
    ▼
┌─────────────────────┐
│ Load Task YAML      │ ← TaskParser.load_from_yaml()
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Validate Task       │ ← TaskParser.validate_task()
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Load Input Data     │ ← Read from file or stdin
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Build Prompts       │ ← ModelExecutor.build_prompt()
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Execute Models      │ ← ModelExecutor.evaluate_multiple()
│  (for each model)   │
│  ┌───────────────┐  │
│  │ API Call      │  │ ← OpenRouterClient.complete()
│  └───────────────┘  │
│  ┌───────────────┐  │
│  │ Parse Response│  │ ← CompletionResponse
│  └───────────────┘  │
│  ┌───────────────┐  │
│  │ Calculate Cost│  │ ← CostTracker.calculate_cost()
│  └───────────────┘  │
│  ┌───────────────┐  │
│  │ Create Result │  │ ← EvaluationResult
│  └───────────────┘  │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Display Results     │ ← Rich tables and cost summary
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Save to JSON        │ ← Optional output file
└─────────────────────┘
```

### 2. LLM-as-Judge Flow

```
Evaluation Results
    │
    ▼
┌─────────────────────┐
│ For Each Result     │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Build Judge Prompt  │ ← LLMJudge.build_judge_prompt()
│  - Task criteria    │
│  - Model output     │
│  - Input data       │
│  - Scoring rubric   │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Call Judge Model    │ ← OpenRouterClient.complete_with_json()
│  (JSON mode)        │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Parse JSON Response │
│  - accuracy_score   │
│  - format_score     │
│  - compliance_score │
│  - overall_score    │
│  - violations       │
│  - reasoning        │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Create JudgeScore   │ ← Validated Pydantic model
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Aggregate Scores    │ ← All models scored
└─────────────────────┘
```

### 3. Recommendation Flow

```
Results + Scores
    │
    ▼
┌─────────────────────┐
│ Merge Data          │ ← ModelComparison.compare_results()
│  - Join by model ID │
│  - Calculate value  │
│  - Rank by score    │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Classify Tiers      │ ← RecommendationEngine.generate_recommendations()
│  - Excellent (90+)  │
│  - Good (80-89)     │
│  - Acceptable (70+) │
│  - Poor (<70)       │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Identify Best       │
│  - Best Overall     │
│  - Best Value       │
│  - Budget Option    │
│  - Premium Option   │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Generate Use Cases  │
│  - Production       │
│  - Cost-Sensitive   │
│  - Budget           │
│  - Development      │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Format Output       │ ← Rich panels and tables
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Display & Export    │ ← Console + optional JSON
└─────────────────────┘
```

---

## Design Decisions

### 1. YAML for Task Definitions

**Decision**: Use YAML files for task definitions instead of JSON or Python code.

**Rationale:**
- Human-readable and easy to edit
- Supports comments for documentation
- No programming knowledge required
- Clear structure for nested data
- Standard format for configuration

**Trade-offs:**
- Slightly more parsing overhead than JSON
- Need for validation layer
- Less programmatic than Python configs

### 2. Pydantic for Data Validation

**Decision**: Use Pydantic models for all data structures.

**Rationale:**
- Automatic validation at runtime
- Clear error messages for invalid data
- Type safety throughout the codebase
- Easy serialization/deserialization
- Self-documenting code

**Benefits:**
- Catches errors early
- Reduces boilerplate validation code
- Excellent IDE support
- Built-in JSON schema generation

### 3. OpenRouter as Primary API

**Decision**: Use OpenRouter as the unified API gateway instead of direct provider APIs.

**Rationale:**
- Single API for multiple providers
- Unified authentication and billing
- Consistent request/response format
- Automatic fallback and routing
- No need for multiple API keys

**Trade-offs:**
- Extra hop in the network path
- Dependency on third-party service
- Potential slight cost overhead

**Mitigation:**
- Design allows easy swapping to direct APIs
- Client abstraction enables different implementations

### 4. LLM-as-Judge Evaluation

**Decision**: Use LLM-based evaluation instead of rule-based or human evaluation.

**Rationale:**
- Scalable to any task type
- Nuanced understanding of quality
- Can assess semantic correctness
- No need for manual labeling
- Consistent evaluation criteria

**Implementation:**
- Structured JSON output for parsing
- Multi-dimensional scoring
- Detailed reasoning provided
- Violation tracking

**Limitations:**
- Judge model quality matters
- Can be expensive for large volumes
- Potential bias from judge model

### 5. Async/Await Architecture

**Decision**: Build on asyncio for API interactions.

**Rationale:**
- Efficient handling of I/O-bound operations
- Better resource utilization
- Supports concurrent requests
- Modern Python best practice
- Non-blocking progress updates

**Complexity:**
- Requires async/await throughout
- More complex error handling
- Testing requires async support

### 6. CLI-First Interface

**Decision**: Start with CLI instead of GUI or web interface.

**Rationale:**
- Easier to automate and script
- Works in any environment
- No frontend dependencies
- Better for CI/CD integration
- Rich library provides beautiful output

**Future Extensions:**
- Web UI could be added later
- API server for programmatic access
- Library usage in Python code

### 7. Modular Component Architecture

**Decision**: Separate concerns into distinct modules with clear responsibilities.

**Rationale:**
- Easy to test components in isolation
- Simple to extend or replace components
- Clear separation of concerns
- Maintainable codebase
- Reusable components

**Structure:**
- Core: Data models and parsing
- API: External communication
- Evaluation: Execution and judging
- CLI: User interface
- Utils: Shared utilities

---

## Technology Choices

### Python 3.11+

**Why:**
- Industry standard for AI/ML
- Excellent library ecosystem
- Type hints for better code quality
- Async/await support
- Good performance for I/O-bound tasks

### Pydantic v2

**Why:**
- Best-in-class data validation
- Excellent performance
- Type safety
- JSON schema generation
- Wide adoption

**Alternatives Considered:**
- Dataclasses: Less validation
- Attrs: Less integrated with type system
- Manual validation: Too much boilerplate

### HTTPX

**Why:**
- Modern async HTTP client
- Better than requests for async
- HTTP/2 support
- Connection pooling
- Timeout handling

**Alternatives Considered:**
- aiohttp: More complex API
- requests: No async support

### Typer

**Why:**
- Built on Click
- Automatic help generation
- Type hints for arguments
- Great error messages
- Easy to use

**Alternatives Considered:**
- Click: More verbose
- argparse: Less user-friendly
- Fire: Less control

### Rich

**Why:**
- Beautiful terminal output
- Progress bars and tables
- Color and styling
- Panels and layouts
- Markdown rendering

**Alternatives Considered:**
- Click rich markup: Less features
- Colorama: Too basic
- tqdm: Only progress bars

### PyYAML

**Why:**
- Standard YAML library
- Simple API
- Good performance
- Wide compatibility

**Alternatives Considered:**
- ruamel.yaml: More complex
- strictyaml: Too restrictive

### pytest

**Why:**
- Industry standard
- Fixtures for setup/teardown
- Async test support
- Coverage integration
- Excellent plugins

**Alternatives Considered:**
- unittest: More verbose
- nose: Deprecated

### Project Structure

```
llm-taskbench/
├── src/taskbench/           # Source code
│   ├── core/                # Core data models and parsing
│   ├── api/                 # API client and retry logic
│   ├── evaluation/          # Execution, judging, cost tracking
│   ├── cli/                 # Command-line interface
│   └── utils/               # Shared utilities
├── tests/                   # Test suite
├── config/                  # Configuration files
│   └── models.yaml          # Model pricing configuration
├── tasks/                   # Task definition templates
├── docs/                    # Documentation
├── examples/                # Example usage and scripts
├── pyproject.toml          # Project metadata and dependencies
└── requirements.txt        # Pinned dependencies
```

---

## Future Architectural Considerations

### Scalability

**Current State:**
- Sequential model execution
- Single-threaded CLI
- In-memory result storage

**Future Enhancements:**
- Parallel model execution with semaphores
- Database for result storage
- Caching layer for repeated evaluations
- Queue-based processing for large batches

### Extensibility Points

1. **Custom Evaluators**: Plugin system for evaluation methods
2. **Additional APIs**: Support for direct provider APIs
3. **Custom Metrics**: User-defined scoring dimensions
4. **Export Formats**: Additional output formats (Excel, Parquet)
5. **Visualization**: Charts and graphs for results

### Monitoring and Observability

**Future Additions:**
- Structured logging for analysis
- Metrics collection (Prometheus)
- Distributed tracing (OpenTelemetry)
- Cost alerts and budgets
- Performance dashboards

---

## Conclusion

LLM TaskBench is architected for:
- **Simplicity**: Easy to understand and use
- **Reliability**: Robust error handling and retries
- **Extensibility**: Easy to add new features
- **Performance**: Async operations for efficiency
- **Developer Experience**: Rich feedback and clear errors

The architecture balances immediate usability with long-term maintainability, providing a solid foundation for task-specific LLM evaluation.
