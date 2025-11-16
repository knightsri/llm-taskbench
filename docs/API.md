# LLM TaskBench API Reference

Complete API reference for all public classes and methods in LLM TaskBench.

## Table of Contents

- [Core Models](#core-models)
  - [TaskDefinition](#taskdefinition)
  - [CompletionResponse](#completionresponse)
  - [EvaluationResult](#evaluationresult)
  - [JudgeScore](#judgescore)
  - [ModelConfig](#modelconfig)
- [Task Management](#task-management)
  - [TaskParser](#taskparser)
- [API Client](#api-client)
  - [OpenRouterClient](#openrouterclient)
  - [OpenRouterAPIError](#openrouterapierror)
- [Retry Logic](#retry-logic)
  - [retry_with_backoff](#retry_with_backoff)
  - [RateLimiter](#ratelimiter)
- [Evaluation Engine](#evaluation-engine)
  - [ModelExecutor](#modelexecutor)
  - [LLMJudge](#llmjudge)
  - [CostTracker](#costtracker)
  - [LLMOrchestrator](#llmorchestrator)
- [Analysis and Reporting](#analysis-and-reporting)
  - [ModelComparison](#modelcomparison)
  - [RecommendationEngine](#recommendationengine)

---

## Core Models

Located in `taskbench.core.models`

### TaskDefinition

Represents a user-defined evaluation task with validation.

```python
from taskbench.core.models import TaskDefinition

task = TaskDefinition(
    name="lecture_analysis",
    description="Extract teaching concepts from lecture transcripts",
    input_type="transcript",
    output_format="csv",
    evaluation_criteria=["Accuracy", "Format compliance"],
    constraints={"min_duration_minutes": 2, "max_duration_minutes": 7},
    examples=[],
    judge_instructions="Evaluate based on accuracy and format"
)
```

#### Constructor Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | `str` | Yes | Unique name for this task |
| `description` | `str` | Yes | Human-readable description |
| `input_type` | `str` | Yes | Type of input: "transcript", "text", "csv", "json" |
| `output_format` | `str` | Yes | Expected output: "csv", "json", "markdown" |
| `evaluation_criteria` | `List[str]` | Yes | List of criteria for evaluation |
| `constraints` | `Dict[str, Any]` | No | Task-specific constraints (default: {}) |
| `examples` | `List[Dict[str, Any]]` | No | Example inputs/outputs (default: []) |
| `judge_instructions` | `str` | Yes | Instructions for the LLM judge |

#### Validation Rules

- `input_type` must be one of: `["transcript", "text", "csv", "json"]`
- `output_format` must be one of: `["csv", "json", "markdown"]`
- All required fields must be non-empty

#### Methods

None - this is a data model.

#### Example

```python
task = TaskDefinition(
    name="sentiment_analysis",
    description="Classify customer feedback as positive, negative, or neutral",
    input_type="text",
    output_format="json",
    evaluation_criteria=[
        "Correct sentiment classification",
        "Confidence score provided",
        "JSON format compliance"
    ],
    constraints={
        "required_fields": ["sentiment", "confidence"],
        "valid_sentiments": ["positive", "negative", "neutral"]
    },
    examples=[
        {
            "input": "This product is amazing!",
            "expected_output": '{"sentiment": "positive", "confidence": 0.95}',
            "notes": "Clear positive sentiment"
        }
    ],
    judge_instructions="Check sentiment accuracy and JSON format"
)
```

---

### CompletionResponse

API response from LLM completion with token tracking.

```python
from taskbench.core.models import CompletionResponse

response = CompletionResponse(
    content="The analysis results...",
    model="anthropic/claude-sonnet-4.5",
    input_tokens=1000,
    output_tokens=500,
    total_tokens=1500,
    latency_ms=2500
)
```

#### Constructor Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `content` | `str` | Yes | The model's generated text response |
| `model` | `str` | Yes | Model identifier used for generation |
| `input_tokens` | `int` | Yes | Number of input tokens consumed |
| `output_tokens` | `int` | Yes | Number of output tokens generated |
| `total_tokens` | `int` | Yes | Total tokens (input + output) |
| `latency_ms` | `float` | Yes | Response latency in milliseconds |
| `timestamp` | `datetime` | No | When response received (default: now) |

#### Methods

None - this is a data model.

---

### EvaluationResult

Complete evaluation result for one model on one task.

```python
from taskbench.core.models import EvaluationResult

result = EvaluationResult(
    model_name="anthropic/claude-sonnet-4.5",
    task_name="lecture_analysis",
    output="concept,start_time,end_time\nIntro,00:00:00,00:03:15",
    input_tokens=1000,
    output_tokens=500,
    total_tokens=1500,
    cost_usd=0.36,
    latency_ms=2500,
    status="success"
)
```

#### Constructor Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model_name` | `str` | Yes | Name/ID of the model evaluated |
| `task_name` | `str` | Yes | Name of the task |
| `output` | `str` | Yes | The model's generated output |
| `input_tokens` | `int` | Yes | Number of input tokens used |
| `output_tokens` | `int` | Yes | Number of output tokens generated |
| `total_tokens` | `int` | Yes | Total tokens consumed |
| `cost_usd` | `float` | Yes | Cost in USD |
| `latency_ms` | `float` | Yes | Time taken in milliseconds |
| `timestamp` | `datetime` | No | When evaluation performed (default: now) |
| `status` | `str` | No | Status: "success", "failed", "timeout" (default: "success") |
| `error` | `str` | No | Error message if failed (default: None) |

#### Methods

None - this is a data model.

---

### JudgeScore

LLM-as-judge scoring result with detailed breakdowns.

```python
from taskbench.core.models import JudgeScore

score = JudgeScore(
    model_evaluated="anthropic/claude-sonnet-4.5",
    accuracy_score=95,
    format_score=100,
    compliance_score=98,
    overall_score=97,
    violations=[],
    reasoning="Excellent performance with accurate concept extraction"
)
```

#### Constructor Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model_evaluated` | `str` | Yes | Name of the model being judged |
| `accuracy_score` | `int` | Yes | Accuracy score (0-100) |
| `format_score` | `int` | Yes | Format compliance score (0-100) |
| `compliance_score` | `int` | Yes | Constraint compliance score (0-100) |
| `overall_score` | `int` | Yes | Overall weighted score (0-100) |
| `violations` | `List[str]` | No | List of constraint violations (default: []) |
| `reasoning` | `str` | Yes | Detailed reasoning for scores |
| `timestamp` | `datetime` | No | When judging performed (default: now) |

#### Validation Rules

- All scores must be integers between 0 and 100 (inclusive)

#### Methods

None - this is a data model.

---

### ModelConfig

Model pricing and configuration information.

```python
from taskbench.core.models import ModelConfig

config = ModelConfig(
    model_id="anthropic/claude-sonnet-4.5",
    display_name="Claude Sonnet 4.5",
    input_price_per_1m=3.00,
    output_price_per_1m=15.00,
    context_window=200000,
    provider="Anthropic"
)
```

#### Constructor Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model_id` | `str` | Yes | Unique identifier for the model |
| `display_name` | `str` | Yes | Human-readable name |
| `input_price_per_1m` | `float` | Yes | Price per 1M input tokens (USD) |
| `output_price_per_1m` | `float` | Yes | Price per 1M output tokens (USD) |
| `context_window` | `int` | Yes | Maximum context window in tokens |
| `provider` | `str` | Yes | Model provider (e.g., "Anthropic") |

#### Methods

##### `calculate_cost(input_tokens: int, output_tokens: int) -> float`

Calculate the cost for given token usage.

**Parameters:**
- `input_tokens` (`int`): Number of input tokens
- `output_tokens` (`int`): Number of output tokens

**Returns:**
- `float`: Total cost in USD, rounded to 2 decimal places

**Example:**
```python
config = ModelConfig(...)
cost = config.calculate_cost(1000, 500)
print(f"Cost: ${cost:.2f}")  # Cost: $0.01
```

---

## Task Management

Located in `taskbench.core.task`

### TaskParser

Parser for task definition YAML files with validation.

```python
from taskbench.core.task import TaskParser

parser = TaskParser()
task = parser.load_from_yaml("tasks/my_task.yaml")
```

#### Constructor Parameters

None - TaskParser uses static methods.

#### Methods

##### `load_from_yaml(yaml_path: str) -> TaskDefinition`

Load a task definition from a YAML file.

**Parameters:**
- `yaml_path` (`str`): Path to the YAML file

**Returns:**
- `TaskDefinition`: Validated task definition object

**Raises:**
- `FileNotFoundError`: If the YAML file doesn't exist
- `ValueError`: If the YAML is malformed or invalid
- `yaml.YAMLError`: If the YAML cannot be parsed

**Example:**
```python
parser = TaskParser()
try:
    task = parser.load_from_yaml("tasks/sentiment.yaml")
    print(f"Loaded task: {task.name}")
except FileNotFoundError:
    print("Task file not found")
except ValueError as e:
    print(f"Invalid task: {e}")
```

##### `validate_task(task: TaskDefinition) -> Tuple[bool, List[str]]`

Validate a task definition for logical consistency.

**Parameters:**
- `task` (`TaskDefinition`): Task to validate

**Returns:**
- `Tuple[bool, List[str]]`: (is_valid, list_of_errors)

**Example:**
```python
task = TaskDefinition(...)
is_valid, errors = TaskParser.validate_task(task)
if not is_valid:
    for error in errors:
        print(f"Error: {error}")
```

##### `save_to_yaml(task: TaskDefinition, yaml_path: str) -> None`

Save a task definition to a YAML file.

**Parameters:**
- `task` (`TaskDefinition`): Task to save
- `yaml_path` (`str`): Path where to save the YAML

**Raises:**
- `IOError`: If the file cannot be written

**Example:**
```python
task = TaskDefinition(...)
TaskParser.save_to_yaml(task, "tasks/new_task.yaml")
```

##### `load_all_from_directory(directory: str) -> List[TaskDefinition]`

Load all task definitions from a directory.

**Parameters:**
- `directory` (`str`): Path to directory containing YAML files

**Returns:**
- `List[TaskDefinition]`: List of loaded tasks

**Example:**
```python
parser = TaskParser()
tasks = parser.load_all_from_directory("tasks/")
print(f"Loaded {len(tasks)} tasks")
```

---

## API Client

Located in `taskbench.api.client`

### OpenRouterClient

Async HTTP client for OpenRouter API with error handling.

```python
from taskbench.api.client import OpenRouterClient

async with OpenRouterClient() as client:
    response = await client.complete(
        model="anthropic/claude-sonnet-4.5",
        prompt="Explain quantum computing"
    )
    print(response.content)
```

#### Constructor Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `api_key` | `str` | No | OpenRouter API key (reads from env if not provided) |
| `timeout` | `float` | No | Request timeout in seconds (default: 120.0) |
| `app_name` | `str` | No | App name for headers (default: "LLM-TaskBench") |
| `site_url` | `str` | No | Site URL for headers |

#### Methods

##### `async complete(model: str, prompt: str, temperature: float = 0.7, max_tokens: int = None, **kwargs) -> CompletionResponse`

Send a completion request to OpenRouter API.

**Parameters:**
- `model` (`str`): Model identifier (e.g., "anthropic/claude-sonnet-4.5")
- `prompt` (`str`): The prompt text
- `temperature` (`float`, optional): Sampling temperature 0.0-2.0 (default: 0.7)
- `max_tokens` (`int`, optional): Maximum tokens to generate
- `**kwargs`: Additional parameters for the API

**Returns:**
- `CompletionResponse`: Model response with metadata

**Raises:**
- `OpenRouterAPIError`: If the API request fails

**Example:**
```python
async with OpenRouterClient() as client:
    response = await client.complete(
        model="anthropic/claude-sonnet-4.5",
        prompt="What is Python?",
        temperature=0.5,
        max_tokens=500
    )
    print(f"Response: {response.content}")
    print(f"Tokens: {response.total_tokens}")
    print(f"Latency: {response.latency_ms}ms")
```

##### `async complete_with_json(model: str, prompt: str, temperature: float = 0.7, max_tokens: int = None, **kwargs) -> CompletionResponse`

Send a completion request with JSON mode enabled.

**Parameters:**
- Same as `complete()`

**Returns:**
- `CompletionResponse`: Model response (content should be valid JSON)

**Raises:**
- `OpenRouterAPIError`: If the API request fails

**Note:** Not all models support JSON mode.

**Example:**
```python
async with OpenRouterClient() as client:
    response = await client.complete_with_json(
        model="openai/gpt-4o",
        prompt="Return a JSON object with name and age fields"
    )
    import json
    data = json.loads(response.content)
```

##### `async close() -> None`

Close the HTTP client and cleanup resources.

**Example:**
```python
client = OpenRouterClient()
# ... use client ...
await client.close()
```

---

### OpenRouterAPIError

Exception class for OpenRouter API errors.

```python
from taskbench.api.client import OpenRouterAPIError

try:
    response = await client.complete(...)
except OpenRouterAPIError as e:
    print(f"API error: {e}")
    print(f"Status code: {e.status_code}")
    print(f"Response: {e.response_body}")
```

#### Constructor Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `message` | `str` | Yes | Error message |
| `status_code` | `int` | No | HTTP status code |
| `response_body` | `str` | No | API response body |

#### Attributes

- `status_code` (`int | None`): HTTP status code if available
- `response_body` (`str | None`): Response body if available

---

## Retry Logic

Located in `taskbench.api.retry`

### retry_with_backoff

Decorator that adds exponential backoff retry logic to async functions.

```python
from taskbench.api.retry import retry_with_backoff

@retry_with_backoff(max_retries=5, initial_delay=2.0)
async def call_api():
    return await client.complete(model="...", prompt="...")

result = await call_api()  # Will retry on transient errors
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_retries` | `int` | 3 | Maximum number of retry attempts |
| `initial_delay` | `float` | 1.0 | Initial delay before first retry (seconds) |
| `max_delay` | `float` | 60.0 | Maximum delay between retries (seconds) |
| `exponential_base` | `float` | 2.0 | Base for exponential backoff |
| `retryable_status_codes` | `Set[int]` | {429, 500, 502, 503, 504} | HTTP codes to retry |

#### Retry Behavior

- Only retries on transient errors (rate limits, server errors)
- Does NOT retry on client errors (401, 403, 400)
- Uses exponential backoff: `delay = min(initial_delay * (base ^ attempt), max_delay)`
- Logs all retry attempts

#### Example

```python
from taskbench.api.retry import retry_with_backoff

@retry_with_backoff(max_retries=5, initial_delay=2.0, max_delay=30.0)
async def fetch_data():
    async with OpenRouterClient() as client:
        return await client.complete(
            model="anthropic/claude-sonnet-4.5",
            prompt="Process this data..."
        )

# Will automatically retry on 429, 500, 502, 503, 504 errors
result = await fetch_data()
```

---

### RateLimiter

Token bucket rate limiter for controlling API request rates.

```python
from taskbench.api.retry import RateLimiter

limiter = RateLimiter(requests_per_minute=60)

async def make_requests():
    for i in range(100):
        await limiter.acquire()  # Wait if necessary
        result = await api_call()
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `requests_per_minute` | `int` | 60 | Maximum requests allowed per minute |
| `burst_size` | `int` | None | Max burst size (default: same as requests_per_minute) |

#### Methods

##### `async acquire(tokens: int = 1) -> None`

Acquire tokens from the bucket, waiting if necessary.

**Parameters:**
- `tokens` (`int`): Number of tokens to acquire (default: 1)

**Blocks until:**
- Requested tokens are available

**Example:**
```python
limiter = RateLimiter(requests_per_minute=60)
await limiter.acquire()  # Wait for 1 token
# Make API call
```

##### `async try_acquire(tokens: int = 1) -> bool`

Try to acquire tokens without waiting.

**Parameters:**
- `tokens` (`int`): Number of tokens to acquire (default: 1)

**Returns:**
- `bool`: True if tokens acquired, False otherwise

**Example:**
```python
limiter = RateLimiter(requests_per_minute=60)
if await limiter.try_acquire():
    # Make API call
else:
    # Handle rate limit
    print("Rate limited")
```

##### `get_available_tokens() -> float`

Get the current number of available tokens.

**Returns:**
- `float`: Number of tokens currently available

##### `reset() -> None`

Reset the rate limiter to full capacity.

---

## Evaluation Engine

Located in `taskbench.evaluation`

### ModelExecutor

Execute evaluation tasks on multiple LLM models.

```python
from taskbench.evaluation.executor import ModelExecutor

executor = ModelExecutor()
result = await executor.execute("anthropic/claude-sonnet-4.5", task, input_data)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|----------|-------------|
| `api_key` | `str` | None | OpenRouter API key (reads from env if not provided) |
| `cost_tracker` | `CostTracker` | None | Cost tracker instance (creates new if not provided) |
| `timeout` | `float` | 120.0 | Request timeout in seconds |

#### Methods

##### `build_prompt(task: TaskDefinition, input_data: str) -> str`

Build a comprehensive prompt from task and input data.

**Parameters:**
- `task` (`TaskDefinition`): Task specifications
- `input_data` (`str`): Input data to process

**Returns:**
- `str`: Formatted prompt ready for the model

**Example:**
```python
executor = ModelExecutor()
prompt = executor.build_prompt(task, "Input text...")
print(prompt)
```

##### `async execute(model_id: str, task: TaskDefinition, input_data: str) -> EvaluationResult`

Execute a task on a single model.

**Parameters:**
- `model_id` (`str`): Model identifier
- `task` (`TaskDefinition`): Task to execute
- `input_data` (`str`): Input data

**Returns:**
- `EvaluationResult`: Result with output, tokens, cost, status

**Example:**
```python
executor = ModelExecutor()
result = await executor.execute(
    "anthropic/claude-sonnet-4.5",
    task,
    "Lecture transcript..."
)
if result.status == "success":
    print(f"Output: {result.output}")
    print(f"Cost: ${result.cost_usd:.4f}")
```

##### `async evaluate_multiple(model_ids: List[str], task: TaskDefinition, input_data: str, show_progress: bool = True) -> List[EvaluationResult]`

Execute a task on multiple models sequentially.

**Parameters:**
- `model_ids` (`List[str]`): List of model identifiers
- `task` (`TaskDefinition`): Task to execute
- `input_data` (`str`): Input data
- `show_progress` (`bool`): Show progress bar (default: True)

**Returns:**
- `List[EvaluationResult]`: Results for all models

**Example:**
```python
executor = ModelExecutor()
models = [
    "anthropic/claude-sonnet-4.5",
    "openai/gpt-4o",
    "google/gemini-2.0-flash-exp"
]
results = await executor.evaluate_multiple(models, task, input_data)
for result in results:
    print(f"{result.model_name}: ${result.cost_usd:.4f}")
```

##### `get_cost_summary() -> str`

Get a formatted cost summary.

**Returns:**
- `str`: Formatted cost statistics

##### `reset_tracker() -> None`

Reset the cost tracker statistics.

---

### LLMJudge

LLM-as-Judge evaluator for scoring model outputs.

```python
from taskbench.evaluation.judge import LLMJudge
from taskbench.api.client import OpenRouterClient

client = OpenRouterClient()
judge = LLMJudge(client, judge_model="anthropic/claude-sonnet-4.5")
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|----------|-------------|
| `api_client` | `OpenRouterClient` | Required | OpenRouter client instance |
| `judge_model` | `str` | "anthropic/claude-sonnet-4.5" | Model to use for judging |

#### Methods

##### `build_judge_prompt(task: TaskDefinition, model_output: str, input_data: str) -> str`

Build a detailed evaluation prompt for the judge model.

**Parameters:**
- `task` (`TaskDefinition`): Task with evaluation criteria
- `model_output` (`str`): Output to evaluate
- `input_data` (`str`): Original input for context

**Returns:**
- `str`: Formatted judge prompt

##### `async evaluate(task: TaskDefinition, result: EvaluationResult, input_data: str) -> JudgeScore`

Evaluate a model's output using the judge model.

**Parameters:**
- `task` (`TaskDefinition`): Task with criteria
- `result` (`EvaluationResult`): Result to evaluate
- `input_data` (`str`): Original input

**Returns:**
- `JudgeScore`: Detailed scoring and violations

**Raises:**
- `OpenRouterAPIError`: If API call fails
- `ValueError`: If response cannot be parsed

**Example:**
```python
judge = LLMJudge(client)
score = await judge.evaluate(task, result, input_data)
print(f"Overall: {score.overall_score}")
print(f"Accuracy: {score.accuracy_score}")
print(f"Format: {score.format_score}")
print(f"Compliance: {score.compliance_score}")
print(f"Violations: {score.violations}")
```

##### `parse_violations(violations: List[str]) -> Dict[str, List[str]]`

Categorize violations by type.

**Parameters:**
- `violations` (`List[str]`): List of violation strings

**Returns:**
- `Dict[str, List[str]]`: Violations categorized by type

**Categories:**
- `under_min`: Below minimum constraints
- `over_max`: Exceeding maximum constraints
- `format`: Format-related violations
- `missing_field`: Missing required fields
- `other`: Other violations

##### `count_violations_by_type(violations: List[str]) -> Dict[str, int]`

Count violations by category.

**Parameters:**
- `violations` (`List[str]`): List of violation strings

**Returns:**
- `Dict[str, int]`: Count of violations per type

##### `get_violation_summary(scores: List[JudgeScore]) -> str`

Generate a text summary of violations across models.

**Parameters:**
- `scores` (`List[JudgeScore]`): Scores to analyze

**Returns:**
- `str`: Formatted summary

---

### CostTracker

Track and calculate costs for LLM API usage.

```python
from taskbench.evaluation.cost import CostTracker

tracker = CostTracker()
cost = tracker.calculate_cost("anthropic/claude-sonnet-4.5", 1000, 500)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|----------|-------------|
| `models_config_path` | `str` | None | Path to models.yaml (default: config/models.yaml) |

#### Methods

##### `get_model_config(model_id: str) -> ModelConfig | None`

Get the configuration for a specific model.

**Parameters:**
- `model_id` (`str`): Model identifier

**Returns:**
- `ModelConfig | None`: Config if found, None otherwise

##### `calculate_cost(model_id: str, input_tokens: int, output_tokens: int) -> float`

Calculate cost for given token usage.

**Parameters:**
- `model_id` (`str`): Model identifier
- `input_tokens` (`int`): Input tokens
- `output_tokens` (`int`): Output tokens

**Returns:**
- `float`: Cost in USD

**Raises:**
- `ValueError`: If model not found

**Example:**
```python
tracker = CostTracker()
cost = tracker.calculate_cost("anthropic/claude-sonnet-4.5", 1000, 500)
print(f"Cost: ${cost:.2f}")
```

##### `track_evaluation(evaluation: EvaluationResult) -> None`

Track an evaluation and update statistics.

**Parameters:**
- `evaluation` (`EvaluationResult`): Result to track

##### `get_total_cost() -> float`

Get total cost of all tracked evaluations.

**Returns:**
- `float`: Total cost in USD

##### `get_cost_breakdown() -> Dict[str, Dict[str, float]]`

Get cost breakdown by model.

**Returns:**
```python
{
    "model_name": {
        "cost": total_cost,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "evaluations": count
    }
}
```

##### `get_statistics() -> Dict[str, Any]`

Get comprehensive cost statistics.

**Returns:**
- Dictionary with total_cost, total_evaluations, averages, breakdown

##### `reset() -> None`

Reset all tracked evaluations and statistics.

##### `export_summary() -> str`

Export a formatted summary of cost statistics.

**Returns:**
- `str`: Formatted summary

---

### LLMOrchestrator

Intelligent model selection based on task characteristics.

```python
from taskbench.evaluation.orchestrator import LLMOrchestrator
from taskbench.api.client import OpenRouterClient

client = OpenRouterClient()
orchestrator = LLMOrchestrator(client)
```

#### Constructor Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `api_client` | `OpenRouterClient` | Client for API calls |

#### Methods

##### `async create_evaluation_plan(task: TaskDefinition, budget: float = None) -> List[str]`

Create an evaluation plan by suggesting appropriate models.

**Parameters:**
- `task` (`TaskDefinition`): Task specifications
- `budget` (`float`, optional): Maximum budget per evaluation (USD)

**Returns:**
- `List[str]`: Recommended model IDs

**Example:**
```python
orchestrator = LLMOrchestrator(client)
models = await orchestrator.create_evaluation_plan(task, budget=0.10)
print(f"Recommended: {models}")
```

##### `get_model_category(model_id: str) -> str`

Categorize a model by its characteristics.

**Parameters:**
- `model_id` (`str`): Model identifier

**Returns:**
- `str`: Category: "premium", "budget", "large_context", or "default"

##### `estimate_cost_range(model_ids: List[str], estimated_input_tokens: int = 2000, estimated_output_tokens: int = 500) -> Dict[str, float]`

Estimate cost range for a list of models.

**Parameters:**
- `model_ids` (`List[str]`): Model identifiers
- `estimated_input_tokens` (`int`): Estimated input (default: 2000)
- `estimated_output_tokens` (`int`): Estimated output (default: 500)

**Returns:**
```python
{
    "min": minimum_cost,
    "max": maximum_cost,
    "average": average_cost
}
```

---

## Analysis and Reporting

Located in `taskbench.evaluation`

### ModelComparison

Compare and analyze evaluation results across models.

```python
from taskbench.evaluation.comparison import ModelComparison

comparison = ModelComparison()
compared = comparison.compare_results(results, scores)
```

#### Methods

##### `compare_results(results: List[EvaluationResult], scores: List[JudgeScore]) -> Dict[str, Any]`

Combine results with scores and calculate rankings.

**Parameters:**
- `results` (`List[EvaluationResult]`): Evaluation results
- `scores` (`List[JudgeScore]`): Judge scores

**Returns:**
```python
{
    "models": List[Dict],  # Combined data, ranked by score
    "total_models": int,
    "successful_models": int,
    "failed_models": int
}
```

**Example:**
```python
comparison = ModelComparison()
compared = comparison.compare_results(results, scores)
for model in compared['models']:
    print(f"{model['name']}: {model['overall_score']}")
```

##### `identify_best(comparison: Dict[str, Any]) -> str`

Identify the model with highest overall score.

**Parameters:**
- `comparison` (`Dict`): Output from compare_results()

**Returns:**
- `str`: Name of best model

##### `identify_best_value(comparison: Dict[str, Any], max_cost: float = None) -> str`

Identify model with best score-to-cost ratio.

**Parameters:**
- `comparison` (`Dict`): Output from compare_results()
- `max_cost` (`float`, optional): Maximum cost filter

**Returns:**
- `str`: Name of best value model

##### `generate_comparison_table(comparison: Dict[str, Any]) -> str`

Generate a beautiful comparison table.

**Parameters:**
- `comparison` (`Dict`): Output from compare_results()

**Returns:**
- `str`: Formatted table

##### `get_summary_statistics(comparison: Dict[str, Any]) -> Dict[str, Any]`

Calculate summary statistics.

**Returns:**
```python
{
    "total_models": int,
    "successful_models": int,
    "failed_models": int,
    "average_score": float,
    "average_cost": float,
    "total_cost": float,
    "best_score": int,
    "worst_score": int,
    "best_value": float
}
```

---

### RecommendationEngine

Generate intelligent recommendations from evaluation results.

```python
from taskbench.evaluation.recommender import RecommendationEngine

engine = RecommendationEngine()
recommendations = engine.generate_recommendations(comparison_data)
```

#### Methods

##### `generate_recommendations(comparison: Dict[str, Any]) -> Dict[str, Any]`

Generate comprehensive recommendations.

**Parameters:**
- `comparison` (`Dict`): Output from ModelComparison.compare_results()

**Returns:**
```python
{
    "tiers": {
        "Excellent": List[Dict],  # Models scoring 90+
        "Good": List[Dict],       # Models scoring 80-89
        "Acceptable": List[Dict], # Models scoring 70-79
        "Poor": List[Dict]        # Models scoring < 70
    },
    "best_overall": Dict,       # Highest scoring model
    "best_value": Dict,         # Best score/cost ratio
    "budget_option": Dict,      # Cheapest acceptable model
    "premium_option": Dict,     # Same as best_overall
    "recommendations": {
        "general": str,
        "production": str,
        "cost_sensitive": str,
        "budget": str,
        "development": str
    }
}
```

**Example:**
```python
engine = RecommendationEngine()
recs = engine.generate_recommendations(compared)
print(f"Best overall: {recs['best_overall']['name']}")
print(f"Best value: {recs['best_value']['name']}")
print(f"Production: {recs['recommendations']['production']}")
```

##### `format_recommendations(recs: Dict[str, Any]) -> str`

Format recommendations into beautiful Rich display.

**Parameters:**
- `recs` (`Dict`): Output from generate_recommendations()

**Returns:**
- `str`: Formatted recommendations with Rich markup

##### `export_recommendations_json(recs: Dict[str, Any]) -> Dict[str, Any]`

Export recommendations in JSON-friendly format.

**Parameters:**
- `recs` (`Dict`): Output from generate_recommendations()

**Returns:**
- `Dict`: Simplified JSON-serializable dictionary

---

## Error Handling

### Common Exceptions

1. **OpenRouterAPIError** - API request failures
   - Check `status_code` for HTTP error code
   - Check `response_body` for detailed error

2. **ValueError** - Validation errors
   - Invalid task definitions
   - Missing required fields
   - Malformed data

3. **FileNotFoundError** - Missing files
   - Task YAML files
   - Config files
   - Input data files

4. **yaml.YAMLError** - YAML parsing errors
   - Invalid YAML syntax
   - Malformed structure

### Best Practices

1. **Always use async context managers for clients:**
```python
async with OpenRouterClient() as client:
    # Use client
    pass
# Automatically closed
```

2. **Handle API errors gracefully:**
```python
try:
    result = await executor.execute(...)
except OpenRouterAPIError as e:
    if e.status_code == 429:
        # Rate limited, wait and retry
        pass
    elif e.status_code in (500, 502, 503):
        # Server error, retry
        pass
    else:
        # Client error, don't retry
        raise
```

3. **Validate tasks before evaluation:**
```python
task = parser.load_from_yaml("task.yaml")
is_valid, errors = parser.validate_task(task)
if not is_valid:
    print(f"Errors: {errors}")
    exit(1)
```

4. **Track costs to avoid surprises:**
```python
executor = ModelExecutor()
results = await executor.evaluate_multiple(...)
print(executor.get_cost_summary())
```

---

## Type Hints

All public APIs include comprehensive type hints for better IDE support and type checking. Use mypy for static type checking:

```bash
mypy src/taskbench
```

---

## Complete Example

Here's a complete example using multiple APIs together:

```python
import asyncio
from taskbench.core.task import TaskParser
from taskbench.api.client import OpenRouterClient
from taskbench.evaluation.executor import ModelExecutor
from taskbench.evaluation.judge import LLMJudge
from taskbench.evaluation.comparison import ModelComparison
from taskbench.evaluation.recommender import RecommendationEngine

async def main():
    # Load task
    parser = TaskParser()
    task = parser.load_from_yaml("tasks/sentiment.yaml")

    # Validate
    is_valid, errors = parser.validate_task(task)
    if not is_valid:
        print(f"Errors: {errors}")
        return

    # Load input
    with open("input.txt") as f:
        input_data = f.read()

    # Execute evaluations
    executor = ModelExecutor()
    models = [
        "anthropic/claude-sonnet-4.5",
        "openai/gpt-4o"
    ]
    results = await executor.evaluate_multiple(models, task, input_data)

    # Judge outputs
    async with OpenRouterClient() as client:
        judge = LLMJudge(client)
        scores = []
        for result in results:
            if result.status == "success":
                score = await judge.evaluate(task, result, input_data)
                scores.append(score)

    # Compare and recommend
    comparison = ModelComparison()
    compared = comparison.compare_results(results, scores)

    engine = RecommendationEngine()
    recs = engine.generate_recommendations(compared)
    print(engine.format_recommendations(recs))

    # Show costs
    print(executor.get_cost_summary())

if __name__ == "__main__":
    asyncio.run(main())
```

---

For more examples, see the [USAGE.md](USAGE.md) documentation and the `/examples` directory.
