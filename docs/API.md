# LLM TaskBench API Reference

## Table of Contents

- [Core Models](#core-models)
- [Task Parser](#task-parser)
- [API Client](#api-client)
- [Retry Logic](#retry-logic)
- [Model Executor](#model-executor)
- [LLM Judge](#llm-judge)
- [Model Comparison](#model-comparison)
- [Cost Tracker](#cost-tracker)
- [Error Handling](#error-handling)

---

## Core Models

Location: `taskbench.core.models`

All core models are Pydantic BaseModel subclasses with automatic validation.

### TaskDefinition

Represents a user-defined evaluation task.

**Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | `str` | Yes | Unique identifier for the task |
| `description` | `str` | Yes | Human-readable task description |
| `input_type` | `str` | Yes | Type of input data: "transcript", "text", "csv", "json" |
| `output_format` | `str` | Yes | Expected output format: "csv", "json", "markdown" |
| `evaluation_criteria` | `List[str]` | Yes | List of criteria for evaluation |
| `constraints` | `Dict[str, Any]` | No | Constraints that outputs must satisfy |
| `examples` | `List[Dict[str, Any]]` | No | Example inputs and expected outputs |
| `judge_instructions` | `str` | Yes | Instructions for the LLM-as-judge evaluator |

**Validation:**
- `input_type` must be one of: "transcript", "text", "csv", "json"
- `output_format` must be one of: "csv", "json", "markdown"

**Example:**

```python
from taskbench.core.models import TaskDefinition

task = TaskDefinition(
    name="lecture_concept_extraction",
    description="Extract teaching concepts from lecture transcripts",
    input_type="transcript",
    output_format="csv",
    evaluation_criteria=[
        "Timestamp accuracy",
        "Duration compliance",
        "Concept clarity"
    ],
    constraints={
        "min_duration_minutes": 2,
        "max_duration_minutes": 7,
        "required_csv_columns": ["concept", "start_time", "end_time"]
    },
    examples=[],
    judge_instructions="Evaluate based on accuracy, format, and compliance..."
)
```

---

### CompletionResponse

API response from an LLM completion.

**Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `content` | `str` | Yes | The model's response text |
| `model` | `str` | Yes | Model identifier (e.g., "anthropic/claude-sonnet-4.5") |
| `input_tokens` | `int` | Yes | Number of input tokens consumed |
| `output_tokens` | `int` | Yes | Number of output tokens generated |
| `total_tokens` | `int` | Yes | Total tokens (input + output) |
| `latency_ms` | `float` | Yes | Response latency in milliseconds |
| `timestamp` | `datetime` | No | When the response was received (auto-generated) |

**Example:**

```python
from taskbench.core.models import CompletionResponse

response = CompletionResponse(
    content="concept,start_time,end_time\n01_Introduction,00:00:00,00:05:30",
    model="anthropic/claude-sonnet-4.5",
    input_tokens=1500,
    output_tokens=500,
    total_tokens=2000,
    latency_ms=2345.67
)

print(f"Model: {response.model}")
print(f"Tokens: {response.total_tokens}")
print(f"Latency: {response.latency_ms:.2f}ms")
```

---

### EvaluationResult

Single model evaluation result.

**Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model_name` | `str` | Yes | Model identifier |
| `task_name` | `str` | Yes | Task identifier |
| `output` | `str` | Yes | Model's output for the task |
| `input_tokens` | `int` | Yes | Input tokens consumed |
| `output_tokens` | `int` | Yes | Output tokens generated |
| `total_tokens` | `int` | Yes | Total tokens used |
| `cost_usd` | `float` | Yes | Cost in USD for this evaluation |
| `latency_ms` | `float` | Yes | Execution latency in milliseconds |
| `timestamp` | `datetime` | No | When the evaluation was performed (auto-generated) |
| `status` | `str` | No | Evaluation status: "success" or "failed" (default: "success") |
| `error` | `Optional[str]` | No | Error message if status is "failed" |

**Example:**

```python
from taskbench.core.models import EvaluationResult

result = EvaluationResult(
    model_name="anthropic/claude-sonnet-4.5",
    task_name="lecture_concept_extraction",
    output="concept,start_time,end_time\n...",
    input_tokens=1500,
    output_tokens=500,
    total_tokens=2000,
    cost_usd=0.36,
    latency_ms=2345.67,
    status="success"
)

if result.status == "success":
    print(f"Output: {result.output[:100]}...")
    print(f"Cost: ${result.cost_usd:.4f}")
```

---

### JudgeScore

LLM-as-judge scoring result.

**Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model_evaluated` | `str` | Yes | Model that was evaluated |
| `accuracy_score` | `int` | Yes | Accuracy score (0-100) |
| `format_score` | `int` | Yes | Format compliance score (0-100) |
| `compliance_score` | `int` | Yes | Constraint compliance score (0-100) |
| `overall_score` | `int` | Yes | Overall score (0-100) |
| `violations` | `List[str]` | No | List of constraint violations found |
| `reasoning` | `str` | Yes | Detailed explanation of the scores |
| `timestamp` | `datetime` | No | When the evaluation was performed (auto-generated) |

**Validation:**
- All scores must be in range 0-100

**Example:**

```python
from taskbench.core.models import JudgeScore

score = JudgeScore(
    model_evaluated="anthropic/claude-sonnet-4.5",
    accuracy_score=95,
    format_score=100,
    compliance_score=90,
    overall_score=95,
    violations=["One segment slightly over 7 minutes"],
    reasoning="Excellent extraction with minor duration issue..."
)

print(f"Overall Score: {score.overall_score}/100")
print(f"Violations: {len(score.violations)}")
for violation in score.violations:
    print(f"  - {violation}")
```

---

### ModelConfig

Model pricing and configuration.

**Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model_id` | `str` | Yes | Unique model identifier for API calls |
| `display_name` | `str` | Yes | Human-readable model name |
| `input_price_per_1m` | `float` | Yes | Input price per 1M tokens in USD |
| `output_price_per_1m` | `float` | Yes | Output price per 1M tokens in USD |
| `context_window` | `int` | Yes | Maximum context window size in tokens |
| `provider` | `str` | Yes | Model provider (e.g., "Anthropic", "OpenAI") |

**Example:**

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

print(f"{config.display_name} ({config.provider})")
print(f"Input: ${config.input_price_per_1m}/1M tokens")
print(f"Output: ${config.output_price_per_1m}/1M tokens")
```

---

## Task Parser

Location: `taskbench.core.task`

### TaskParser

Parser and validator for task definitions.

#### `load_from_yaml(yaml_path: str) -> TaskDefinition`

Load a task definition from a YAML file.

**Parameters:**
- `yaml_path` (str): Path to the YAML file containing the task definition

**Returns:**
- `TaskDefinition`: Parsed task definition object

**Raises:**
- `FileNotFoundError`: If the YAML file doesn't exist
- `yaml.YAMLError`: If the YAML is malformed
- `ValidationError`: If the YAML doesn't match TaskDefinition schema

**Example:**

```python
from taskbench.core.task import TaskParser

parser = TaskParser()
task = parser.load_from_yaml("tasks/lecture_analysis.yaml")

print(f"Loaded task: {task.name}")
print(f"Input type: {task.input_type}")
print(f"Output format: {task.output_format}")
```

---

#### `validate_task(task: TaskDefinition) -> Tuple[bool, List[str]]`

Validate a task definition for logical consistency.

**Parameters:**
- `task` (TaskDefinition): Task definition to validate

**Returns:**
- `Tuple[bool, List[str]]`: (is_valid, list_of_errors)
  - `is_valid`: True if task is valid, False otherwise
  - `list_of_errors`: List of error messages (empty if valid)

**Validation Checks:**
- Evaluation criteria is non-empty
- Judge instructions is non-empty
- Min/max constraints satisfy min < max
- CSV output has required_csv_columns constraint
- All constraint values have correct types

**Example:**

```python
from taskbench.core.task import TaskParser

parser = TaskParser()
task = parser.load_from_yaml("tasks/my_task.yaml")

is_valid, errors = parser.validate_task(task)
if not is_valid:
    print("Validation errors:")
    for error in errors:
        print(f"  - {error}")
else:
    print("Task is valid!")
```

---

#### `save_to_yaml(task: TaskDefinition, yaml_path: str) -> None`

Save a task definition to a YAML file.

**Parameters:**
- `task` (TaskDefinition): Task definition to save
- `yaml_path` (str): Path where the YAML file should be saved

**Raises:**
- `IOError`: If the file cannot be written

**Example:**

```python
from taskbench.core.task import TaskParser
from taskbench.core.models import TaskDefinition

parser = TaskParser()
task = TaskDefinition(
    name="custom_task",
    description="My custom task",
    input_type="text",
    output_format="json",
    evaluation_criteria=["Accuracy"],
    judge_instructions="Evaluate the output..."
)

parser.save_to_yaml(task, "tasks/custom_task.yaml")
print("Task saved successfully!")
```

---

## API Client

Location: `taskbench.api.client`

### OpenRouterClient

Async HTTP client for OpenRouter API.

#### `__init__(api_key: str, base_url: str = "https://openrouter.ai/api/v1", timeout: float = 120.0)`

Initialize the OpenRouter client.

**Parameters:**
- `api_key` (str): OpenRouter API key
- `base_url` (str): Base URL for OpenRouter API (default: official endpoint)
- `timeout` (float): Request timeout in seconds (default: 120s)

**Example:**

```python
from taskbench.api.client import OpenRouterClient

async with OpenRouterClient(api_key="your-key") as client:
    # Use client for API calls
    pass
```

---

#### `complete(model: str, prompt: str, max_tokens: int = 1000, temperature: float = 0.7, **kwargs) -> CompletionResponse`

Send a completion request to OpenRouter.

**Parameters:**
- `model` (str): Model identifier (e.g., "anthropic/claude-sonnet-4.5")
- `prompt` (str): The prompt to send to the model
- `max_tokens` (int): Maximum tokens to generate (default: 1000)
- `temperature` (float): Sampling temperature 0-1 (default: 0.7)
- `**kwargs`: Additional parameters to pass to the API

**Returns:**
- `CompletionResponse`: Response with model output and metadata

**Raises:**
- `AuthenticationError`: If API key is invalid
- `RateLimitError`: If rate limit is exceeded
- `BadRequestError`: If request is malformed
- `OpenRouterError`: For other API errors

**Example:**

```python
from taskbench.api.client import OpenRouterClient

async with OpenRouterClient(api_key="your-key") as client:
    response = await client.complete(
        model="anthropic/claude-sonnet-4.5",
        prompt="Explain Python lists in 2 sentences",
        max_tokens=100,
        temperature=0.7
    )

    print(f"Response: {response.content}")
    print(f"Tokens: {response.total_tokens}")
    print(f"Latency: {response.latency_ms:.2f}ms")
```

---

#### `complete_with_json(model: str, prompt: str, max_tokens: int = 1000, temperature: float = 0.7, **kwargs) -> CompletionResponse`

Request a completion in JSON mode.

Adds JSON formatting instructions to the prompt and validates that the response is valid JSON.

**Parameters:**
- Same as `complete()`

**Returns:**
- `CompletionResponse`: Response with JSON content (cleaned of markdown blocks)

**Raises:**
- `OpenRouterError`: If response is not valid JSON
- Other exceptions same as `complete()`

**Example:**

```python
from taskbench.api.client import OpenRouterClient
import json

async with OpenRouterClient(api_key="your-key") as client:
    response = await client.complete_with_json(
        model="anthropic/claude-sonnet-4.5",
        prompt="List 3 programming languages with their year of creation",
        max_tokens=500,
        temperature=0.5
    )

    data = json.loads(response.content)
    print(json.dumps(data, indent=2))
```

---

#### `close() -> None`

Close the HTTP client and cleanup resources.

**Example:**

```python
from taskbench.api.client import OpenRouterClient

client = OpenRouterClient(api_key="your-key")
# Use client...
await client.close()
```

---

### Exception Classes

#### `OpenRouterError`

Base exception for OpenRouter API errors.

#### `RateLimitError`

Raised when API rate limit is exceeded (HTTP 429).

#### `AuthenticationError`

Raised when API authentication fails (HTTP 401).

#### `BadRequestError`

Raised when the request is malformed (HTTP 400).

---

## Retry Logic

Location: `taskbench.api.retry`

### RateLimiter

Token bucket rate limiter for API requests.

#### `__init__(max_requests_per_minute: int = 60)`

Initialize the rate limiter.

**Parameters:**
- `max_requests_per_minute` (int): Maximum requests allowed per minute

**Example:**

```python
from taskbench.api.retry import RateLimiter

limiter = RateLimiter(max_requests_per_minute=60)
```

---

#### `acquire() -> None`

Acquire permission to make a request.

Sleeps if making a request now would exceed the rate limit.

**Example:**

```python
from taskbench.api.retry import RateLimiter

limiter = RateLimiter(max_requests_per_minute=60)

async def make_api_call():
    await limiter.acquire()  # Wait if rate limit would be exceeded
    # Make API request
    pass
```

---

### retry_with_backoff

Decorator for retrying async functions with exponential backoff.

#### `retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0, retryable_exceptions: Optional[Set[Type[Exception]]] = None, non_retryable_exceptions: Optional[Set[Type[Exception]]] = None)`

**Parameters:**
- `max_retries` (int): Maximum number of retry attempts (default: 3)
- `base_delay` (float): Initial delay in seconds (default: 1.0)
- `max_delay` (float): Maximum delay in seconds (default: 60.0)
- `retryable_exceptions` (Set[Type[Exception]]): Exceptions to retry (default: RateLimitError, OpenRouterError, TimeoutError, ConnectionError)
- `non_retryable_exceptions` (Set[Type[Exception]]): Exceptions to never retry (default: AuthenticationError, BadRequestError, ValueError, TypeError)

**Returns:**
- Decorated function with retry logic

**Retry Strategy:**
- Exponential backoff: delay = min(base_delay * (2 ** attempt), max_delay)
- Retries transient errors (rate limits, timeouts, server errors)
- Immediately raises non-retryable errors (auth, bad requests)

**Example:**

```python
from taskbench.api.retry import retry_with_backoff
from taskbench.api.client import OpenRouterClient

@retry_with_backoff(max_retries=3, base_delay=2.0)
async def make_api_call(client: OpenRouterClient):
    return await client.complete(
        model="anthropic/claude-sonnet-4.5",
        prompt="Hello, world!"
    )

# If the call fails with a retryable error, it will retry up to 3 times
# with delays of 2s, 4s, 8s
```

---

### with_rate_limit

Decorator to enforce rate limiting on async functions.

#### `with_rate_limit(limiter: RateLimiter)`

**Parameters:**
- `limiter` (RateLimiter): RateLimiter instance to use

**Returns:**
- Decorated function with rate limiting

**Example:**

```python
from taskbench.api.retry import RateLimiter, with_rate_limit

limiter = RateLimiter(max_requests_per_minute=60)

@with_rate_limit(limiter)
async def make_request():
    # This function will automatically respect the rate limit
    pass
```

---

## Model Executor

Location: `taskbench.evaluation.executor`

### ModelExecutor

Execute tasks on LLM models and collect results.

#### `__init__(api_client: OpenRouterClient, cost_tracker: CostTracker)`

Initialize the model executor.

**Parameters:**
- `api_client` (OpenRouterClient): OpenRouter client for making API calls
- `cost_tracker` (CostTracker): Cost tracker for calculating costs

**Example:**

```python
from taskbench.api.client import OpenRouterClient
from taskbench.evaluation.cost import CostTracker
from taskbench.evaluation.executor import ModelExecutor

async with OpenRouterClient(api_key="your-key") as client:
    cost_tracker = CostTracker()
    executor = ModelExecutor(client, cost_tracker)
```

---

#### `build_prompt(task: TaskDefinition, input_data: str) -> str`

Build a comprehensive prompt from task definition and input data.

**Parameters:**
- `task` (TaskDefinition): Task definition describing the task
- `input_data` (str): Input data to process

**Returns:**
- `str`: Complete prompt string to send to the model

**Prompt Structure:**
1. Task name and description
2. Output format requirements
3. CRITICAL CONSTRAINTS section (emphasized)
4. Examples of good outputs
5. Evaluation criteria
6. Input data
7. Final instructions

**Example:**

```python
from taskbench.evaluation.executor import ModelExecutor
from taskbench.core.task import TaskParser

parser = TaskParser()
task = parser.load_from_yaml("tasks/lecture_analysis.yaml")
input_data = "Lecture transcript content..."

# Assuming executor is already initialized
prompt = executor.build_prompt(task, input_data)
print(prompt[:500])  # Preview first 500 chars
```

---

#### `execute(model_id: str, task: TaskDefinition, input_data: str, max_tokens: int = 2000, temperature: float = 0.7) -> EvaluationResult`

Execute a task on a single model.

**Parameters:**
- `model_id` (str): Model identifier (e.g., "anthropic/claude-sonnet-4.5")
- `task` (TaskDefinition): Task definition describing the task
- `input_data` (str): Input data to process
- `max_tokens` (int): Maximum tokens to generate (default: 2000)
- `temperature` (float): Sampling temperature (default: 0.7)

**Returns:**
- `EvaluationResult`: Evaluation result with output and metadata

**Error Handling:**
- On success: Returns EvaluationResult with status="success"
- On error: Returns EvaluationResult with status="failed" and error message

**Example:**

```python
from taskbench.evaluation.executor import ModelExecutor
from taskbench.core.task import TaskParser

parser = TaskParser()
task = parser.load_from_yaml("tasks/lecture_analysis.yaml")
input_data = open("data/transcript.txt").read()

# Assuming executor is already initialized
result = await executor.execute(
    model_id="anthropic/claude-sonnet-4.5",
    task=task,
    input_data=input_data,
    max_tokens=2000,
    temperature=0.7
)

if result.status == "success":
    print(f"Output: {result.output[:200]}...")
    print(f"Cost: ${result.cost_usd:.4f}")
else:
    print(f"Error: {result.error}")
```

---

#### `evaluate_multiple(model_ids: List[str], task: TaskDefinition, input_data: str, max_tokens: int = 2000, temperature: float = 0.7) -> List[EvaluationResult]`

Execute a task on multiple models with progress tracking.

**Parameters:**
- `model_ids` (List[str]): List of model identifiers
- `task` (TaskDefinition): Task definition describing the task
- `input_data` (str): Input data to process
- `max_tokens` (int): Maximum tokens to generate per model (default: 2000)
- `temperature` (float): Sampling temperature (default: 0.7)

**Returns:**
- `List[EvaluationResult]`: List of evaluation results, one per model

**Features:**
- Displays progress bar with Rich
- Shows real-time status updates
- Prints summary after completion

**Example:**

```python
from taskbench.evaluation.executor import ModelExecutor

# Assuming executor is already initialized
results = await executor.evaluate_multiple(
    model_ids=[
        "anthropic/claude-sonnet-4.5",
        "openai/gpt-4o",
        "qwen/qwen-2.5-72b-instruct"
    ],
    task=task,
    input_data=input_data,
    max_tokens=2000,
    temperature=0.7
)

for result in results:
    if result.status == "success":
        print(f"{result.model_name}: ${result.cost_usd:.4f}")
```

---

## LLM Judge

Location: `taskbench.evaluation.judge`

### LLMJudge

Use an LLM to evaluate model outputs.

#### `__init__(api_client: OpenRouterClient, judge_model: str = "anthropic/claude-sonnet-4.5")`

Initialize the LLM judge.

**Parameters:**
- `api_client` (OpenRouterClient): OpenRouter client for making API calls
- `judge_model` (str): Model to use as judge (default: "anthropic/claude-sonnet-4.5")

**Example:**

```python
from taskbench.api.client import OpenRouterClient
from taskbench.evaluation.judge import LLMJudge

async with OpenRouterClient(api_key="your-key") as client:
    judge = LLMJudge(client, judge_model="anthropic/claude-sonnet-4.5")
```

---

#### `build_judge_prompt(task: TaskDefinition, model_output: str, input_data: str) -> str`

Build evaluation prompt for the judge model.

**Parameters:**
- `task` (TaskDefinition): Task definition with evaluation criteria
- `model_output` (str): The output to evaluate
- `input_data` (str): Original input data for context

**Returns:**
- `str`: Complete judge prompt

**Prompt Structure:**
1. Judge role and task description
2. Evaluation criteria from task
3. Constraints to check
4. Original input data (for context)
5. Model output to evaluate
6. Judge instructions from task
7. JSON response format specification

**Example:**

```python
from taskbench.evaluation.judge import LLMJudge

# Assuming judge is already initialized
prompt = judge.build_judge_prompt(task, result.output, input_data)
```

---

#### `evaluate(task: TaskDefinition, result: EvaluationResult, input_data: str) -> JudgeScore`

Evaluate a model's output using LLM-as-judge.

**Parameters:**
- `task` (TaskDefinition): Task definition with evaluation criteria
- `result` (EvaluationResult): Evaluation result to evaluate
- `input_data` (str): Original input data

**Returns:**
- `JudgeScore`: Score with accuracy, format, compliance scores and violations

**Raises:**
- `Exception`: If judge fails to return valid JSON

**Judge Configuration:**
- Uses JSON mode for structured output
- Temperature: 0.3 (for consistency)
- Max tokens: 2000

**Example:**

```python
from taskbench.evaluation.judge import LLMJudge

# Assuming judge is already initialized
score = await judge.evaluate(
    task=task,
    result=result,
    input_data=input_data
)

print(f"Overall Score: {score.overall_score}/100")
print(f"Accuracy: {score.accuracy_score}/100")
print(f"Format: {score.format_score}/100")
print(f"Compliance: {score.compliance_score}/100")
print(f"Violations: {score.violations}")
print(f"Reasoning: {score.reasoning}")
```

---

#### `parse_violations(violations: List[str]) -> Dict[str, List[str]]`

Categorize violations by type.

**Parameters:**
- `violations` (List[str]): List of violation strings

**Returns:**
- `Dict[str, List[str]]`: Dictionary mapping violation types to specific violations

**Categories:**
- `under_min`: Below minimum requirements
- `over_max`: Exceeds maximum limits
- `format`: Format specification violations
- `missing_field`: Required fields absent
- `other`: Miscellaneous issues

**Example:**

```python
from taskbench.evaluation.judge import LLMJudge

judge = LLMJudge(client)
violations = [
    "Segment duration under 2 minutes",
    "Missing required CSV column: end_time",
    "Timestamp format invalid"
]

categorized = judge.parse_violations(violations)
print(categorized)
# {
#   "under_min": ["Segment duration under 2 minutes"],
#   "missing_field": ["Missing required CSV column: end_time"],
#   "format": ["Timestamp format invalid"],
#   "over_max": [],
#   "other": []
# }
```

---

## Model Comparison

Location: `taskbench.evaluation.judge`

### ModelComparison

Compare and rank model evaluation results.

#### `compare_results(results: List[EvaluationResult], scores: List[JudgeScore]) -> List[Dict[str, Any]]`

Combine results and scores into comparison data.

**Parameters:**
- `results` (List[EvaluationResult]): List of evaluation results
- `scores` (List[JudgeScore]): List of corresponding judge scores

**Returns:**
- `List[Dict[str, Any]]`: List of dicts with combined data, sorted by overall_score descending

**Raises:**
- `ValueError`: If results and scores lists have different lengths

**Comparison Data Fields:**
- `rank`: Ranking (1 = best)
- `model`: Model identifier
- `overall_score`: Overall score (0-100)
- `accuracy_score`, `format_score`, `compliance_score`: Subscores
- `violations`: Number of violations
- `violation_list`: List of violation strings
- `cost_usd`: Cost in USD
- `tokens`: Total tokens used
- `latency_ms`: Latency in milliseconds
- `status`: Evaluation status
- `reasoning`: Judge's detailed reasoning

**Example:**

```python
from taskbench.evaluation.judge import ModelComparison

comparison = ModelComparison.compare_results(results, scores)

for item in comparison:
    print(f"Rank {item['rank']}: {item['model']}")
    print(f"  Score: {item['overall_score']}/100")
    print(f"  Cost: ${item['cost_usd']:.4f}")
    print(f"  Violations: {item['violations']}")
```

---

#### `identify_best(comparison: List[Dict[str, Any]]) -> str`

Identify model with highest overall score.

**Parameters:**
- `comparison` (List[Dict[str, Any]]): Comparison data from compare_results()

**Returns:**
- `str`: Model identifier of the best model

**Example:**

```python
from taskbench.evaluation.judge import ModelComparison

comparison = ModelComparison.compare_results(results, scores)
best_model = ModelComparison.identify_best(comparison)

print(f"Best model: {best_model}")
```

---

#### `identify_best_value(comparison: List[Dict[str, Any]], max_cost: float = None) -> str`

Identify model with best score/cost ratio.

**Parameters:**
- `comparison` (List[Dict[str, Any]]): Comparison data from compare_results()
- `max_cost` (float, optional): Optional maximum cost filter

**Returns:**
- `str`: Model identifier with best value

**Value Calculation:**
- If cost > 0: value_score = overall_score / cost_usd
- If cost = 0: value_score = overall_score * 1000 (free models get bonus)

**Example:**

```python
from taskbench.evaluation.judge import ModelComparison

comparison = ModelComparison.compare_results(results, scores)

# Best value overall
best_value = ModelComparison.identify_best_value(comparison)
print(f"Best value: {best_value}")

# Best value under $0.50
best_cheap = ModelComparison.identify_best_value(comparison, max_cost=0.50)
print(f"Best value under $0.50: {best_cheap}")
```

---

#### `generate_comparison_table(comparison: List[Dict[str, Any]]) -> Table`

Generate Rich table for comparison display.

**Parameters:**
- `comparison` (List[Dict[str, Any]]): Comparison data from compare_results()

**Returns:**
- `rich.table.Table`: Rich Table object

**Table Columns:**
- Rank
- Model (short name)
- Score (color-coded: green >=90, yellow >=80, red <80)
- Violations (color-coded: green =0, yellow <=2, red >2)
- Cost (USD)
- Tokens
- Value (P/PP/PPP rating based on score/cost ratio)

**Example:**

```python
from rich.console import Console
from taskbench.evaluation.judge import ModelComparison

console = Console()
comparison = ModelComparison.compare_results(results, scores)
table = ModelComparison.generate_comparison_table(comparison)

console.print(table)
```

---

## Cost Tracker

Location: `taskbench.evaluation.cost`

### CostTracker

Calculate and track costs for LLM evaluations.

#### `__init__(models_config_path: str = "config/models.yaml")`

Initialize the cost tracker.

**Parameters:**
- `models_config_path` (str): Path to YAML file containing model pricing (default: "config/models.yaml")

**Raises:**
- `FileNotFoundError`: If config file doesn't exist
- `ValueError`: If config file is invalid

**Example:**

```python
from taskbench.evaluation.cost import CostTracker

tracker = CostTracker("config/models.yaml")
```

---

#### `calculate_cost(model_id: str, input_tokens: int, output_tokens: int) -> float`

Calculate cost for a specific API call.

**Parameters:**
- `model_id` (str): Model identifier (e.g., "anthropic/claude-sonnet-4.5")
- `input_tokens` (int): Number of input tokens consumed
- `output_tokens` (int): Number of output tokens generated

**Returns:**
- `float`: Cost in USD, rounded to $0.01 precision

**Raises:**
- `ValueError`: If model_id is not found in pricing database

**Formula:**
```
cost = (input_tokens / 1,000,000) * input_price_per_1m
     + (output_tokens / 1,000,000) * output_price_per_1m
```

**Example:**

```python
from taskbench.evaluation.cost import CostTracker

tracker = CostTracker()

cost = tracker.calculate_cost(
    model_id="anthropic/claude-sonnet-4.5",
    input_tokens=1000,
    output_tokens=500
)

print(f"Cost: ${cost:.4f}")
# Cost: $0.0105
# Calculation: (1000/1M * $3.00) + (500/1M * $15.00) = $0.0030 + $0.0075 = $0.0105
```

---

#### `track_evaluation(result: EvaluationResult) -> None`

Track an evaluation result for cost analysis.

**Parameters:**
- `result` (EvaluationResult): Evaluation result to track

**Example:**

```python
from taskbench.evaluation.cost import CostTracker

tracker = CostTracker()
tracker.track_evaluation(result)
```

---

#### `get_total_cost() -> float`

Get total cost of all tracked evaluations.

**Returns:**
- `float`: Total cost in USD

**Example:**

```python
from taskbench.evaluation.cost import CostTracker

tracker = CostTracker()
# ... track some evaluations ...

total = tracker.get_total_cost()
print(f"Total cost: ${total:.2f}")
```

---

#### `get_cost_breakdown() -> Dict[str, float]`

Get per-model cost breakdown.

**Returns:**
- `Dict[str, float]`: Dictionary mapping model names to their total costs

**Example:**

```python
from taskbench.evaluation.cost import CostTracker

tracker = CostTracker()
# ... track some evaluations ...

breakdown = tracker.get_cost_breakdown()
for model, cost in breakdown.items():
    print(f"{model}: ${cost:.4f}")
```

---

#### `get_statistics() -> Dict[str, Any]`

Get comprehensive cost statistics.

**Returns:**
- `Dict[str, Any]`: Dictionary with statistics:
  - `total_cost`: Total cost in USD
  - `total_tokens`: Total tokens across all evaluations
  - `total_evaluations`: Number of evaluations tracked
  - `avg_cost_per_eval`: Average cost per evaluation
  - `avg_tokens_per_eval`: Average tokens per evaluation
  - `cost_by_model`: Per-model cost breakdown

**Example:**

```python
from taskbench.evaluation.cost import CostTracker

tracker = CostTracker()
# ... track some evaluations ...

stats = tracker.get_statistics()
print(f"Total cost: ${stats['total_cost']:.2f}")
print(f"Total tokens: {stats['total_tokens']:,}")
print(f"Total evaluations: {stats['total_evaluations']}")
print(f"Average cost: ${stats['avg_cost_per_eval']:.4f}")
print(f"Average tokens: {stats['avg_tokens_per_eval']:,}")
```

---

#### `get_model_config(model_id: str) -> Optional[ModelConfig]`

Get configuration for a specific model.

**Parameters:**
- `model_id` (str): Model identifier

**Returns:**
- `Optional[ModelConfig]`: ModelConfig if found, None otherwise

**Example:**

```python
from taskbench.evaluation.cost import CostTracker

tracker = CostTracker()
config = tracker.get_model_config("anthropic/claude-sonnet-4.5")

if config:
    print(f"{config.display_name}")
    print(f"Input: ${config.input_price_per_1m}/1M tokens")
    print(f"Output: ${config.output_price_per_1m}/1M tokens")
```

---

#### `list_models() -> List[ModelConfig]`

Get list of all available models.

**Returns:**
- `List[ModelConfig]`: List of all model configurations

**Example:**

```python
from taskbench.evaluation.cost import CostTracker

tracker = CostTracker()
models = tracker.list_models()

for model in models:
    print(f"{model.display_name} ({model.provider})")
    print(f"  Input: ${model.input_price_per_1m}/1M")
    print(f"  Output: ${model.output_price_per_1m}/1M")
```

---

## Error Handling

### Exception Hierarchy

```
Exception
├── OpenRouterError (base for all API errors)
│   ├── AuthenticationError (401)
│   ├── BadRequestError (400)
│   └── RateLimitError (429)
├── FileNotFoundError (task/config files)
├── yaml.YAMLError (YAML parsing)
└── pydantic.ValidationError (data validation)
```

### Best Practices

1. **Always use async context managers for API clients:**
```python
async with OpenRouterClient(api_key="key") as client:
    # client will be properly closed even if errors occur
    pass
```

2. **Check evaluation status before using results:**
```python
if result.status == "success":
    process(result.output)
else:
    print(f"Error: {result.error}")
```

3. **Validate tasks before running evaluations:**
```python
is_valid, errors = parser.validate_task(task)
if not is_valid:
    for error in errors:
        print(f"Error: {error}")
    return
```

4. **Handle missing models gracefully:**
```python
try:
    cost = tracker.calculate_cost(model_id, input_tokens, output_tokens)
except ValueError as e:
    print(f"Model not found: {e}")
```

5. **Use retry decorators for resilience:**
```python
@retry_with_backoff(max_retries=3)
async def robust_api_call():
    return await client.complete(...)
```

---

## Complete Example

Here's a complete example using all major components:

```python
import asyncio
from taskbench.api.client import OpenRouterClient
from taskbench.core.task import TaskParser
from taskbench.evaluation.cost import CostTracker
from taskbench.evaluation.executor import ModelExecutor
from taskbench.evaluation.judge import LLMJudge, ModelComparison
from rich.console import Console

async def main():
    console = Console()

    # Load task
    parser = TaskParser()
    task = parser.load_from_yaml("tasks/lecture_analysis.yaml")

    # Validate task
    is_valid, errors = parser.validate_task(task)
    if not is_valid:
        console.print("[red]Task validation failed:[/red]")
        for error in errors:
            console.print(f"  - {error}")
        return

    # Load input
    with open("data/transcript.txt") as f:
        input_data = f.read()

    # Initialize components
    async with OpenRouterClient(api_key="your-key") as client:
        cost_tracker = CostTracker()
        executor = ModelExecutor(client, cost_tracker)
        judge = LLMJudge(client)

        # Evaluate models
        model_ids = [
            "anthropic/claude-sonnet-4.5",
            "openai/gpt-4o",
            "qwen/qwen-2.5-72b-instruct"
        ]

        results = await executor.evaluate_multiple(
            model_ids=model_ids,
            task=task,
            input_data=input_data
        )

        # Judge results
        scores = []
        for result in results:
            if result.status == "success":
                score = await judge.evaluate(task, result, input_data)
                scores.append(score)
            else:
                scores.append(None)

        # Compare results
        valid_results = [r for r, s in zip(results, scores) if s is not None]
        valid_scores = [s for s in scores if s is not None]

        comparison = ModelComparison.compare_results(valid_results, valid_scores)
        table = ModelComparison.generate_comparison_table(comparison)
        console.print(table)

        # Show best models
        best_model = ModelComparison.identify_best(comparison)
        best_value = ModelComparison.identify_best_value(comparison)

        console.print(f"\nBest Overall: {best_model}")
        console.print(f"Best Value: {best_value}")

        # Cost statistics
        stats = cost_tracker.get_statistics()
        console.print(f"\nTotal Cost: ${stats['total_cost']:.2f}")
        console.print(f"Total Tokens: {stats['total_tokens']:,}")

if __name__ == "__main__":
    asyncio.run(main())
```

This completes the API reference documentation for LLM TaskBench.
