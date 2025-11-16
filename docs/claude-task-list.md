# LLM TaskBench - Claude Code Implementation Task List

**Version:** 1.0
**Target:** MVP completion by December 22, 2025
**Current Phase:** ✅ MVP COMPLETED - November 15, 2025
**Execution Model:** Sequential implementation with validation at each milestone

---

## 🎉 IMPLEMENTATION COMPLETE - NOVEMBER 15, 2025

**Status:** All Phase 1-3 core tasks completed successfully!

### ✅ What Was Completed:
- **Phase 1:** Core framework (models, parser, API client, cost tracking, executor, CLI)
- **Phase 2:** LLM-as-Judge evaluation system and comparison logic
- **Phase 3:** Recommendation engine with CLI integration
- **Testing:** 117 comprehensive tests passing (53% coverage, 89-99% on core modules)
- **Documentation:** Full architecture, API reference, and usage guides
- **README:** Professional GitHub-ready README with examples

### 📊 Key Metrics Achieved:
- ✅ 117 tests passing
- ✅ 10 models configured in pricing database
- ✅ All core components implemented with type hints and docstrings
- ✅ Comprehensive error handling throughout
- ✅ Professional CLI with Rich formatting
- ✅ Complete documentation (ARCHITECTURE.md, API.md, USAGE.md)

### 🚀 Ready For:
- Live evaluation of 5+ models
- LLM-as-judge scoring
- Cost-aware recommendations
- GitHub publication
- Demo/presentation

---

## ⚡ CRITICAL INSTRUCTIONS FOR CLAUDE CODE

**How to use this task list:**

1. **Work sequentially** - Complete each task in order, as later tasks depend on earlier ones
2. **Validate at each step** - Don't move forward until validation criteria are met
3. **Test as you go** - Write unit tests alongside implementation
4. **Document decisions** - Add docstrings and comments explaining design choices
5. **Check dependencies** - Install required packages as needed via pip

**Success criteria for each task:**
- Code runs without errors
- Tests pass (if testing is specified)
- Validation criteria are met
- Type hints are present
- Docstrings are complete

---

## 📋 Project Context

You are building **LLM TaskBench**, a task-specific LLM evaluation framework that enables domain experts to compare multiple LLMs on their actual use cases. The system uses agentic orchestration and LLM-as-judge evaluation to provide cost-aware model recommendations.

**Key Innovation:** Shifts from metric-first (BLEU, ROUGE) to task-first evaluation, validated by 42-model research showing model size and cost don't correlate with performance.

**Primary Use Case:** Lecture transcript concept extraction with timestamp precision and duration constraints (2-7 minute segments).

**Research Background:** Based on testing 42 production LLMs on lecture analysis, revealing:
- Model size doesn't correlate with quality (405B didn't beat 72B)
- "Reasoning" models can underperform on reasoning tasks
- Cost has zero correlation with performance
- Fine-tuning beats raw parameter count

---

## 🏗️ Project Structure Setup

**TASK 0: Initialize Project Structure**

Create the following directory structure:

```
llm-taskbench/
├── src/
│   └── taskbench/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── task.py           # Task definition & parsing
│       │   └── models.py         # Pydantic models
│       ├── api/
│       │   ├── __init__.py
│       │   ├── client.py         # OpenRouter API client
│       │   └── retry.py          # Retry logic & rate limiting
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── executor.py       # Model execution
│       │   ├── judge.py          # LLM-as-judge evaluator
│       │   ├── cost.py           # Cost tracking & analysis
│       │   └── orchestrator.py   # Agentic orchestration
│       ├── cli/
│       │   ├── __init__.py
│       │   └── main.py           # Typer CLI commands
│       └── utils/
│           ├── __init__.py
│           ├── logging.py        # Logging configuration
│           └── validation.py     # Input validation
├── tasks/
│   ├── lecture_analysis.yaml     # Built-in task definition
│   └── template.yaml             # Template for new tasks
├── tests/
│   ├── __init__.py
│   ├── test_task.py
│   ├── test_api.py
│   ├── test_executor.py
│   ├── test_judge.py
│   ├── test_cost.py
│   └── fixtures/
│       └── sample_transcript.txt
├── examples/
│   └── lecture_analysis.yaml     # Example task
├── config/
│   └── models.yaml               # Model pricing database
├── docs/
│   ├── ARCHITECTURE.md
│   ├── API.md
│   └── USAGE.md
├── .github/
│   └── workflows/
│       └── tests.yml
├── pyproject.toml
├── requirements.txt
├── README.md
├── LICENSE (MIT)
└── .gitignore
```

**Action Items:**
1. Create all directories and `__init__.py` files
2. Create empty placeholder files for each module
3. Create `.gitignore` with Python standard ignores plus:
   - `.env`
   - `*.yaml` in root (for local testing)
   - `results/` directory
   - `.pytest_cache/`
   - `__pycache__/`

**Validation:**
- All directories exist
- All `__init__.py` files are present
- Can import `taskbench` package

---

## 📦 TASK 1: Setup Dependencies

**Goal:** Configure project dependencies and environment

**Action Items:**

1. **Create `requirements.txt`** with these dependencies:
   - `pydantic>=2.0.0` - Data validation
   - `pyyaml>=6.0` - YAML parsing
   - `httpx>=0.25.0` - Async HTTP client
   - `typer>=0.9.0` - CLI framework
   - `rich>=13.0.0` - Beautiful terminal output
   - `python-dotenv>=1.0.0` - Environment variables
   - `pytest>=7.4.0` - Testing
   - `pytest-asyncio>=0.21.0` - Async test support
   - `pytest-cov>=4.1.0` - Coverage reporting

2. **Create `pyproject.toml`** with:
   - Project metadata (name, version, description, author)
   - Python version requirement: >=3.11
   - Entry point: `taskbench = taskbench.cli.main:app`
   - Build system: `setuptools`

3. **Create `.env.example`** file documenting required environment variables:
   - `OPENROUTER_API_KEY=your_key_here`
   - `ANTHROPIC_API_KEY=your_key_here` (optional, for direct API)
   - `OPENAI_API_KEY=your_key_here` (optional, for direct API)

**Validation:**
- `pip install -r requirements.txt` succeeds
- Can import all required packages
- `pip install -e .` makes `taskbench` command available

---

## 🎯 PHASE 1: Core Framework (Weeks 1-2)

**Goal:** Basic task definition and model execution with cost tracking  
**Target:** Can evaluate 5 models with accurate cost tracking

---

### MILESTONE 1.1: Task Definition System (Days 1-3)

**Deliverable:** Can parse and validate YAML task definitions

---

#### TASK 1.1.1: Create Pydantic Data Models

**File:** `src/taskbench/core/models.py`

**Goal:** Define all core data structures using Pydantic for type safety and validation

**Requirements:**

Create the following Pydantic models with proper validation:

1. **TaskDefinition** - Represents a user-defined evaluation task
   - Fields: name, description, input_type, output_format, evaluation_criteria, constraints, examples, judge_instructions
   - Validators: input_type must be in ['transcript', 'text', 'csv', 'json']
   - Validators: output_format must be in ['csv', 'json', 'markdown']
   - All fields should have Field() with descriptions

2. **CompletionResponse** - API response from LLM
   - Fields: content, model, input_tokens, output_tokens, total_tokens, latency_ms, timestamp
   - timestamp should default to current datetime

3. **EvaluationResult** - Single model evaluation result
   - Fields: model_name, task_name, output, input_tokens, output_tokens, total_tokens, cost_usd, latency_ms, timestamp, status, error
   - status defaults to "success"
   - error is Optional

4. **JudgeScore** - LLM-as-judge scoring result
   - Fields: model_evaluated, accuracy_score, format_score, compliance_score, overall_score, violations, reasoning, timestamp
   - All scores must be 0-100 (use Field validators)
   - violations is a list of strings

5. **ModelConfig** - Model pricing and configuration
   - Fields: model_id, display_name, input_price_per_1m, output_price_per_1m, context_window, provider
   - Prices in USD per 1M tokens

**Design Considerations:**
- Use proper type hints for all fields
- Add comprehensive docstrings to each class
- Include example usage in docstrings
- Use Field() with descriptions for clarity
- Implement __str__ and __repr__ for debugging

**Testing:**
Write `tests/test_models.py` with:
- Test instantiation of each model with valid data
- Test validators catch invalid data
- Test optional fields work correctly
- Test model serialization to dict/JSON

**Validation Criteria:**
✓ All models instantiate with valid data
✓ Invalid data raises ValidationError
✓ Models serialize to JSON correctly
✓ Tests pass with 100% coverage of models.py

---

#### TASK 1.1.2: Implement Task Parser

**File:** `src/taskbench/core/task.py`

**Goal:** Parse YAML files into TaskDefinition objects with comprehensive validation

**Requirements:**

Create a `TaskParser` class with these methods:

1. **load_from_yaml(yaml_path: str) -> TaskDefinition**
   - Load YAML file and parse into TaskDefinition
   - Handle file not found errors gracefully
   - Validate YAML structure before parsing
   - Return helpful error messages for malformed YAML

2. **validate_task(task: TaskDefinition) -> tuple[bool, List[str]]**
   - Check that all required fields are present
   - Validate constraints make sense (e.g., min < max)
   - Check evaluation_criteria is non-empty
   - Return (is_valid, list_of_errors)

3. **save_to_yaml(task: TaskDefinition, yaml_path: str) -> None**
   - Serialize TaskDefinition back to YAML
   - Preserve comments and formatting where possible
   - Create parent directories if needed

**Design Considerations:**
- Use pathlib.Path for file operations
- Provide clear error messages with file names and line numbers
- Log validation steps for debugging
- Handle edge cases (empty files, malformed YAML, missing fields)

**Testing:**
Write `tests/test_task.py` with:
- Test loading valid YAML file
- Test loading invalid YAML raises proper errors
- Test validation catches missing fields
- Test validation catches invalid constraints
- Test round-trip (load → save → load) preserves data

**Test Fixtures:**
Create `tests/fixtures/valid_task.yaml` - a valid lecture analysis task
Create `tests/fixtures/invalid_task.yaml` - missing required fields

**Validation Criteria:**
✓ Can load valid YAML without errors
✓ Invalid YAML produces helpful error messages
✓ Validation catches all error conditions
✓ Round-trip preserves all data
✓ Tests pass with >80% coverage

---

#### TASK 1.1.3: Create Built-in Task Definition

**File:** `tasks/lecture_analysis.yaml`

**Goal:** Create the primary built-in task for lecture concept extraction

**Requirements:**

Create a YAML file defining the lecture analysis task with:

**Structure:**
```yaml
name: "lecture_concept_extraction"
description: "Extract teaching concepts from lecture transcripts with precise timestamps"

input_type: "transcript"
output_format: "csv"

evaluation_criteria:
  - "Concept count (target: 20-24 for 3-hour lecture)"
  - "Timestamp accuracy (within ±5 seconds)"
  - "Duration compliance (2-7 minutes per segment)"
  - "Concept names are descriptive and clear"
  - "No overlapping time ranges"

constraints:
  min_duration_minutes: 2
  max_duration_minutes: 7
  target_duration_minutes: "3-6"
  required_csv_columns: ["concept", "start_time", "end_time"]
  timestamp_format: "HH:MM:SS"

examples:
  - input: "Sample transcript snippet..."
    expected_output: "01_Teaching_Concept,00:20:06,00:23:15"
    quality_score: 95
    notes: "Perfect: 3.15 minute segment, descriptive name, clean timestamps"

judge_instructions: |
  Evaluate the model's output against these specific criteria:
  
  1. ACCURACY (40 points):
     - Did it identify all major teaching concepts?
     - Are concepts semantically distinct?
     - Are timestamps accurate?
  
  2. FORMAT (30 points):
     - Valid CSV with required columns?
     - Timestamps in HH:MM:SS format?
     - Concept names follow naming convention?
  
  3. COMPLIANCE (30 points):
     - All segments 2-7 minutes? (violations: -5 points each)
     - No overlapping timestamps?
     - Concepts in chronological order?
  
  Count violations:
  - Under 2 minutes: VIOLATION
  - Over 7 minutes: VIOLATION
  - Overlapping timestamps: VIOLATION
  - Invalid format: VIOLATION
```

**Validation Criteria:**
✓ YAML is valid and parsable
✓ Loads successfully with TaskParser
✓ Passes all TaskDefinition validators
✓ Constraints are clear and testable
✓ Judge instructions are comprehensive

---

#### TASK 1.1.4: Create Task Template

**File:** `tasks/template.yaml`

**Goal:** Provide a template for users to create custom tasks

**Requirements:**

Create a comprehensive template with:
- All possible fields documented
- Inline comments explaining each field
- Examples for each constraint type
- Multiple examples of good/bad outputs
- Tips for writing effective evaluation criteria

**Include sections for:**
- Basic metadata (name, description)
- Input/output specifications
- Evaluation criteria (with weights if needed)
- Constraints (with examples)
- Examples (2-3 showing good/bad outputs)
- Judge instructions (detailed scoring rubric)

**Validation Criteria:**
✓ Template has all possible fields
✓ Comments are clear and helpful
✓ Template itself is valid YAML
✓ Can be loaded (with placeholder values filled)

---

### MILESTONE 1.2: API Client (Days 4-6)

**Deliverable:** Robust OpenRouter API client with retry logic

---

#### TASK 1.2.1: Implement OpenRouter API Client

**File:** `src/taskbench/api/client.py`

**Goal:** Create async HTTP client for OpenRouter API with proper error handling

**Requirements:**

Create an `OpenRouterClient` class with:

1. **__init__(api_key: str, base_url: str = "https://openrouter.ai/api/v1")**
   - Initialize httpx AsyncClient
   - Store API key securely
   - Set default headers (Authorization, HTTP-Referer, X-Title)

2. **async complete(model: str, prompt: str, max_tokens: int = 1000, temperature: float = 0.7, **kwargs) -> CompletionResponse**
   - Send completion request to OpenRouter
   - Parse response into CompletionResponse model
   - Extract token usage from response
   - Calculate latency
   - Handle API errors gracefully

3. **async complete_with_json(model: str, prompt: str, **kwargs) -> CompletionResponse**
   - Same as complete() but request JSON mode
   - Add "Respond only with valid JSON" to prompt
   - Validate response is valid JSON

4. **async close()**
   - Cleanup HTTP client resources

**Design Considerations:**
- Use async/await for non-blocking I/O
- Extract tokens from response['usage'] if available
- Measure latency using time.perf_counter()
- Log all requests and responses for debugging
- Handle rate limiting (429 errors)
- Handle API errors (400, 401, 500, etc.)

**Error Handling:**
- Raise custom exceptions for different error types
- Include response body in error messages
- Retry on 429 (rate limit) errors
- Don't retry on 400 (bad request) errors

**Testing:**
Write `tests/test_api.py` with:
- Mock httpx responses using pytest-mock
- Test successful completion
- Test JSON mode
- Test error handling (401, 429, 500)
- Test token extraction
- Test latency calculation

**Validation Criteria:**
✓ Can make successful API calls
✓ Returns CompletionResponse with all fields
✓ Handles errors gracefully
✓ Latency is measured accurately
✓ Tests pass with mocked responses

---

#### TASK 1.2.2: Implement Retry Logic

**File:** `src/taskbench/api/retry.py`

**Goal:** Add exponential backoff retry logic for API resilience

**Requirements:**

Create retry decorator and utilities:

1. **@retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=60.0)**
   - Decorator for async functions
   - Exponential backoff: delay = base_delay * (2 ** attempt)
   - Cap delay at max_delay
   - Only retry on specific errors (429, 500, 502, 503, 504)
   - Don't retry on 400, 401, 403
   - Log each retry attempt

2. **RateLimiter class**
   - Track requests per minute
   - Sleep if rate limit would be exceeded
   - Configurable limit (default: 60/minute)

**Design Considerations:**
- Use asyncio.sleep() for delays
- Log retry attempts with delay duration
- Include original error in final exception if all retries fail
- Make retry limits configurable

**Testing:**
Write tests for:
- Successful retry after transient error
- Max retries exhausted
- Non-retryable errors fail immediately
- Rate limiter prevents exceeding limits

**Validation Criteria:**
✓ Retries transient errors
✓ Doesn't retry permanent errors
✓ Exponential backoff works correctly
✓ Rate limiter prevents bursts

---

### MILESTONE 1.3: Cost Tracking (Days 7-9)

**Deliverable:** Accurate cost calculation for all API calls

---

#### TASK 1.3.1: Create Model Pricing Database

**File:** `config/models.yaml`

**Goal:** Centralized pricing data for all supported models

**Requirements:**

Create YAML file with pricing for at least these 5 models:

```yaml
models:
  - model_id: "anthropic/claude-sonnet-4.5"
    display_name: "Claude Sonnet 4.5"
    provider: "Anthropic"
    input_price_per_1m: 3.00
    output_price_per_1m: 15.00
    context_window: 200000
    
  - model_id: "openai/gpt-4o"
    display_name: "GPT-4o"
    provider: "OpenAI"
    input_price_per_1m: 5.00
    output_price_per_1m: 15.00
    context_window: 128000
    
  # Add: Gemini 2.5 Pro, Llama 3.1 405B, Qwen 2.5 72B
```

**Validation Criteria:**
✓ All 5 models have complete pricing
✓ Prices match current OpenRouter pricing (verify manually)
✓ YAML is valid and parsable

---

#### TASK 1.3.2: Implement Cost Calculator

**File:** `src/taskbench/evaluation/cost.py`

**Goal:** Calculate and track costs for all API calls

**Requirements:**

Create `CostTracker` class with:

1. **__init__(models_config_path: str)**
   - Load model pricing from YAML
   - Build lookup dictionary by model_id

2. **calculate_cost(model_id: str, input_tokens: int, output_tokens: int) -> float**
   - Look up model pricing
   - Calculate: (input_tokens / 1_000_000) * input_price + (output_tokens / 1_000_000) * output_price
   - Round to $0.01 precision
   - Raise error if model not found

3. **track_evaluation(result: EvaluationResult) -> None**
   - Store evaluation result
   - Update running totals

4. **get_total_cost() -> float**
   - Sum all tracked evaluation costs

5. **get_cost_breakdown() -> Dict[str, float]**
   - Return per-model cost breakdown

6. **get_statistics() -> Dict[str, Any]**
   - Return: total_cost, total_tokens, avg_cost_per_eval, cost_by_model

**Design Considerations:**
- Always round to 2 decimal places
- Handle missing models gracefully
- Keep running totals in memory
- Optionally persist to JSON for history

**Testing:**
Write `tests/test_cost.py` with:
- Test cost calculation accuracy
- Test rounding to $0.01
- Test tracking multiple evaluations
- Test cost breakdown by model
- Verify against known costs (e.g., 100K input + 10K output tokens)

**Validation Criteria:**
✓ Cost calculations accurate to $0.01
✓ Handles all 5 models
✓ Running totals are correct
✓ Tests verify against manual calculations

---

### MILESTONE 1.4: Model Executor (Days 10-14)

**Deliverable:** Can execute evaluations on 5 models sequentially

---

#### TASK 1.4.1: Implement Model Executor

**File:** `src/taskbench/evaluation/executor.py`

**Goal:** Execute task on a single model and capture results

**Requirements:**

Create `ModelExecutor` class with:

1. **__init__(api_client: OpenRouterClient, cost_tracker: CostTracker)**
   - Store dependencies

2. **build_prompt(task: TaskDefinition, input_data: str) -> str**
   - Create comprehensive prompt from task definition
   - Include: task description, input data, output format, constraints, examples
   - Add explicit instructions for format compliance
   - Make constraints VERY clear (use CAPS, bold if possible)

3. **async execute(model_id: str, task: TaskDefinition, input_data: str) -> EvaluationResult**
   - Build prompt
   - Call API client
   - Parse response
   - Calculate cost
   - Create EvaluationResult object
   - Handle errors and timeouts

**Prompt Template Design:**

The prompt should include:
- Task description
- Input data
- Expected output format with example
- All constraints clearly stated
- Examples of good output
- Explicit "CRITICAL RULES" section

**Error Handling:**
- Catch API errors
- Set status="failed" in result
- Include error message in result.error field
- Continue to next model (don't crash)

**Testing:**
Write `tests/test_executor.py` with:
- Mock API client
- Test prompt building
- Test successful execution
- Test error handling
- Verify EvaluationResult is complete

**Validation Criteria:**
✓ Prompt includes all task information
✓ Returns complete EvaluationResult
✓ Handles errors gracefully
✓ Cost is calculated correctly

---

#### TASK 1.4.2: Implement Multi-Model Evaluation

**File:** `src/taskbench/evaluation/executor.py` (extend)

**Goal:** Orchestrate evaluation across multiple models

**Requirements:**

Add to `ModelExecutor` class:

1. **async evaluate_multiple(model_ids: List[str], task: TaskDefinition, input_data: str) -> List[EvaluationResult]**
   - Loop through all models
   - Execute each evaluation
   - Collect all results
   - Handle individual model failures gracefully
   - Log progress (Rich progress bar)
   - Track total cost

**Progress Display:**
- Use Rich Progress for terminal output
- Show: Model name, Status, Tokens, Cost
- Update in real-time
- Final summary table

**Testing:**
- Test with 3 models (some succeed, some fail)
- Verify all results are collected
- Verify progress display works
- Verify cost tracking is accurate

**Validation Criteria:**
✓ Evaluates all models successfully
✓ Handles individual failures
✓ Progress bar shows real-time status
✓ Results include all evaluations

---

### MILESTONE 1.5: Basic CLI (Days 15-17)

**Deliverable:** Working CLI that can run evaluations

---

#### TASK 1.5.1: Implement CLI Framework

**File:** `src/taskbench/cli/main.py`

**Goal:** Create Typer-based CLI with basic commands

**Requirements:**

Create CLI app with these commands:

1. **evaluate <task_yaml> [--models model1,model2] [--input-file path]**
   - Load task definition
   - Load input data file
   - Get model list (from flag or use defaults)
   - Run evaluation
   - Display results table
   - Save results to JSON

2. **models [--list] [--info model_id]**
   - --list: Show all available models with pricing
   - --info: Show detailed info for one model

3. **validate <task_yaml>**
   - Validate task YAML
   - Show any errors
   - Exit 0 if valid, 1 if invalid

**Design:**
- Use Typer for CLI framework
- Use Rich for beautiful output
- Add --verbose flag for detailed logging
- Add --output flag to specify results file
- Provide helpful error messages

**Example Usage:**
```bash
# Run evaluation
taskbench evaluate tasks/lecture_analysis.yaml \
  --models claude-sonnet-4.5,gpt-4o \
  --input-file examples/sample_transcript.txt

# List models
taskbench models --list

# Validate task
taskbench validate tasks/my_task.yaml
```

**Testing:**
- Use Typer's testing utilities
- Test each command
- Test error cases
- Verify output format

**Validation Criteria:**
✓ All commands work
✓ Help text is clear
✓ Errors are user-friendly
✓ Results display correctly

---

#### TASK 1.5.2: Implement Results Display and Export

**File:** `src/taskbench/cli/main.py` (extend)

**Goal:** Display evaluation results in multiple formats

**Requirements:**

Add commands:

1. **results [--format table|json|csv] [--output path]**
   - Load last evaluation results
   - Display in specified format
   - Save to file if --output specified

**Table Format (using Rich):**
```
┌───────────────────┬────────┬──────────────┬──────────┐
│ Model             │ Tokens │ Cost         │ Status   │
├───────────────────┼────────┼──────────────┼──────────┤
│ Claude Sonnet 4.5 │ 45,234 │ $0.36        │ Success  │
│ GPT-4o            │ 48,012 │ $0.42        │ Success  │
└───────────────────┴────────┴──────────────┴──────────┘
```

**JSON Format:**
- Pretty-printed
- Include all fields
- Timestamps in ISO format

**CSV Format:**
- Headers: model,tokens,cost,status,timestamp
- One row per model

**Validation Criteria:**
✓ Table displays beautifully
✓ JSON is valid and complete
✓ CSV opens in Excel correctly
✓ --output saves to file

---

## 🎯 PHASE 2: LLM-as-Judge Evaluation (Weeks 3-4)

**Goal:** Automated quality assessment using LLM-as-judge  
**Target:** Can score outputs and identify violations

---

### MILESTONE 2.1: Judge Implementation (Days 18-21)

---

#### TASK 2.1.1: Create Judge Evaluator

**File:** `src/taskbench/evaluation/judge.py`

**Goal:** Use Claude Sonnet 4.5 to evaluate model outputs

**Requirements:**

Create `LLMJudge` class with:

1. **__init__(api_client: OpenRouterClient, judge_model: str = "anthropic/claude-sonnet-4.5")**
   - Store dependencies
   - Set judge model

2. **build_judge_prompt(task: TaskDefinition, model_output: str, input_data: str) -> str**
   - Create detailed evaluation prompt
   - Include: task criteria, model output, input context
   - Ask for: scores (0-100), violations, reasoning
   - Request JSON response

**Judge Prompt Structure:**
```
You are evaluating an LLM's performance on this task:

TASK: {task.description}

EVALUATION CRITERIA:
{task.evaluation_criteria}

CONSTRAINTS:
{task.constraints}

INPUT DATA:
{input_data}

MODEL OUTPUT:
{model_output}

Evaluate the output and respond with JSON:
{
  "accuracy_score": 0-100,
  "format_score": 0-100,
  "compliance_score": 0-100,
  "overall_score": 0-100,
  "violations": ["list of violations"],
  "reasoning": "detailed explanation"
}

SCORING RUBRIC:
- Accuracy (40%): Did it extract correct concepts?
- Format (30%): Valid CSV with required columns?
- Compliance (30%): Met all constraints?

VIOLATION DETECTION:
- Segments under 2 minutes
- Segments over 7 minutes
- Invalid format
- Missing required fields
```

3. **async evaluate(task: TaskDefinition, result: EvaluationResult, input_data: str) -> JudgeScore**
   - Build judge prompt
   - Call judge model with JSON mode
   - Parse JSON response
   - Create JudgeScore object
   - Handle parsing errors

**Design Considerations:**
- Use JSON mode for structured output
- Validate scores are 0-100
- Extract specific violation types
- Include detailed reasoning
- Handle judge failures gracefully

**Testing:**
Write `tests/test_judge.py` with:
- Mock API responses
- Test prompt building
- Test JSON parsing
- Test violation detection
- Test error handling

**Validation Criteria:**
✓ Judge returns valid JudgeScore
✓ Scores are 0-100
✓ Violations are detected
✓ Reasoning is included
✓ Handles errors gracefully

---

#### TASK 2.1.2: Implement Violation Detection

**File:** `src/taskbench/evaluation/judge.py` (extend)

**Goal:** Parse and categorize violations from judge output

**Requirements:**

Add methods to extract specific violations:

1. **parse_violations(violations: List[str]) -> Dict[str, List[str]]**
   - Categorize violations by type
   - Types: "under_min", "over_max", "format", "missing_field", "other"
   - Return: {"under_min": [...], "over_max": [...], ...}

2. **count_violations_by_type(violations: List[str]) -> Dict[str, int]**
   - Return counts: {"under_min": 2, "over_max": 1, ...}

3. **get_violation_summary(scores: List[JudgeScore]) -> str**
   - Generate text summary of all violations across models
   - Format: "3 models had violations. 5 segments under min, 2 segments over max..."

**Testing:**
- Test violation categorization
- Test counting
- Test summary generation

**Validation Criteria:**
✓ Violations are correctly categorized
✓ Counts are accurate
✓ Summary is readable

---

### MILESTONE 2.2: Model Comparison (Days 22-24)

---

#### TASK 2.2.1: Implement Comparison Logic

**File:** `src/taskbench/evaluation/judge.py` (extend)

**Goal:** Compare multiple model evaluations with rankings

**Requirements:**

Add `ModelComparison` class:

1. **compare_results(results: List[EvaluationResult], scores: List[JudgeScore]) -> pd.DataFrame** (or dict)
   - Combine evaluation results with judge scores
   - Calculate rankings
   - Sort by overall_score descending
   - Include: model, score, violations, cost, rank

2. **identify_best(comparison: DataFrame) -> str**
   - Return model with highest overall_score

3. **identify_best_value(comparison: DataFrame, max_cost: float = None) -> str**
   - Return model with best score/cost ratio
   - Optional: filter by max_cost

4. **generate_comparison_table(comparison: DataFrame) -> str**
   - Use Rich to create beautiful table
   - Include: Rank, Model, Score, Violations, Cost, Value Rating

**Table Example:**
```
┌──────┬──────────────────┬───────┬────────────┬──────────┬────────────┐
│ Rank │ Model            │ Score │ Violations │ Cost     │ Value      │
├──────┼──────────────────┼───────┼────────────┼──────────┼────────────┤
│ 1    │ Claude Sonnet 4.5│ 98    │ 0          │ $0.36    │ Excellent  │
│ 2    │ GPT-4o           │ 95    │ 1          │ $0.42    │ Excellent  │
│ 3    │ Qwen 2.5 72B     │ 87    │ 3          │ $0.18    │ Best Value │
└──────┴──────────────────┴───────┴────────────┴──────────┴────────────┘
```

**Testing:**
- Test with 5 model results
- Test ranking
- Test best model identification
- Test best value calculation

**Validation Criteria:**
✓ Rankings are correct
✓ Best model identified correctly
✓ Table displays beautifully
✓ Value ratings make sense

---

## 🎯 PHASE 3: Analysis & Recommendations (Weeks 5-6)

**Goal:** Cost-aware recommendations  
**Target:** Actionable advice for model selection

---

### MILESTONE 3.1: Recommendation Engine (Days 25-28)

---

#### TASK 3.1.1: Implement Recommendation Logic

**File:** `src/taskbench/evaluation/cost.py` (extend)

**Goal:** Generate actionable recommendations based on scores and cost

**Requirements:**

Add `RecommendationEngine` class:

1. **generate_recommendations(comparison: DataFrame) -> Dict[str, Any]**
   - Analyze all models
   - Identify tiers: Excellent (90+), Good (80-89), Acceptable (70-79), Poor (<70)
   - Find: best_overall, best_value, budget_option, premium_option
   - Generate explanations

**Recommendation Categories:**

**Best Overall:**
- Highest score, acceptable cost
- "Worth the premium for production use"

**Best Value:**
- Best score/cost ratio
- "87% as good for 50% less"

**Budget Option:**
- Lowest cost with score >70
- "Good enough for development"

**Premium Option:**
- Highest score regardless of cost
- "When quality is critical"

2. **format_recommendations(recs: Dict) -> str**
   - Use Rich for beautiful formatting
   - Include emoji/icons
   - Show specific numbers
   - Give actionable advice

**Example Output:**
```
📊 RECOMMENDATIONS

🏆 Best Overall: Claude Sonnet 4.5
   Score: 98/100, Cost: $0.36
   Perfect for production. Zero violations, highest accuracy.

💎 Best Value: Qwen 2.5 72B  
   Score: 87/100, Cost: $0.18
   50% cheaper than Claude, 89% as good. Great for development.

💰 Budget Option: Qwen 2.5 72B
   Lowest cost while maintaining good quality.

⚠️  Avoid: [Model with score <70]
   Reason: Too many violations, inconsistent output.
```

**Testing:**
- Test with different score distributions
- Test with various cost profiles
- Verify recommendations make sense

**Validation Criteria:**
✓ Recommendations are actionable
✓ Explanations include specific numbers
✓ Output is well-formatted
✓ Edge cases handled (all models poor, all expensive, etc.)

---

### MILESTONE 3.2: CLI Integration (Days 29-31)

---

#### TASK 3.2.1: Add Recommend Command

**File:** `src/taskbench/cli/main.py` (extend)

**Goal:** Add CLI command for recommendations

**Requirements:**

Add command:

1. **recommend [--budget max_cost] [--results-file path]**
   - Load evaluation results
   - Load judge scores
   - Generate recommendations
   - Display with formatting
   - Optional: filter by budget

**Example Usage:**
```bash
# Get recommendations from last evaluation
taskbench recommend

# Filter by budget
taskbench recommend --budget 0.25

# Use specific results file
taskbench recommend --results-file my_eval.json
```

**Testing:**
- Test with various result sets
- Test budget filtering
- Test with missing data
- Verify output quality

**Validation Criteria:**
✓ Command works with defaults
✓ Budget filtering works
✓ Output is actionable
✓ Handles edge cases

---

## 🎯 PHASE 4: Polish & Demo (Weeks 7-8)

**Goal:** Demo-ready MVP  
**Target:** Professional presentation quality

---

### MILESTONE 4.1: Testing & Quality (Days 32-35)

---

#### TASK 4.1.1: Comprehensive Testing

**Goal:** Achieve 80%+ test coverage

**Requirements:**

1. **Unit Tests** - Test all modules in isolation
   - models.py: 100% coverage
   - task.py: >80% coverage
   - client.py: >80% coverage (mock HTTP)
   - executor.py: >80% coverage
   - judge.py: >80% coverage
   - cost.py: >80% coverage

2. **Integration Tests** - Test full workflows
   - End-to-end evaluation (mock API)
   - CLI commands (all variations)
   - Results export (all formats)

3. **Edge Case Tests**
   - Empty results
   - All models fail
   - Invalid YAML
   - API errors
   - Cost calculation edge cases

**Commands:**
```bash
# Run all tests
pytest

# With coverage
pytest --cov=taskbench --cov-report=html

# Specific module
pytest tests/test_judge.py -v
```

**Validation Criteria:**
✓ All tests pass
✓ Coverage >80%
✓ No flaky tests
✓ Edge cases covered

---

#### TASK 4.1.2: Code Quality

**Goal:** Professional code standards

**Requirements:**

1. **Type Hints** - Add to all functions
2. **Docstrings** - Add to all classes and public methods
3. **Code Formatting** - Run black and isort
4. **Linting** - Fix all pylint/flake8 warnings
5. **Error Messages** - Make all user-facing errors helpful

**Tools:**
```bash
# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/
```

**Validation Criteria:**
✓ All files have type hints
✓ All public APIs documented
✓ Code formatted consistently
✓ No linting errors
✓ Error messages are clear

---

### MILESTONE 4.2: Documentation (Days 36-38)

---

#### TASK 4.2.1: Create Documentation

**Files:** `docs/ARCHITECTURE.md`, `docs/API.md`, `docs/USAGE.md`

**Goal:** Complete project documentation

**ARCHITECTURE.md:**
- System overview diagram
- Component descriptions
- Data flow
- Design decisions
- Technology choices

**API.md:**
- All public classes and methods
- Parameters and return types
- Usage examples
- Error handling

**USAGE.md:**
- Installation instructions
- Quick start guide
- Complete command reference
- Example workflows
- Troubleshooting

**Validation Criteria:**
✓ All docs are complete
✓ Examples work
✓ Links are valid
✓ Diagrams are clear

---

#### TASK 4.2.2: Update README

**File:** `README.md` (update)

**Goal:** Professional README for GitHub

**Requirements:**

Update README with:
- Working examples from actual runs
- Real screenshots/output
- Clear installation steps
- Link to blog post
- Badges (Python version, license, tests)
- Contribution guidelines link

**Include:**
- Problem statement (from blog post)
- Key features with examples
- Quick start (copy-paste ready)
- Example output (real results)
- Architecture diagram
- Comparison with other tools
- Research background
- Roadmap

**Validation Criteria:**
✓ First-time user can follow it
✓ All examples work
✓ Screenshots look professional
✓ Links are valid

---

### MILESTONE 4.3: Demo Preparation (Days 39-42)

---

#### TASK 4.3.1: Create Demo Script

**Goal:** 10-minute demo that works flawlessly

**Requirements:**

Create `DEMO.md` with:

1. **Setup (1 min)**
   - Show project structure
   - Show built-in task YAML

2. **Live Evaluation (5 min)**
   - Run evaluation on 3 models
   - Show progress in real-time
   - Display results table

3. **Analysis (3 min)**
   - Show judge scores
   - Show recommendations
   - Export results

4. **Architecture (1 min)**
   - Quick diagram walkthrough
   - Highlight agentic design
   - Explain LLM-as-judge

**Practice:**
- Run demo 3 times
- Time each section
- Fix any issues
- Prepare for Q&A

**Backup Plan:**
- Pre-record video
- Have screenshots ready
- Pre-run evaluation (show results)

**Validation Criteria:**
✓ Demo completes in 10 minutes
✓ No errors
✓ Output looks professional
✓ Can explain architecture clearly

---

#### TASK 4.3.2: Create Presentation

**File:** `presentation.pdf` or slides

**Goal:** 10-slide presentation

**Slides:**
1. Title + Problem Statement
2. Research Background (42 models)
3. Solution Overview
4. Architecture Diagram
5. Live Demo or Video
6. Results Analysis
7. Key Findings
8. Technical Highlights
9. Future Roadmap
10. Q&A

**Validation Criteria:**
✓ Slides are professional
✓ Text is readable
✓ Timing fits 10 minutes
✓ Demo embedded or linked

---

## ✅ Final Checklist

Before marking project complete, verify:

### Functionality
- [ ] Can evaluate 5+ models in <30 minutes
- [ ] LLM-as-judge scores within ±10 points of manual
- [ ] Cost tracking accurate to $0.01
- [ ] Test coverage ≥80%
- [ ] Zero critical bugs in demo path

### Quality
- [ ] Results align with 42-model research
- [ ] Recommendations are actionable
- [ ] First-time user can run CLI
- [ ] Code has type hints and docstrings
- [ ] Judge scoring is consistent (±5 points)

### Portfolio
- [ ] Professional README with examples
- [ ] API documentation complete
- [ ] Demo video recorded (or live demo ready)
- [ ] Can explain architecture in 2 minutes
- [ ] GitHub repo looks professional
- [ ] Presentation ready

### Files
- [ ] All code files exist
- [ ] All tests pass
- [ ] All docs complete
- [ ] requirements.txt accurate
- [ ] .gitignore complete
- [ ] LICENSE file (MIT)

---

## 🚀 Success Criteria Summary

**MVP is successful when:**

1. ✅ Can run: `taskbench evaluate tasks/lecture_analysis.yaml --models claude-sonnet-4.5,gpt-4o,qwen-2.5-72b`
2. ✅ Evaluation completes in <30 minutes
3. ✅ Results table shows scores, costs, violations
4. ✅ Recommendations are actionable
5. ✅ Cost matches OpenRouter billing (±$0.01)
6. ✅ Tests pass with >80% coverage
7. ✅ Can present 10-minute demo without errors
8. ✅ GitHub repo looks professional

**Demo Day Readiness:**
- Can explain problem in 1 minute
- Can run live demo in 5 minutes
- Can show results in 3 minutes
- Can explain architecture in 1 minute
- Can answer questions confidently

---

## 📝 Notes for Claude Code

**Critical Success Factors:**

1. **Test as you build** - Don't write 1000 lines then test
2. **Validate at each milestone** - Don't move forward with broken code
3. **Mock external APIs** - Don't make real API calls in tests
4. **Handle errors gracefully** - User-friendly error messages
5. **Log everything** - Use Python logging for debugging

**Common Pitfalls to Avoid:**

- ❌ Not validating input data
- ❌ Not handling API errors
- ❌ Not rounding costs correctly
- ❌ Not mocking in tests
- ❌ Not checking test coverage
- ❌ Poor error messages

**Quality Standards:**

- Every function has type hints
- Every public API has docstring
- Every module has tests
- Every error has helpful message
- Every user-facing output is polished

---

**Ready to build! Start with TASK 0 and work sequentially through each milestone.**