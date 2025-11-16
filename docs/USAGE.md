# LLM TaskBench Usage Guide

Complete guide to installing, configuring, and using LLM TaskBench.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Command Reference](#command-reference)
- [Example Workflows](#example-workflows)
- [Task Definition Guide](#task-definition-guide)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

---

## Installation

### Prerequisites

- **Python 3.11 or higher**
- **pip** (Python package manager)
- **OpenRouter API key** (get one at [openrouter.ai](https://openrouter.ai/keys))

### Option 1: Install from Source (Development)

```bash
# Clone the repository
git clone https://github.com/knightsri/llm-taskbench.git
cd llm-taskbench

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in editable mode
pip install -e .
```

### Option 2: Install Dependencies Only

If you just want to use the library without CLI:

```bash
pip install pydantic pyyaml httpx typer rich python-dotenv
```

### Verify Installation

```bash
# Check that the CLI is available
taskbench --help

# Should output:
# LLM TaskBench - Evaluate LLMs on custom tasks
#
# Commands:
#   evaluate    Evaluate one or more models on a task
#   models      Show available models and pricing information
#   validate    Validate a task definition YAML file
#   results     Display evaluation results from a JSON file
#   recommend   Generate model recommendations from evaluation results
```

---

## Quick Start

### 1. Set Up Environment Variables

Create a `.env` file in your project directory:

```bash
# Copy the example
cp .env.example .env

# Edit .env and add your API key
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

### 2. Validate a Task Definition

Check that your task definition is valid:

```bash
taskbench validate tasks/lecture_analysis.yaml
```

Output:
```
✓ Task validation passed!

Task: lecture_concept_extraction
Description: Extract teaching concepts from lecture transcripts
Input Type: transcript
Output Format: csv
...
```

### 3. List Available Models

See which models you can evaluate:

```bash
taskbench models --list
```

Output:
```
                        Available Models
┌─────────────────────────┬──────────────┬──────────┬─────────┐
│ Model ID                │ Display Name │ Provider │ Cost    │
├─────────────────────────┼──────────────┼──────────┼─────────┤
│ anthropic/claude-son... │ Claude Son...│ Anthropic│ $3/$15  │
│ openai/gpt-4o           │ GPT-4o       │ OpenAI   │ $2.5/$10│
...
```

### 4. Run Your First Evaluation

Evaluate models on a task:

```bash
taskbench evaluate tasks/lecture_analysis.yaml \
  --models "anthropic/claude-sonnet-4.5,openai/gpt-4o" \
  --input-file examples/sample_transcript.txt \
  --output results.json
```

Output:
```
Loading task definition from tasks/lecture_analysis.yaml...
✓ Task loaded: lecture_concept_extraction

Loading input data from examples/sample_transcript.txt...
✓ Loaded 15243 characters of input data

Evaluating 2 model(s):
  • anthropic/claude-sonnet-4.5
  • openai/gpt-4o

Evaluating models...
━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 2/2 ✓ 1500 tokens, $0.0225

                     Evaluation Results
┌──────────────────┬─────────┬────────┬─────────┬──────────┐
│ Model            │ Status  │ Tokens │ Cost    │ Latency  │
├──────────────────┼─────────┼────────┼─────────┼──────────┤
│ claude-sonnet... │ Success │ 1,500  │ $0.0225 │ 2,450ms  │
│ gpt-4o           │ Success │ 1,350  │ $0.0169 │ 1,980ms  │
└──────────────────┴─────────┴────────┴─────────┴──────────┘

✓ Results saved to results.json
```

---

## Configuration

### Environment Variables

LLM TaskBench uses environment variables for configuration. Create a `.env` file:

```bash
# Required: OpenRouter API Key
OPENROUTER_API_KEY=sk-or-v1-your-key-here

# Optional: Direct API Keys (if using direct provider access)
ANTHROPIC_API_KEY=sk-ant-your-key-here
OPENAI_API_KEY=sk-your-key-here
```

### Model Configuration

Model pricing and metadata are configured in `config/models.yaml`:

```yaml
models:
  - model_id: "anthropic/claude-sonnet-4.5"
    display_name: "Claude Sonnet 4.5"
    provider: "Anthropic"
    input_price_per_1m: 3.00
    output_price_per_1m: 15.00
    context_window: 200000
    description: "Anthropic's most balanced model"

  - model_id: "openai/gpt-4o"
    display_name: "GPT-4o"
    provider: "OpenAI"
    input_price_per_1m: 2.50
    output_price_per_1m: 10.00
    context_window: 128000
    description: "OpenAI's flagship model"
```

To add a new model:
1. Add an entry to `config/models.yaml`
2. Use the `model_id` in evaluation commands

---

## Command Reference

### `taskbench evaluate`

Evaluate one or more models on a task.

#### Syntax

```bash
taskbench evaluate TASK_YAML [OPTIONS]
```

#### Arguments

- `TASK_YAML` (required): Path to task definition YAML file

#### Options

| Option | Short | Type | Description |
|--------|-------|------|-------------|
| `--models` | `-m` | string | Comma-separated model IDs (default: 3 common models) |
| `--input-file` | `-i` | string | Path to input data file (default: stdin) |
| `--output` | `-o` | string | Path to save results JSON |
| `--verbose` | `-v` | flag | Enable verbose logging |

#### Examples

**Basic evaluation with default models:**
```bash
taskbench evaluate tasks/sentiment.yaml --input-file data.txt
```

**Evaluate specific models:**
```bash
taskbench evaluate tasks/sentiment.yaml \
  --models "anthropic/claude-sonnet-4.5,openai/gpt-4o,google/gemini-2.0-flash-exp" \
  --input-file input.txt \
  --output results.json
```

**Read from stdin:**
```bash
cat input.txt | taskbench evaluate tasks/sentiment.yaml
```

**Verbose output for debugging:**
```bash
taskbench evaluate tasks/sentiment.yaml -i input.txt -v
```

---

### `taskbench models`

Show available models and pricing information.

#### Syntax

```bash
taskbench models [OPTIONS]
```

#### Options

| Option | Short | Type | Description |
|--------|-------|------|-------------|
| `--list` | `-l` | flag | List all available models |
| `--info` | | string | Show detailed info about a model |
| `--verbose` | `-v` | flag | Enable verbose logging |

#### Examples

**List all models:**
```bash
taskbench models --list
```

Output:
```
                        Available Models
┌─────────────────────────┬──────────────┬──────────┬─────────┐
│ Model ID                │ Display Name │ Provider │ Cost    │
├─────────────────────────┼──────────────┼──────────┼─────────┤
│ anthropic/claude-son... │ Claude Son...│ Anthropic│ $3/$15  │
...
```

**Get details about a specific model:**
```bash
taskbench models --info "anthropic/claude-sonnet-4.5"
```

Output:
```
╭────────────────────────────────────────────────╮
│    Model Info: anthropic/claude-sonnet-4.5     │
├────────────────────────────────────────────────┤
│                                                │
│ Claude Sonnet 4.5                              │
│ Provider: Anthropic                            │
│ Model ID: anthropic/claude-sonnet-4.5          │
│                                                │
│ Pricing:                                       │
│   Input:  $3.00 per 1M tokens                  │
│   Output: $15.00 per 1M tokens                 │
│                                                │
│ Context Window: 200,000 tokens                 │
│                                                │
╰────────────────────────────────────────────────╯
```

---

### `taskbench validate`

Validate a task definition YAML file.

#### Syntax

```bash
taskbench validate TASK_YAML [OPTIONS]
```

#### Arguments

- `TASK_YAML` (required): Path to task definition file

#### Options

| Option | Short | Type | Description |
|--------|-------|------|-------------|
| `--verbose` | `-v` | flag | Enable verbose logging |

#### Examples

**Validate a task:**
```bash
taskbench validate tasks/sentiment.yaml
```

Success output:
```
Validating task definition: tasks/sentiment.yaml

✓ Task YAML loaded successfully
  Name: sentiment_analysis
  Input Type: text
  Output Format: json

✓ Task validation passed!

╭─────────────────────────────────────────╮
│             Task Details                │
├─────────────────────────────────────────┤
│ Task: sentiment_analysis                │
│ Description: Classify sentiment         │
│ Input Type: text                        │
│ Output Format: json                     │
│                                         │
│ Evaluation Criteria (3):                │
│   • Correct classification              │
│   • Confidence score                    │
│   • JSON format                         │
│                                         │
│ Constraints (2):                        │
│   • required_fields: [...list...]       │
│   • valid_sentiments: [...list...]      │
╰─────────────────────────────────────────╯
```

Error output:
```
✗ Task validation failed:

  ✗ Task name is required
  ✗ min_duration_minutes (10) cannot be greater than max_duration_minutes (7)
```

---

### `taskbench results`

Display evaluation results from a JSON file.

#### Syntax

```bash
taskbench results RESULTS_FILE [OPTIONS]
```

#### Arguments

- `RESULTS_FILE` (required): Path to results JSON file (default: results.json)

#### Options

| Option | Short | Type | Description |
|--------|-------|------|-------------|
| `--format` | `-f` | string | Output format: table, json, csv (default: table) |
| `--output` | `-o` | string | Save output to file |
| `--verbose` | `-v` | flag | Enable verbose logging |

#### Examples

**Display as table (default):**
```bash
taskbench results results.json
```

**Export as CSV:**
```bash
taskbench results results.json --format csv --output results.csv
```

**Display as JSON:**
```bash
taskbench results results.json --format json
```

---

### `taskbench recommend`

Generate model recommendations from evaluation results.

#### Syntax

```bash
taskbench recommend [OPTIONS]
```

#### Options

| Option | Short | Type | Description |
|--------|-------|------|-------------|
| `--results-file` | `-r` | string | Path to results JSON (default: results.json) |
| `--scores-file` | `-s` | string | Path to judge scores JSON (default: scores.json) |
| `--budget` | `-b` | float | Maximum budget per request in USD |
| `--output` | `-o` | string | Save recommendations to JSON |
| `--verbose` | `-v` | flag | Enable verbose logging |

#### Examples

**Basic recommendations:**
```bash
taskbench recommend --results-file results.json --scores-file scores.json
```

**Filter by budget (max 10 cents per request):**
```bash
taskbench recommend --budget 0.10
```

**Save recommendations:**
```bash
taskbench recommend --output recommendations.json
```

Output:
```
╭───────────────────────────────────────────╮
│        Model Recommendations              │
╰───────────────────────────────────────────╯

Performance Tier Summary
┌─────────────┬────────┬─────────────┐
│ Tier        │ Models │ Score Range │
├─────────────┼────────┼─────────────┤
│ 🏆 Excellent│   2    │    90-100   │
│ ⭐ Good     │   1    │    80-89    │
│ ✓ Acceptable│   0    │    70-79    │
│ ✗ Poor      │   0    │     0-69    │
└─────────────┴────────┴─────────────┘

╭───────────────────────────────────────────╮
│    🏆 Best Overall Performance            │
├───────────────────────────────────────────┤
│ anthropic/claude-sonnet-4.5               │
│                                           │
│ Overall Score: 97/100                     │
│ Accuracy: 95 | Format: 100 | Compliance: 98│
│ Cost: $0.0225                             │
│ Violations: 0                             │
╰───────────────────────────────────────────╯

Use Case Recommendations
┌────────────────┬─────────────────────────────┐
│ Use Case       │ Recommendation              │
├────────────────┼─────────────────────────────┤
│ 🎯 General     │ claude-sonnet-4.5 (97 pts)  │
│ 🚀 Production  │ claude-sonnet-4.5 (highest) │
│ 💰 Cost        │ gemini-flash (free, 90 pts) │
│ 💵 Budget      │ gpt-4o-mini ($0.003, 85)    │
│ 🔧 Development │ gemini-flash (free)         │
└────────────────┴─────────────────────────────┘

✓ Recommendations saved to recommendations.json
```

---

## Example Workflows

### Workflow 1: Evaluate Models on a Custom Task

**Goal:** Compare 3 models on a sentiment analysis task.

```bash
# Step 1: Create task definition
cat > tasks/sentiment.yaml << 'EOF'
name: "sentiment_analysis"
description: "Classify customer feedback sentiment"
input_type: "text"
output_format: "json"
evaluation_criteria:
  - "Correct sentiment classification"
  - "Confidence score between 0-1"
  - "Valid JSON format"
constraints:
  required_fields: ["sentiment", "confidence"]
  valid_sentiments: ["positive", "negative", "neutral"]
judge_instructions: |
  Check:
  1. Sentiment is correct (positive/negative/neutral)
  2. Confidence is between 0 and 1
  3. Output is valid JSON with required fields
EOF

# Step 2: Validate the task
taskbench validate tasks/sentiment.yaml

# Step 3: Create input data
echo "This product exceeded my expectations!" > input.txt

# Step 4: Run evaluation
taskbench evaluate tasks/sentiment.yaml \
  --models "anthropic/claude-sonnet-4.5,openai/gpt-4o,google/gemini-2.0-flash-exp" \
  --input-file input.txt \
  --output results.json

# Step 5: View results
taskbench results results.json
```

---

### Workflow 2: Budget-Constrained Model Selection

**Goal:** Find the cheapest model that meets quality requirements.

```bash
# Step 1: Run evaluation on budget models
taskbench evaluate tasks/my_task.yaml \
  --models "google/gemini-2.0-flash-exp,openai/gpt-4o-mini,anthropic/claude-haiku-3.5" \
  --input-file input.txt \
  --output budget_results.json

# Step 2: Generate recommendations with budget limit
taskbench recommend \
  --results-file budget_results.json \
  --budget 0.01 \
  --output budget_recs.json

# The recommendation will show:
# - Best model within budget
# - Cost savings vs premium models
# - Performance trade-offs
```

---

### Workflow 3: Production Model Selection

**Goal:** Select the best model for production use.

```bash
# Step 1: Evaluate premium models
taskbench evaluate tasks/production_task.yaml \
  --models "anthropic/claude-sonnet-4.5,openai/gpt-4o,anthropic/claude-opus-4" \
  --input-file production_input.txt \
  --output prod_results.json

# Step 2: Analyze with judge scores (manual scoring for now)
# In the future, this would be automated

# Step 3: Get recommendations
taskbench recommend \
  --results-file prod_results.json \
  --output prod_recs.json

# Look for "Production" recommendation in output
```

---

### Workflow 4: Batch Evaluation

**Goal:** Evaluate models on multiple inputs.

```bash
#!/bin/bash
# batch_evaluate.sh

TASK="tasks/sentiment.yaml"
MODELS="anthropic/claude-sonnet-4.5,openai/gpt-4o"
INPUT_DIR="data/inputs"
OUTPUT_DIR="results"

mkdir -p "$OUTPUT_DIR"

for input_file in "$INPUT_DIR"/*.txt; do
  filename=$(basename "$input_file" .txt)
  echo "Processing $filename..."

  taskbench evaluate "$TASK" \
    --models "$MODELS" \
    --input-file "$input_file" \
    --output "$OUTPUT_DIR/${filename}_results.json"
done

echo "Batch evaluation complete!"
```

Run it:
```bash
chmod +x batch_evaluate.sh
./batch_evaluate.sh
```

---

## Task Definition Guide

### Task YAML Structure

A task definition is a YAML file with the following structure:

```yaml
name: "unique_task_name"
description: "What the task does"

input_type: "transcript"  # or "text", "csv", "json"
output_format: "csv"       # or "json", "markdown"

evaluation_criteria:
  - "First criterion to evaluate"
  - "Second criterion"
  - "Third criterion"

constraints:
  # Task-specific constraints
  min_duration_minutes: 2
  max_duration_minutes: 7
  required_csv_columns: ["column1", "column2"]
  timestamp_format: "HH:MM:SS"

examples:
  - input: "Example input text..."
    expected_output: "Expected output format..."
    quality_score: 95
    notes: "Notes about this example"

judge_instructions: |
  Detailed instructions for the LLM judge on how to evaluate outputs.

  Include:
  - Scoring rubrics
  - What to look for
  - How to calculate scores
  - What constitutes violations
```

### Field Descriptions

#### Required Fields

- **name**: Unique identifier for the task (use snake_case)
- **description**: Human-readable description of what the task does
- **input_type**: Type of input data
  - `transcript`: Long-form transcriptions
  - `text`: General text
  - `csv`: CSV data
  - `json`: JSON data
- **output_format**: Expected output format
  - `csv`: Comma-separated values
  - `json`: JSON object/array
  - `markdown`: Markdown formatted text
- **evaluation_criteria**: List of what to evaluate (3-5 items recommended)
- **judge_instructions**: Detailed rubric for LLM-as-judge

#### Optional Fields

- **constraints**: Dictionary of task-specific constraints
  - Use `min_*` and `max_*` for ranges
  - Use `required_*` for mandatory fields
  - Use descriptive names
- **examples**: List of example inputs/outputs
  - Helps models understand expectations
  - Include quality scores for reference

### Examples

#### Example 1: Text Classification Task

```yaml
name: "email_priority"
description: "Classify emails by priority (urgent, normal, low)"

input_type: "text"
output_format: "json"

evaluation_criteria:
  - "Correct priority assignment"
  - "Reasoning provided"
  - "JSON format compliance"

constraints:
  required_fields: ["priority", "reasoning"]
  valid_priorities: ["urgent", "normal", "low"]

examples:
  - input: "URGENT: Server down! Production impacted!"
    expected_output: '{"priority": "urgent", "reasoning": "Critical system failure"}'
    quality_score: 100

judge_instructions: |
  Evaluate based on:
  1. ACCURACY (50%): Is the priority correct?
  2. REASONING (30%): Is the explanation clear?
  3. FORMAT (20%): Valid JSON with required fields?
```

#### Example 2: Data Extraction Task

```yaml
name: "invoice_extraction"
description: "Extract structured data from invoices"

input_type: "text"
output_format: "json"

evaluation_criteria:
  - "All required fields extracted"
  - "Amounts are numeric"
  - "Dates in correct format"
  - "JSON structure correct"

constraints:
  required_fields: ["invoice_number", "date", "total", "items"]
  date_format: "YYYY-MM-DD"
  amount_format: "numeric (not string)"

judge_instructions: |
  Check for:
  - All required fields present
  - Date in YYYY-MM-DD format
  - Total is a number, not string
  - Items is an array

  Violations:
  - Missing field: -20 points
  - Wrong format: -10 points
  - Invalid JSON: 0 points
```

---

## Troubleshooting

### Common Issues

#### 1. API Key Not Found

**Error:**
```
ValueError: OpenRouter API key not found. Please provide api_key parameter
or set OPENROUTER_API_KEY environment variable.
```

**Solution:**
```bash
# Create .env file
echo "OPENROUTER_API_KEY=sk-or-v1-your-key" > .env

# Or export directly
export OPENROUTER_API_KEY=sk-or-v1-your-key
```

---

#### 2. Task Validation Failed

**Error:**
```
✗ Task validation failed:
  ✗ min_duration_minutes (10) cannot be greater than max_duration_minutes (7)
```

**Solution:**
Fix the constraints in your task YAML:
```yaml
constraints:
  min_duration_minutes: 2    # Must be less than max
  max_duration_minutes: 7
```

---

#### 3. Model Not Found

**Error:**
```
ValueError: Model 'anthropic/claude-4' not found in configuration.
Available models: [...]
```

**Solution:**
- Check the model ID spelling
- List available models: `taskbench models --list`
- Add the model to `config/models.yaml` if needed

---

#### 4. Rate Limited

**Error:**
```
OpenRouterAPIError: Rate limit exceeded. Please retry after a delay.
```

**Solution:**
The retry logic should handle this automatically. If it persists:
- Wait a few minutes
- Reduce the number of concurrent requests
- Check your OpenRouter rate limits

---

#### 5. Timeout Errors

**Error:**
```
OpenRouterAPIError: Request timed out after 120 seconds
```

**Solution:**
```python
# Increase timeout programmatically
from taskbench.evaluation.executor import ModelExecutor

executor = ModelExecutor(timeout=300.0)  # 5 minutes
```

Or for very long inputs, consider chunking the data.

---

#### 6. Out of Memory

**Error:**
```
MemoryError: Unable to allocate memory
```

**Solution:**
- Process smaller batches of inputs
- Use streaming for large files
- Reduce the number of parallel evaluations

---

#### 7. Invalid JSON from Judge

**Error:**
```
ValueError: Judge model returned invalid JSON
```

**Solution:**
This usually happens with smaller models. Try:
- Use a more capable judge model (Claude Sonnet 4.5 recommended)
- Simplify your judge instructions
- Add more explicit formatting requirements

---

### Debug Mode

Enable verbose logging to see detailed information:

```bash
taskbench evaluate tasks/my_task.yaml -i input.txt -v
```

This shows:
- API requests and responses
- Token counts and costs
- Detailed error messages
- Internal processing steps

---

### Getting Help

If you encounter issues:

1. **Check the logs**: Use `-v` flag for verbose output
2. **Validate your task**: Run `taskbench validate` first
3. **Check model availability**: Run `taskbench models --list`
4. **Review examples**: Check `/examples` directory
5. **Read the docs**: See [ARCHITECTURE.md](ARCHITECTURE.md) and [API.md](API.md)
6. **Open an issue**: [GitHub Issues](https://github.com/knightsri/llm-taskbench/issues)

---

## Best Practices

### 1. Task Definition

**DO:**
- Write clear, specific descriptions
- Include 3-5 evaluation criteria
- Provide examples
- Use explicit constraints
- Write detailed judge instructions

**DON'T:**
- Be vague about requirements
- Over-constrain the task
- Skip validation
- Forget to specify output format

### 2. Model Selection

**DO:**
- Start with a diverse set of models
- Consider budget constraints early
- Use the orchestrator for suggestions
- Test with small inputs first

**DON'T:**
- Evaluate too many models at once (costs add up)
- Ignore cost tracking
- Skip budget-friendly options

### 3. Evaluation

**DO:**
- Validate tasks before evaluation
- Use appropriate input sizes
- Save results to files
- Track costs carefully
- Test with sample data first

**DON'T:**
- Run expensive evaluations on large batches without testing
- Ignore failed evaluations
- Skip cost summaries

### 4. Cost Management

**DO:**
- Check model pricing before evaluation
- Use `--budget` flag for cost control
- Start with cheaper models for testing
- Monitor the cost summary after each run

**DON'T:**
- Run evaluations without checking costs
- Use expensive models for development/testing
- Ignore cost breakdowns

### 5. Production Use

**DO:**
- Use consistent task definitions
- Version your task YAML files
- Save all results and scores
- Monitor costs over time
- Set up automated evaluations

**DON'T:**
- Change task definitions between evaluations
- Lose track of results
- Skip validation steps

---

## Advanced Usage

### Using as a Python Library

```python
import asyncio
from taskbench.core.task import TaskParser
from taskbench.evaluation.executor import ModelExecutor

async def main():
    # Load task
    parser = TaskParser()
    task = parser.load_from_yaml("tasks/my_task.yaml")

    # Execute
    executor = ModelExecutor()
    results = await executor.evaluate_multiple(
        ["anthropic/claude-sonnet-4.5", "openai/gpt-4o"],
        task,
        "Input data here"
    )

    # Process results
    for result in results:
        print(f"{result.model_name}: ${result.cost_usd:.4f}")

    # Get cost summary
    print(executor.get_cost_summary())

if __name__ == "__main__":
    asyncio.run(main())
```

### Custom Model Configuration

Add your own models to `config/models.yaml`:

```yaml
models:
  # ... existing models ...

  - model_id: "custom/my-model"
    display_name: "My Custom Model"
    provider: "CustomProvider"
    input_price_per_1m: 1.00
    output_price_per_1m: 5.00
    context_window: 100000
    description: "My custom fine-tuned model"
```

### Batch Processing Script

```python
#!/usr/bin/env python3
import asyncio
import json
from pathlib import Path
from taskbench.core.task import TaskParser
from taskbench.evaluation.executor import ModelExecutor

async def batch_evaluate(task_file, input_dir, output_dir, models):
    """Evaluate multiple inputs in batch."""
    parser = TaskParser()
    task = parser.load_from_yaml(task_file)

    executor = ModelExecutor()
    input_files = Path(input_dir).glob("*.txt")

    for input_file in input_files:
        print(f"Processing {input_file.name}...")

        input_data = input_file.read_text()
        results = await executor.evaluate_multiple(models, task, input_data)

        # Save results
        output_file = Path(output_dir) / f"{input_file.stem}_results.json"
        output_file.parent.mkdir(exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump([r.model_dump() for r in results], f, indent=2, default=str)

        print(f"  Saved to {output_file}")

    # Final cost summary
    print("\n" + executor.get_cost_summary())

if __name__ == "__main__":
    asyncio.run(batch_evaluate(
        task_file="tasks/sentiment.yaml",
        input_dir="data/inputs",
        output_dir="results",
        models=["anthropic/claude-sonnet-4.5", "openai/gpt-4o"]
    ))
```

---

## Next Steps

1. **Explore Examples**: Check the `/examples` directory for more use cases
2. **Read Architecture**: See [ARCHITECTURE.md](ARCHITECTURE.md) to understand the system design
3. **API Reference**: Check [API.md](API.md) for detailed API documentation
4. **Create Custom Tasks**: Design tasks specific to your use case
5. **Contribute**: Submit issues and pull requests on GitHub

---

## Resources

- **GitHub Repository**: https://github.com/knightsri/llm-taskbench
- **OpenRouter API**: https://openrouter.ai/docs
- **Issue Tracker**: https://github.com/knightsri/llm-taskbench/issues
- **Model Pricing**: https://openrouter.ai/models

---

For questions or support, please open an issue on GitHub or check the documentation in the `/docs` directory.
