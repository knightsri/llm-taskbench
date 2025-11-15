# LLM TaskBench User Guide

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Command Reference](#command-reference)
- [Task Definition Guide](#task-definition-guide)
- [Example Workflows](#example-workflows)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

---

## Installation

### Prerequisites

- **Python**: 3.11 or higher
- **OpenRouter API Key**: Get one at [openrouter.ai](https://openrouter.ai)

### Install from Source

1. Clone the repository:
```bash
git clone https://github.com/yourusername/llm-taskbench.git
cd llm-taskbench
```

2. Install dependencies:
```bash
pip install -e .
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

### Verify Installation

```bash
taskbench --help
```

You should see the help message with available commands.

---

## Quick Start

### 1. Set Your API Key

Create a `.env` file in the project root:

```bash
OPENROUTER_API_KEY=your-api-key-here
```

Or export it as an environment variable:

```bash
export OPENROUTER_API_KEY=your-api-key-here
```

### 2. List Available Models

```bash
taskbench models --list
```

This shows all available models with their pricing.

### 3. Validate a Task Definition

```bash
taskbench validate tasks/lecture_analysis.yaml
```

This checks if your task definition is valid.

### 4. Run Your First Evaluation

```bash
taskbench evaluate tasks/lecture_analysis.yaml \
  --models anthropic/claude-sonnet-4.5,openai/gpt-4o \
  --input-file examples/lecture_transcript.txt \
  --output results/my_evaluation.json
```

This evaluates two models on the lecture analysis task.

### 5. View Results

Results are saved to `results/my_evaluation.json` and displayed in your terminal with:
- Comparison table showing scores, costs, and violations
- Best overall model recommendation
- Best value model recommendation
- Total cost and token usage

---

## Command Reference

### `taskbench evaluate`

Evaluate multiple LLMs on a specific task.

**Syntax:**
```bash
taskbench evaluate TASK_YAML [OPTIONS]
```

**Arguments:**
- `TASK_YAML`: Path to task definition YAML file (required)

**Options:**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--models` | `-m` | claude-sonnet-4.5,gpt-4o,qwen-2.5-72b-instruct | Comma-separated list of model IDs |
| `--input-file` | `-i` | None | Path to input data file |
| `--output` | `-o` | results/evaluation_results.json | Output file for results |
| `--judge/--no-judge` | | --judge | Run LLM-as-judge evaluation |
| `--verbose` | `-v` | False | Enable verbose logging |

**Examples:**

1. **Basic evaluation:**
```bash
taskbench evaluate tasks/my_task.yaml
```

2. **Evaluate specific models:**
```bash
taskbench evaluate tasks/my_task.yaml \
  --models anthropic/claude-sonnet-4.5,openai/gpt-4o
```

3. **With custom input file:**
```bash
taskbench evaluate tasks/my_task.yaml \
  --input-file data/input.txt \
  --output results/output.json
```

4. **Skip judge evaluation:**
```bash
taskbench evaluate tasks/my_task.yaml \
  --no-judge
```

5. **Verbose output:**
```bash
taskbench evaluate tasks/my_task.yaml \
  --verbose
```

**Output:**

The command displays:
- Progress bar during evaluation
- Real-time status for each model
- Judge scores and violations
- Comparison table
- Recommendations (best overall, best value)
- Total cost and tokens

Results are saved as JSON containing:
```json
{
  "task": { ... },
  "results": [ ... ],
  "scores": [ ... ],
  "statistics": {
    "total_cost": 0.42,
    "total_tokens": 5000,
    "total_evaluations": 3,
    "avg_cost_per_eval": 0.14,
    "avg_tokens_per_eval": 1666,
    "cost_by_model": { ... }
  }
}
```

---

### `taskbench models`

Show available models and pricing information.

**Syntax:**
```bash
taskbench models [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--list` | `-l` | List all available models with pricing |
| `--info` | `-i` | Show detailed info for specific model |

**Examples:**

1. **List all models:**
```bash
taskbench models --list
```

Output:
```
                          Available Models
┌─────────────────────────────┬──────────────────┬───────────┬───────────┬────────────┐
│ Model ID                     │ Display Name     │ Provider  │ Input $/1M│ Output $/1M│
├─────────────────────────────┼──────────────────┼───────────┼───────────┼────────────┤
│ anthropic/claude-sonnet-4.5  │ Claude Sonnet 4.5│ Anthropic │ $3.00     │ $15.00     │
│ openai/gpt-4o                │ GPT-4o           │ OpenAI    │ $5.00     │ $15.00     │
│ qwen/qwen-2.5-72b-instruct   │ Qwen 2.5 72B     │ Alibaba   │ $0.35     │ $0.40      │
└─────────────────────────────┴──────────────────┴───────────┴───────────┴────────────┘
```

2. **Get model details:**
```bash
taskbench models --info anthropic/claude-sonnet-4.5
```

Output:
```
Claude Sonnet 4.5
ID: anthropic/claude-sonnet-4.5
Provider: Anthropic
Context Window: 200,000 tokens
Input Price: $3.00 per 1M tokens
Output Price: $15.00 per 1M tokens
```

---

### `taskbench validate`

Validate a task definition file.

**Syntax:**
```bash
taskbench validate TASK_YAML
```

**Arguments:**
- `TASK_YAML`: Path to task definition YAML file (required)

**Examples:**

1. **Validate a task:**
```bash
taskbench validate tasks/my_task.yaml
```

**Success Output:**
```
✓ YAML is valid and parseable
✓ Task 'my_task' passed all validation checks

Task: my_task
Input type: text
Output format: json
Evaluation criteria: 5
Constraints: 3
```

**Error Output:**
```
✗ Task validation failed with 2 error(s):
  - min_duration_minutes (5) must be less than max_duration_minutes (3)
  - evaluation_criteria cannot be empty
```

**Common Validation Errors:**

| Error | Cause | Fix |
|-------|-------|-----|
| "input_type must be one of..." | Invalid input type | Use: "transcript", "text", "csv", or "json" |
| "output_format must be one of..." | Invalid output format | Use: "csv", "json", or "markdown" |
| "min_X must be less than max_X" | Min >= Max | Ensure min < max for all constraints |
| "evaluation_criteria cannot be empty" | No criteria specified | Add at least one evaluation criterion |
| "judge_instructions cannot be empty" | No instructions | Add judge instructions |

---

## Task Definition Guide

### Task Definition Structure

Task definitions are YAML files that describe what you want LLMs to accomplish.

**Required Fields:**
```yaml
name: "unique_task_identifier"
description: "What the LLM should accomplish"
input_type: "text"  # or "transcript", "csv", "json"
output_format: "json"  # or "csv", "markdown"
evaluation_criteria:
  - "Criterion 1"
  - "Criterion 2"
judge_instructions: |
  Detailed instructions for how to score outputs...
```

**Optional Fields:**
```yaml
constraints:
  min_duration_minutes: 2
  max_duration_minutes: 7
  required_csv_columns: ["col1", "col2"]
  custom_constraint: "value"

examples:
  - input: "Example input..."
    expected_output: "Example output..."
    quality_score: 95
    notes: "Why this is good..."
```

### Creating a Task Definition

#### Step 1: Choose a Template

Start with the provided template:
```bash
cp tasks/template.yaml tasks/my_task.yaml
```

#### Step 2: Define Basic Information

```yaml
name: "sentiment_analysis"
description: "Analyze customer reviews and classify sentiment"
input_type: "text"
output_format: "json"
```

#### Step 3: Specify Evaluation Criteria

Be specific and measurable:
```yaml
evaluation_criteria:
  - "Sentiment correctly classified (positive/negative/neutral)"
  - "Confidence score provided (0-100)"
  - "Key phrases extracted from review"
  - "Output is valid JSON"
  - "All required fields present"
```

#### Step 4: Add Constraints

Define hard requirements:
```yaml
constraints:
  required_json_keys: ["sentiment", "confidence", "key_phrases"]
  valid_sentiments: ["positive", "negative", "neutral"]
  min_confidence: 0
  max_confidence: 100
  min_key_phrases: 2
  max_key_phrases: 10
```

#### Step 5: Provide Examples

Include 2-3 examples:
```yaml
examples:
  - input: |
      "This product exceeded my expectations! Great quality and fast shipping."
    expected_output: |
      {
        "sentiment": "positive",
        "confidence": 95,
        "key_phrases": ["exceeded expectations", "great quality", "fast shipping"]
      }
    quality_score: 100
    notes: "Perfect: correct sentiment, high confidence, relevant phrases"

  - input: |
      "The item was okay, nothing special."
    expected_output: |
      {
        "sentiment": "neutral",
        "confidence": 70,
        "key_phrases": ["okay", "nothing special"]
      }
    quality_score: 90
    notes: "Good: neutral sentiment correctly identified"
```

#### Step 6: Write Judge Instructions

Be explicit about scoring:
```yaml
judge_instructions: |
  Evaluate the sentiment analysis output using this rubric:

  ACCURACY (40 points):
  - Correct sentiment classification: 30 points
  - Confidence score matches sentiment strength: 10 points
  - Deduct 30 points for wrong sentiment
  - Deduct 10 points for unrealistic confidence

  FORMAT (30 points):
  - Valid JSON: 10 points
  - All required keys present: 10 points
  - Correct data types: 10 points
  - Deduct 10 points per format violation

  COMPLIANCE (30 points):
  - Confidence in valid range (0-100): 10 points
  - 2-10 key phrases extracted: 10 points
  - Sentiment is valid value: 10 points
  - Deduct 10 points per constraint violation

  VIOLATIONS:
  List any violations found:
  - "Invalid sentiment value" if not in [positive, negative, neutral]
  - "Confidence out of range" if < 0 or > 100
  - "Too few key phrases" if < 2
  - "Too many key phrases" if > 10
  - "Missing required key" if any key is absent
  - "Invalid JSON" if not parseable

  Provide detailed reasoning explaining your scores and any violations.
```

### Common Task Types

#### 1. Text Classification

```yaml
name: "email_classification"
description: "Classify emails into categories"
input_type: "text"
output_format: "json"
evaluation_criteria:
  - "Category correctly identified"
  - "Confidence score provided"
constraints:
  valid_categories: ["urgent", "important", "normal", "spam"]
  required_json_keys: ["category", "confidence"]
```

#### 2. Data Extraction

```yaml
name: "invoice_extraction"
description: "Extract structured data from invoices"
input_type: "text"
output_format: "json"
evaluation_criteria:
  - "All fields accurately extracted"
  - "Dates in correct format"
  - "Amounts properly parsed"
constraints:
  required_json_keys: ["invoice_number", "date", "total", "vendor", "items"]
  date_format: "YYYY-MM-DD"
```

#### 3. Content Generation

```yaml
name: "blog_outline"
description: "Generate blog post outline from topic"
input_type: "text"
output_format: "markdown"
evaluation_criteria:
  - "Clear hierarchical structure"
  - "5-7 main sections"
  - "2-4 subsections per main section"
  - "Logical flow of topics"
constraints:
  min_sections: 5
  max_sections: 7
  min_subsections_per_section: 2
  max_subsections_per_section: 4
```

#### 4. Transcript Analysis

```yaml
name: "lecture_concept_extraction"
description: "Extract teaching concepts from lecture transcripts"
input_type: "transcript"
output_format: "csv"
evaluation_criteria:
  - "Timestamp accuracy"
  - "Duration compliance"
  - "Concept clarity"
constraints:
  min_duration_minutes: 2
  max_duration_minutes: 7
  required_csv_columns: ["concept", "start_time", "end_time"]
  timestamp_format: "HH:MM:SS"
```

---

## Example Workflows

### Workflow 1: Compare Models on Custom Task

**Scenario**: You want to find the best model for classifying customer support tickets.

**Steps:**

1. **Create task definition** (`tasks/ticket_classification.yaml`):
```yaml
name: "ticket_classification"
description: "Classify customer support tickets into categories"
input_type: "text"
output_format: "json"
evaluation_criteria:
  - "Category correctly identified"
  - "Priority correctly assigned"
  - "Suggested department is appropriate"
constraints:
  valid_categories: ["technical", "billing", "general", "feature_request"]
  valid_priorities: ["low", "medium", "high", "urgent"]
  required_json_keys: ["category", "priority", "department", "summary"]
judge_instructions: |
  Score based on:
  - Accuracy (40 pts): Correct category and priority
  - Format (30 pts): Valid JSON with all required keys
  - Compliance (30 pts): Values match valid options
```

2. **Prepare input data** (`data/sample_ticket.txt`):
```
Customer reports that they cannot access their account after the recent update.
They tried resetting their password but still cannot log in. They mention this
is urgent as they have an important deadline tomorrow.
```

3. **Run evaluation**:
```bash
taskbench evaluate tasks/ticket_classification.yaml \
  --models anthropic/claude-sonnet-4.5,openai/gpt-4o,qwen/qwen-2.5-72b-instruct \
  --input-file data/sample_ticket.txt \
  --output results/ticket_comparison.json
```

4. **Review results**:
- Check the comparison table in terminal
- Review detailed scores in `results/ticket_comparison.json`
- Note the best model and best value recommendations

5. **Iterate**:
- If results are poor, update judge_instructions to be more specific
- Add more examples to the task definition
- Adjust constraints based on actual output

---

### Workflow 2: Optimize Cost While Maintaining Quality

**Scenario**: You need to process thousands of documents but want to minimize costs.

**Steps:**

1. **Test with multiple models**:
```bash
taskbench evaluate tasks/document_summarization.yaml \
  --models anthropic/claude-sonnet-4.5,openai/gpt-4o,qwen/qwen-2.5-72b-instruct,google/gemini-2.0-flash-exp:free \
  --input-file data/sample_document.txt
```

2. **Analyze cost vs. quality**:
- Check the comparison table's "Value" column
- Note models with "PPP" (best value)
- Compare overall_score vs. cost_usd

3. **Test edge cases** with cheaper models:
```bash
# Test on difficult input
taskbench evaluate tasks/document_summarization.yaml \
  --models qwen/qwen-2.5-72b-instruct \
  --input-file data/difficult_document.txt
```

4. **Choose model**:
- Use expensive model (Claude, GPT-4) for critical documents
- Use cheaper model (Qwen, Gemini) for routine processing
- Free models (Gemini Free) for high-volume, lower-stakes tasks

---

### Workflow 3: Develop and Refine a Task

**Scenario**: You're creating a new task and want to ensure it's well-defined.

**Steps:**

1. **Start with template**:
```bash
cp tasks/template.yaml tasks/code_review.yaml
```

2. **Fill in basic fields**:
```yaml
name: "code_review"
description: "Review Python code and identify issues"
input_type: "text"
output_format: "json"
```

3. **Validate early and often**:
```bash
taskbench validate tasks/code_review.yaml
```

4. **Add one criterion at a time**:
```yaml
evaluation_criteria:
  - "Identify syntax errors"
```

Validate again. Then add more:
```yaml
evaluation_criteria:
  - "Identify syntax errors"
  - "Detect potential bugs"
  - "Suggest improvements"
  - "Rate code quality (0-100)"
```

5. **Test with a single model**:
```bash
taskbench evaluate tasks/code_review.yaml \
  --models anthropic/claude-sonnet-4.5 \
  --input-file examples/sample_code.py \
  --no-judge
```

6. **Review raw output**, then refine constraints:
```yaml
constraints:
  required_json_keys: ["syntax_errors", "bugs", "improvements", "quality_score"]
  min_quality_score: 0
  max_quality_score: 100
```

7. **Add judge evaluation**:
```bash
taskbench evaluate tasks/code_review.yaml \
  --models anthropic/claude-sonnet-4.5 \
  --input-file examples/sample_code.py
```

8. **Refine judge instructions** based on results.

9. **Compare models** once task is stable:
```bash
taskbench evaluate tasks/code_review.yaml \
  --models anthropic/claude-sonnet-4.5,openai/gpt-4o,qwen/qwen-2.5-72b-instruct \
  --input-file examples/sample_code.py
```

---

### Workflow 4: Batch Evaluation

**Scenario**: Evaluate multiple test cases to ensure consistency.

**Steps:**

1. **Prepare multiple input files**:
```
data/
  test_case_1.txt
  test_case_2.txt
  test_case_3.txt
```

2. **Create a shell script** (`scripts/batch_evaluate.sh`):
```bash
#!/bin/bash

MODELS="anthropic/claude-sonnet-4.5,openai/gpt-4o"
TASK="tasks/my_task.yaml"

for input_file in data/test_case_*.txt; do
  basename=$(basename "$input_file" .txt)
  output_file="results/${basename}_results.json"

  echo "Evaluating $input_file..."

  taskbench evaluate "$TASK" \
    --models "$MODELS" \
    --input-file "$input_file" \
    --output "$output_file"

  echo "Results saved to $output_file"
  echo "---"
done

echo "Batch evaluation complete!"
```

3. **Run batch evaluation**:
```bash
chmod +x scripts/batch_evaluate.sh
./scripts/batch_evaluate.sh
```

4. **Analyze results** by comparing JSON files.

---

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Required
OPENROUTER_API_KEY=your-api-key-here

# Optional (future use)
# LOG_LEVEL=INFO
# RESULTS_DIR=./results
```

### Model Pricing Configuration

Edit `config/models.yaml` to add or update model pricing:

```yaml
models:
  - model_id: "anthropic/claude-sonnet-4.5"
    display_name: "Claude Sonnet 4.5"
    provider: "Anthropic"
    input_price_per_1m: 3.00
    output_price_per_1m: 15.00
    context_window: 200000

  - model_id: "custom/my-model"
    display_name: "My Custom Model"
    provider: "Custom Provider"
    input_price_per_1m: 1.00
    output_price_per_1m: 2.00
    context_window: 100000
```

**Note**: Prices are in USD per 1 million tokens.

### Directory Structure

```
llm-taskbench/
├── tasks/              # Task definition YAML files
├── data/               # Input data files (not committed)
├── results/            # Evaluation output JSON files (not committed)
├── config/             # Configuration files
│   └── models.yaml     # Model pricing database
├── .env                # Environment variables (not committed)
└── .env.example        # Example environment file
```

---

## Troubleshooting

### Common Issues

#### 1. "OPENROUTER_API_KEY not set in environment"

**Cause**: API key is missing or not loaded.

**Solutions**:
- Check that `.env` file exists and contains `OPENROUTER_API_KEY=...`
- Verify the `.env` file is in the project root directory
- Try exporting the key directly:
  ```bash
  export OPENROUTER_API_KEY=your-key-here
  taskbench evaluate ...
  ```

#### 2. "Model 'X' not found in pricing database"

**Cause**: The model is not in `config/models.yaml`.

**Solutions**:
- List available models: `taskbench models --list`
- Check the exact model ID (case-sensitive)
- Add the model to `config/models.yaml` if it's new
- Verify you're using the OpenRouter model ID format: `provider/model-name`

#### 3. "Task validation failed"

**Cause**: Task definition has errors.

**Solutions**:
- Run `taskbench validate tasks/your_task.yaml` to see specific errors
- Common fixes:
  - Ensure `min_X < max_X` for all constraints
  - Check `input_type` is one of: "transcript", "text", "csv", "json"
  - Check `output_format` is one of: "csv", "json", "markdown"
  - Ensure `evaluation_criteria` is not empty
  - Ensure `judge_instructions` is not empty

#### 4. "Rate limit exceeded"

**Cause**: Too many requests in a short time.

**Solutions**:
- Wait a few minutes and retry
- The system has built-in retry logic with exponential backoff
- For batch evaluation, add delays between runs
- Consider upgrading your OpenRouter plan

#### 5. "Request timeout"

**Cause**: Model took too long to respond (>120s default).

**Solutions**:
- Reduce input size
- Reduce `max_tokens` parameter
- Try a faster model
- The timeout is configurable in code if needed

#### 6. "Judge returned invalid JSON"

**Cause**: Judge model didn't follow JSON format instructions.

**Solutions**:
- This is rare with Claude Sonnet 4.5
- Check `judge_instructions` are clear about JSON format
- If persistent, there may be an issue with the judge prompt
- Try running without judge (`--no-judge`) to see if models work

#### 7. "FileNotFoundError: Task definition file not found"

**Cause**: Path to task file is incorrect.

**Solutions**:
- Check the file path (use tab completion)
- Verify the file exists: `ls tasks/`
- Use relative or absolute paths correctly
- Ensure file has `.yaml` extension

#### 8. High Costs

**Cause**: Models using many tokens or expensive models chosen.

**Solutions**:
- Check token usage in results: look at `input_tokens` and `output_tokens`
- Use cheaper models for testing: `qwen/qwen-2.5-72b-instruct`
- Use free models: `google/gemini-2.0-flash-exp:free`
- Reduce input size or `max_tokens` parameter
- Use `--no-judge` to skip judge evaluation costs

#### 9. Models Performing Poorly

**Cause**: Task definition may be unclear or models not suitable.

**Solutions**:
- Add more examples to task definition
- Make evaluation criteria more specific
- Emphasize constraints more in prompt
- Try different models (some are better at certain tasks)
- Review the actual model outputs (in JSON results file)
- Refine `judge_instructions` to match your expectations

---

## Best Practices

### Task Definition

1. **Be Specific**:
   - Bad: "Output should be good"
   - Good: "Output must include exactly 5 items, each with a title and description"

2. **Use Examples**:
   - Include 2-3 high-quality examples
   - Show both good and bad outputs
   - Explain why each example is good or bad

3. **Clear Constraints**:
   - Define min/max ranges clearly
   - Use realistic constraints based on actual needs
   - Test constraints with example inputs

4. **Detailed Judge Instructions**:
   - Specify exact point values for each criterion
   - Explain what constitutes a violation
   - Provide clear scoring rubric
   - Give examples of good vs. bad outputs

5. **Iterate**:
   - Start simple, add complexity gradually
   - Test with one model before comparing many
   - Refine based on actual results

### Model Selection

1. **Start with Quality**:
   - Test with Claude Sonnet 4.5 or GPT-4o first
   - Establish baseline performance
   - Then test cheaper alternatives

2. **Consider Trade-offs**:
   - Accuracy vs. Cost
   - Speed vs. Quality
   - Context window requirements

3. **Task-Specific Models**:
   - Code tasks: Claude Sonnet 4.5 often excels
   - Creative writing: GPT-4o, Claude
   - Data extraction: Qwen, Gemini can be cost-effective
   - High-volume: Free models (Gemini Free)

### Cost Management

1. **Use Free Models for Testing**:
   ```bash
   taskbench evaluate tasks/my_task.yaml \
     --models google/gemini-2.0-flash-exp:free \
     --no-judge
   ```

2. **Start Small**:
   - Test with small input samples
   - Use `max_tokens` to limit output
   - Validate task before full evaluation

3. **Monitor Costs**:
   - Check cost statistics in output
   - Review `results/*.json` for per-model costs
   - Set budget alerts on OpenRouter

4. **Optimize**:
   - Reduce input size if possible
   - Use cheaper models for routine tasks
   - Skip judge evaluation when not needed

### Evaluation

1. **Validate First**:
   ```bash
   taskbench validate tasks/my_task.yaml
   ```

2. **Test Incrementally**:
   - Single model, no judge
   - Single model, with judge
   - Multiple models, with judge

3. **Review Outputs**:
   - Check actual model outputs in JSON files
   - Verify judge scores make sense
   - Look for patterns in violations

4. **Compare Fairly**:
   - Use same input for all models
   - Use same temperature/max_tokens
   - Run multiple times if results vary

### Production Use

1. **Version Control**:
   - Commit task definitions to git
   - Document changes to tasks
   - Track which models performed best

2. **Automation**:
   - Create scripts for batch evaluation
   - Set up scheduled evaluations
   - Automate result analysis

3. **Monitoring**:
   - Track costs over time
   - Monitor model performance trends
   - Alert on unexpected results

4. **Documentation**:
   - Document why certain models were chosen
   - Keep notes on task definition iterations
   - Share results with team

---

## Advanced Usage

### Custom Judge Model

You can use a different model as the judge by modifying the code:

```python
from taskbench.evaluation.judge import LLMJudge

judge = LLMJudge(client, judge_model="openai/gpt-4o")
```

### Custom Temperature/Max Tokens

Control generation parameters:

```python
results = await executor.evaluate_multiple(
    model_ids=model_ids,
    task=task,
    input_data=input_data,
    max_tokens=3000,  # Custom
    temperature=0.5   # Custom (lower = more deterministic)
)
```

### Programmatic Usage

Use TaskBench as a library:

```python
import asyncio
from taskbench.api.client import OpenRouterClient
from taskbench.core.task import TaskParser
from taskbench.evaluation.cost import CostTracker
from taskbench.evaluation.executor import ModelExecutor

async def evaluate_models():
    # Load task
    parser = TaskParser()
    task = parser.load_from_yaml("tasks/my_task.yaml")

    # Load input
    with open("data/input.txt") as f:
        input_data = f.read()

    # Run evaluation
    async with OpenRouterClient(api_key="your-key") as client:
        cost_tracker = CostTracker()
        executor = ModelExecutor(client, cost_tracker)

        results = await executor.evaluate_multiple(
            model_ids=["anthropic/claude-sonnet-4.5"],
            task=task,
            input_data=input_data
        )

        # Process results
        for result in results:
            print(f"{result.model_name}: {result.output}")

asyncio.run(evaluate_models())
```

---

## Getting Help

### Resources

- **GitHub Issues**: Report bugs and request features
- **Documentation**: Read ARCHITECTURE.md and API.md for technical details
- **Examples**: Check the `examples/` directory for sample tasks and workflows

### Debugging

1. **Enable Verbose Mode**:
   ```bash
   taskbench evaluate tasks/my_task.yaml --verbose
   ```

2. **Check Log Files**:
   - Logs are printed to console by default
   - Enable file logging if needed (future feature)

3. **Inspect Results**:
   - Open `results/*.json` to see detailed outputs
   - Check `scores` array for judge evaluations
   - Review `statistics` for cost breakdown

4. **Test Components Separately**:
   ```bash
   # Just validate
   taskbench validate tasks/my_task.yaml

   # Just list models
   taskbench models --list

   # Evaluate without judge
   taskbench evaluate tasks/my_task.yaml --no-judge
   ```

---

## Next Steps

Now that you're familiar with LLM TaskBench:

1. **Create your first task** using `tasks/template.yaml`
2. **Run an evaluation** on your task
3. **Iterate on your task definition** based on results
4. **Compare models** to find the best fit for your use case
5. **Integrate into your workflow** using the CLI or Python API

For technical details, see:
- [ARCHITECTURE.md](ARCHITECTURE.md) - System design and components
- [API.md](API.md) - Complete API reference

Happy evaluating!
