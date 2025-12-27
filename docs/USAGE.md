# LLM TaskBench User Guide

## Install & Configure

```bash
pip install -e .
export OPENROUTER_API_KEY=sk-or-...
```

Optional environment variables:
- `TASKBENCH_MAX_CONCURRENCY` (default 5)
- `TASKBENCH_PROMPT_GEN_MODEL` (default anthropic/claude-sonnet-4.5)
- `TASKBENCH_MAX_TOKENS` (default 4000)
- `TASKBENCH_TEMPERATURE` (default 0.7)
- `TASKBENCH_USE_GENERATION_LOOKUP` (true/false, default true)

### Docker (CLI)

```bash
cp .env.example .env
# add your OPENROUTER_API_KEY to .env
docker compose -f docker-compose.cli.yml build
docker compose -f docker-compose.cli.yml run --rm taskbench-cli --help
```

### Docker (UI)

```bash
cp .env.example .env
# add your OPENROUTER_API_KEY to .env
docker compose -f docker-compose.ui.yml up --build
# API at http://localhost:8000, UI at http://localhost:5173
```

## Folder-Based Use Cases

Use cases are organized in folders with human-friendly Markdown descriptions:

```
sample-usecases/
├── 00-lecture-concept-extraction/
│   ├── USE-CASE.md           # Human-readable description
│   ├── data/                 # Input files
│   │   ├── lecture-01-python-basics.txt
│   │   └── lecture-02-ml-fundamentals.txt
│   ├── ground-truth/         # Expected outputs
│   │   ├── lecture-01-concepts.csv
│   │   └── lecture-02-concepts.csv
│   └── generated-prompts.json  # Auto-generated (optional)
└── 01-meeting-action-items/
    └── ...
```

### USE-CASE.md Format

```markdown
# Use Case: Concepts Extraction from Lecture Transcript

## Metadata
- **Difficulty:** Moderate to Hard
- **Primary Capability:** Reasoning + Structured Extraction

## Goal
Extract distinct teaching concepts from lecture transcripts...

## LLM Evaluation Notes
**What this tests:**
- Understanding of teaching flow
- Precise timestamp boundary identification
- Rule following (duration constraints)

**Edge cases to watch:**
- Mid-lecture breaks
- Q&A sections
- Tangential stories

## Expected Output Schema
```csv
concept,start_time,end_time
01_Introduction,00:00:00,00:04:32
```

## Quality Criteria
**"Excellent" extraction:**
- All segments 2-7 minutes
- Descriptive, accurate concept names
```

## Commands

### List Available Use Cases

```bash
taskbench list-usecases
# or specify a different folder
taskbench list-usecases my-usecases/
```

### Run Evaluation

```bash
taskbench run <usecase_folder> [options]
```

Options:
- `--models` / `-m` - Comma-separated model IDs (required)
- `--data` / `-d` - Specific data file to use
- `--output` / `-o` - Output file path
- `--regenerate-prompts` - Force regenerate prompts
- `--skip-judge` - Skip judge evaluation

Examples:

```bash
# Basic evaluation with two models
taskbench run sample-usecases/00-lecture-concept-extraction \
  --models anthropic/claude-sonnet-4,openai/gpt-4o

# Use specific data file
taskbench run sample-usecases/00-lecture-concept-extraction \
  --models anthropic/claude-sonnet-4 \
  --data sample-usecases/00-lecture-concept-extraction/data/lecture-02-ml-fundamentals.txt

# Skip judge for quick runs
taskbench run sample-usecases/00-lecture-concept-extraction \
  --models anthropic/claude-sonnet-4 \
  --skip-judge

# Force regenerate prompts
taskbench run sample-usecases/00-lecture-concept-extraction \
  --models anthropic/claude-sonnet-4 \
  --regenerate-prompts
```

### Generate Prompts

Generate prompts without running evaluation:

```bash
taskbench generate-prompts <usecase_folder> [--force]
```

This creates:
- `generated-prompts.json` - All prompts and analysis
- `prompts/task-prompt.txt` - Task prompt for models
- `prompts/judge-prompt.txt` - Judge evaluation instructions
- `prompts/rubric.json` - Scoring criteria
- `prompts/analysis.json` - Use case analysis

### Legacy Commands

For backward compatibility with YAML task definitions:

```bash
# Evaluate with YAML task
taskbench evaluate tasks/lecture_analysis.yaml \
  --models anthropic/claude-sonnet-4,openai/gpt-4o \
  --input-file tests/fixtures/sample_transcript.txt

# Load saved results and recommend
taskbench recommend --results results/run.json

# List available models
taskbench models --list

# Validate task YAML
taskbench validate tasks/my_task.yaml

# Run sample task
taskbench sample --models anthropic/claude-sonnet-4 --no-judge
```

## How Prompt Generation Works

When you run a use case, the framework:

1. **Parses USE-CASE.md**
   - Extracts goal, difficulty, primary capability
   - Identifies evaluation notes and edge cases
   - Determines expected output format

2. **Analyzes Data/Ground-Truth**
   - Scans `data/` folder for input files
   - Scans `ground-truth/` folder for expected outputs
   - Matches files by naming patterns (e.g., `lecture-01` matches `lecture-01-*`)

3. **Generates Prompts via LLM**
   - Analyzes the transformation required
   - Creates task prompt with specific instructions
   - Creates judge prompt with evaluation criteria
   - Derives rubric from ground truth analysis

4. **Saves to Folder**
   - Prompts saved in `generated-prompts.json`
   - Individual prompt files in `prompts/` subfolder

## Web UI

The web UI provides an interactive interface:

1. **Select Use Case**
   - Browse available use cases from `sample-usecases/`
   - View goal, difficulty, and capability
   - See data files and ground truth pairs

2. **Configure Evaluation**
   - Select data file (if multiple available)
   - Choose models to evaluate
   - Set options (skip judge, regenerate prompts)

3. **View Results**
   - Model rankings by accuracy
   - Cost breakdown
   - Quality violations
   - Detailed comparison

## Cost Tracking

- Inline usage requested on every API call
- Generation lookup for billed costs when available
- Per-model and per-run cost totals
- Judge evaluation costs tracked separately

## Concurrency & Resilience

- Parallel model execution via `TASKBENCH_MAX_CONCURRENCY`
- Automatic retries with exponential backoff
- Rate limiting support
- JSON mode enforced for judge calls

## Results Organization

Results are automatically organized by use case:

```
results/
├── 00-lecture-concept-extraction/
│   ├── 2025-12-26_233901_lecture-01-python-basics.json
│   └── 2025-12-26_235012_lecture-02-ml-fundamentals.json
├── 01-meeting-action-items/
│   └── 2025-12-26_234802_meeting-01-standup.json
└── _legacy/
    └── (old YAML-based results)
```

Naming convention: `{YYYY-MM-DD}_{HHMMSS}_{data-file-name}.json`

Override with `--output` to specify a custom path.

## Result File Format

Each result JSON file contains comprehensive evaluation data:

```json
{
  "usecase": {
    "name": "Use Case Name",
    "folder": "path/to/use-case",
    "data_file": "path/to/input/data.txt",
    "ground_truth_file": "path/to/ground-truth/expected.json"
  },
  "prompts": {
    "analysis": {
      "transformation_type": "Description of what the LLM is doing",
      "key_fields": ["field1", "field2"],
      "quality_indicators": ["What good output looks like"],
      "comparison_strategy": "How to compare against ground truth"
    },
    "task_prompt": "The prompt sent to candidate models",
    "judge_prompt": "The prompt sent to the judge model",
    "rubric": {
      "critical_requirements": [...],
      "compliance_checks": [...],
      "weights": {"accuracy": 40, "format": 20, "compliance": 40}
    }
  },
  "results": [
    {
      "model_name": "provider/model-name",
      "task_name": "task_identifier",
      "output": "The model's raw output",
      "input_tokens": 5450,
      "output_tokens": 292,
      "total_tokens": 5742,
      "cost_usd": 0.0024,
      "billed_cost_usd": 0.002365,
      "latency_ms": 3125.32,
      "timestamp": "2025-12-27 08:44:11.602350",
      "generation_id": "gen-xxx",
      "status": "success",
      "error": null
    }
  ],
  "scores": [
    {
      "model_evaluated": "provider/model-name",
      "accuracy_score": 85,
      "format_score": 95,
      "compliance_score": 80,
      "overall_score": 85,
      "violations": ["List of specific issues found"],
      "reasoning": "Judge's explanation of the scores"
    }
  ],
  "statistics": {
    "total_cost": 0.0024,
    "total_tokens": 5742,
    "total_input_tokens": 5450,
    "total_output_tokens": 292,
    "total_evaluations": 1,
    "avg_cost_per_eval": 0.0024,
    "avg_tokens_per_eval": 5742,
    "cost_by_model": {
      "google/gemini-2.5-flash": {
        "cost": 0.0024,
        "input_tokens": 5450,
        "output_tokens": 292,
        "evaluations": 1
      }
    }
  }
}
```

### Key Fields Explained

| Section | Field | Description |
|---------|-------|-------------|
| `usecase` | `name` | Human-readable use case name |
| `usecase` | `data_file` | Input data that was processed |
| `usecase` | `ground_truth_file` | Reference output for comparison |
| `prompts` | `task_prompt` | Exact prompt sent to models |
| `prompts` | `rubric` | Scoring criteria with penalties |
| `results[]` | `output` | Raw model response |
| `results[]` | `cost_usd` | Calculated cost |
| `results[]` | `billed_cost_usd` | Actual cost from OpenRouter |
| `results[]` | `status` | "success" or "failed" |
| `scores[]` | `overall_score` | 0-100 judge score |
| `scores[]` | `violations` | Specific issues found |
| `statistics` | `total_cost` | Sum of all evaluation costs |

### Notes

- `scores` array is empty when using `--skip-judge`
- `billed_cost_usd` is the actual cost from OpenRouter (may differ from calculated)
- `generation_id` can be used for cost auditing via OpenRouter API
- Multi-model runs have multiple entries in `results[]` and `scores[]`

## Creating Your Own Use Case

### Step 1: Create Folder Structure

```bash
mkdir -p my-usecases/invoice-extraction/{data,ground-truth}
```

### Step 2: Write USE-CASE.md

```markdown
# Use Case: Invoice Data Extraction

## Metadata
- **Difficulty:** Moderate
- **Primary Capability:** Structured Extraction

## Goal
Extract key fields from invoice PDFs/images: vendor name, invoice number,
date, line items, subtotal, tax, and total.

## LLM Evaluation Notes
**What this tests:**
- Structured data extraction from semi-structured documents
- Numerical accuracy (amounts must match exactly)
- Date format normalization

**Edge cases to watch:**
- Handwritten invoices
- Multi-page invoices
- Foreign currency

## Expected Output Schema
```json
{
  "vendor": "string",
  "invoice_number": "string",
  "date": "YYYY-MM-DD",
  "line_items": [{"description": "string", "amount": number}],
  "subtotal": number,
  "tax": number,
  "total": number
}
```

## Quality Criteria
**"Excellent" extraction:**
- All amounts match exactly
- Dates normalized to ISO format
- No missing required fields
```

### Step 3: Add Data Files

Place input files in `data/`:
```
my-usecases/invoice-extraction/data/
├── invoice-001.txt
├── invoice-002.txt
└── invoice-003.txt
```

### Step 4: Add Ground Truth

Place expected outputs in `ground-truth/` with matching names:
```
my-usecases/invoice-extraction/ground-truth/
├── invoice-001.json
├── invoice-002.json
└── invoice-003.json
```

### Step 5: Generate Prompts

```bash
taskbench generate-prompts my-usecases/invoice-extraction
```

This analyzes your use case and creates optimized prompts.

### Step 6: Run Evaluation

```bash
taskbench run my-usecases/invoice-extraction \
  --models anthropic/claude-sonnet-4,openai/gpt-4o-mini
```

## Sample Benchmark Results

Results from running all sample use cases (Claude Sonnet 4 vs GPT-4o-mini):

| Use Case | Claude Sonnet 4 | GPT-4o-mini | Cost Ratio |
|----------|-----------------|-------------|------------|
| 00-lecture-concept-extraction | 93/100 | 35/100 | 31x |
| 01-meeting-action-items | 82/100 | 66/100 | 46x |
| 02-bug-report-triage | 86/100 | 75/100 | 32x |
| 03-regex-generation | 97/100 | 0/100 | 63x |
| 04-data-cleaning-rules | 88/100 | 76/100 | 38x |

Key observations:
- Claude Sonnet 4 consistently outperforms on quality
- GPT-4o-mini offers better value for simpler tasks
- Regex generation shows largest capability gap

## Files

- `sample-usecases/` - Use case folder collection
- `config/models.yaml` - Model pricing catalog
- `results/` - Evaluation results (organized by use case)
- `docs/openrouter-cost-tracking-guide.md` - Cost API details
