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

## Files

- `sample-usecases/` - Use case folder collection
- `config/models.yaml` - Model pricing catalog
- `results/` - Evaluation results
- `docs/openrouter-cost-tracking-guide.md` - Cost API details
