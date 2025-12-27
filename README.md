# LLM TaskBench

Task-first LLM evaluation with folder-based use cases, automatic prompt generation, LLM-as-judge, and cost-aware recommendations.

## What it does
- Define use cases in human-friendly Markdown (USE-CASE.md)
- Automatically derive prompts from use case analysis + ground truth
- Run multiple LLMs in parallel on the same task
- Judge outputs against ground truth for accuracy, format, and compliance
- Compare and recommend best overall and best value models
- Track all costs with inline and billed cost tracking

## Quick start

1) Install deps
```bash
pip install -e .
```

2) Set your key
```bash
export OPENROUTER_API_KEY=sk-or-...
```

3) List available use cases
```bash
taskbench list-usecases
```

4) Run evaluation on a use case
```bash
taskbench run sample-usecases/00-lecture-concept-extraction \
  --models anthropic/claude-sonnet-4,openai/gpt-4o
```

5) Generate prompts for a use case (without running)
```bash
taskbench generate-prompts sample-usecases/00-lecture-concept-extraction
```

## Folder-Based Use Cases

Use cases are now organized in folders with:
- `USE-CASE.md` - Human-friendly description with goal, evaluation notes, edge cases
- `data/` - Input data files
- `ground-truth/` - Expected outputs for comparison

```
sample-usecases/
├── 00-lecture-concept-extraction/
│   ├── USE-CASE.md
│   ├── data/
│   │   ├── lecture-01-python-basics.txt
│   │   ├── lecture-02-ml-fundamentals.txt
│   │   └── lecture-03-system-design.txt
│   └── ground-truth/
│       ├── lecture-01-concepts.csv
│       ├── lecture-02-concepts.csv
│       └── lecture-03-concepts.csv
├── 01-meeting-action-items/
│   └── ...
└── 02-bug-report-triage/
    └── ...
```

The framework automatically:
1. Parses USE-CASE.md for goal, evaluation notes, edge cases
2. Matches data files to ground truth by naming patterns
3. Uses LLM to analyze and generate task prompts and judge rubrics
4. Saves generated prompts to `generated-prompts.json` for reuse

## Key Commands

### Run Evaluation
```bash
taskbench run <usecase_folder> [options]
```
Options:
- `--models` / `-m` - Comma-separated model IDs
- `--data` / `-d` - Specific data file to use (if multiple)
- `--output` / `-o` - Output file path
- `--regenerate-prompts` - Force regenerate prompts
- `--skip-judge` - Skip judge evaluation

### List Use Cases
```bash
taskbench list-usecases [folder]
```

### Generate Prompts
```bash
taskbench generate-prompts <usecase_folder> [--force]
```

### Legacy Commands
- `taskbench evaluate` - Run with YAML task definition
- `taskbench recommend` - Load saved results and recommend
- `taskbench models` - List priced models
- `taskbench validate` - Validate task YAML
- `taskbench sample` - Run bundled sample task

## Docker

```bash
cp .env.example .env
# add your OPENROUTER_API_KEY to .env

# CLI mode
docker compose -f docker-compose.cli.yml build
docker compose -f docker-compose.cli.yml run --rm taskbench-cli list-usecases

# UI mode
docker compose -f docker-compose.ui.yml up --build
# API at http://localhost:8000, UI at http://localhost:5173
```

## UI

The web UI provides:
- Browse and select use cases from `sample-usecases/`
- View use case details, data files, and ground truth
- Generate and preview prompts
- Select models to evaluate
- Run evaluations and view results
- Compare model performance with cost tracking

## How Prompt Generation Works

When you run a use case, the framework:

1. **Parses USE-CASE.md** - Extracts goal, evaluation notes, edge cases
2. **Analyzes Data/Ground-Truth** - Matches input files to expected outputs
3. **Generates Prompts via LLM** - Creates:
   - Task prompt for model execution
   - Judge prompt for output evaluation
   - Rubric with compliance checks and scoring weights
4. **Saves to Folder** - Prompts saved in `generated-prompts.json`

Example generated rubric:
```json
{
  "critical_requirements": [
    {"name": "duration_bounds", "description": "Segments 2-7 minutes", "penalty": 8}
  ],
  "compliance_checks": [
    {"check": "timestamp_format", "severity": "HIGH", "penalty": 5}
  ],
  "weights": {"accuracy": 40, "format": 20, "compliance": 40}
}
```

## Environment Variables

- `OPENROUTER_API_KEY` (required)
- `TASKBENCH_MAX_CONCURRENCY` (default 5)
- `TASKBENCH_PROMPT_GEN_MODEL` (default anthropic/claude-sonnet-4.5)
- `TASKBENCH_MAX_TOKENS` (default 4000)
- `TASKBENCH_TEMPERATURE` (default 0.7)
- `TASKBENCH_USE_GENERATION_LOOKUP` (true/false, default true)

## Cost Tracking

- Inline usage requested on every call
- Billed cost fetched from `/generation?id=...` when available
- Results store token counts, generation IDs, and per-model totals
- Judge evaluation costs tracked separately

## Sample Use Cases

| # | Use Case | Difficulty | Capability |
|---|----------|------------|------------|
| 00 | Lecture Concept Extraction | Moderate-Hard | Reasoning + Structured Extraction |
| 01 | Meeting Action Items | Moderate | Extraction + Inference |
| 02 | Bug Report Triage | Moderate-Hard | Classification + Reasoning |
| 03 | Regex Generation | Hard | Pattern Recognition + Logic |
| 04 | Data Cleaning Rules | Moderate-Hard | Pattern Recognition |

## Status

Folder-based use cases, automatic prompt generation, parallel executor, judge integration, cost tracking, and UI are implemented. Legacy YAML-based tasks still supported.
