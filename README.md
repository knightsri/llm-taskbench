# LLM TaskBench

**Task-first LLM evaluation framework** for domain experts to benchmark models on real use cases—not generic metrics.

[![Status](https://img.shields.io/badge/status-MVP%20Complete-green)]()
[![Python](https://img.shields.io/badge/python-3.11+-blue)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()

## The Problem

Existing LLM evaluation tools (DeepEval, Promptfoo, Eleuther AI) focus on generic benchmarks and are built for AI engineers. When a domain expert needs to know "which LLM best extracts action items from meeting notes" or "the most cost-effective model for bug triage," generic BLEU/ROUGE scores don't help.

## The Solution

LLM TaskBench shifts from **metric-first to task-first** evaluation:

- **Define use cases in Markdown** - Human-readable USE-CASE.md with goals and edge cases
- **Auto-generate prompts** - Framework analyzes ground truth to create optimal prompts
- **LLM-as-judge scoring** - Claude/GPT-4 evaluates outputs against your criteria
- **Cost-aware recommendations** - Not just "best" but "best for your budget"

## Key Insight

Based on testing 42+ production LLMs:

| Conventional Wisdom | Reality |
|---------------------|---------|
| Bigger models = better | 405B didn't beat 72B on our tasks |
| "Reasoning-optimized" = better reasoning | Sometimes performed *worse* |
| Higher price = higher quality | Zero correlation found |

**What actually matters:** Task-specific evaluation reveals which models excel at *your* use case.

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

## Results Organization

Results are automatically saved to `results/{usecase-name}/`:

```
results/
├── 00-lecture-concept-extraction/
│   └── 2025-12-26_233901_lecture-01-python-basics.json
├── 01-meeting-action-items/
│   └── 2025-12-26_234802_meeting-01-standup.json
└── ...
```

## Sample Use Cases & Benchmark Results

| # | Use Case | Claude Sonnet 4 | GPT-4o-mini | Key Finding |
|---|----------|-----------------|-------------|-------------|
| 00 | [Lecture Concepts](sample-usecases/00-lecture-concept-extraction/) | **93**/100 | 35/100 | GPT ignores duration constraints |
| 01 | [Meeting Actions](sample-usecases/01-meeting-action-items/) | **82**/100 | 66/100 | GPT misses implicit tasks |
| 02 | [Bug Triage](sample-usecases/02-bug-report-triage/) | **86**/100 | 75/100 | Both usable |
| 03 | [Regex Generation](sample-usecases/03-regex-generation/) | **97**/100 | 0/100 | GPT fails entirely |
| 04 | [Data Cleaning](sample-usecases/04-data-cleaning-rules/) | **88**/100 | 76/100 | Both usable |

See detailed results in each use case's `taskbench-results.md`.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CLI: taskbench run/list-usecases         │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────┴───────────────────────────────┐
│               Folder-Based Use Case Processing              │
│  UseCaseParser → DataAnalyzer → PromptGenerator (LLM)       │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────┴───────────────────────────────┐
│                    Core Evaluation                          │
│  Executor (parallel) → Judge (LLM scoring) → CostTracker    │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────┴───────────────────────────────┐
│                    OpenRouter API                           │
│          100+ LLM models (Claude, GPT-4, Gemini, etc.)      │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

| Component | Purpose | Location |
|-----------|---------|----------|
| Use Case Parser | Parse USE-CASE.md folders | `src/taskbench/usecase_parser.py` |
| Prompt Generator | LLM-driven prompt creation | `src/taskbench/prompt_generator.py` |
| Executor | Parallel model execution | `src/taskbench/evaluation/executor.py` |
| Judge | LLM-as-judge scoring | `src/taskbench/evaluation/judge.py` |
| Cost Tracker | Token/cost tracking | `src/taskbench/evaluation/cost.py` |
| CLI | Command interface | `src/taskbench/cli/main.py` |

## Documentation

- **[USAGE.md](USAGE.md)** - Full user guide with examples
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Technical architecture
- **[docs/API.md](docs/API.md)** - API reference
- **[sample-usecases/](sample-usecases/)** - Example use cases with results

## License

MIT

## Author

**Sri Bolisetty** ([@KnightSri](https://github.com/KnightSri))
