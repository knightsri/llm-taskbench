# LLM TaskBench User Guide

## Install & configure
```bash
pip install -e .
export OPENROUTER_API_KEY=sk-or-...
```
Optional envs:
- `TASKBENCH_MAX_CONCURRENCY` (default 5)
- `TASKBENCH_USE_GENERATION_LOOKUP` (true/false, default true)
- `GENERAL_TASK_LLM` (default judge/model fallback)

Docker (CLI):
```bash
cp .env.example .env
# add your OPENROUTER_API_KEY to .env
docker compose -f docker-compose.cli.yml build
docker compose -f docker-compose.cli.yml run --rm taskbench-cli --help
```

UI (FastAPI + Streamlit):
```bash
cp .env.example .env
# add your OPENROUTER_API_KEY to .env
docker compose -f docker-compose.ui.yml up --build
# API at http://localhost:8000, UI at http://localhost:8501
```
Capabilities:
- Create/select use-cases, upload inputs.
- Auto-recommend models or pick manually.
- Run evaluations, judge now or later, view comparisons/recommendations.
- Per-use-case run list and cost summaries.

## Commands

### Evaluate
```bash
taskbench evaluate TASK_YAML \
  --usecase usecases/concepts_extraction.yaml \
  --models auto \
  --input-file tests/fixtures/sample_transcript.txt \
  --output results/run.json \
  --skip-judge        # optional: defer judging
```
Flags:
- `--usecase` path to use-case YAML (goal/notes drive prompts)
- `--models auto` to use use-case–driven recommendations
- `--judge/--no-judge` (default judge on), `--skip-judge` overrides
- `--models` comma-separated IDs
- `--input-file`, `--output`

### Recommend
```bash
taskbench recommend --results results/run.json
```

### Models
```bash
taskbench models --list
taskbench models --info anthropic/claude-sonnet-4.5
```

### Validate
```bash
taskbench validate tasks/my_task.yaml
```

### Sample
```bash
taskbench sample --models anthropic/claude-sonnet-4.5,openai/gpt-4o --no-judge
```

## Human-in-the-loop flow
1) `evaluate --skip-judge` to gather outputs/costs.
2) Review/edit outputs.
3) `taskbench recommend --results <file>` to judge/compare without re-calling models.

## Cost tracking
- Inline usage requested; generation lookup if inline cost missing.
- Stats include total/input/output tokens and per-model breakdowns; UI shows per-use-case rollups.

## Concurrency & resilience
- Parallel via `TASKBENCH_MAX_CONCURRENCY`.
- Retries/backoff on API calls; judge enforces JSON mode.

## Task/use-case definitions
- Tasks: `tasks/*.yaml` (see `tasks/lecture_analysis.yaml`).
- Use-cases: `usecases/*.yaml` drive prompts, judge rubric, and model recommendations.

## Files
- `tasks/lecture_analysis.yaml` – sample task
- `usecases/concepts_extraction.yaml` – sample use-case
- `tests/fixtures/sample_transcript.txt` – sample input
- `config/models.yaml` – pricing catalog
- `docs/openrouter-cost-tracking-guide.md` – cost API details
