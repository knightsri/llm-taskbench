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
- `TASKBENCH_MODELS_CACHE_TTL` (hours, default 24) - Model catalog cache duration
- `MODEL_SELECTOR_LLM` (default openai/gpt-4o) - LLM used for model selection

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
- **Tier-based model selection**: Select from Quality/Value/Budget/Speed tiers.
- Run evaluations, judge now or later, view comparisons/recommendations.
- Per-use-case run list and cost summaries.

## Intelligent Model Selection

TaskBench includes an AI-powered model selector that recommends models based on your task:

**Tiers:**
- 💎 **Quality** - Premium: Claude Opus, o1, GPT-4-turbo (>$25/1M tokens)
- ⚖️ **Value** - Mid-tier: Claude Sonnet 4.5, GPT-4o, Gemini Pro ($3-25/1M)
- 💰 **Budget** - Low-cost and free models (<$3/1M tokens)
- ⚡ **Speed** - Fast: Gemini Flash, GPT-4o-mini, Claude Haiku

**CLI usage:**
```bash
# Auto-select models based on use case
taskbench evaluate tasks/my_task.yaml --models auto --usecase usecases/my_usecase.yaml
```

**Programmatic usage:**
```python
from taskbench.evaluation.model_selector import select_models_for_task

# Default tiers (quality, value, budget)
result = await select_models_for_task("Extract concepts from lectures")

# Specific tiers
result = await select_models_for_task(
    "Extract concepts from lectures",
    tiers=["value", "budget", "speed"]
)

print(result["models"])  # List of recommended models with costs
print(result["suggested_test_order"])  # Top 3 to test first
```

**Performance:**
- Time: ~8 seconds (2 LLM calls)
- Cost: ~$0.007 per selection
- Cache: Model catalog cached for 24 hours (configurable)

## Commands

### Evaluate
```bash
taskbench evaluate TASK_YAML \
  --usecase usecases/concepts_extraction.yaml \
  --models auto \
  --input-file tests/fixtures/sample_transcript.txt \
  --output results/run.json \
  --chunked --chunk-chars 20000 --chunk-overlap 500 \
  --dynamic-chunk \
  --skip-judge        # optional: defer judging
```
Flags:
- `--usecase` path to use-case YAML (goal/notes drive prompts)
- `--models auto` to use use-case-driven recommendations
- `--judge/--no-judge` (default judge on), `--skip-judge` overrides
- `--models` comma-separated IDs
- `--input-file`, `--output`
- `--chunked` for long inputs; `--chunk-chars`, `--chunk-overlap` to tune sizes
- `--dynamic-chunk/--no-dynamic-chunk` to derive chunk size from selected models' context windows (default on)

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
- `tasks/lecture_analysis.yaml` â€“ sample task
- `usecases/concepts_extraction.yaml` â€“ sample use-case
- `tests/fixtures/sample_transcript.txt` â€“ sample input
- `config/models.yaml` â€“ pricing catalog
- `docs/openrouter-cost-tracking-guide.md` â€“ cost API details



