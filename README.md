# LLM TaskBench

Task-first LLM evaluation with agentic orchestration, parallel execution, LLM-as-judge, and cost-aware recommendations.

## What it does
- Run multiple LLMs in parallel on the same task input.
- Capture all artifacts: prompts, outputs, timing, token usage, inline/billed costs, generation IDs.
- Judge outputs (automated or human-in-the-loop) for accuracy, format, compliance, coverage, and depth.
- Compare and recommend best overall and best value models.
- Replay or re-judge saved runs without re-calling models.

## Quick start
1) Install deps  
```bash
pip install -e .
```

2) Set your key  
```bash
export OPENROUTER_API_KEY=sk-or-...
```

Docker (optional)  
```bash
cp .env.example .env
# add your OPENROUTER_API_KEY to .env
docker compose -f docker-compose.cli.yml build
docker compose -f docker-compose.cli.yml run --rm taskbench-cli --help
```

3) Run a sample evaluation (parallel, no judge)  
```bash
taskbench sample \
  --models anthropic/claude-sonnet-4.5,openai/gpt-4o,qwen/qwen-2.5-72b-instruct \
  --no-judge
```

4) Run your own task  
```bash
taskbench evaluate tasks/lecture_analysis.yaml \
 --models anthropic/claude-sonnet-4.5,openai/gpt-4o \
 --input-file tests/fixtures/sample_transcript.txt \
 --output results/my_run.json \
  --chunked --chunk-chars 20000 --chunk-overlap 500 --dynamic-chunk \
  --skip-judge      # or omit to judge automatically
```

5) Judge/recommend later from saved results  
```bash
taskbench recommend --results results/my_run.json
```

## Key commands
- `taskbench evaluate` – run models on a task; `--skip-judge` to defer judging; parallel by default.
- `--usecase` to load use-case goals/notes and drive prompts; `--models auto` to pick recommended models.
- `--chunked` (plus `--chunk-chars/--chunk-overlap` and `--dynamic-chunk/--no-dynamic-chunk`) for long inputs; dynamic chunking sizes to the smallest selected model’s context window by default.
- `taskbench recommend` – load a saved results JSON, render comparison, and recommendations.
- `taskbench models` – list priced models from `config/models.yaml`.
- `taskbench validate` – validate task YAML.
- `taskbench sample` – run the bundled lecture task on the sample transcript.

## Concurrency & resilience
- Parallelism via `TASKBENCH_MAX_CONCURRENCY` (default 5).
- Built-in retries/backoff and optional rate limiting.
- JSON mode enforced for judge calls.

## Cost tracking
- Inline usage requested on every call (`usage.include=true`).
- Billed cost fetched from `/generation?id=...` when available.
- Results store `generation_id`, `billed_cost_usd`, token counts, and per-model totals.

## Human-in-the-loop judging
- Run `evaluate --skip-judge`, inspect/edit outputs, then `recommend` to judge and compare.
- Saved runs include all artifacts needed to re-judge without re-calling models.

## Files to know
- `tasks/lecture_analysis.yaml` – sample task.
- `tests/fixtures/sample_transcript.txt` – sample input.
- `config/models.yaml` – pricing catalog.
- `docs/openrouter-cost-tracking-guide.md` – billing details.

## Environment variables
- `OPENROUTER_API_KEY` (required)
- `TASKBENCH_MAX_CONCURRENCY` (default 5)
- `TASKBENCH_USE_GENERATION_LOOKUP` (true/false, default true)

Docker notes:
- Compose file: `docker-compose.cli.yml`
- Volumes: mounts `tasks/`, `config/`, `results/`, `tests/fixtures/`
- Override command, e.g.:
  ```bash
  docker compose -f docker-compose.cli.yml run --rm taskbench-cli \
    evaluate tasks/lecture_analysis.yaml --models anthropic/claude-sonnet-4.5 --input-file tests/fixtures/sample_transcript.txt --skip-judge
  ```

## UI status
- Available: FastAPI backend + Streamlit UI for task selection/upload, model selection, run launch, and judging/recommendations.
- Run UI via Docker:
  ```bash
  cp .env.example .env
  # add your OPENROUTER_API_KEY
  docker compose -f docker-compose.ui.yml up --build
  # API at http://localhost:8000, UI at http://localhost:8501
  ```
- The UI reuses the same executor/judge flow; runs are stored under `results/` and can be re-judged later.
- Use-cases can be created/selected in UI; runs are grouped per use-case with cost summaries.

## How comparison works
- Each model run yields `EvaluationResult` with timings, tokens, inline/billed costs.
- Judge produces `JudgeScore` (accuracy, format, compliance, violations, reasoning).
- `ModelComparison` merges results + scores, ranks by overall_score, shows value (score/cost).
- `RecommendationEngine` surfaces best overall, best value, budget-friendly options.

## Status
Core CLI, parallel executor, judge integration, cost tracking, comparison, and recommendations are implemented. Docs reflect current behavior; additional polish (more tasks, UI) can be layered on top.
