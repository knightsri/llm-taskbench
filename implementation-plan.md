# Implementation Plan (Use-Case Driven TaskBench)

## Goals
- Use-case–driven execution: use-case specs drive prompts, judge rubric, and model recommendations (no hard-wired models/constraints).
- Robust UI flow: create/switch use-cases, upload inputs, run/re-run with selected or recommended models, judge runs individually or all, show latest judgments.
- Cost visibility: per-run and per-use-case rollups (inline + generation billing) surfaced in UI/CLI.
- Cleanup legacy artifacts and keep compose/docs aligned with the current architecture.

## Tasks
1) Use-case & task alignment
   - Finalize use-case schema (goal, chunk/coverage notes, output schema hints, judge/model defaults via env `GENERAL_TASK_LLM`).
   - Align lecture task YAML to the desired output schema (JSON array with title, description, start_time, end_time).
   - Ensure executor/judge prompt builders consume use-case text (no hard-wired coverage logic).

2) Judge prompt/rubric
   - Build judge prompt from: use-case doc, generated model prompt, and input.
   - Keep rubric generic but driven by use-case notes/constraints (coverage, chunk size, depth) without hard-coded rules.
   - Wire `GENERAL_TASK_LLM` as default judge when not provided in use-case.

3) Model recommendation
   - Keep heuristics in orchestrator; no fixed allow/deny; preference order only.
   - Expose `GENERAL_TASK_LLM` as default judge/model fallback; allow auto models in CLI/UI via use-case traits.

4) Cost rollups
   - Aggregate costs/tokens per use-case across runs (from stored results).
   - Expose aggregation in API and display in UI (per-run and cumulative).

5) UI flow (Streamlit + FastAPI)
   - Create new use-case from UI (save YAML).
   - Select/switch use-case; upload input; request recommended models; edit list.
   - Launch runs; view run list per use-case; re-judge a run; judge all runs; show latest judgments.
   - Display comparison/recommendations per run; display per-use-case cost rollups.

6) Cleanup
   - Remove legacy backend/frontend compose/services; keep only CLI/UI stacks.
   - Update docs (ARCHITECTURE/USAGE/README) to match use-case layer, UI flow, cost rollups, envs.

## Notes
- Storage stays file-based under `results/<usecase>/<run_id>/`.
- No curated deny/allow list for models (future enhancement in TODO).
