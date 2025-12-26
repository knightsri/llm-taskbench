# Run Analysis – Lecture Concept Extraction (Chunked, Dynamic)

Command executed:
```
docker compose -f docker-compose.cli.yml run --rm taskbench-cli \
  evaluate tasks/lecture_analysis.yaml \
  --usecase usecases/concepts_extraction.yaml \
  --models "anthropic/claude-sonnet-4.5,openai/gpt-4o,google/gemini-2.5-flash,z-ai/glm-4.7" \
  --input-file lecture_transcript.txt \
  --output results/lecture_run.json \
  --chunked --chunk-chars 20000 --chunk-overlap 500
```

Overall:
- All 4 model calls succeeded; total cost ≈ $0.52.
- Judge scores: GPT-4o 35/100; others 0/100. Primary failures were fabricated timestamps/content beyond the transcript window, duration violations, and malformed/missing JSON.

Per-model findings:
- google/gemini-2.5-flash: Score 0/100. Fabricated timestamps far beyond transcript; segments 25–40+ minutes (exceed 8-minute cap); invented coverage. JSON shape ok but timing/coverage unusable. 11 violations.
- openai/gpt-4o: Score 35/100. JSON largely valid, but first segment under 4 minutes; many segments extend past provided window; coverage/timing issues; 18 violations. Some content drift beyond the visible transcript.
- anthropic/claude-sonnet-4.5: Score 0/100. Output truncated/invalid JSON (array not closed); timestamps fabricated up to ~2h42m while transcript shows ~22:14–25:30; fabricated topics. 7 violations.
- z-ai/glm-4.7: Score 0/100. Returned only chunk separator comments, no JSON data; missing required fields and zero coverage. 7 violations.

Underlying issues observed:
- Models hallucinated timestamps/content beyond the provided transcript range.
- Duration constraint repeatedly violated (many segments >> 8 minutes).
- Some outputs were malformed or empty (invalid JSON or missing data).

Notes on chunking:
- Chunked mode with dynamic chunk sizing (based on model context windows) was enabled. Chunk sizing worked; failures stemmed from hallucination/coverage/timestamp misuse, not chunk errors.
