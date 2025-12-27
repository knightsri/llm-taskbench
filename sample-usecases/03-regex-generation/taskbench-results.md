# TaskBench Results: Regex Pattern Generation

**Last Run:** 2025-12-26
**Models Tested:** Claude Sonnet 4, GPT-4o-mini
**Input:** regex-challenges.json (10 pattern challenges)

## Summary

| Model | Overall Score | Accuracy | Format | Compliance | Cost |
|-------|--------------|----------|--------|------------|------|
| **Claude Sonnet 4** | 97/100 | 95 | 100 | 96 | $0.0315 |
| GPT-4o-mini | 0/100 | 0 | 0 | 0 | $0.0005 |

**Winner:** Claude Sonnet 4 (Excellent quality tier)

## Key Findings

### Claude Sonnet 4 (97/100) - Excellent

**Performance:**
- Generated working regex patterns for all 10 challenges
- Patterns correctly match positive examples
- Patterns correctly reject negative examples
- Clear explanations provided

**What This Tests:**
- Pure logical reasoning
- Pattern recognition from examples
- Generalization without overfitting
- Technical accuracy (regex syntax)

### GPT-4o-mini (0/100) - Failed

**Critical Issues:**
- Invalid JSON output or schema violations
- Failed to generate functional regex patterns
- Could not process the challenge format correctly

**Note:** This task requires strong logical reasoning, which is challenging for smaller models.

## Cost Analysis

| Model | Cost | Quality/$ |
|-------|------|-----------|
| Claude Sonnet 4 | $0.0315 | 3,079 pts/$ |
| GPT-4o-mini | $0.0005 | 0 pts/$ (failed) |

**Recommendation:** Claude Sonnet 4 (or equivalent) is required for regex generation tasks. Smaller models cannot reliably produce functional patterns.

## Reproduction

```bash
taskbench run sample-usecases/03-regex-generation \
  --models anthropic/claude-sonnet-4,openai/gpt-4o-mini

# Quick test
taskbench run sample-usecases/03-regex-generation \
  --models anthropic/claude-sonnet-4 \
  --skip-judge
```

## Result Files

- `results/03-regex-generation/2025-12-26_235300_regex-challenges.json`
