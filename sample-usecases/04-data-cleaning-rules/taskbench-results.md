# TaskBench Results: Data Cleaning Rule Generation

**Last Run:** 2025-12-26
**Models Tested:** Claude Sonnet 4, GPT-4o-mini
**Input:** customers-messy.csv (32 rows with quality issues)

## Summary

| Model | Overall Score | Cost |
|-------|--------------|------|
| **Claude Sonnet 4** | 88/100 | $0.0683 |
| GPT-4o-mini | 76/100 | $0.0018 |

**Winner:** Claude Sonnet 4 (Good quality tier)

## Key Findings

### Claude Sonnet 4 (88/100)

**Strengths:**
- Identified all major data quality issues
- Generated actionable cleaning rules with examples
- Correct priority ordering of rules
- Included before/after transformation examples

**What This Tests:**
- Pattern recognition in noisy data
- Generating executable transformations
- Prioritization of issues
- Handling ambiguous cases

### GPT-4o-mini (76/100)

**Strengths:**
- Valid JSON output structure
- Identified most obvious issues
- Cost-effective at $0.0018

**Areas for Improvement:**
- Missed some subtle data quality patterns
- Less detailed transformation rules
- Fewer examples provided

## Cost Analysis

| Model | Cost | Quality/$ |
|-------|------|-----------|
| Claude Sonnet 4 | $0.0683 | 1,289 pts/$ |
| GPT-4o-mini | $0.0018 | 42,222 pts/$ |

**Recommendation:** Both models produce usable output. Choose based on volume:
- High-stakes/low-volume: Claude Sonnet 4
- Initial discovery/high-volume: GPT-4o-mini with human review

## Reproduction

```bash
taskbench run sample-usecases/04-data-cleaning-rules \
  --models anthropic/claude-sonnet-4,openai/gpt-4o-mini

# Quick test
taskbench run sample-usecases/04-data-cleaning-rules \
  --models anthropic/claude-sonnet-4 \
  --skip-judge
```

## Result Files

- `results/04-data-cleaning-rules/2025-12-26_235612_customers-messy.json`
