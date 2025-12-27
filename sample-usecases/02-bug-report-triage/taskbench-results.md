# TaskBench Results: Bug Report Triage Classification

**Last Run:** 2025-12-26
**Models Tested:** Claude Sonnet 4, GPT-4o-mini
**Input:** bug-reports.csv (25 bug reports)

## Summary

| Model | Overall Score | Accuracy | Format | Compliance | Cost | Violations |
|-------|--------------|----------|--------|------------|------|------------|
| **Claude Sonnet 4** | 86/100 | 82 | 100 | 85 | $0.0617 | 11 |
| GPT-4o-mini | 75/100 | 72 | 100 | 68 | $0.0019 | 24 |

**Winner:** Claude Sonnet 4 (Good quality tier)

## Key Findings

### Claude Sonnet 4 (86/100)

**Strengths:**
- Accurate severity classification for most bug reports
- Consistent component attribution
- Fewer compliance violations (11 vs 24)
- Better reasoning in classifications

**Areas for Improvement:**
- Some severity misclassifications on edge cases
- Occasional root cause ambiguity

### GPT-4o-mini (75/100)

**Strengths:**
- Valid JSON output with correct schema
- Cost-effective ($0.0019 vs $0.0617)
- Reasonable accuracy for clear-cut cases

**Issues:**
- Higher violation count (24 violations)
- Inconsistent severity assessments
- Some component misattributions

## Cost Analysis

| Model | Cost | Quality/$ |
|-------|------|-----------|
| Claude Sonnet 4 | $0.0617 | 1,394 pts/$ |
| GPT-4o-mini | $0.0019 | 39,474 pts/$ |

**Recommendation:** For production bug triage, Claude Sonnet 4's higher accuracy justifies the cost. For high-volume triage with human review, GPT-4o-mini may be sufficient as a first pass.

## Reproduction

```bash
taskbench run sample-usecases/02-bug-report-triage \
  --models anthropic/claude-sonnet-4,openai/gpt-4o-mini

# Quick test
taskbench run sample-usecases/02-bug-report-triage \
  --models anthropic/claude-sonnet-4 \
  --skip-judge
```

## Result Files

- `results/02-bug-report-triage/2025-12-26_235101_bug-reports.json`
