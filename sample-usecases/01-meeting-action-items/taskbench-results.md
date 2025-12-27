# TaskBench Results: Meeting Action Item Extraction

**Last Run:** 2025-12-26
**Models Tested:** Claude Sonnet 4, GPT-4o-mini
**Input:** meeting-01-standup.txt (daily standup transcript)

## Summary

| Model | Overall Score | Accuracy | Format | Compliance | Cost | Latency |
|-------|--------------|----------|--------|------------|------|---------|
| **Claude Sonnet 4** | 82/100 | 89 | 100 | 71 | $0.0278 | 30.6s |
| GPT-4o-mini | 66/100 | 78 | 100 | 44 | $0.0006 | 27.2s |

**Winner:** Claude Sonnet 4 (Good quality tier)

## Detailed Results

### Claude Sonnet 4 - Score: 82/100 (Good)

**Output Preview:**
```json
{
  "meeting_id": "daily-standup-2025-01-06",
  "meeting_date": "2025-01-06",
  "action_items": [
    {
      "id": "AI-001",
      "action": "Start on the database migration script for the user table changes",
      "owner": "Sarah",
      "owner_confidence": "explicit",
      "priority": "medium",
      "status": "new"
    },
    // ... 9 more action items extracted
  ],
  "decisions": [
    {
      "decision": "Priya will work on desktop integration first while waiting for mobile specs",
      "stakeholders": ["Lisa", "Priya"]
    }
  ],
  "open_questions": [...]
}
```

**Strengths:**
- 88.9% recall (8/9 action items captured)
- 100% precision (no hallucinated items)
- Perfect JSON structure
- Correct owner attribution for all items

**Issues:**
- Some deadline type classifications differ from ground truth
- Reversed dependency relationship in one case
- Missed one open question

---

### GPT-4o-mini - Score: 66/100 (Acceptable)

**Output Preview:**
```json
{
  "meeting_id": "daily_standup_2025_01_06",
  "meeting_date": "2025-01-06",
  "action_items": [
    // 5 action items extracted (missed 4)
  ],
  "decisions": [1 decision],
  "open_questions": [1 question]
}
```

**Critical Issues:**
- **44% recall** - missed 4 of 9 action items:
  - Database migration script task
  - Stripe webhook validation task
  - Dashboard component integration
  - Security issue review
- One decision missing
- One open question missing

**What Worked:**
- Valid JSON with correct schema
- 100% precision (all extracted items are legitimate)
- Correct deadline formats

---

## Key Insights

### Why Claude Sonnet 4 Won

1. **Superior Recall**: Captured nearly all action items vs half for GPT-4o-mini
2. **Implicit Commitment Detection**: Better at finding "I'm going to..." and "Today I'll..." patterns
3. **Context Understanding**: Correctly identified task dependencies

### Why GPT-4o-mini Struggled

1. **Missed Implicit Tasks**: Didn't capture tasks stated as intentions rather than explicit assignments
2. **Conservative Extraction**: Only extracted clearly stated action items
3. **Missing Context**: Missed conversational cues about urgency

## Cost Analysis

| Model | Cost per Eval | 100x Scale | Quality/$ Ratio |
|-------|--------------|-----------|-----------------|
| Claude Sonnet 4 | $0.0278 | $2.78 | 2,950 pts/$ |
| GPT-4o-mini | $0.0006 | $0.06 | 110,000 pts/$ |

**Recommendation:** Despite higher cost, Claude Sonnet 4's superior recall makes it necessary for production use. Missing half the action items makes GPT-4o-mini output unreliable for meeting follow-ups.

## Reproduction

```bash
# Run with judge evaluation
taskbench run sample-usecases/01-meeting-action-items \
  --models anthropic/claude-sonnet-4,openai/gpt-4o-mini

# Quick test (skip judge)
taskbench run sample-usecases/01-meeting-action-items \
  --models anthropic/claude-sonnet-4 \
  --skip-judge
```

## Result Files

Full evaluation results stored in:
- `results/01-meeting-action-items/2025-12-26_234802_meeting-01-standup.json`
