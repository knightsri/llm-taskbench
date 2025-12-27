# Use Case: Bug Report Triage Classification

## Metadata
- **Name:** Bug-Report-Triage-Classification
- **Difficulty:** Moderate to Hard
- **Primary Capability:** Classification + Reasoning under Ambiguity
- **Token Range:** 200-800 tokens per bug report

## Goal

Classify incoming bug reports with the following attributes:
- **Severity:** Critical / High / Medium / Low
- **Component:** UI / Backend / Database / Auth / Payments / API / Mobile / Infrastructure
- **Type:** Bug / Feature Request / Question / Documentation
- **Root Cause Category:** Code Defect / Configuration / Data Issue / Third-Party / User Error / Unknown
- **Reproducibility:** Always / Sometimes / Rare / Cannot Reproduce
- **Potential Duplicates:** List of bug IDs that might be duplicates

Output should be structured JSON suitable for automated ticketing system ingestion.

## LLM Evaluation Notes

**What this tests:**
- Classification consistency across similar reports
- Reasoning under ambiguity (vague reports)
- Identifying duplicates with different phrasings
- Distinguishing bugs from feature requests and user errors
- Handling emotional/frustrated language without bias

**Comparison metrics:**
1. **Severity Accuracy:** Match against human-labeled ground truth
2. **Component Accuracy:** Correct assignment rate
3. **Duplicate Detection:** Precision and recall on known duplicate pairs
4. **Consistency:** Same classification for rephrased versions of same issue

**Edge cases to watch:**
- Vague reports: "app crashes sometimes"
- Multi-component issues spanning several areas
- User error masquerading as bugs
- Feature requests disguised as bugs
- Reports with misleading titles

## Expected Output Schema

```json
{
  "bug_id": "BUG-XXX",
  "classification": {
    "severity": "critical|high|medium|low",
    "severity_rationale": "Brief explanation",
    "component": "ui|backend|database|auth|payments|api|mobile|infrastructure",
    "type": "bug|feature_request|question|documentation",
    "root_cause_category": "code_defect|configuration|data_issue|third_party|user_error|unknown",
    "reproducibility": "always|sometimes|rare|cannot_reproduce"
  },
  "potential_duplicates": [
    {
      "bug_id": "BUG-YYY",
      "confidence": "high|medium|low",
      "rationale": "Why this might be a duplicate"
    }
  ],
  "missing_information": ["List of info that would help triage"],
  "suggested_assignee_team": "Team name based on component"
}
```

## Severity Guidelines

| Severity | Criteria |
|----------|----------|
| Critical | Production down, data loss, security vulnerability, payments broken |
| High | Major feature broken, significant user impact, no workaround |
| Medium | Feature partially broken, workaround available, moderate impact |
| Low | Minor issues, cosmetic problems, edge cases |

## Sample Data Files

| File | Description |
|------|-------------|
| `data/bug-reports.csv` | 25 bug reports of varying quality and complexity |

## Ground Truth

Located in `ground-truth/bug-reports-labeled.json` with classifications for all reports.
