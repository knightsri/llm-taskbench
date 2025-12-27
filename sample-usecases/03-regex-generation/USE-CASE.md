# Use Case: Regex Pattern Generation from Examples

## Metadata
- **Name:** Regex-Pattern-Generation
- **Difficulty:** Hard
- **Primary Capability:** Logical Reasoning + Pattern Recognition
- **Token Range:** 500-1500 tokens per challenge

## Goal

Given sets of positive examples (strings that MUST match) and negative examples (strings that must NOT match), generate a working regular expression that:
- Matches ALL positive examples
- Rejects ALL negative examples
- Is reasonably concise (not overly complex)
- Uses standard regex syntax (PCRE or JavaScript compatible)

## LLM Evaluation Notes

**What this tests:**
- Pure logical reasoning with no domain knowledge shortcuts
- Pattern recognition in string data
- Understanding of regex syntax and semantics
- Ability to generalize from examples without overfitting

**Comparison metrics:**
1. **Correctness:** Binary pass/fail - does the regex work on all test cases?
2. **Generalization:** Performance on held-out test cases not shown to the model
3. **Conciseness:** Length of regex (shorter is better for equivalent correctness)
4. **Validity:** Is the regex syntactically valid and executable?

**Why this task is hard:**
- Models often produce regex that looks plausible but fails edge cases
- Temptation to overfit (just OR all positive examples)
- Requires understanding what makes examples similar/different
- Must balance specificity vs generality

## Expected Output Schema

```json
{
  "challenge_id": "REGEX-XXX",
  "regex": "^pattern$",
  "explanation": "Brief description of what the pattern matches",
  "confidence": "high|medium|low",
  "test_results": {
    "positive_matches": 10,
    "positive_total": 10,
    "negative_rejected": 5,
    "negative_total": 5
  }
}
```

## Sample Data Files

| File | Description |
|------|-------------|
| `data/regex-challenges.json` | 10 pattern challenges with positive/negative examples |

## Ground Truth

Located in `ground-truth/regex-solutions.json` with working regex patterns and additional test cases.

## Evaluation Script

```python
import re

def evaluate_regex(pattern, positives, negatives):
    try:
        compiled = re.compile(pattern)
    except re.error:
        return {"valid": False, "error": "Invalid regex syntax"}
    
    results = {
        "valid": True,
        "positive_pass": sum(1 for s in positives if compiled.fullmatch(s)),
        "positive_total": len(positives),
        "negative_pass": sum(1 for s in negatives if not compiled.fullmatch(s)),
        "negative_total": len(negatives)
    }
    results["perfect"] = (
        results["positive_pass"] == results["positive_total"] and
        results["negative_pass"] == results["negative_total"]
    )
    return results
```
