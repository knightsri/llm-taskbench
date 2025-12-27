# Use Case: Data Cleaning Rule Generation

## Metadata
- **Name:** Data-Cleaning-Rule-Generation
- **Difficulty:** Moderate to Hard
- **Primary Capability:** Pattern Recognition + Structured Output
- **Token Range:** 2K-10K tokens per dataset

## Goal

Given a messy CSV with data quality issues, generate a list of executable cleaning rules that can transform the data into a consistent, clean format. Rules should be:
- **Actionable:** Specific enough to implement programmatically
- **Complete:** Cover all identified issues
- **Prioritized:** Most critical issues first
- **Non-destructive:** Preserve original values where uncertain

Output should include both human-readable descriptions and machine-executable transformations.

## LLM Evaluation Notes

**What this tests:**
- Pattern recognition in noisy data
- Generating actionable (not vague) transformations
- Balancing automation with human review for ambiguous cases
- Understanding data types and validation rules
- Identifying relationships between columns

**Comparison metrics:**
1. **Detection Rate:** % of known issues identified
2. **Rule Correctness:** Do rules fix issues without breaking valid data?
3. **Actionability:** Are rules specific enough to implement?
4. **False Positives:** Did model flag valid data as problematic?

**Common issues to detect:**
- Inconsistent date formats (MM/DD/YYYY vs YYYY-MM-DD vs DD-Mon-YY)
- Mixed case in categorical fields (CA, ca, California, Calif.)
- Null value variations (NULL, null, N/A, NA, -, empty string, "None")
- Encoding issues (special characters, mojibake)
- Typos in categorical fields
- Invalid values (negative ages, future dates for historical data)
- Unit inconsistencies (USD vs $, kg vs lbs)

## Expected Output Schema

```json
{
  "dataset_id": "dataset-xxx",
  "summary": {
    "total_rows": 500,
    "columns_analyzed": 10,
    "issues_found": 25,
    "critical_issues": 5
  },
  "column_analysis": [
    {
      "column": "column_name",
      "inferred_type": "date|string|number|categorical|email|phone|etc",
      "issues": [
        {
          "issue_type": "inconsistent_format|invalid_value|null_variation|typo|etc",
          "severity": "critical|high|medium|low",
          "affected_rows": 45,
          "examples": ["value1", "value2"],
          "suggested_rule": {
            "description": "Human readable description",
            "transform": "Executable transformation logic",
            "confidence": "high|medium|low",
            "requires_review": false
          }
        }
      ]
    }
  ],
  "cleaning_rules": [
    {
      "rule_id": "RULE-001",
      "priority": 1,
      "column": "column_name",
      "description": "What this rule does",
      "condition": "When to apply (SQL-like or regex)",
      "transformation": "What to do",
      "examples": {
        "before": ["messy1", "messy2"],
        "after": ["clean1", "clean2"]
      }
    }
  ],
  "flagged_for_review": [
    {
      "column": "column_name",
      "reason": "Why human review needed",
      "sample_values": ["ambiguous1", "ambiguous2"]
    }
  ]
}
```

## Sample Data Files

| File | Description |
|------|-------------|
| `data/customers-messy.csv` | Customer data with various quality issues |
| `data/transactions-messy.csv` | Transaction data with date/currency issues |
| `data/products-messy.csv` | Product catalog with categorization issues |

## Ground Truth

Located in `ground-truth/` with:
- `*-issues.json`: Known issues in each dataset
- `*-clean.csv`: Expected output after cleaning
