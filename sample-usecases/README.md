# LLM TaskBench - Use Case Collection

A collection of evaluation use cases for benchmarking LLM reasoning and text processing capabilities.

## Use Cases

| # | Use Case | Difficulty | Primary Capability | Files | Results |
|---|----------|------------|-------------------|-------|---------|
| 00 | [Lecture Concept Extraction](00-lecture-concept-extraction/) | Moderate-Hard | Reasoning + Structured Extraction | 3 lectures | [📊 Results](00-lecture-concept-extraction/taskbench-results.md) |
| 01 | [Meeting Notes → Action Items](01-meeting-action-items/) | Moderate | Extraction + Inference | 4 transcripts | [📊 Results](01-meeting-action-items/taskbench-results.md) |
| 02 | [Bug Report Triage](02-bug-report-triage/) | Moderate-Hard | Classification + Reasoning | 25 reports | [📊 Results](02-bug-report-triage/taskbench-results.md) |
| 03 | [Regex Generation](03-regex-generation/) | Hard | Pattern Recognition + Logic | 10 challenges | [📊 Results](03-regex-generation/taskbench-results.md) |
| 04 | [Data Cleaning Rules](04-data-cleaning-rules/) | Moderate-Hard | Pattern Recognition + Structured Output | 1 dataset | [📊 Results](04-data-cleaning-rules/taskbench-results.md) |

## Benchmark Summary

Latest results comparing **Claude Sonnet 4** vs **GPT-4o-mini**:

| Use Case | Winner | Winner Score | Runner-up Score | Key Insight |
|----------|--------|--------------|-----------------|-------------|
| 00 - Lecture Concepts | Claude Sonnet 4 | 93/100 | 35/100 | GPT-4o-mini over-segments, ignoring duration constraints |
| 01 - Meeting Actions | Claude Sonnet 4 | 82/100 | 66/100 | GPT-4o-mini misses implicit commitments (44% recall) |
| 02 - Bug Triage | Claude Sonnet 4 | 86/100 | 75/100 | Both usable; Claude has fewer misclassifications |
| 03 - Regex Generation | Claude Sonnet 4 | 97/100 | 0/100 | GPT-4o-mini fails entirely on logical reasoning tasks |
| 04 - Data Cleaning | Claude Sonnet 4 | 88/100 | 76/100 | Both usable; Claude generates more complete rules |

**Overall:** Claude Sonnet 4 outperforms GPT-4o-mini across all use cases, with the gap being most significant on tasks requiring constraint following (00) and logical reasoning (03).

## Folder Structure

```
llm-taskbench/
├── README.md                          # This file
├── 00-lecture-concept-extraction/
│   ├── USE-CASE.md                    # Use case description & evaluation notes
│   ├── data/                          # Input data
│   │   ├── lecture-01-python-basics.txt
│   │   ├── lecture-02-ml-fundamentals.txt
│   │   └── lecture-03-system-design.txt
│   └── ground-truth/                  # Expected outputs
│       ├── lecture-01-concepts.csv
│       ├── lecture-01-metadata.json
│       ├── lecture-02-concepts.csv
│       ├── lecture-02-metadata.json
│       ├── lecture-03-concepts.csv
│       └── lecture-03-metadata.json
├── 01-meeting-action-items/
│   ├── USE-CASE.md                    # Use case description & evaluation notes
│   ├── data/                          # Input data
│   │   ├── meeting-01-standup.txt
│   │   ├── meeting-02-planning.txt
│   │   ├── meeting-03-incident.txt
│   │   └── meeting-04-strategy.txt
│   └── ground-truth/                  # Expected outputs
│       ├── meeting-01-standup.json
│       ├── meeting-02-planning.json
│       ├── meeting-03-incident.json
│       └── meeting-04-strategy.json
├── 02-bug-report-triage/
│   ├── USE-CASE.md
│   ├── data/
│   │   └── bug-reports.csv
│   └── ground-truth/
│       └── bug-reports-labeled.json
├── 03-regex-generation/
│   ├── USE-CASE.md
│   ├── data/
│   │   └── regex-challenges.json
│   └── ground-truth/
│       └── regex-solutions.json
└── 04-data-cleaning-rules/
    ├── USE-CASE.md
    ├── data/
    │   └── customers-messy.csv
    └── ground-truth/
        ├── customers-issues.json
        └── customers-clean.csv
```

## How to Use

### For Each Use Case:

1. **Read the USE-CASE.md** - Contains:
   - Goal and context
   - Expected output schema
   - Evaluation metrics
   - LLM notes (what this tests)

2. **Examine the data/** folder - Input files for the LLM

3. **Compare against ground-truth/** - Expected outputs for scoring

### Evaluation Flow:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ USE-CASE.md     │────▶│ Prompt + Data   │────▶│ LLM Response    │
│ (read first)    │     │ (to LLM)        │     │                 │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                        ┌─────────────────┐              │
                        │ Ground Truth    │◀─────────────┘
                        │ (compare)       │    Evaluate
                        └─────────────────┘
```

## Use Case Details

### 00 - Lecture Concept Extraction

**Input:** Lecture transcript with timestamps (text)
**Output:** CSV with concept name, start time, end time

**What it tests:**
- Understanding teaching flow and pedagogical structure
- Precise timestamp boundary identification
- Rule following (2-7 minute duration constraints)
- Handling breaks, Q&A, tangents

**Difficulty progression:**
- lecture-01: Easy (linear structure, clear transitions)
- lecture-02: Medium (includes 5-min break, more complex topics)
- lecture-03: Hard (tangents, inline Q&A, stories, break)

---

### 01 - Meeting Notes Action Item Extraction

**Input:** Meeting transcript (text)
**Output:** Structured JSON with action items, owners, deadlines, decisions

**What it tests:**
- Inference (implicit deadlines like "by end of sprint")
- Attribution (who owns ambiguous tasks)
- Distinguishing discussion from commitments

**Difficulty progression:**
- meeting-01: Easy (explicit items, clear owners)
- meeting-02: Medium (dependencies, conditional items)
- meeting-03: Hard (urgent items, unclear ownership)
- meeting-04: Hard (vague timelines, strategic decisions)

---

### 02 - Bug Report Triage Classification

**Input:** Bug report (title + description)
**Output:** Severity, component, type, root cause, duplicates

**What it tests:**
- Classification consistency
- Handling vague reports
- Duplicate detection across phrasings
- Distinguishing bugs from feature requests

---

### 03 - Regex Pattern Generation

**Input:** Positive examples (must match) + Negative examples (must not match)
**Output:** Working regex pattern

**What it tests:**
- Pure logical reasoning
- Pattern recognition
- Generalization without overfitting
- Binary pass/fail validation

---

### 04 - Data Cleaning Rule Generation

**Input:** Messy CSV with quality issues
**Output:** Actionable cleaning rules

**What it tests:**
- Pattern recognition in noisy data
- Generating executable (not vague) transformations
- Prioritization of issues
- Handling ambiguous cases

---

## Scoring Guidelines

Each use case has specific metrics in its USE-CASE.md. General quality tiers:

| Rating | Criteria |
|--------|----------|
| Excellent | Matches ground truth 90%+, handles edge cases |
| Good | Matches 75-90%, minor misses on edge cases |
| Acceptable | Matches 60-75%, usable with cleanup |
| Poor | Below 60%, significant errors |

## Adding New Use Cases

1. Create folder: `XX-use-case-name/`
2. Add `USE-CASE.md` with standard sections
3. Add `data/` folder with input files
4. Add `ground-truth/` folder with expected outputs
5. Update this README
