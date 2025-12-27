# Use Case: Concepts Extraction from Lecture Transcript

## Metadata
- **Name:** Concepts-Extraction-from-Lecture-Transcript
- **Difficulty:** Moderate to Hard
- **Primary Capability:** Reasoning + Structured Extraction + Rule Following
- **Token Range:** 10K-100K+ tokens per transcript

## Goal

Extract distinct teaching concepts from lecture transcripts that can be turned into bite-size video clips. Each concept should be:
- **Self-contained:** A complete teaching moment that makes sense on its own
- **Appropriately sized:** 3-6 minutes ideal, 2 minute minimum, 7 minute maximum
- **Searchable:** Clear, descriptive concept names for easy discovery
- **Gapless:** Full transcript coverage with no content lost between concepts

Output should be a clean CSV with concept name, start time, and end time for video clipping.

## LLM Evaluation Notes

**What this tests:**
- Understanding of teaching flow and pedagogical structure
- Precise timestamp boundary identification
- Rule following (duration constraints)
- Balancing granularity vs coherence
- Handling natural lecture patterns (tangents, Q&A, breaks)

**Comparison metrics:**
1. **Concept Count:** Compare to baseline (not too few, not over-segmented)
2. **Duration Violations:** Segments under 2 min or over 7 min
3. **Concept Quality:** Are names descriptive and accurate?
4. **Coverage:** Does extraction cover full lecture without gaps?
5. **Boundary Precision:** Do clips start/end at natural transition points?

**Edge cases to watch:**
- Mid-lecture breaks (should be excluded, not create gaps)
- Q&A sections (may be separate concept or merged with preceding topic)
- Tangential stories (include with relevant concept or separate?)
- Repeated references to earlier concepts
- Transitions between major topics

## Duration Rules

| Duration | Action |
|----------|--------|
| < 2 minutes | Too short - combine with adjacent concept |
| 2-3 minutes | Acceptable if concept is truly atomic |
| 3-6 minutes | Ideal range - target this |
| 6-7 minutes | Acceptable for complex topics |
| > 7 minutes | Too long - must split into sub-concepts |

## Expected Output Schema

```csv
concept,start_time,end_time
01_Introduction_to_Topic_Name,00:00:00,00:04:32
02_Core_Concept_Definition,00:04:32,00:08:15
03_Practical_Example_Walkthrough,00:08:15,00:14:22
...
```

**Naming convention:**
- Format: `XX_Concept_Name_In_Title_Case`
- Use underscores, no spaces
- Be descriptive but concise (3-7 words)
- Number sequentially (01, 02, 03...)

## Sample Data Files

| File | Description | Duration | Complexity |
|------|-------------|----------|------------|
| `data/lecture-01-python-basics.txt` | Intro programming lecture | ~45 min | Easy |
| `data/lecture-02-ml-fundamentals.txt` | Machine learning overview | ~60 min | Medium |
| `data/lecture-03-system-design.txt` | System design with tangents | ~75 min | Hard |

## Ground Truth

Located in `ground-truth/` folder with:
- `lecture-XX-concepts.csv`: Expected concept extraction
- `lecture-XX-metadata.json`: Additional context and reasoning

## Quality Criteria

**"Excellent" extraction:**
- All segments 2-7 minutes
- Descriptive, accurate concept names
- Natural boundary points
- Complete coverage

**"Good" extraction:**
- 1-2 minor duration violations
- Names mostly accurate
- Minor boundary issues
- Near-complete coverage

**"Acceptable" extraction:**
- 3-5 duration violations
- Some vague concept names
- Some awkward boundaries
- Small gaps acceptable

**"Poor" extraction:**
- 6+ duration violations OR
- Major gaps in coverage OR
- Inaccurate concept names OR
- Ignores lecture structure

## Evaluation Script

```python
import csv
from datetime import datetime

def parse_time(time_str):
    """Parse HH:MM:SS to seconds."""
    parts = time_str.split(':')
    return int(parts[0])*3600 + int(parts[1])*60 + int(parts[2])

def evaluate_extraction(csv_path):
    results = {
        "total_concepts": 0,
        "violations_under_2min": 0,
        "violations_over_7min": 0,
        "durations": []
    }
    
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            start = parse_time(row['start_time'])
            end = parse_time(row['end_time'])
            duration_min = (end - start) / 60
            
            results["total_concepts"] += 1
            results["durations"].append(duration_min)
            
            if duration_min < 2:
                results["violations_under_2min"] += 1
            if duration_min > 7:
                results["violations_over_7min"] += 1
    
    results["avg_duration"] = sum(results["durations"]) / len(results["durations"])
    results["total_violations"] = results["violations_under_2min"] + results["violations_over_7min"]
    
    return results
```
