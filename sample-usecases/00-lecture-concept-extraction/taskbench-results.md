# TaskBench Results: Lecture Concept Extraction

**Last Run:** 2025-12-26
**Models Tested:** Claude Sonnet 4, GPT-4o-mini
**Input:** lecture-01-python-basics.txt (27 min lecture)

## Summary

| Model | Overall Score | Accuracy | Format | Compliance | Cost | Latency |
|-------|--------------|----------|--------|------------|------|---------|
| **Claude Sonnet 4** | 93/100 | 85 | 95 | 100 | $0.0186 | 4.2s |
| GPT-4o-mini | 35/100 | 65 | 95 | 12 | $0.0006 | 8.1s |

**Winner:** Claude Sonnet 4 (Excellent quality tier)

## Detailed Results

### Claude Sonnet 4 - Score: 93/100 (Excellent)

**Output Preview:**
```csv
concept,start_time,end_time
01_Introduction_And_Variables_Basics,00:00:00,00:02:42
02_Python_Data_Types_Overview,00:02:42,00:07:45
03_Operators_And_Type_Conversion,00:07:45,00:11:20
04_Conditional_Statements_And_If_Logic,00:11:20,00:15:10
05_Loops_For_And_While,00:15:10,00:20:15
06_Practical_Programming_Example,00:20:15,00:23:30
07_Summary_And_Course_Wrap_Up,00:23:30,00:27:02
```

**Strengths:**
- All 7 segments within 2-7 minute constraint (ideal 3-5 min each)
- Perfect gapless coverage from 00:00:00 to 00:27:02
- Correct naming format (XX_Title_Case_With_Underscores)
- Natural pedagogical boundaries

**Minor Issues:**
- Some boundary misalignments with ground truth (30s-2min off)
- 7 segments vs ground truth 8 (slight under-segmentation)
- Semantic differences in concept naming

---

### GPT-4o-mini - Score: 35/100 (Poor)

**Output Preview:**
```csv
concept,start_time,end_time
01_Introduction_to_Lecture,00:00:00,00:00:25
02_Variables_And_Their_Usage,00:00:25,00:02:15
03_Data_Types_In_Python,00:02:15,00:05:38
...
17_Conclusion_Of_Lecture,00:26:50,00:27:02
```

**Critical Issues:**
- **12 of 17 segments under 2-minute minimum** (critical violations)
- Severe over-segmentation (17 segments vs 8 expected)
- Many sub-1-minute segments (25s, 32s, 12s)

**What Worked:**
- Correct formatting and naming convention
- Complete coverage with no gaps
- Accurate timestamp extraction

---

## Key Insights

### Why Claude Sonnet 4 Won

1. **Constraint Adherence**: Understood and followed the 2-7 minute requirement
2. **Appropriate Granularity**: Created pedagogically meaningful segments
3. **Trade-off Balance**: Chose slightly fewer segments rather than violating duration constraints

### Why GPT-4o-mini Failed

1. **Over-Literal Interpretation**: Split on every topic shift regardless of duration
2. **Ignored Constraints**: Created segments as short as 12 seconds
3. **Lost Pedagogical Value**: Segments too short to be standalone video clips

## Cost Analysis

| Model | Cost per Eval | 30x Scale | Quality/$ Ratio |
|-------|--------------|-----------|-----------------|
| Claude Sonnet 4 | $0.0186 | $0.56 | 5,000 pts/$ |
| GPT-4o-mini | $0.0006 | $0.02 | 58,333 pts/$ |

**Recommendation:** Despite 30x lower cost, GPT-4o-mini's output is unusable for this task. Claude Sonnet 4 is the clear choice for production use.

## Reproduction

```bash
# Run with judge evaluation
taskbench run sample-usecases/00-lecture-concept-extraction \
  --models anthropic/claude-sonnet-4,openai/gpt-4o-mini

# Quick test (skip judge)
taskbench run sample-usecases/00-lecture-concept-extraction \
  --models anthropic/claude-sonnet-4 \
  --skip-judge
```

## Result Files

Full evaluation results stored in:
- `results/00-lecture-concept-extraction/lecture-eval-run-v3.json`
