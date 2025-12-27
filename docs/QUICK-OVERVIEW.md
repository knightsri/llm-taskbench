# LLM TaskBench - Quick Reference Guide

**Last Updated:** December 2025

---

## 🎯 Project at a Glance

**What:** Task-specific LLM evaluation framework
**Why:** Existing tools use generic metrics, not real use cases
**How:** Folder-based use cases + Auto-generated prompts + LLM-as-judge + Cost analysis
**Status:** MVP Complete ✅

---

## 📝 Core Concepts

### Task-First Evaluation
Instead of asking "What's the BLEU score?", ask "Which model best extracts teaching concepts from lectures?"

### Folder-Based Use Cases
Define use cases in human-readable Markdown:
```
sample-usecases/00-lecture-concept-extraction/
├── USE-CASE.md           # Human-readable description
├── data/                 # Input files
└── ground-truth/         # Expected outputs
```

### LLM-as-Judge
Claude Sonnet 4.5 evaluates model outputs:
- Scores 0-100 for accuracy, format, compliance
- Detects violations
- Provides detailed reasoning

### Cost-Aware Recommendations
Not just "which is best" but "which is best for my budget":
- Best Overall (highest score)
- Best Value (best score/cost ratio)
- Budget Option (lowest cost, acceptable quality)

---

## 🏗️ Key Components

| Component | Purpose | File |
|-----------|---------|------|
| **Use Case Parser** | Parse USE-CASE.md folders | `src/taskbench/usecase_parser.py` |
| **Prompt Generator** | LLM-driven prompt generation | `src/taskbench/prompt_generator.py` |
| **API Client** | Call OpenRouter API | `src/taskbench/api/client.py` |
| **Executor** | Run model evaluations | `src/taskbench/evaluation/executor.py` |
| **Judge** | Score model outputs | `src/taskbench/evaluation/judge.py` |
| **Cost Tracker** | Calculate & track costs | `src/taskbench/evaluation/cost.py` |
| **CLI** | Command-line interface | `src/taskbench/cli/main.py` |

---

## 🎬 Quick Start

```bash
# Install
pip install -e .
export OPENROUTER_API_KEY=sk-or-...

# List available use cases
taskbench list-usecases

# Run evaluation on a use case
taskbench run sample-usecases/00-lecture-concept-extraction \
  --models anthropic/claude-sonnet-4,openai/gpt-4o-mini

# Generate prompts without running
taskbench generate-prompts sample-usecases/00-lecture-concept-extraction
```

---

## 📊 Sample Benchmark Results

| Use Case | Claude Sonnet 4 | GPT-4o-mini |
|----------|-----------------|-------------|
| 00-lecture-concept-extraction | 93/100 | 35/100 |
| 01-meeting-action-items | 82/100 | 66/100 |
| 02-bug-report-triage | 86/100 | 75/100 |
| 03-regex-generation | 97/100 | 0/100 |
| 04-data-cleaning-rules | 88/100 | 76/100 |

**Key Findings:**
- Claude Sonnet 4 consistently outperforms on quality
- GPT-4o-mini offers better value for simpler tasks
- Regex generation shows largest capability gap

---

## 🎓 Sample Use Cases

| # | Use Case | Difficulty | Capability |
|---|----------|------------|------------|
| 00 | Lecture Concept Extraction | Moderate-Hard | Reasoning + Structured Extraction |
| 01 | Meeting Action Items | Moderate | Extraction + Inference |
| 02 | Bug Report Triage | Moderate-Hard | Classification + Reasoning |
| 03 | Regex Generation | Hard | Pattern Recognition + Logic |
| 04 | Data Cleaning Rules | Moderate-Hard | Pattern Recognition |

---

## 📦 Project Structure

```
llm-taskbench/
├── src/taskbench/           # Main package
│   ├── core/                # Task & data models
│   ├── api/                 # API clients
│   ├── evaluation/          # Executor, judge, cost
│   ├── cli/                 # CLI commands
│   ├── usecase_parser.py    # Parse folder-based use cases
│   └── prompt_generator.py  # LLM-driven prompt generation
├── sample-usecases/         # Sample use cases (folder-based)
├── results/                 # Evaluation results (by use case)
├── config/                  # Model pricing
└── docs/                    # Documentation
```

---

## 🔧 Key Commands

| Command | Description |
|---------|-------------|
| `taskbench run <folder>` | Run evaluation on use case |
| `taskbench list-usecases` | List available use cases |
| `taskbench generate-prompts <folder>` | Generate prompts only |
| `taskbench models --list` | List available models |
| `taskbench evaluate <yaml>` | Legacy YAML-based evaluation |

---

## ⚙️ Environment Variables

```bash
# Required
OPENROUTER_API_KEY=sk-or-...

# Optional
TASKBENCH_MAX_CONCURRENCY=5
TASKBENCH_PROMPT_GEN_MODEL=anthropic/claude-sonnet-4.5
TASKBENCH_MAX_TOKENS=4000
TASKBENCH_TEMPERATURE=0.7
```

---

## 📁 Results Organization

Results auto-saved by use case:
```
results/
├── 00-lecture-concept-extraction/
│   └── 2025-12-26_233901_lecture-01-python-basics.json
├── 01-meeting-action-items/
│   └── 2025-12-26_234802_meeting-01-standup.json
└── ...
```

---

## 💡 Creating Your Own Use Case

1. Create folder: `mkdir -p my-usecases/my-task/{data,ground-truth}`
2. Write `USE-CASE.md` with goal, evaluation notes, expected output
3. Add input files to `data/`
4. Add expected outputs to `ground-truth/`
5. Run: `taskbench run my-usecases/my-task --models model1,model2`

See `docs/USAGE.md` for detailed guide.

---

## 🔗 Quick Links

- **GitHub:** https://github.com/knightsri/llm-taskbench
- **Usage Guide:** `docs/USAGE.md`
- **Architecture:** `docs/ARCHITECTURE.md`
- **OpenRouter:** https://openrouter.ai

---

**Ready to evaluate! 🚀**
