# LLM TaskBench - Quick Reference Guide

**Last Updated:** November 2025

---

## 🎯 Project at a Glance

**What:** Task-specific LLM evaluation framework  
**Why:** Existing tools use generic metrics, not real use cases  
**How:** YAML task definition + Multi-model evaluation + LLM-as-judge + Cost analysis  
**Timeline:** 6-8 weeks (Oct 27 - Dec 22, 2025)  
**Status:** Phase 0 complete ✅, Phase 1 starting 🔜

---

## 📝 Core Concepts

### Task-First Evaluation
Instead of asking "What's the BLEU score?", ask "Which model best extracts teaching concepts from lectures?"

### LLM-as-Judge
Use Claude Sonnet 4.5 to evaluate other models' outputs:
- Scores 0-100 for accuracy, format, compliance
- Detects violations
- Provides reasoning

### Cost-Aware Recommendations
Not just "which is best" but "which is best for my budget":
- Best Overall (highest score, acceptable cost)
- Best Value (best score/cost ratio)
- Budget Option (lowest cost, good enough quality)

---

## 🏗️ Key Components

| Component | Purpose | File |
|-----------|---------|------|
| **Task Parser** | Load & validate YAML tasks | `src/taskbench/core/task.py` |
| **API Client** | Call OpenRouter/Anthropic/OpenAI | `src/taskbench/api/client.py` |
| **Executor** | Run model evaluations | `src/taskbench/evaluation/executor.py` |
| **Judge** | Score model outputs | `src/taskbench/evaluation/judge.py` |
| **Cost Tracker** | Calculate & track costs | `src/taskbench/evaluation/cost.py` |
| **CLI** | Command-line interface | `src/taskbench/cli/main.py` |

---

## 🎬 Quick Start (When Complete)

```bash
# Install
pip install llm-taskbench

# Evaluate models
taskbench evaluate tasks/lecture_analysis.yaml \
  --models claude-sonnet-4.5,gpt-4o,qwen-2.5-72b

# Get recommendations
taskbench recommend

# Export results
taskbench results --format csv --output results.csv
```

---

## 📊 Research Background

**42-Model Study Results:**

| Finding | Implication |
|---------|-------------|
| Model size ≠ quality | Llama 405B didn't beat Qwen 72B |
| Reasoning models can underperform | Some scored 31% worse on reasoning tasks |
| Cost ≠ performance | Zero correlation across 36X price range |
| Fine-tuning > size | Specialized training beats parameter count |

**Sweet Spot:** Mid-tier models (Qwen 72B, Claude Sonnet 4) often deliver 85-90% of premium quality at 50% of the cost.

---

## 🎓 Primary Use Case

**Lecture Transcript Analysis:**
- **Input:** 3-hour lecture transcript (plain text)
- **Output:** CSV with teaching concepts and timestamps
- **Rules:** 2-7 minute segments, no overlaps
- **Baseline:** Claude Sonnet 4.5 extracts 24 concepts, 0 violations, $0.36

**Why This Task:**
- Requires complex reasoning
- Demands precise text processing
- Tests rule following
- Validates structured output

---

## 🏆 Success Criteria

**Functional:**
- ✅ Evaluate 5+ models in <30 minutes
- ✅ Judge scores within ±10 points of manual
- ✅ Cost tracking accurate to $0.01
- ✅ Test coverage ≥80%

**Portfolio:**
- ✅ Professional README
- ✅ Working demo (video or live)
- ✅ Can explain architecture in 2 minutes
- ✅ Zero critical bugs in demo path

---

## 📅 Timeline Milestones

| Phase | Dates | Key Deliverable |
|-------|-------|-----------------|
| **Phase 0** | Oct 20-24 | ✅ Planning & docs complete |
| **Phase 1** | Oct 27 - Nov 10 | Basic evaluation working |
| **Phase 2** | Nov 11 - Nov 24 | LLM-as-judge functional |
| **Phase 3** | Nov 25 - Dec 8 | Recommendations engine |
| **Phase 4** | Dec 9 - Dec 22 | Demo-ready MVP |

**Buffer Week:** Week 7 (Dec 9-15) if behind schedule  
**Demo Day:** January 17, 2026

---

## ⚡ Critical Path

**Must Complete On Time:**
1. Week 1-2: Core framework (foundation for everything)
2. Week 3: LLM-as-judge (highest risk, start simple)

**Can Compress if Needed:**
- Phase 3: Simplified recommendations
- Phase 4: Basic documentation, shorter demo

**Cannot Drop:**
- Task parser and API client
- Model execution
- Cost tracking
- Basic CLI
- Core tests

---

## 🔧 Tech Stack Quick Ref

```python
# Data validation
from pydantic import BaseModel, Field

# CLI framework
import typer
from rich.console import Console

# HTTP client
import httpx

# YAML parsing
import yaml

# Testing
import pytest
```

**Key Dependencies:**
- `pydantic>=2.0.0` - Data validation
- `typer>=0.9.0` - CLI framework
- `rich>=13.0.0` - Terminal output
- `httpx>=0.25.0` - Async HTTP
- `pytest>=7.4.0` - Testing

---

## 📋 Must-Have vs Nice-to-Have

### Must Have (MVP)
- [x] 1 built-in task (lecture analysis)
- [ ] 5 model evaluation
- [ ] LLM-as-judge scoring
- [ ] Cost tracking ($0.01 precision)
- [ ] Basic CLI
- [ ] Basic recommendations
- [ ] 80% test coverage
- [ ] Demo video or live demo

### Nice to Have (Post-MVP)
- [ ] 2nd built-in task
- [ ] Multiple export formats
- [ ] Parallel execution
- [ ] Advanced CLI features
- [ ] PyPI publication
- [ ] Extensive docs
- [ ] Web UI

---

## 🚨 Risk Management

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| API rate limits | Medium | Medium | Use caching, free tier |
| Judge inconsistency | Medium | High | Manual validation, prompt tuning |
| Time overrun | Medium | High | Week 7 buffer, scope reduction |
| Demo day failures | Low | Critical | Video backup, screenshots |

---

## 📦 Project Structure

```
llm-taskbench/
├── src/taskbench/           # Main package
│   ├── core/                # Task & data models
│   ├── api/                 # API clients
│   ├── evaluation/          # Executor, judge, cost
│   ├── cli/                 # CLI commands
│   └── utils/               # Helpers
├── tasks/                   # Built-in tasks
├── tests/                   # Test suite
├── config/                  # Model pricing
└── docs/                    # Documentation
```

---

## 🎯 Demo Script (10 minutes)

**1. Problem (1 min)**
- Show blog post screenshot
- "Which LLM for my use case?"

**2. Solution Overview (1 min)**
- Show task YAML
- Explain task-first approach

**3. Live Demo (5 min)**
- Run evaluation on 3 models
- Show real-time progress
- Display results table

**4. Results Analysis (2 min)**
- Show judge scores
- Show recommendations
- Explain best value pick

**5. Architecture (1 min)**
- Show diagram
- Highlight agentic orchestration
- Explain LLM-as-judge

---

## 🔍 Key Metrics to Track

**During Development:**
- Lines of code
- Test coverage %
- API calls made (cost)
- Time spent per phase

**At Demo:**
- Evaluation time (should be <30 min)
- Cost accuracy (vs OpenRouter billing)
- Judge consistency (±5 points across runs)
- Test coverage (should be >80%)

---

## 📚 Documentation Files

| File | Purpose |
|------|---------|
| `PROJECT-OVERVIEW.md` | High-level summary (this file) |
| `claude-code-task-list.md` | Detailed implementation tasks |
| `technical-spec.md` | Technical architecture & design |
| `LLMTaskBench-Vision.md` | Full vision document |
| `Todo.md` | Detailed checklist tracker |

---

## 🤝 Comparison with Competitors

| Feature | TaskBench | DeepEval | Promptfoo |
|---------|-----------|----------|-----------|
| Target User | Domain experts | AI engineers | Developers |
| Eval Focus | Task-specific | Metric-based | Prompt testing |
| LLM-as-Judge | ✅ Built-in | ❌ No | ⚠️ Limited |
| Cost Tracking | ✅ Real-time | ❌ No | ❌ No |
| No-code Setup | ✅ YAML | ❌ Python | ⚠️ Config |

---

## 💡 Design Principles

1. **Task-First:** Start with use case, not metrics
2. **Cost-Aware:** Always show quality/cost tradeoff
3. **Research-Backed:** Validate against 42-model study
4. **User-Friendly:** Domain experts, not just engineers
5. **Production-Ready:** 80% test coverage, proper error handling

---

## ⚙️ Environment Variables

```bash
# Required
OPENROUTER_API_KEY=sk-or-...

# Optional (for direct API access)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

---

## 🔗 Quick Links

**Project Resources:**
- GitHub Repo: (to be created)
- Blog Post: [How a $3.45 API Call...](https://www.linkedin.com/pulse/how-345-api-call-taught-me-everything-llm-cost-sri-bolisetty-vxate)
- Research Data: `analysis.csv`, `models.csv`

**External APIs:**
- OpenRouter: <https://openrouter.ai>
- Anthropic: <https://docs.anthropic.com>
- OpenAI: <https://platform.openai.com>

---

## 🎯 Remember

**What Makes This Project Special:**
1. Real research (42 models) validates the approach
2. Solves actual problem (from $3.45 blog post)
3. Task-first is genuinely novel
4. Cost-awareness is critical but rare
5. Agentic + LLM-as-judge demonstrates advanced skills

**Keys to Success:**
- Start simple, iterate
- Test continuously
- Document decisions
- Practice demo 3x
- Have backup plan

---

**Ready to build! 🚀**

See `claude-code-task-list.md` for detailed implementation tasks.