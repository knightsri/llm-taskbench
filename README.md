# LLM TaskBench

> **Evaluate LLMs for YOUR task, not generic benchmarks**

Task-first LLM evaluation framework for domain experts. Compare models on your actual use case—lecture analysis, medical triage, support tickets—with cost-aware recommendations backed by research testing 42 production LLMs.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 🎯 The Problem

Existing LLM evaluation tools are built for AI engineers, not domain experts:

- **DeepEval, Promptfoo, Eleuther AI** → Focus on generic metrics (BLEU, ROUGE)
- **Academic benchmarks** → Test synthetic tasks, not your real use case
- **No cost awareness** → Miss the sweet spot between quality and budget

**Real example:** A medical professional needs to know "which LLM accurately triages patient symptoms," not "which model scores 0.85 on ROUGE-L."

### The Research That Started This

After spending **$3.45 on a single chat session** processing 4 lecture transcripts, I discovered my workflow was accumulating every file in context—costing 2x what it should. This led to comprehensive research:

**I tested 42 LLMs on lecture transcript analysis. The findings were surprising:**

- ❌ Model size doesn't correlate with quality
- ❌ "Reasoning-optimized" models can perform *worse* on reasoning tasks
- ❌ Cost shows zero correlation with performance
- ✅ Fine-tuning proves more impactful than raw model size

**Read the full story:** ["How a $3.45 API Call Taught Me Everything About LLM Cost Optimization"](https://www.linkedin.com/pulse/how-345-api-call-taught-me-everything-llm-cost-sri-bolisetty-vxate)

LLM TaskBench grew from this research—a tool to help others avoid costly mistakes and make data-driven model choices for *their* specific tasks.

---

## ✨ The Solution

LLM TaskBench shifts evaluation from **metric-first to task-first**:

```bash
# Define YOUR task (not a generic benchmark)
$ cat lecture_analysis.yaml

# Evaluate multiple models on it
$ taskbench evaluate lecture_analysis.yaml \
  --models claude-sonnet-4.5,gpt-4o,gemini-2.5-pro,llama-3.1-405b,qwen-2.5-72b

# Get cost-aware recommendations
Evaluating Claude Sonnet 4.5... ✓ (24 concepts, 0 violations, $0.36)
Evaluating GPT-4o...           ✓ (23 concepts, 1 violation, $0.42)
Evaluating Gemini 2.5 Pro...   ✓ (22 concepts, 2 violations, $0.38)
Evaluating Llama 3.1 405B...   ✓ (20 concepts, 4 violations, $0.25)
Evaluating Qwen 2.5 72B...     ✓ (21 concepts, 3 violations, $0.18)

# See actionable recommendations
$ taskbench recommend

┌─────────────────────┬───────┬────────────┬──────────┬───────────┐
│ Model               │ Score │ Violations │ Cost     │ Tier      │
├─────────────────────┼───────┼────────────┼──────────┼───────────┤
│ Claude Sonnet 4.5   │ 98    │ 0          │ $0.36    │ Excellent │
│ GPT-4o              │ 95    │ 1          │ $0.42    │ Excellent │
│ Gemini 2.5 Pro      │ 92    │ 2          │ $0.38    │ Excellent │
│ Qwen 2.5 72B        │ 87    │ 3          │ $0.18    │ Good      │
│ Llama 3.1 405B      │ 83    │ 4          │ $0.25    │ Good      │
└─────────────────────┴───────┴────────────┴──────────┴───────────┘

📊 Recommendations:

  🏆 Best Overall: Claude Sonnet 4.5
     Highest accuracy, zero violations, worth the $0.36 for production

  💎 Best Value: Qwen 2.5 72B  
     87% as good as Claude but 50% cheaper - perfect for development

  💰 Budget Option: Qwen 2.5 72B
     Good enough quality at lowest cost
```

---

## 🚀 Key Features

### 1. **Task-First Evaluation**
Define your actual use case, not a proxy metric:
- Lecture concept extraction with timestamps
- Medical case triage
- Support ticket categorization
- Contract analysis
- *Your domain-specific task*

### 2. **LLM-as-Judge Quality Assessment**
Let Claude/GPT-4 score outputs on *your* criteria:
- Accuracy scores (0-100)
- Format compliance
- Rule violation detection
- Reasoning for scores

### 3. **Cost-Aware Recommendations**
Balance quality and budget:
- Real-time token tracking (accurate to $0.01)
- Cost-quality tradeoff analysis
- Production vs. development recommendations
- "Sweet spot" identification

### 4. **Research-Validated**
Built on findings from testing 42 models:
- Results validated against published research
- No model size assumptions
- Real-world performance data
- Surprising insights (reasoning models can underperform!)

### 5. **Agentic Architecture**
Smart orchestration, not manual coordination:
- LLM creates evaluation plan
- Dynamic model selection
- Intelligent error handling
- Automated comparison

---

## 📦 Installation

> **Note:** This is a MVP project currently in development. Installation instructions will be available upon v1.0 release (December 2025).

**Prerequisites:**
- Python 3.11+
- OpenRouter API key (or Anthropic/OpenAI direct)

**Coming Soon:**
```bash
# Install from PyPI (post-MVP)
pip install llm-taskbench

# Or install from source
git clone https://github.com/KnightSri/llm-taskbench.git
cd llm-taskbench
pip install -e .
```

---

## 🎯 Quick Start

### 1. Define Your Task

Create a YAML file describing your evaluation task:

```yaml
# tasks/my_task.yaml
name: "customer_support_triage"
description: "Categorize support tickets by urgency and department"

input_type: "text"
output_format: "json"

evaluation_criteria:
  - "Correctly identifies urgency (low/medium/high/critical)"
  - "Routes to appropriate department"
  - "Suggests initial response actions"

constraints:
  response_time_ms: 3000
  required_fields: ["urgency", "department", "suggested_actions"]
```

### 2. Run Evaluation

```bash
# Evaluate with specific models
taskbench evaluate tasks/my_task.yaml \
  --models claude-sonnet-4.5,gpt-4o,gemini-2.5-pro

# Let TaskBench suggest appropriate models
taskbench evaluate tasks/my_task.yaml
```

### 3. Analyze Results

```bash
# View results table
taskbench results --format table

# Export to CSV for further analysis
taskbench results --format csv --output results.csv

# Get recommendations
taskbench recommend --budget 0.50  # Models under $0.50/eval
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   LLM TaskBench                          │
│                                                           │
│  User (CLI) → Task Parser → LLM Orchestrator (Agentic)  │
│                                    ↓                      │
│                         Model Execution                   │
│                    (OpenRouter/Direct APIs)               │
│                                    ↓                      │
│                     LLM-as-Judge Evaluator               │
│                    (Claude/GPT-4 scoring)                 │
│                                    ↓                      │
│                    Cost Analysis Engine                   │
│                (Tradeoffs + Recommendations)              │
│                                    ↓                      │
│                    Results + Export                       │
│                  (Tables, CSV, JSON)                      │
└─────────────────────────────────────────────────────────┘
```

**Core Components:**

1. **Task Definition System** - Parse & validate YAML task specs
2. **Agentic Orchestrator** - LLM-driven evaluation planning
3. **LLM-as-Judge** - Meta-evaluation with structured scoring
4. **Cost Analyzer** - Real-time tracking + quality tradeoffs
5. **Results Presenter** - Professional output & export

**Technology Stack:**
- **Language:** Python 3.11+
- **CLI:** Typer + Rich (beautiful terminal output)
- **AI Gateway:** OpenRouter (30+ models, unified API)
- **Testing:** pytest (80%+ coverage target)
- **Type Safety:** mypy + Pydantic validation

---

## 📊 Example: Lecture Analysis

**Task:** Extract teaching concepts from lecture transcripts with precise timestamps

**Built-in Task:** `lecture_analysis.yaml`

```yaml
name: "lecture_concept_extraction"
description: "Extract teaching concepts from lecture transcripts with timestamps"

evaluation_criteria:
  - "Concept count (target: 20-24 for 3-hour lecture)"
  - "Timestamp accuracy"
  - "Duration compliance (2-7 minutes per segment)"

constraints:
  min_duration_minutes: 2
  max_duration_minutes: 7
  target_duration_minutes: 3-6
```

**Results from 42-Model Research:**

| Model | Concepts | Violations | Cost | Quality Tier |
|-------|----------|------------|------|--------------|
| **Claude Sonnet 4.5** | 24 | 0 | $0.36 | Excellent (100%) |
| **GPT-4o** | 23 | 1 | $0.42 | Excellent (96%) |
| **Claude Sonnet 4** | 22 | 7 minor | $0.18 | Good (92%) |
| **Qwen 2.5 72B** | 21 | 3 | $0.18 | Good (87%) |
| **Claude Haiku 4** | 17 | 11 | $0.03 | Acceptable (71%) |

**Surprising Finding:** The 405B Llama model didn't outperform the 72B Qwen model despite being 5.6x larger. Model size ≠ quality.

---

## 🆚 Comparison with Existing Tools

| Feature | LLM TaskBench | DeepEval | Promptfoo | Eleuther AI |
|---------|---------------|----------|-----------|-------------|
| **Target User** | Domain experts | AI engineers | Developers | Researchers |
| **Evaluation Focus** | Task-specific | Metric-based | Prompt testing | Academic benchmarks |
| **LLM-as-Judge** | ✅ Built-in | ❌ No | ⚠️ Limited | ❌ No |
| **Cost Tracking** | ✅ Real-time | ❌ No | ❌ No | ❌ No |
| **Agentic Orchestration** | ✅ Yes | ❌ No | ❌ No | ❌ No |
| **Domain Examples** | ✅ Medical, edu, support | ❌ Generic | ⚠️ Dev-focused | ❌ Academic |
| **No-code Setup** | ✅ YAML config | ❌ Python code | ⚠️ Config files | ❌ Code |

**Why TaskBench?**
- Start with YOUR task, not a generic benchmark
- Get cost-aware recommendations, not just scores
- Research-backed (42-model validation)
- Accessible to non-technical domain experts

---

## 🎓 Research Background

This project emerged from comprehensive LLM research:

### The Study
- **Tested:** 42 production LLMs (Oct 2025)
- **Task:** Lecture transcript concept extraction
- **Tiers:** Premium (Opus 4.1, o1) → Efficient (Haiku, Phi-4)
- **Providers:** Anthropic, OpenAI, Google, xAI, Meta, DeepSeek, Qwen, Mistral

### Key Findings

**1. Model Size Doesn't Predict Quality**
- Llama 3.1 405B didn't beat Qwen 2.5 72B (5.6x smaller)
- Smaller fine-tuned models often outperform larger base models

**2. "Reasoning" Models Can Underperform**
- Some reasoning-optimized LLMs scored worse on reasoning tasks
- General models sometimes better than specialists

**3. Cost Has Zero Correlation with Performance**
- $0.03 models sometimes beat $0.42 models
- Sweet spot: mid-tier models (Qwen 72B, Claude Sonnet 4)

**4. Fine-Tuning > Raw Size**
- Nous Hermes 3 405B vs base Llama 405B shows fine-tuning value
- Specialized training beats parameter count

**Read the full research:** [LinkedIn Article](https://www.linkedin.com/pulse/how-345-api-call-taught-me-everything-llm-cost-sri-bolisetty-vxate)

---

## 📚 Built-in Tasks

LLM TaskBench includes real-world task definitions you can use immediately:

### 1. Lecture Concept Extraction
**File:** `tasks/lecture_analysis.yaml`  
**Use Case:** Educational technology, course creation, content summarization  
**Validated:** ✅ Against 42-model research

### 2. Support Ticket Categorization
**File:** `tasks/ticket_categorization.yaml`  
**Use Case:** Customer support automation, triage systems  
**Coming:** December 2025

### 3. Medical Case Analysis *(Post-MVP)*
**Use Case:** Clinical decision support, triage assistance  
**Status:** Planned for v1.1

---

## 🛠️ Development Status

**Current Phase:** Phase 0 Complete ✅ (Planning & Documentation)

**Timeline:**
- **Phase 1** (Nov 2025) → Core framework
- **Phase 2** (Nov 2025) → LLM-as-judge evaluation
- **Phase 3** (Dec 2025) → Cost analysis & recommendations
- **Phase 4** (Dec 2025) → Polish & demo

**v1.0 Release:** December 22, 2025

### Progress Tracking

| Component | Status | Target |
|-----------|--------|--------|
| Task Definition System | 🔜 Next | Week 1-2 |
| API Client (OpenRouter) | 🔜 Next | Week 1-2 |
| LLM-as-Judge | ⏳ Planned | Week 3-4 |
| Cost Analysis | ⏳ Planned | Week 5-6 |
| Built-in Tasks | ⏳ Planned | Week 7 |
| Documentation | 🔄 In Progress | Week 8 |

---

## 🎯 Use Cases

### Education Technology
**Example:** Lecture summarization feature for online learning platform

**Before TaskBench:**
- Test a few popular models by "feel"
- No data-driven comparison
- Unknown costs at scale

**With TaskBench:**
- Define task: "Extract 3-6 min teaching concepts"
- Test 10 models
- Get recommendation: "Claude Sonnet 4 balances quality ($0.18/lecture) and accuracy (92% baseline). Haiku saves 83% but needs manual cleanup."
- **Decision:** Use Sonnet 4, avoid Haiku despite cost savings

### Healthcare AI
**Example:** Evaluate models for medical case triage

**With TaskBench:**
- Define task with medical accuracy criteria
- Test 15 models including specialized medical LLMs
- Get compliance report on diagnostic accuracy
- **Decision:** Data-driven model selection with audit trail

### Enterprise SaaS
**Example:** Choose LLM for customer support chatbot

**With TaskBench:**
- Test on real support ticket corpus
- Balance quality, cost, and response time
- Validate against SLA requirements
- **Decision:** Qwen 72B hits quality bar at 50% the cost

---

## 🤝 Contributing

> **Note:** Project opens for contributions after v1.0 release (Dec 2025)

**Planned Contribution Areas:**
- Additional built-in tasks (legal, finance, etc.)
- New evaluation criteria
- Integration with other LLM platforms
- Documentation improvements

**Contact:** Sri Bolisetty - [@KnightSri](https://github.com/KnightSri)

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details

---

## 🙏 Acknowledgments

**Inspiration:**
- The $3.45 API call that started this journey
- 42-model research that validated the approach
- Domain experts who need better LLM evaluation tools

**Technical Foundation:**
- OpenRouter for unified LLM access
- Anthropic Claude for orchestration & judging
- Open-source LLM community

**Research:**
- 42 LLM providers & models tested
- Real-world validation on production tasks

---

## 📞 Contact

**Sri Bolisetty**
- GitHub: [@KnightSri](https://github.com/KnightSri)
- LinkedIn: [Article on LLM Cost Optimization](https://www.linkedin.com/pulse/how-345-api-call-taught-me-everything-llm-cost-sri-bolisetty-vxate)

---

## 🗺️ Roadmap

### v1.0 (December 2025) - MVP
- [x] Project planning & architecture
- [ ] Core evaluation framework
- [ ] LLM-as-judge implementation
- [ ] Cost analysis engine
- [ ] 2 built-in tasks
- [ ] CLI interface
- [ ] Documentation

### v1.1 (Q1 2026) - Enhancement
- [ ] Web UI (Streamlit)
- [ ] Medical case analysis task
- [ ] Batch evaluation (50+ models)
- [ ] Historical tracking
- [ ] PyPI package

### v2.0 (Q2 2026) - Platform
- [ ] Custom judge LLM selection
- [ ] A/B testing framework
- [ ] Team collaboration
- [ ] API access
- [ ] Integration with LangSmith/Helicone

---

## ⭐ Star This Project

If LLM TaskBench helps you make better model choices, please star the repository!

---

<div align="center">

**Built with 🧠 by [Sri Bolisetty](https://github.com/KnightSri)**
*Making LLM evaluation accessible to domain experts*

</div>
