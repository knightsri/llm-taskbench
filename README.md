# LLM TaskBench

> **Evaluate LLMs for YOUR task, not generic benchmarks**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-MVP%20Development-yellow.svg)]()

**TaskBench** is a task-first LLM evaluation framework that helps domain experts evaluate models for their specific use cases—without needing to understand complex AI metrics or academic benchmarks.

## 🎯 The Problem

Existing LLM evaluation tools force you to:
- Pick from 14+ generic metrics (hallucination, toxicity, relevance...)
- Run academic benchmarks (MMLU, HumanEval) that don't match your real work
- Interpret scores without clear guidance on which model to actually use

**Result:** Domain experts struggle to evaluate LLMs for their specific tasks.

## ✨ The Solution

**TaskBench** flips the approach:

1. **You describe your task** → TaskBench suggests evaluation criteria
2. **We run the evaluation** → Across multiple models automatically  
3. **You get recommendations** → "Use Model X because..." with cost/quality trade-offs

```bash
# Install
pip install taskbench

# Evaluate models for YOUR task
taskbench run --task lecture-analysis --models claude-sonnet-4.5,gpt-4o,deepseek-v3

# Get clear recommendations
taskbench recommend

╔═══════════════════════════════════════════════════════════╗
║ ✅ BEST CHOICE: Claude Sonnet 4.5                        ║
║    - Perfect quality (100% accuracy)                      ║
║    - Cost: $0.36 per lecture                             ║
║    - Worth the premium for this task                      ║
╠═══════════════════════════════════════════════════════════╣
║ 💰 BUDGET OPTION: Claude Sonnet 4                        ║
║    - 50% cheaper, 92% quality                            ║
║    - Minor issues easily fixable                          ║
║    - Best cost/quality balance                            ║
╚═══════════════════════════════════════════════════════════╝
```

## 🚀 Key Features

### ✅ Task-First, Not Metric-First
Define your specific task (or use built-in tasks). TaskBench determines what to evaluate—you don't need to understand evaluation metrics.

### 🎓 Built-in Domain Tasks
- **Lecture Transcript Analysis** (Education)
- **Customer Support Ticket Categorization** (Support)
- **Medical Case Extraction** (Healthcare)

### 💰 Cost-Aware Recommendations
Every evaluation includes:
- Real cost per task (not just per token)
- Cost projections (50 lectures = $X)
- Cost/quality trade-off analysis
- "Is the premium model worth it?" guidance

### 🔧 Extensible Architecture
- Add custom tasks via simple YAML files
- Support for multiple LLM providers (OpenRouter, Anthropic, OpenAI, local models)
- Community task marketplace (coming soon)

### 📊 Comprehensive Analysis
- Quality scoring (rule-based + LLM-as-judge)
- Comparative analysis across models
- Detailed reports with visualizations
- Reproducible evaluations

## 🏆 Why TaskBench?

### vs. DeepEval
- **DeepEval:** "Here are 14 metrics, pick which ones apply"
- **TaskBench:** "What's your task? We'll evaluate models for it"

### vs. Promptfoo
- **Promptfoo:** Great for A/B testing prompts
- **TaskBench:** Built for comparing models on complex domain tasks

### vs. Academic Benchmarks (Eleuther AI)
- **Academic:** 200+ pre-defined benchmarks for researchers
- **TaskBench:** Create benchmarks for YOUR domain in minutes

## 📖 Real-World Example

### Background: The $3.45 API Call
TaskBench was born from real experience evaluating 42 LLM models for lecture transcript analysis. The results revealed:

- **Claude Sonnet 4.5**: Perfect quality, 24/24 concepts, 0 violations → $0.36/lecture
- **Claude Sonnet 4**: Good quality, 22/24 concepts, minor issues → $0.18/lecture  
- **Claude Haiku 4**: Poor quality, 17/24 concepts, major errors → $0.03/lecture

**Key insight:** The cheapest model ($0.03) required manual cleanup that cost more in time than using the best model ($0.36).

> 💡 **TaskBench makes these comparisons easy** for any domain-specific task.

Read the full story: [Blog post coming soon]

## 🎯 Who Is This For?

- **Educators** evaluating models for course content generation
- **Medical professionals** assessing clinical documentation tools
- **Legal teams** testing contract analysis systems  
- **Customer support managers** benchmarking ticket categorization
- **Researchers** creating reproducible domain-specific benchmarks

**You don't need AI expertise**—just a task and sample data.

## 🚦 Quick Start

### Installation

```bash
pip install taskbench
```

### Run Your First Evaluation

```bash
# Initialize with a built-in task
taskbench init --task lecture-analysis

# Add your sample data
taskbench add-sample my-lecture-transcript.txt

# Run evaluation across multiple models
taskbench run --models claude-sonnet-4.5,gpt-4o,deepseek-v3

# View results and recommendations
taskbench report
taskbench recommend
```

### Create a Custom Task

```bash
# Generate task template
taskbench new-task --name "my-domain-task"

# Edit my-domain-task.yaml with your task definition
# (TaskBench provides validation and examples)

# Validate your task
taskbench validate my-domain-task.yaml

# Run evaluation
taskbench run --task-file my-domain-task.yaml
```

## 📁 Project Structure

```
taskbench/
├── src/taskbench/
│   ├── cli.py              # Command-line interface
│   ├── orchestrator.py     # Evaluation orchestration
│   ├── runner.py           # Model execution
│   ├── evaluator.py        # Quality scoring
│   ├── analyzer.py         # Cost/quality analysis
│   ├── recommender.py      # Recommendation engine
│   └── tasks/              # Built-in task definitions
│       ├── lecture_analysis.yaml
│       ├── ticket_categorization.yaml
│       └── medical_case_summary.yaml
├── tests/                  # Comprehensive test suite
├── docs/                   # Documentation
└── examples/               # Example tasks and usage
```

## 🔬 How It Works

### 1. Task Definition
Define your task via YAML:

```yaml
task:
  name: "Lecture Transcript Analysis"
  description: "Extract teaching concepts from lecture transcripts"
  
input:
  type: "text"
  format: "transcript with timestamps"
  
expected_output:
  format: "csv"
  columns: ["concept", "start_time", "end_time"]
  
evaluation:
  - type: "rule-based"
    rules:
      - name: "concept_count"
        target: {min: 20, max: 30}
      - name: "segment_duration"  
        target: {min_seconds: 120, max_seconds: 420}
        
  - type: "llm-judge"
    criteria:
      - "Content coverage"
      - "Concept clarity"
      - "Timestamp accuracy"
```

### 2. Evaluation Pipeline

```
Your Input → TaskBench Orchestrator
    ↓
    ├─→ Model A (Claude Sonnet 4.5)
    ├─→ Model B (GPT-4o)  
    └─→ Model C (DeepSeek V3)
    ↓
Results → Evaluator Agent
    ├─→ Rule-based scoring
    └─→ LLM-as-judge scoring
    ↓
Analysis → Analyzer Agent
    ├─→ Quality comparison
    ├─→ Cost analysis
    └─→ Trade-off analysis
    ↓
Recommendations → Report Generator
```

### 3. Get Results

TaskBench produces:
- **Detailed comparison report** (Markdown with charts)
- **Model recommendations** with clear reasoning
- **Cost projections** for your actual usage
- **Raw evaluation data** (JSON) for further analysis

## 🛠️ Technology Stack

- **Python 3.9+** - Core framework
- **Click** - CLI interface
- **PyYAML** - Task definitions
- **Pydantic** - Data validation
- **Requests** - API calls (OpenRouter, Anthropic, OpenAI)
- **Rich** - Beautiful terminal output
- **Jinja2** - Report templates
- **Pytest** - Testing

## 📊 Current Status

**Phase:** MVP Development (Week 1-8)

- [x] Technical specification complete
- [ ] Phase 1: Core Framework (Weeks 1-2)
- [ ] Phase 2: Evaluation Engine (Weeks 3-4)
- [ ] Phase 3: Analysis & Recommendations (Weeks 5-6)
- [ ] Phase 4: Built-in Tasks & Polish (Weeks 7-8)

See [TODO.md](TODO.md) for detailed development roadmap.

## 🎓 Academic Context

This project is the capstone for an AI Engineering course, demonstrating:
- **Agentic architecture** (orchestrator, runner, evaluator, analyzer)
- **LLM-as-judge** evaluation patterns
- **Production-ready** LLM application design
- **Real-world problem solving** (based on actual research)

The lecture transcript analysis task is based on research comparing 42 LLM models for educational content processing.

## 🤝 Contributing

TaskBench is currently in MVP development. Contributions welcome after v1.0 release!

**Planned for post-MVP:**
- Community task marketplace
- Additional built-in tasks (legal, financial, etc.)
- Web UI for non-CLI users
- Local model support (Ollama)

## 📝 License

Apache License 2.0 - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built by [@KnightSri](https://github.com/KnightSri) as an AI Engineering capstone project
- Inspired by real-world LLM evaluation challenges in education
- Based on comparative analysis of 42 production LLMs (October 2025)

## 📬 Contact

- **GitHub:** [@KnightSri](https://github.com/KnightSri)
- **Project:** [llm-taskbench](https://github.com/KnightSri/llm-taskbench)

---

**⚡ TaskBench: Because your task deserves a custom benchmark, not a generic one.**
