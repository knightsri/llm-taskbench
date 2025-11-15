# LLM TaskBench 🚀

**Task-specific LLM evaluation framework with agentic orchestration and LLM-as-judge evaluation**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 📖 Overview

LLM TaskBench shifts from **metric-first** to **task-first** LLM evaluation. Instead of relying on generic metrics like BLEU or ROUGE, evaluate models on **your actual use cases** with task-specific criteria.

### Why LLM TaskBench?

Traditional LLM benchmarks don't tell you which model is best for **your** task. Our research on 42 production LLMs revealed:

- ❌ **Model size doesn't correlate with quality** - 405B models didn't beat 72B models
- ❌ **"Reasoning" models can underperform** on reasoning tasks
- ❌ **Cost has zero correlation** with performance
- ✅ **Fine-tuning beats raw parameter count**

**LLM TaskBench** lets you discover these insights for your specific use case.

### Key Features

- 🎯 **Task-First Evaluation** - Define your own evaluation tasks with custom criteria
- 🤖 **LLM-as-Judge** - Automated quality assessment using Claude Sonnet 4.5
- 💰 **Cost-Aware Recommendations** - Find the best value model for your budget
- 🔄 **Multi-Model Comparison** - Evaluate 5+ models simultaneously
- 📊 **Detailed Analytics** - Scores, violations, token usage, and cost breakdowns
- 🚀 **Production Ready** - Retry logic, rate limiting, comprehensive error handling

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/knightsri/llm-taskbench.git
cd llm-taskbench

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .

# Set up your API key
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

### Run Your First Evaluation

```bash
# Evaluate 3 models on lecture concept extraction
taskbench evaluate tasks/lecture_analysis.yaml \
  --models anthropic/claude-sonnet-4.5,openai/gpt-4o,qwen/qwen-2.5-72b-instruct \
  --input-file tests/fixtures/sample_transcript.txt
```

**Output:**
```
Evaluating 3 models on task 'lecture_concept_extraction'

✓ anthropic/claude-sonnet-4.5: 15,234 tokens, $0.36, 2,145ms
✓ openai/gpt-4o: 16,012 tokens, $0.42, 1,876ms
✓ qwen/qwen-2.5-72b-instruct: 14,567 tokens, $0.18, 3,201ms

Running LLM-as-judge evaluation...

✓ anthropic/claude-sonnet-4.5: Score 98/100, 0 violations
✓ openai/gpt-4o: Score 95/100, 1 violations
✓ qwen/qwen-2.5-72b-instruct: Score 87/100, 3 violations

                      Model Comparison Results
┌──────┬────────────────────┬───────┬────────────┬──────────┬────────┬───────┐
│ Rank │ Model              │ Score │ Violations │ Cost     │ Tokens │ Value │
├──────┼────────────────────┼───────┼────────────┼──────────┼────────┼───────┤
│ 1    │ claude-sonnet-4.5  │  98   │     0      │ $0.3600  │ 15,234 │ ⭐⭐⭐ │
│ 2    │ gpt-4o             │  95   │     1      │ $0.4200  │ 16,012 │ ⭐⭐   │
│ 3    │ qwen-2.5-72b       │  87   │     3      │ $0.1800  │ 14,567 │ ⭐⭐⭐ │
└──────┴────────────────────┴───────┴────────────┴──────────┴────────┴───────┘

📊 RECOMMENDATIONS

🏆 Best Overall: anthropic/claude-sonnet-4.5
   Score: 98/100, Cost: $0.3600

💎 Best Value: qwen/qwen-2.5-72b-instruct
   Score: 87/100, Cost: $0.1800

✓ Results saved to results/evaluation_results.json
```

---

## 📚 Documentation

- **[Architecture Guide](docs/ARCHITECTURE.md)** - System design and component overview
- **[API Reference](docs/API.md)** - Complete API documentation
- **[Usage Guide](docs/USAGE.md)** - Detailed tutorials and examples

---

## 🎯 Core Concepts

### 1. Task Definitions

Define evaluation tasks using YAML:

```yaml
name: "lecture_concept_extraction"
description: "Extract teaching concepts from lecture transcripts with precise timestamps"
input_type: "transcript"
output_format: "csv"

evaluation_criteria:
  - "Timestamp accuracy (within ±5 seconds)"
  - "Duration compliance (2-7 minutes per segment)"
  - "Concept names are descriptive and clear"

constraints:
  min_duration_minutes: 2
  max_duration_minutes: 7
  required_csv_columns: ["concept", "start_time", "end_time"]

judge_instructions: |
  Evaluate the model's output based on:
  1. Accuracy (40%): Are concepts correctly identified?
  2. Format (30%): Valid CSV with required columns?
  3. Compliance (30%): Meet duration constraints?
```

### 2. LLM-as-Judge Evaluation

Automatically evaluate outputs using Claude Sonnet 4.5:

- **Accuracy Score** (0-100): Content quality
- **Format Score** (0-100): Structure compliance
- **Compliance Score** (0-100): Constraint adherence
- **Violations**: Specific issues found

### 3. Cost-Aware Recommendations

Get actionable recommendations based on:

- **Best Overall**: Highest quality (98/100)
- **Best Value**: Best score/cost ratio (87/100 for 50% less)
- **Budget Option**: Acceptable quality at lowest cost

---

## 💻 CLI Commands

### Evaluate Models

```bash
taskbench evaluate <task.yaml> --models <model-list> --input-file <input.txt>
```

### List Available Models

```bash
taskbench models --list
```

### Validate Task Definition

```bash
taskbench validate <task.yaml>
```

---

## 🏗️ Project Structure

```
llm-taskbench/
├── src/taskbench/
│   ├── core/           # Data models and task parsing
│   ├── api/            # OpenRouter API client
│   ├── evaluation/     # Executor, judge, cost tracking
│   └── cli/            # Command-line interface
├── tasks/              # Built-in task definitions
├── tests/              # Test suite
├── docs/               # Documentation
└── config/             # Model pricing database
```

---

## 🔬 Research Background

Based on evaluating **42 production LLMs** on lecture analysis:

| Finding | Impact |
|---------|--------|
| Model size ≠ quality | 72B models beat 405B models |
| "Reasoning" ≠ better reasoning | Some reasoning models scored lower |
| Cost ≠ quality | Zero correlation found |
| Fine-tuning > parameters | Specialized models outperform larger general models |

**Conclusion**: You need task-specific evaluation to find the right model.

---

## 🎓 Use Cases

### 1. Lecture Transcript Analysis
Extract teaching concepts with timestamps - perfect for educational platforms.

### 2. Customer Support Classification
Evaluate models on classifying support tickets with your categories.

### 3. Code Generation
Test models on generating code for your specific framework/library.

### 4. Content Moderation
Compare models on detecting violations according to your guidelines.

### 5. Custom NLP Tasks
Any task where generic benchmarks don't tell the full story.

---

## 🛠️ Development

### Run Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=taskbench --cov-report=html

# Specific module
pytest tests/test_models.py -v
```

### Code Quality

```bash
# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/
```

---

## 🗺️ Roadmap

### Phase 1: MVP (Current)
- ✅ Core framework
- ✅ LLM-as-judge evaluation
- ✅ CLI interface
- ✅ Cost tracking

### Phase 2: Enhanced Features
- ⏳ Batch evaluation
- ⏳ Custom judge models
- ⏳ Results visualization
- ⏳ Web interface

### Phase 3: Advanced Analytics
- 📋 Historical tracking
- 📋 A/B testing
- 📋 Regression detection
- 📋 Fine-tuning guidance

---

## 🤝 Contributing

Contributions welcome! Please see our contributing guidelines (coming soon).

### Ways to Contribute

- 🐛 Report bugs
- 💡 Suggest features
- 📝 Improve documentation
- 🔧 Submit pull requests
- 📊 Share your task definitions

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **OpenRouter** for unified LLM API access
- **Anthropic** for Claude Sonnet 4.5
- **Research participants** who tested 42 models

---

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/knightsri/llm-taskbench/issues)
- **Discussions**: [GitHub Discussions](https://github.com/knightsri/llm-taskbench/discussions)

---

## ⭐ Star History

If you find LLM TaskBench useful, please consider giving it a star! ⭐

---

**Built with ❤️ for developers who need real-world LLM evaluation**
