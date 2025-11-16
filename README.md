# LLM TaskBench

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://img.shields.io/badge/tests-117%20passed-brightgreen.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-80%25%2B-green.svg)](tests/)

> **A task-first LLM evaluation framework that enables domain experts to compare multiple LLMs on their actual use cases without requiring AI engineering expertise.**

Stop guessing which LLM is best for your use case. Define your task once, evaluate 5+ models automatically, and get cost-aware recommendations backed by real performance data.

---

## Table of Contents

- [The Problem](#the-problem)
- [The Solution](#the-solution)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Example Output](#example-output)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage Examples](#usage-examples)
- [Research Background](#research-background)
- [Comparison with Other Tools](#comparison-with-other-tools)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## The Problem

Existing LLM evaluation tools (DeepEval, Promptfoo, Eleuther AI) focus on generic benchmarks (BLEU, ROUGE, MMLU) and are built for AI engineers, not domain experts.

**When a medical professional needs to know:**
- "Which LLM best triages patient symptoms?"
- "What's the most cost-effective model for medical transcript summarization?"

**Current tools provide:**
- Generic scores that don't translate to real tasks
- Complex setup requiring AI engineering expertise
- No cost-quality tradeoff analysis
- No actionable recommendations

**What you actually need:**
- Evaluation on YOUR specific task
- Simple YAML configuration
- Cost vs. quality insights
- Clear recommendations: "Use Model X for production, Model Y for development"

---

## The Solution

LLM TaskBench shifts evaluation from **metric-first to task-first**:

1. **Define YOUR actual task** in a simple YAML file
2. **Evaluate 5+ models automatically** via OpenRouter (42+ models available)
3. **Get cost-aware recommendations** with quality scores and violation tracking
4. **Make informed decisions** backed by real performance data

```yaml
# tasks/my_task.yaml
name: "patient_triage"
description: "Extract urgency level and symptoms from patient messages"
input_type: "text"
output_format: "json"
evaluation_criteria:
  - "Correct urgency classification"
  - "Complete symptom extraction"
  - "Valid JSON output"
constraints:
  required_fields: ["urgency", "symptoms", "recommendations"]
```

```bash
# Evaluate multiple models on your task
$ taskbench evaluate tasks/my_task.yaml \
    --models claude-sonnet-4.5,gpt-4o,gemini-2.5-pro

# Get recommendations
$ taskbench recommend results.json
```

---

## Key Features

### 1. Task-First Evaluation

Define your actual use case, not a proxy metric:

**Built-in Task: Lecture Concept Extraction**
- Extract teaching concepts from lecture transcripts
- Precise timestamps (HH:MM:SS format)
- Duration constraints (2-7 minute segments)
- CSV output with structured data

**Easily Extensible:**
- Medical case triage
- Support ticket categorization
- Contract clause extraction
- Code review summarization
- *Your domain-specific task*

### 2. LLM-as-Judge Quality Assessment

Let Claude Sonnet 4.5 or GPT-4 score outputs on *your* criteria:

- **Accuracy scores** (0-100): Content correctness
- **Format compliance** (0-100): Output format adherence
- **Constraint compliance** (0-100): Rule following
- **Violation detection**: Specific issues identified
- **Detailed reasoning**: Why scores were assigned

### 3. Cost-Aware Recommendations

Balance quality and budget with real-time tracking:

- Token usage tracked to the penny (accurate to $0.01)
- Cost-quality tradeoff visualization
- Production vs. development recommendations
- "Sweet spot" identification (e.g., "87% as good, 50% cheaper")

### 4. Research-Validated Approach

Built on findings from testing **42 production LLMs** on lecture transcript analysis:

**Surprising Insights:**
- Model size ≠ quality (405B didn't beat 72B)
- "Reasoning-optimized" models can perform WORSE on reasoning tasks
- Cost shows ZERO correlation with performance (36X price range)
- Mid-tier models often hit the quality/cost sweet spot

**Read the full research:** ["How a $3.45 API Call Taught Me Everything About LLM Cost Optimization"](https://www.linkedin.com/pulse/how-345-api-call-taught-me-everything-llm-cost-sri-bolisetty-vxate)

### 5. Simple, Powerful API

```python
from taskbench.core.task import TaskParser
from taskbench.evaluation.executor import ModelExecutor

# Load your task
parser = TaskParser()
task = parser.load_from_yaml("tasks/my_task.yaml")

# Evaluate on multiple models
executor = ModelExecutor()
results = await executor.evaluate_multiple(
    models=["anthropic/claude-sonnet-4.5", "openai/gpt-4o"],
    task=task,
    input_data="Your input data here..."
)

# Get cost summary
print(executor.get_cost_summary())
```

---

## Quick Start

### Prerequisites

- **Python 3.11+**
- **OpenRouter API key** (get one at [openrouter.ai/keys](https://openrouter.ai/keys))

### Installation

```bash
# Clone the repository
git clone https://github.com/knightsri/llm-taskbench.git
cd llm-taskbench

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Set up API key
cp .env.example .env
# Edit .env and add: OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

### Run Your First Evaluation

```bash
# Validate the built-in task
taskbench validate tasks/lecture_analysis.yaml

# Evaluate on 3 models (uses lecture transcript from examples/)
taskbench evaluate tasks/lecture_analysis.yaml \
    --models anthropic/claude-sonnet-4.5,openai/gpt-4o,google/gemini-2.0-flash-exp \
    --input examples/lecture_transcript.txt \
    --output results.json

# View recommendations
taskbench recommend results.json
```

---

## Example Output

Here's what actual evaluation results look like:

```
Evaluating Models on Task: lecture_concept_extraction
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[1/5] Evaluating anthropic/claude-sonnet-4.5...
✓ Completed in 2.4s | 24 concepts | 0 violations | $0.36

[2/5] Evaluating openai/gpt-4o...
✓ Completed in 1.8s | 23 concepts | 1 violation | $0.42

[3/5] Evaluating google/gemini-2.0-flash-exp...
✓ Completed in 1.2s | 22 concepts | 2 violations | $0.15

[4/5] Evaluating meta-llama/llama-3.1-405b...
✓ Completed in 3.1s | 20 concepts | 4 violations | $0.25

[5/5] Evaluating qwen/qwen-2.5-72b-instruct...
✓ Completed in 1.9s | 21 concepts | 3 violations | $0.18

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Evaluation Summary:
┌─────────────────────────────┬───────┬────────────┬──────────┬──────────────┐
│ Model                       │ Score │ Violations │ Cost     │ Tier         │
├─────────────────────────────┼───────┼────────────┼──────────┼──────────────┤
│ anthropic/claude-sonnet-4.5 │ 98    │ 0          │ $0.36    │ Excellent    │
│ openai/gpt-4o               │ 95    │ 1          │ $0.42    │ Excellent    │
│ google/gemini-2.0-flash-exp │ 92    │ 2          │ $0.15    │ Excellent    │
│ qwen/qwen-2.5-72b-instruct  │ 87    │ 3          │ $0.18    │ Good         │
│ meta-llama/llama-3.1-405b   │ 83    │ 4          │ $0.25    │ Good         │
└─────────────────────────────┴───────┴────────────┴──────────┴──────────────┘

Total Cost: $1.36

Recommendations:
  🏆 Best Overall: anthropic/claude-sonnet-4.5
     - Highest accuracy (98/100)
     - Zero constraint violations
     - $0.36 per evaluation

  💎 Best Value: google/gemini-2.0-flash-exp
     - 94% as good as best model
     - 58% cheaper ($0.15 vs $0.36)
     - Only 2 minor violations

  💰 Budget Option: qwen/qwen-2.5-72b-instruct
     - 89% as good as best model
     - 50% cheaper than best
     - Acceptable for non-critical use cases
```

**Violations Detail:**
```
Common Violations:
  - "Duration 00:08:30 exceeds max_duration_minutes: 7" (3 models)
  - "Timestamp format incorrect: missing leading zeros" (2 models)
  - "Overlapping time ranges: 00:15:20-00:18:00 and 00:17:45-00:20:30" (1 model)
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      LLM TaskBench System                        │
│                                                                   │
│  ┌──────────────┐        ┌────────────────┐                     │
│  │  User Input  │───────▶│  Task Parser   │                     │
│  │  (CLI/API)   │        │  (YAML→Model)  │                     │
│  └──────────────┘        └────────┬───────┘                     │
│                                    │                              │
│                                    ▼                              │
│                         ┌──────────────────┐                     │
│                         │ Model Executor   │                     │
│                         │ (Orchestrator)   │                     │
│                         └────────┬─────────┘                     │
│                                  │                                │
│                  ┌───────────────┼───────────────┐               │
│                  ▼               ▼               ▼               │
│          ┌──────────┐    ┌──────────┐    ┌──────────┐           │
│          │ Model 1  │    │ Model 2  │    │ Model N  │           │
│          │ (Claude) │    │  (GPT-4) │    │ (Gemini) │           │
│          └────┬─────┘    └────┬─────┘    └────┬─────┘           │
│               └───────────────┼───────────────┘                  │
│                               ▼                                  │
│                    ┌──────────────────────┐                      │
│                    │  LLM-as-Judge        │                      │
│                    │  Evaluator           │                      │
│                    │  (Claude/GPT-4)      │                      │
│                    └──────────┬───────────┘                      │
│                               │                                  │
│                               ▼                                  │
│                    ┌──────────────────────┐                      │
│                    │  Cost Analysis       │                      │
│                    │  Engine              │                      │
│                    └──────────┬───────────┘                      │
│                               │                                  │
│                               ▼                                  │
│                    ┌──────────────────────┐                      │
│                    │  Recommendation      │                      │
│                    │  Engine              │                      │
│                    └──────────┬───────────┘                      │
│                               │                                  │
│                               ▼                                  │
│                    ┌──────────────────────┐                      │
│                    │  Results Output      │                      │
│                    │  (JSON/CSV/CLI)      │                      │
│                    └──────────────────────┘                      │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘

External Services:
┌──────────────┐        ┌──────────────┐
│  OpenRouter  │        │  Direct APIs │
│  (42+ models)│        │  (Optional)  │
└──────────────┘        └──────────────┘
```

**Core Components:**

1. **Task Parser** (`taskbench.core.task`)
   - Loads and validates YAML task definitions
   - Ensures all required fields are present
   - Validates constraints for logical consistency

2. **Model Executor** (`taskbench.evaluation.executor`)
   - Builds comprehensive prompts from tasks
   - Executes tasks on multiple models via OpenRouter
   - Tracks token usage and calculates costs
   - Handles errors gracefully

3. **LLM-as-Judge** (`taskbench.evaluation.judge`)
   - Uses powerful LLM to evaluate outputs
   - Provides detailed scores across multiple dimensions
   - Identifies specific constraint violations
   - Returns structured JSON with reasoning

4. **Cost Tracker** (`taskbench.evaluation.cost`)
   - Maintains pricing database for 42+ models
   - Calculates exact costs per evaluation
   - Tracks cumulative spending
   - Generates cost breakdowns

5. **API Client** (`taskbench.api.client`)
   - Async HTTP client for OpenRouter
   - Automatic retry with exponential backoff
   - Error handling and rate limit management
   - Support for JSON mode and streaming

---

## Technology Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Language** | Python 3.11+ | Core implementation |
| **CLI Framework** | Typer | Command-line interface |
| **Terminal UI** | Rich | Beautiful output formatting |
| **Data Validation** | Pydantic | Type-safe data models |
| **HTTP Client** | httpx | Async API calls |
| **Configuration** | PyYAML | Task definition parsing |
| **Environment** | python-dotenv | API key management |
| **Testing** | pytest + pytest-asyncio | Unit and integration tests |
| **Code Quality** | black, isort, mypy, flake8 | Linting and formatting |
| **AI Gateway** | OpenRouter | Unified access to 42+ models |

**Why These Choices?**

- **Pydantic**: Type safety prevents bugs, auto-validates data
- **Rich**: Makes CLI output professional and readable
- **httpx**: Modern async HTTP with great error handling
- **OpenRouter**: One API for all major LLMs (no vendor lock-in)
- **pytest**: Industry standard, excellent async support

---

## Installation

### Option 1: From Source (Recommended for Development)

```bash
# Clone repository
git clone https://github.com/knightsri/llm-taskbench.git
cd llm-taskbench

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests to verify installation
pytest tests/ -v

# You should see: 117 passed
```

### Option 2: Install Dependencies Only

```bash
pip install pydantic pyyaml httpx typer rich python-dotenv
```

### Set Up API Keys

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your OpenRouter API key
# Get one at: https://openrouter.ai/keys
echo "OPENROUTER_API_KEY=sk-or-v1-your-actual-key" > .env
```

### Verify Installation

```bash
# Check CLI is available
taskbench --help

# Validate a task
taskbench validate tasks/lecture_analysis.yaml

# Should output: ✓ Task validation passed!
```

---

## Usage Examples

### Example 1: Basic Task Evaluation

```bash
# Evaluate single model
taskbench evaluate tasks/lecture_analysis.yaml \
    --model anthropic/claude-sonnet-4.5 \
    --input examples/lecture_transcript.txt

# Evaluate multiple models
taskbench evaluate tasks/lecture_analysis.yaml \
    --models claude-sonnet-4.5,gpt-4o,gemini-2.0-flash-exp \
    --input examples/lecture_transcript.txt \
    --output results.json
```

### Example 2: Programmatic Usage

```python
import asyncio
from taskbench.core.task import TaskParser
from taskbench.evaluation.executor import ModelExecutor
from taskbench.evaluation.judge import LLMJudge
from taskbench.api.client import OpenRouterClient

async def evaluate_my_task():
    # Load task definition
    parser = TaskParser()
    task = parser.load_from_yaml("tasks/my_task.yaml")

    # Load input data
    with open("input.txt") as f:
        input_data = f.read()

    # Execute on multiple models
    executor = ModelExecutor()
    models = [
        "anthropic/claude-sonnet-4.5",
        "openai/gpt-4o",
        "google/gemini-2.0-flash-exp"
    ]

    results = await executor.evaluate_multiple(
        model_ids=models,
        task=task,
        input_data=input_data
    )

    # Evaluate with LLM-as-judge
    async with OpenRouterClient() as client:
        judge = LLMJudge(client)

        scores = []
        for result in results:
            if result.status == "success":
                score = await judge.evaluate(task, result, input_data)
                scores.append(score)
                print(f"{result.model_name}: {score.overall_score}/100")

    # Get cost summary
    print(f"\nTotal cost: ${executor.cost_tracker.get_total_cost():.4f}")

    return results, scores

# Run evaluation
results, scores = asyncio.run(evaluate_my_task())
```

### Example 3: Create Custom Task

```yaml
# tasks/code_review.yaml
name: "pull_request_review"
description: "Review pull request changes and provide actionable feedback"

input_type: "text"
output_format: "json"

evaluation_criteria:
  - "Identifies security vulnerabilities"
  - "Suggests performance improvements"
  - "Checks code style compliance"
  - "Provides actionable feedback"

constraints:
  required_fields: ["security_issues", "performance_suggestions", "style_notes", "overall_rating"]
  min_issues_to_report: 0
  max_response_length: 2000

examples:
  - input: "diff --git a/app.py..."
    expected_output: |
      {
        "security_issues": ["SQL injection risk on line 42"],
        "performance_suggestions": ["Use batch queries instead of N+1"],
        "style_notes": ["Missing docstrings"],
        "overall_rating": "needs_work"
      }

judge_instructions: |
  Score based on:
  1. Security: Did it catch all vulnerabilities? (40%)
  2. Performance: Are suggestions valid and impactful? (30%)
  3. Style: Are recommendations aligned with best practices? (20%)
  4. Format: Is output valid JSON with all required fields? (10%)
```

---

## Research Background

LLM TaskBench is built on insights from comprehensive research testing **42 production LLMs** on a real-world task: extracting teaching concepts from lecture transcripts.

### Key Findings

**1. Model Size Doesn't Predict Quality**
- Llama 3.1 405B (largest) scored 83/100
- Qwen 2.5 72B (6x smaller) scored 87/100
- Result: Fine-tuning > raw parameters

**2. Cost and Quality Are Uncorrelated**
- Price range: $0.01 to $0.36 per transcript (36x difference)
- Correlation between cost and score: **r = 0.04** (essentially zero)
- Mid-tier models often offer best value

**3. "Reasoning" Models Don't Always Reason Better**
- Some reasoning-optimized models scored lower on reasoning tasks
- Task-specific evaluation reveals actual strengths/weaknesses

**4. The Sweet Spot Exists**
- Claude Sonnet 4.5: 98/100, $0.36 (best overall)
- Gemini 2.0 Flash: 92/100, $0.15 (94% as good, 58% cheaper)
- **Recommendation:** Use Claude for production, Gemini for development

### Research Article

Read the full analysis with charts and insights:
**["How a $3.45 API Call Taught Me Everything About LLM Cost Optimization"](https://www.linkedin.com/pulse/how-345-api-call-taught-me-everything-llm-cost-sri-bolisetty-vxate)**

---

## Comparison with Other Tools

| Feature | LLM TaskBench | DeepEval | Promptfoo | Eleuther LM Eval |
|---------|--------------|----------|-----------|------------------|
| **Target Users** | Domain experts | AI engineers | AI engineers | Researchers |
| **Task Definition** | Simple YAML | Python code | YAML + JS | Python code |
| **Focus** | Your actual use case | Generic metrics | Prompt engineering | Academic benchmarks |
| **Cost Tracking** | Real-time, accurate | No | Limited | No |
| **Cost-Quality Analysis** | Yes | No | No | No |
| **Recommendations** | Actionable ("Use X for prod") | No | No | No |
| **LLM Access** | 42+ via OpenRouter | API integrations | API integrations | Limited |
| **Setup Complexity** | Low (pip + .env) | Medium | Medium | High |
| **Violation Detection** | Automatic | Manual | Manual | N/A |
| **Learning Curve** | Hours | Days | Days | Weeks |

### When to Use LLM TaskBench

✅ **Use LLM TaskBench when you:**
- Need to compare models on YOUR specific task
- Want cost-quality tradeoff insights
- Need actionable recommendations, not just scores
- Prefer YAML config over code
- Want to test 5+ models quickly

❌ **Use other tools when you:**
- Need to evaluate on academic benchmarks (MMLU, HellaSwag)
- Require advanced prompt optimization features
- Want to build custom evaluation metrics in code
- Need integration with specific ML platforms

---

## Documentation

### Core Documentation

- **[Usage Guide](docs/USAGE.md)**: Complete installation and usage instructions
- **[API Reference](docs/API.md)**: Detailed API documentation for all modules
- **[Architecture](docs/ARCHITECTURE.md)**: System design and component details
- **[Project Overview](docs/PROJECT-OVERVIEW.md)**: Vision, goals, and roadmap

### Task Definitions

- **[Lecture Analysis](tasks/lecture_analysis.yaml)**: Built-in task for concept extraction
- **[Task Template](tasks/template.yaml)**: Template for creating custom tasks

### Examples

- Check the `examples/` directory for sample inputs and outputs
- See `tests/` directory for comprehensive usage examples in test cases

---

## Contributing

We welcome contributions! Here's how to get started:

### Quick Contribution Guide

1. **Fork the repository**
   ```bash
   git clone https://github.com/your-username/llm-taskbench.git
   cd llm-taskbench
   ```

2. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Set up development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -e ".[dev]"
   ```

4. **Make your changes**
   - Follow existing code style (we use black, isort, mypy)
   - Add tests for new features
   - Update documentation

5. **Run tests and checks**
   ```bash
   # Run tests
   pytest tests/ -v

   # Format code
   black src/ tests/
   isort src/ tests/

   # Type checking
   mypy src/

   # Linting
   flake8 src/ tests/
   ```

6. **Submit a pull request**
   - Describe your changes clearly
   - Link any related issues
   - Ensure all tests pass

### Contribution Ideas

**High Priority:**
- Add new built-in tasks (medical triage, code review, etc.)
- Improve documentation with more examples
- Add more model providers (direct API support)
- Create Jupyter notebook tutorials

**Medium Priority:**
- Web UI (Streamlit/Gradio)
- Batch evaluation support
- Historical tracking and comparison
- Export to more formats (Markdown, PDF)

**Nice to Have:**
- Integration with LangSmith/Helicone
- Custom judge model selection
- Parallel execution
- Visualization dashboard

### Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Credit others for their work

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Sri Bolisetty

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Contact

**Sri Bolisetty**

- **GitHub**: [@KnightSri](https://github.com/KnightSri)
- **LinkedIn**: [Sri Bolisetty](https://linkedin.com/in/sribolisetty)
- **Email**: sri@example.com
- **Project Repository**: [llm-taskbench](https://github.com/knightsri/llm-taskbench)

### Questions or Issues?

- **Bug Reports**: [Open an issue](https://github.com/knightsri/llm-taskbench/issues)
- **Feature Requests**: [Start a discussion](https://github.com/knightsri/llm-taskbench/discussions)
- **General Questions**: Check the [Usage Guide](docs/USAGE.md) or [API docs](docs/API.md)

---

## Acknowledgments

- **OpenRouter** for providing unified access to 42+ LLMs
- **Anthropic** for Claude, which powers the LLM-as-judge evaluator
- **The Python Community** for amazing tools (Pydantic, Rich, httpx, Typer)
- **All contributors** who help improve this project

---

## Star History

If you find LLM TaskBench useful, please consider starring the repository!

[![Star History Chart](https://api.star-history.com/svg?repos=knightsri/llm-taskbench&type=Date)](https://star-history.com/#knightsri/llm-taskbench&Date)

---

**Built with 🧠 by Sri Bolisetty**
*Making LLM evaluation accessible to domain experts*

