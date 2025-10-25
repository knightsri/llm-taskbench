# LLM TaskBench - Development TODO

**Project Timeline:** 6-8 weeks (MVP)  
**Start Date:** [Your start date]  
**Target Demo Date:** [Your demo date]

---

## 📋 MVP Development Phases

### ✅ Phase 0: Project Setup & Planning (Week 0)

**Status:** COMPLETE

- [x] Technical specification finalized
- [x] GitHub repository created (`llm-taskbench`)
- [x] README.md created
- [x] TODO.md created
- [x] Project structure designed
- [x] Technology stack selected

---

## 🏗️ Phase 1: Core Framework (Weeks 1-2)

**Goal:** Get basic task definition and orchestration working with one model provider

### Week 1: Foundation

#### Project Setup
- [ ] Initialize Python project structure
  - [ ] Create `src/taskbench/` directory structure
  - [ ] Setup `pyproject.toml` with dependencies
  - [ ] Configure `ruff` for linting
  - [ ] Setup pytest configuration
  - [ ] Create `.gitignore`
  - [ ] Add LICENSE (Apache 2.0)

#### Task Definition System
- [ ] Create `task.py` - Task data models
  - [ ] `Task` class (Pydantic model)
  - [ ] `TaskInput` schema
  - [ ] `ExpectedOutput` schema
  - [ ] `EvaluationCriteria` schema
  
- [ ] Create `task_loader.py` - YAML loading
  - [ ] Load task from YAML file
  - [ ] Validate task structure
  - [ ] Handle missing/invalid fields
  - [ ] Unit tests for loader

- [ ] Create example task YAML
  - [ ] `lecture_analysis.yaml` (your research task)
  - [ ] Test loading and validation

#### CLI Foundation
- [ ] Create `cli.py` - Click-based CLI
  - [ ] `taskbench init` command (skeleton)
  - [ ] `taskbench run` command (skeleton)
  - [ ] `taskbench list-tasks` command
  - [ ] `--help` documentation
  - [ ] Entry point in `pyproject.toml`

### Week 2: Basic Orchestration & Runner

#### Orchestrator Agent
- [ ] Create `orchestrator.py`
  - [ ] `Orchestrator` class
  - [ ] Load task definition
  - [ ] Validate input sample
  - [ ] Prepare execution plan
  - [ ] Handle multiple models
  - [ ] Error handling
  - [ ] Unit tests

#### Runner Agent (OpenRouter Only)
- [ ] Create `runner.py`
  - [ ] `RunnerAgent` class
  - [ ] OpenRouter API integration
  - [ ] API key management (environment variables)
  - [ ] Build prompts from task definition
  - [ ] Execute single model call
  - [ ] Parse model responses
  - [ ] Cost tracking (token count × pricing)
  - [ ] Error handling and retries
  - [ ] Unit tests (with mocked API calls)

#### Result Storage
- [ ] Create `storage.py`
  - [ ] Save results to JSON files
  - [ ] Directory structure: `results/{task_name}/{timestamp}/`
  - [ ] Store: model output, tokens, cost, metadata
  - [ ] Load results for analysis

#### Integration Testing
- [ ] End-to-end test: Load task → Run on OpenRouter → Save results
- [ ] Test with Claude Sonnet 4.5 (real API call)
- [ ] Verify cost calculation accuracy

#### Phase 1 Demo Checkpoint
- [ ] Can run: `taskbench run --task lecture-analysis --model claude-sonnet-4.5 --input sample.txt`
- [ ] Results saved correctly
- [ ] Costs tracked accurately

---

## ⚖️ Phase 2: Evaluation Engine (Weeks 3-4)

**Goal:** Implement quality scoring (rule-based + LLM-as-judge)

### Week 3: Rule-Based Evaluation

#### Evaluator Framework
- [ ] Create `evaluator.py`
  - [ ] `EvaluatorAgent` base class
  - [ ] Load evaluation criteria from task
  - [ ] Score aggregation logic

#### Rule-Based Evaluator
- [ ] Create `rule_evaluator.py`
  - [ ] `RuleBasedEvaluator` class
  - [ ] Format validation (CSV, JSON, etc.)
  - [ ] Count checks (min/max items)
  - [ ] Duration checks (for time-based tasks)
  - [ ] Pattern matching (regex rules)
  - [ ] Custom rule engine
  - [ ] Scoring: violations → deductions
  - [ ] Unit tests for each rule type

#### Task-Specific Rules
- [ ] Implement lecture analysis rules
  - [ ] Concept count (target: 20-30)
  - [ ] Segment duration (2-7 minutes)
  - [ ] No gaps >5 minutes
  - [ ] Valid timestamp format
  
- [ ] Add rules to `lecture_analysis.yaml`
- [ ] Test on sample outputs

### Week 4: LLM-as-Judge Evaluation

#### LLM Judge Framework
- [ ] Create `llm_judge.py`
  - [ ] `LLMJudgeEvaluator` class
  - [ ] Judge prompt templates
  - [ ] Call judge model (Claude Sonnet 4 recommended)
  - [ ] Parse judge scores (1-10 scale)
  - [ ] Confidence scoring
  - [ ] Cost tracking for judge calls

#### Judge Criteria Implementation
- [ ] Content coverage scoring
  - [ ] "Did the model capture all key concepts?"
  - [ ] Compare to input text
  
- [ ] Concept clarity scoring
  - [ ] "Are concept names descriptive?"
  - [ ] "Would a student understand these?"
  
- [ ] Timestamp accuracy scoring
  - [ ] "Are boundaries logical?"
  - [ ] "Do times match actual content?"

#### Score Aggregation
- [ ] Combine rule-based + LLM-judge scores
- [ ] Weighted scoring (configurable per task)
- [ ] Final quality score (1-10)
- [ ] Generate score breakdown

#### Evaluation Reports
- [ ] Create `evaluation_report.py`
  - [ ] JSON report structure
  - [ ] Include: scores, violations, judge feedback
  - [ ] Save to results directory

#### Integration Testing
- [ ] Test full evaluation pipeline
- [ ] Rule-based + LLM-judge working together
- [ ] Verify score calculations
- [ ] Check cost tracking for judge calls

#### Phase 2 Demo Checkpoint
- [ ] Can run: `taskbench run --task lecture-analysis --models claude-sonnet-4.5,gpt-4o`
- [ ] Both models evaluated with scores
- [ ] Results include quality breakdown

---

## 📊 Phase 3: Analysis & Recommendations (Weeks 5-6)

**Goal:** Generate comparative analysis and model recommendations

### Week 5: Analyzer Agent

#### Cost Analysis
- [ ] Create `analyzer.py`
  - [ ] `AnalyzerAgent` class
  - [ ] Load multiple evaluation results
  - [ ] Calculate cost metrics
    - [ ] Cost per task
    - [ ] Cost per 100 tasks
    - [ ] Cost for typical monthly/semester usage
  
#### Quality Comparison
- [ ] Compare quality scores across models
  - [ ] Rank models by quality
  - [ ] Identify quality gaps
  - [ ] Statistical significance (if multiple samples)
  
#### Trade-off Analysis
- [ ] Create `tradeoff_analyzer.py`
  - [ ] Cost vs quality plotting (text-based for CLI)
  - [ ] Pareto frontier identification
  - [ ] "Sweet spot" detection
  - [ ] Value calculation: quality per dollar

#### Metrics Calculation
- [ ] Per-model metrics
  - [ ] Quality score (1-10)
  - [ ] Cost per task ($)
  - [ ] Violations count
  - [ ] Performance rank
  
- [ ] Comparative metrics
  - [ ] Cost savings vs best quality
  - [ ] Quality loss vs cheapest
  - [ ] ROI for premium models

### Week 6: Recommendation Engine & Reports

#### Recommendation Engine
- [ ] Create `recommender.py`
  - [ ] `RecommendationEngine` class
  - [ ] Decision tree logic
    - [ ] If quality critical → best model
    - [ ] If cost critical → budget model
    - [ ] If balanced → sweet spot model
  
- [ ] Generate recommendations
  - [ ] Best choice (highest quality)
  - [ ] Budget option (best value)
  - [ ] Not recommended (poor value)
  - [ ] Reasoning for each

#### Report Generator
- [ ] Create `report_generator.py`
  - [ ] `ReportGenerator` class
  - [ ] Markdown report template (Jinja2)
  - [ ] Rich terminal output (for CLI)
  
- [ ] Report sections
  - [ ] Executive Summary
  - [ ] Model Comparison Table
  - [ ] Cost Analysis
  - [ ] Quality Breakdown
  - [ ] Recommendations
  - [ ] Cost Projections
  
- [ ] Create `templates/report.md.j2`
  - [ ] Professional formatting
  - [ ] Tables, charts (text-based)
  - [ ] Clear recommendations

#### CLI Commands
- [ ] `taskbench analyze` - Run analysis on saved results
- [ ] `taskbench recommend` - Show recommendations
- [ ] `taskbench report` - Generate full report
- [ ] `taskbench compare --models model1,model2` - Compare specific models

#### Rich Terminal Output
- [ ] Use `rich` library for beautiful output
- [ ] Colored recommendation boxes (like spec example)
- [ ] Progress bars during evaluation
- [ ] Tables for comparison
- [ ] Success/warning/error formatting

#### Integration Testing
- [ ] Test full pipeline: Run → Evaluate → Analyze → Recommend
- [ ] Verify recommendations match expected logic
- [ ] Check report formatting

#### Phase 3 Demo Checkpoint
- [ ] Can run full workflow and get recommendations
- [ ] Reports look professional
- [ ] Recommendations are clear and actionable

---

## 🎯 Phase 4: Built-in Tasks & Polish (Weeks 7-8)

**Goal:** Polish for demo, add 2-3 tasks, documentation, and publish

### Week 7: Additional Tasks & Documentation

#### Built-in Task 1: Lecture Transcript Analysis
- [x] Already defined from your research
- [ ] Add comprehensive examples
- [ ] Document expected input format
- [ ] Add sample transcript for testing
- [ ] Validate on multiple samples

#### Built-in Task 2: Customer Support Ticket Categorization
- [ ] Define task YAML
  - [ ] Input: support ticket text
  - [ ] Output: category, priority, sentiment
  - [ ] Evaluation: accuracy, consistency
  
- [ ] Create sample tickets (10-15)
- [ ] Define evaluation rules
- [ ] Test with 2-3 models
- [ ] Document task

#### Built-in Task 3: Medical Case Summary (Optional)
- [ ] Define task YAML
  - [ ] Input: clinical notes
  - [ ] Output: structured summary
  - [ ] Evaluation: completeness, accuracy
  
- [ ] Create sample cases (5-10)
- [ ] Define evaluation criteria
- [ ] Test with 2-3 models
- [ ] Document task

**NOTE:** If time is tight, focus on Tasks 1 & 2. Task 3 can be post-MVP.

#### Documentation
- [ ] `docs/getting-started.md`
  - [ ] Installation
  - [ ] First evaluation
  - [ ] Understanding results
  
- [ ] `docs/creating-tasks.md`
  - [ ] Task YAML format
  - [ ] Evaluation criteria
  - [ ] Examples
  - [ ] Validation
  
- [ ] `docs/api-reference.md`
  - [ ] CLI commands
  - [ ] Task schema
  - [ ] Configuration options
  
- [ ] `docs/evaluation-guide.md`
  - [ ] Rule-based evaluation
  - [ ] LLM-as-judge
  - [ ] Scoring methodology
  
- [ ] Update main README.md
  - [ ] Add examples
  - [ ] Add screenshots (terminal output)
  - [ ] Add badge updates

### Week 8: Testing, Polish & Release

#### Comprehensive Testing
- [ ] Unit tests for all modules (target: 80%+ coverage)
- [ ] Integration tests for full workflows
- [ ] End-to-end tests with real API calls
- [ ] Test error handling (API failures, invalid inputs)
- [ ] Test cost calculations with known examples
- [ ] Performance testing (1, 5, 10 models)

#### Code Quality
- [ ] Run `ruff` and fix all issues
- [ ] Add type hints everywhere
- [ ] Docstrings for all public functions
- [ ] Code review and cleanup
- [ ] Remove debug code

#### PyPI Package Preparation
- [ ] Finalize `pyproject.toml`
  - [ ] Dependencies locked
  - [ ] Entry points configured
  - [ ] Metadata complete (description, keywords, classifiers)
  
- [ ] Create `MANIFEST.in` (if needed)
- [ ] Test package build: `python -m build`
- [ ] Test installation: `pip install dist/taskbench-*.whl`
- [ ] Verify CLI commands work after install

#### Demo Preparation
- [ ] Create demo script
  - [ ] Show problem (generic benchmarks don't help)
  - [ ] Run TaskBench evaluation
  - [ ] Show results and recommendations
  - [ ] Highlight key differentiators
  
- [ ] Record demo video (5-7 minutes)
  - [ ] Problem statement
  - [ ] Installation
  - [ ] Running evaluation
  - [ ] Interpreting results
  - [ ] Creating custom task
  
- [ ] Prepare slides (if needed)
  - [ ] Architecture overview
  - [ ] Technical achievements
  - [ ] Results and insights

#### Release Checklist
- [ ] All Phase 1-4 tests passing
- [ ] Documentation complete
- [ ] README polished
- [ ] Demo video ready
- [ ] PyPI package tested
- [ ] Git tags created (`v1.0.0`)

#### PyPI Publication
- [ ] Create PyPI account
- [ ] Generate API token
- [ ] Test publish to TestPyPI
- [ ] Publish to PyPI: `twine upload dist/*`
- [ ] Verify installation: `pip install taskbench`
- [ ] Update README with PyPI badge

#### Phase 4 Demo Checkpoint
- [ ] Complete working system
- [ ] 2-3 tasks demonstrable
- [ ] Professional documentation
- [ ] Published to PyPI
- [ ] Demo video recorded
- [ ] Ready for course presentation

---

## 🚀 Post-MVP (Future Work)

### V1.1 Enhancements
- [ ] Web UI (Streamlit or Gradio)
- [ ] More LLM providers (Groq, Together AI)
- [ ] Local model support (Ollama)
- [ ] Parallel evaluation (speed boost)
- [ ] Task marketplace (community tasks)

### V1.2 Features
- [ ] Batch evaluation (multiple samples)
- [ ] A/B testing capabilities
- [ ] Cost budgets and alerts
- [ ] CI/CD integration
- [ ] Slack/email notifications

### V2.0 Vision
- [ ] Managed cloud service
- [ ] Real-time dashboard
- [ ] Historical tracking
- [ ] Team collaboration
- [ ] Enterprise features

---

## 📝 Notes & Decisions

### Key Decisions Made
- **LLM Provider:** Start with OpenRouter (easy access to multiple models)
- **Judge Model:** Claude Sonnet 4 (good balance of quality and cost)
- **Report Format:** Markdown + Rich terminal output
- **Testing:** Mock API calls in unit tests, real calls in integration tests

### Risk Mitigation
- **Risk:** LLM-as-judge too slow/expensive
  - **Mitigation:** Cache judge results, use smaller judge model for testing
  
- **Risk:** Task 3 takes too long
  - **Mitigation:** Focus on Tasks 1-2, make Task 3 post-MVP
  
- **Risk:** API rate limits during testing
  - **Mitigation:** Use rate limiting, mock tests, small test samples

### Time Management
- **Buffer:** Week 8 has buffer time for unexpected issues
- **Priority:** Core functionality > polish > extra features
- **Demo Focus:** Lecture analysis task (your research) is the star

---

## ✅ Success Criteria Checklist

At the end of 8 weeks, you should have:

- [ ] **Working CLI tool** - Install via pip, run evaluations
- [ ] **3 built-in tasks** - Lecture analysis + 2 others (or 1 other if time tight)
- [ ] **5+ models supported** - Via OpenRouter/direct APIs
- [ ] **Quality evaluation** - Rule-based + LLM-as-judge
- [ ] **Cost analysis** - Accurate tracking and projections
- [ ] **Recommendations** - Clear guidance on which model to use
- [ ] **Documentation** - Getting started, task creation, API reference
- [ ] **Tests** - Unit + integration tests passing
- [ ] **Published** - Available on PyPI
- [ ] **Demo ready** - Video + live demo prepared

---

## 📞 Questions & Blockers

**Use this section to track blockers:**

- [ ] Issue: [Description]
  - Status: [Blocked/In Progress/Resolved]
  - Resolution: [How it was solved]

---

**Last Updated:** [Date]  
**Current Phase:** Phase 0 (Setup)  
**Next Milestone:** Phase 1 Week 1 - Project Setup
