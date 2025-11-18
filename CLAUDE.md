# CLAUDE.md

**Context file for AI-assisted development of LLM TaskBench**

This document provides essential context to help Claude (or any AI assistant) make decisions aligned with the project's goals, constraints, and architecture.

---

## Project Overview

**LLM TaskBench** is a flexible, task-specific LLM evaluation framework that enables developers to benchmark multiple LLM models on their actual use cases with Responsible AI metrics and cost tracking.

### Core Purpose

- Enable **task-first** evaluation with automatic quality validation
- Help developers find the **best model for their specific use case** balancing cost and quality
- Provide **LLM-powered quality check generation** - framework analyzes task and auto-generates validation rules
- Support **Responsible AI metrics** (hallucination, bias, safety, factuality) based on task requirements
- Deliver **cost-aware recommendations** with transparent cost breakdowns

### Key Innovation

**LLM-Powered Quality Check Generation**: When user defines a task, an LLM analyzes the problem and automatically generates task-specific quality checks. This makes the framework truly adaptive to any domain.

Example:
```
Task: "Extract medical concepts from clinical transcripts"
→ LLM generates checks:
  ✓ No overlapping timestamps
  ✓ Medical terminology preserved
  ✓ Minimum 2 minutes per concept
  ✓ No PHI (patient identifiable info) leaked
```

---

## Project Goals

### Primary Goals (MVP - 4-6 Weeks)

1. **Task-Specific Evaluation**: YAML-based task definition with UI-assisted configuration
2. **Multi-Model Comparison**: Evaluate 8-10 models simultaneously (OpenRouter primary, direct APIs supported)
3. **Responsible AI Metrics**: 6 core metrics (Accuracy, Hallucination, Completeness, Cost, Instruction Following, Consistency)
4. **LLM-Powered Validation**: Auto-generate quality checks from task descriptions
5. **Cost Tracking**: Token-level tracking with estimate vs. actual comparison
6. **Docker Deployment**: Single-command startup via `RunTaskBench.sh` → localhost:9999
7. **Historical Results**: Store and visualize past benchmark runs

### Secondary Goals (Post-MVP - See docs/TODO.md)

- Per-model prompt optimization
- Advanced parallel execution strategies  
- Deep failure pattern analysis
- Cost optimization suggestions
- Reproducibility versioning system
- Iteration/A/B testing workflow
- Cross-task comparative benchmarking
- Robustness testing (input corruption)
- Batch evaluation (multi-input processing)
- Custom judge models

**Full roadmap**: See `docs/TODO.md` for detailed post-MVP features organized by phase.

### Non-Goals

- Generic benchmarking (we're task-specific)
- Model training or fine-tuning (evaluation only)
- Real-time inference optimization (batch evaluation focus)

---

## Tech Stack

### Core Technologies

**Backend**:
- **Python 3.11+**: Modern Python features, type hints required
- **FastAPI**: REST API framework with automatic OpenAPI docs
- **Pydantic 2.0+**: Data validation and settings management
- **SQLAlchemy + PostgreSQL**: Task history, results storage
- **Redis**: Task queue (Celery workers)
- **Celery**: Async task execution

**Frontend**:
- **React 18+**: UI framework
- **TypeScript**: Type-safe frontend code
- **Vite**: Build tool and dev server
- **Tailwind CSS**: Utility-first styling
- **Recharts**: Data visualization

**Infrastructure**:
- **Docker + Docker Compose**: Container orchestration
- **NGINX** (optional): Reverse proxy for production

### Development Tools

- **pytest**: Testing framework with async support
- **black**: Code formatting (100 char line length)
- **isort**: Import sorting (black profile)
- **mypy**: Type checking (strict mode)
- **pytest-cov**: Code coverage reporting (target: 80%+)

### External Services

- **OpenRouter API**: Primary unified access to 100+ LLM models
- **Anthropic/OpenAI APIs** (optional): Direct API access when user provides keys
- **Claude Sonnet 4.5**: Default LLM for quality check generation and task analysis

---

## Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────┐
│                  INPUT GATHERING                    │
├─────────────────────────────────────────────────────┤
│ 1. Task Definition (UI + YAML generation)          │
│ 2. Gold Data Upload + Validation                   │
│ 3. Model Selection (Custom endpoints supported)    │
│ 4. Metric Configuration (LLM-recommended)          │
│ 5. LLM-Powered Quality Check Generation ⭐         │
│ 6. Budget & Constraints                             │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│              AGENTIC EXECUTION                      │
├─────────────────────────────────────────────────────┤
│ 1. Pre-flight Validation (gold data, prompts)      │
│ 2. Parallel Orchestration (via Celery)             │
│ 3. Real-time Progress Tracking                     │
│ 4. Error Handling (retry 3x, abandon after 5 fails)│
│ 5. Auto-generated Quality Checks Applied           │
│ 6. Checkpointing (resume from failures)            │
│ 7. Cost Tracking (estimate vs. actual)             │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│            OUTPUT ANALYSIS                          │
├─────────────────────────────────────────────────────┤
│ 1. Core Metrics Dashboard (6 metrics)              │
│ 2. Cost Breakdown (by model, by phase)             │
│ 3. Statistical Comparison (confidence intervals)   │
│ 4. Model Ranking (by accuracy, value, cost)        │
│ 5. Failed Examples Log                             │
│ 6. Export (JSON, CSV, PDF report)                  │
└─────────────────────────────────────────────────────┘
```

### Project Structure

```
llm-taskbench/
├── backend/
│   ├── app/
│   │   ├── api/              # FastAPI endpoints
│   │   │   ├── tasks.py      # Task CRUD operations
│   │   │   ├── evaluation.py # Start/monitor evaluations
│   │   │   └── results.py    # Results retrieval
│   │   ├── core/             # Business logic
│   │   │   ├── task_parser.py      # YAML validation
│   │   │   ├── quality_gen.py      # LLM quality check generation ⭐
│   │   │   ├── metric_selector.py  # LLM metric recommendations
│   │   │   └── executor.py         # Orchestrates evaluation
│   │   ├── models/           # SQLAlchemy models
│   │   ├── schemas/          # Pydantic schemas
│   │   ├── workers/          # Celery tasks
│   │   └── utils/            # Helpers, API clients
│   ├── tests/
│   ├── Dockerfile
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── components/       # React components
│   │   │   ├── TaskBuilder/  # UI for task definition
│   │   │   ├── ModelSelector/
│   │   │   ├── MetricConfig/
│   │   │   ├── Results/
│   │   │   └── History/
│   │   ├── api/              # API client
│   │   ├── hooks/            # Custom React hooks
│   │   └── types/            # TypeScript types
│   ├── Dockerfile
│   └── package.json
│
├── docker-compose.yml        # Full stack orchestration
├── RunTaskBench.sh          # One-command startup
├── CLAUDE.md                # This file
├── ARCHITECTURE.md          # Detailed technical design
└── docs/
    ├── TODO.md              # Post-MVP features and roadmap
    ├── HANDOFF.md           # Implementation guide
    └── examples/            # Example task definitions
```

### Data Models

**Core Entities**:
```python
# Task definition
class Task(BaseModel):
    id: UUID
    name: str
    description: str
    domain: str  # healthcare, education, legal, etc.
    input_format: str
    output_format: str
    gold_data_path: str
    quality_checks: list[QualityCheck]  # LLM-generated
    created_at: datetime

# Quality Check (LLM-generated)
class QualityCheck(BaseModel):
    name: str
    description: str
    validation_function: str  # Python code or rule
    severity: str  # critical, warning, info

# Model Configuration (User-provided)
class ModelConfig(BaseModel):
    id: str  # e.g., "gpt-4o" or "custom-model-1"
    endpoint: str  # API endpoint URL
    api_key: str  # User-provided API key
    provider: str  # "openrouter", "anthropic", "openai", "custom"
    
    # Example configurations:
    # OpenRouter:
    #   id: "anthropic/claude-sonnet-4.5"
    #   endpoint: "https://openrouter.ai/api/v1"
    #   api_key: "$OPENROUTER_KEY"
    #   provider: "openrouter"
    #
    # Direct API:
    #   id: "gpt-4o"
    #   endpoint: "https://api.openai.com/v1"
    #   api_key: "sk-..."
    #   provider: "openai"

# Evaluation run
class EvaluationRun(BaseModel):
    id: UUID
    task_id: UUID
    models: list[ModelConfig]  # User-configured models
    metrics: list[str]
    status: str  # pending, running, completed, failed
    estimated_cost: float
    actual_cost: float
    results: list[ModelResult]
    created_at: datetime
    completed_at: Optional[datetime]

# Model result
class ModelResult(BaseModel):
    model_id: str
    accuracy: float
    hallucination_rate: float
    completeness: float
    cost: float
    consistency_std: float  # Standard deviation across consistency runs
    instruction_following: float  # % of constraints followed
    quality_violations: list[str]  # Failed quality checks
    latency: Optional[float]  # Only if using direct APIs
```

---

## Design Principles

### Framework Design Principles

1. **Fair Comparison by Default**: Same prompt for all models (no per-model optimization in MVP)
2. **LLM-Powered Adaptation**: Framework uses LLM to analyze tasks and generate quality checks
3. **Cost Transparency**: Always show estimate before execution, track actual costs
4. **Fail Fast, Continue Smart**: Retry 3x, abandon model after 5 failures, continue with others
5. **User Choice Over Magic**: Recommend configurations, but let users override
6. **User Interpretation Over Prescription**: Framework provides objective data (cost, quality metrics, rankings), users decide value trade-offs based on their specific context. Like musical instruments ($10 guitar vs $10,000 guitar), the "best" model depends on user's priorities, budget, and use case - not framework's opinion.

### Code Quality Standards

1. **Type Safety**: All functions must have type hints (`mypy --strict`)
2. **Pydantic Validation**: Use Pydantic models for all data structures
3. **Error Handling**: Comprehensive try/except with user-friendly messages
4. **Testing**: 80%+ code coverage target
5. **Documentation**: Docstrings for all public functions

### API Design Principles

1. **Async-First**: Use async/await for all I/O operations
2. **Retry Logic**: Exponential backoff for API failures (3 retries max)
3. **Rate Limiting**: Respect provider rate limits
4. **Timeouts**: 60s default, configurable per model
5. **Streaming**: Support streaming for real-time progress (not for judge)

### UI Design Principles

1. **Progressive Disclosure**: Show essentials first, details on demand
2. **Real-time Feedback**: Live progress updates during execution
3. **Cost Awareness**: Always show cost estimates before actions
4. **Collapsible Sections**: Keep UI clean with expandable detail panels
5. **Responsive Design**: Mobile-friendly (even if desktop-primary)

---

## Constraints

### Technical Constraints

- **Python 3.11+**: Cannot support older versions (uses new typing features)
- **Docker Required**: Primary deployment method
- **OpenRouter Primary**: Direct APIs optional (user provides keys)
- **PostgreSQL**: No SQLite support (need JSONB for flexibility)
- **File Size Limits**: Max 100MB per upload (gold data, KB files)

### Evaluation Constraints

- **Concurrent Models**: 5 max in parallel (rate limiting)
- **Output Size**: Max 100KB per model output
- **Timeout**: 60s per model invocation (configurable)
- **Retry Limit**: 3 attempts per call
- **Abandon Threshold**: 5 consecutive failures = skip model

### Cost Constraints

- **Budget Warnings**: Alert if evaluation exceeds $10 estimated cost
- **Budget Circuit Breaker**: Pause if actual cost exceeds 150% of estimate
- **Token Tracking**: Track input + output tokens separately per phase

---

## Key Design Decisions

### Why LLM-Powered Quality Check Generation?

**Traditional Approach**: User manually defines quality checks
**Our Approach**: LLM analyzes task description and auto-generates checks

**Benefits**:
- Works for ANY domain without hardcoding rules
- Captures nuanced requirements (e.g., "no medical abbreviations in patient-facing content")
- Reduces setup time from hours to minutes
- More comprehensive than human-defined checks

**Implementation**:
```python
async def generate_quality_checks(task_description: str) -> list[QualityCheck]:
    """Use LLM to analyze task and generate validation rules."""
    prompt = f"""
    Task: {task_description}
    
    Generate 5-8 specific quality checks for this task's outputs.
    For each check, provide:
    1. Name (concise identifier)
    2. Description (what it validates)
    3. Validation rule (how to check it)
    4. Severity (critical/warning/info)
    
    Focus on:
    - Output format constraints
    - Domain-specific requirements
    - Common failure modes
    - Completeness checks
    """
    
    checks = await llm_analyze(prompt)
    return [QualityCheck(**check) for check in checks]
```

### Why OpenRouter + Direct APIs?

**OpenRouter as Primary**:
- Unified API for 100+ models
- Built-in cost tracking
- Rate limiting handled centrally

**Direct APIs as Option**:
- Lower latency (no routing overhead)
- More control over parameters
- Support for latest model versions

**Configuration**:
```yaml
models:
  - id: "gpt-4o"
    provider: "openrouter"
    endpoint: "https://openrouter.ai/api/v1"
    
  - id: "claude-sonnet-4.5"
    provider: "anthropic"
    endpoint: "https://api.anthropic.com/v1"
    api_key: "user-provided"  # Direct API
```

### Why 6 Core Metrics?

**Selected Metrics** (always evaluated):
1. **Accuracy**: Correctness against gold data
2. **Hallucination Rate**: Fabricated information percentage
3. **Completeness**: Coverage of required content (recall)
4. **Cost**: Dollars per task
5. **Instruction Following**: Adherence to format/constraints
6. **Consistency**: Reproducibility across runs (σ of scores)

**Optional Metrics** (user-enabled based on LLM recommendation):
7. **Latency**: Response time (only with direct APIs)
8. **Safety/Toxicity**: Harmful output detection
9. **Bias & Fairness**: Demographic disparities
10. **Factuality**: Verifiable claims against knowledge base
11. **Robustness**: Performance on corrupted inputs
12. **Contextual Relevance**: For RAG/multi-turn tasks

**Why These 6?**:
- Universal across task types
- Measurable without domain knowledge
- Balance quality (first 3) + operations (last 3)

### Why Docker + FastAPI + React?

**Docker**: One-command deployment, consistent environment
**FastAPI**: Auto-docs, async support, Pydantic integration
**React**: Component reusability, rich ecosystem, TypeScript support
**PostgreSQL**: JSONB for flexible schema, production-ready
**Redis + Celery**: Async task execution, progress tracking

---

## Development Guidelines

### Adding New Features

**Decision Framework**:
1. Is it aligned with task-specific evaluation? (core mission)
2. Does it add significant complexity? (prefer simple)
3. Is it testable without external APIs? (mock when possible)
4. Does it affect user costs? (document clearly)
5. Should it be MVP or FUTURE-TODO? (be ruthless on scope)

**Process**:
1. Add to appropriate TODO section (MVP or FUTURE-TODO.md)
2. Write tests first (TDD when possible)
3. Implement with type hints
4. Update ARCHITECTURE.md if design changes
5. Add integration tests
6. Update CLAUDE.md if context changes

### Code Style

**Backend (Python)**:
```python
# Good: Type hints, Pydantic validation, async, docstrings
async def evaluate_model(
    task: Task,
    model: str,
    input_text: str,
) -> ModelResult:
    """Evaluate a single model on a task.

    Args:
        task: Pydantic Task model with quality checks
        model: Model ID (e.g., "anthropic/claude-sonnet-4.5")
        input_text: Input text to evaluate

    Returns:
        ModelResult with scores, tokens, cost, violations

    Raises:
        APIError: If API call fails after 3 retries
        ValidationError: If output fails quality checks
    """
    try:
        async with APIClient() as client:
            output = await client.complete(model, input_text, timeout=60)
            violations = await check_quality(output, task.quality_checks)
            return ModelResult(
                model_id=model,
                output=output,
                quality_violations=violations,
                cost=calculate_cost(output)
            )
    except APIError as e:
        logger.error(f"API error for {model}: {e}")
        raise
```

**Frontend (TypeScript + React)**:
```typescript
// Good: TypeScript interfaces, async/await, error handling
interface ModelResult {
  modelId: string;
  accuracy: number;
  hallucination: number;
  cost: number;
}

async function fetchResults(taskId: string): Promise<ModelResult[]> {
  try {
    const response = await fetch(`/api/results/${taskId}`);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Failed to fetch results:', error);
    throw error;
  }
}
```

### Testing Guidelines

**Backend Tests**:
- Unit tests: Pure functions, Pydantic models
- Integration tests: API endpoints with TestClient
- Mocked external APIs: Don't hit real OpenRouter/Anthropic
- Fixtures: Sample tasks, gold data in `tests/fixtures/`

**Frontend Tests**:
- Component tests: React Testing Library
- API client tests: Mock fetch responses
- Integration tests: Full user flows with mock backend

**Coverage Target**: 80%+ overall, 90%+ for critical paths (executor, quality checks)

### Git Workflow

**Branch Naming**:
- Feature: `feature/quality-check-generation`
- Bugfix: `bugfix/cost-tracking-error`
- Claude sessions: `claude/session-<date>-<feature>`

**Commit Messages**:
```
Add LLM-powered quality check generation

- Implement generate_quality_checks() using Claude API
- Add QualityCheck Pydantic model
- Create tests with mocked LLM responses
- Update CLAUDE.md with design decision
```

**PR Process**:
1. Run tests: `pytest --cov`
2. Format code: `black . && isort .`
3. Type check: `mypy src/`
4. Update docs if needed
5. Create PR with clear description

---

## Common Patterns

### Error Handling Pattern

```python
# Pattern: Retry with exponential backoff, then abandon
async def call_with_retry(
    func: Callable,
    max_retries: int = 3,
    model_id: str = "unknown"
) -> Any:
    """Call function with retry logic."""
    for attempt in range(max_retries):
        try:
            return await func()
        except APIError as e:
            if attempt == max_retries - 1:
                logger.error(f"{model_id} failed after {max_retries} attempts")
                raise TaskBenchError(
                    f"Model {model_id} failed. Check API key and rate limits."
                )
            wait = 2 ** attempt  # Exponential backoff
            logger.warning(f"Retry {attempt+1}/{max_retries} after {wait}s")
            await asyncio.sleep(wait)
```

### Quality Check Pattern

```python
# Pattern: Apply LLM-generated checks to model outputs
async def check_quality(
    output: str,
    quality_checks: list[QualityCheck]
) -> list[str]:
    """Apply quality checks to model output.
    
    Returns list of violation messages (empty if all pass).
    """
    violations = []
    for check in quality_checks:
        if check.validation_function.startswith("lambda"):
            # Simple lambda check
            validator = eval(check.validation_function)
            if not validator(output):
                violations.append(f"{check.name}: {check.description}")
        else:
            # Complex check - use helper function
            if not await run_validation(check, output):
                violations.append(f"{check.name}: {check.description}")
    return violations
```

### Cost Tracking Pattern

```python
# Pattern: Track costs at multiple levels
class CostTracker:
    """Track costs across evaluation phases."""
    
    def __init__(self):
        self.costs = defaultdict(float)
    
    def add_cost(self, phase: str, model: str, tokens: int, price: float):
        """Add cost entry."""
        cost = (tokens / 1_000_000) * price
        self.costs[f"{phase}:{model}"] += cost
        self.costs[f"phase:{phase}"] += cost
        self.costs[f"model:{model}"] += cost
        self.costs["total"] += cost
    
    def get_breakdown(self) -> dict:
        """Get cost breakdown by model, phase, total."""
        return {
            "by_model": {k: v for k, v in self.costs.items() if k.startswith("model:")},
            "by_phase": {k: v for k, v in self.costs.items() if k.startswith("phase:")},
            "total": self.costs["total"]
        }
```

---

## Critical Implementation Details

### Quality Check Generation

**When**: Immediately after user defines task (before model selection)
**How**: Send task description to Claude Sonnet 4.5
**Validation**: User reviews and can edit generated checks
**Storage**: Saved in taskbench.yaml as part of task definition

**Example Flow**:
```
User enters: "Extract medical diagnoses from patient notes"
↓
LLM analyzes task:
  - Domain: Healthcare (high risk)
  - Output: Structured list
  - Constraints: Medical accuracy critical
↓
Generates checks:
  1. No PHI leakage (CRITICAL)
  2. ICD-10 codes valid (WARNING)
  3. Diagnoses match patient history (CRITICAL)
  4. No speculative diagnoses (WARNING)
↓
User reviews → Approves → Saved to task config
```

### Metric Selection Logic

**Default Enabled** (always):
- Accuracy, Hallucination, Completeness, Cost, Instruction Following, Consistency

**LLM-Recommended Optional Metrics**:

#### Latency/Speed
**When to Enable**: Only when using direct API keys (Anthropic, OpenAI, etc.)
**Why**: OpenRouter adds routing overhead making cross-model latency comparison meaningless
**Implementation**:
```python
latency_config = {
    'enabled': False,  # Default OFF
    'warning': 'Latency comparison only accurate with direct API keys',
    'show_if': user_has_direct_api_keys(),
    'recommendation': 'Disable when using OpenRouter'
}
```

#### Consistency/Reliability
**When to Enable**: LLM recommends minimum N runs based on task complexity
**Factors Considered**:
- Task ambiguity (vague requirements → more runs needed)
- Output format (structured JSON → fewer runs, creative text → more runs)
- Risk level (medical diagnosis → 50+ runs, blog summary → 5 runs)

**Implementation**:
```python
async def recommend_consistency_runs(task_description: str, user_budget: float) -> dict:
    """LLM analyzes task and suggests minimum runs."""
    prompt = f"""
    Task: {task_description}
    
    Recommend minimum consistency runs (range: 5-100) based on:
    - Task ambiguity and complexity
    - Output variability expected
    - Risk/criticality level
    
    Return: min_runs, reasoning, estimated_cost_per_model
    """
    
    recommendation = await llm_analyze(prompt)
    
    return {
        'min_runs': recommendation.min_runs,  # e.g., 10
        'reasoning': 'Moderate task ambiguity, structured output reduces variability',
        'cost_estimate': f"${recommendation.cost_per_model * num_models}",
        'user_override': True  # User can adjust N
    }
```

#### Safety/Toxicity
**When to Enable**: LLM analyzes task for user-facing content or harm potential
**Auto-Recommended For**: Healthcare, customer service, public-facing content
**User Override**: Always allowed

#### Bias & Fairness
**When to Enable**: Task involves demographics or sensitive attributes
**Auto-Recommended For**: Hiring, lending, content moderation
**Skip For**: Internal processing, technical tasks

#### Factuality
**When to Enable**: Task requires verifiable claims
**Methods** (user selects one):

1. **Source Verification** (Recommended)
   - User provides knowledge base path
   - Verify claims against provided documents
   - Cost: Low, Accuracy: High
   ```python
   factuality_config = {
       'method': 'source_verification',
       'kb_path': 'C:/Users/Sri/KB',  # User-provided
       'kb_format': ['pdf', 'txt', 'md', 'docx'],
       'verification': 'claim_extraction_and_match'
   }
   ```

2. **Web Search Verification**
   - Extract claims from output
   - Verify via web search
   - Cost: Medium, Accuracy: Medium
   - Requires: Internet access

3. **LLM Judge**
   - Use stronger model (e.g., GPT-4) to fact-check
   - Cost: High, Accuracy: Medium
   - Requires: Additional API key

4. **Domain-Specific**
   - Custom validation function
   - Example: For concept extraction, verify concepts exist in source transcript
   - Cost: Low, Accuracy: High (when available)

**Implementation**:
```python
async def verify_factuality(output: str, method: str, config: dict) -> float:
    """Verify factual accuracy using selected method."""
    if method == 'source_verification':
        kb_docs = load_knowledge_base(config['kb_path'])
        claims = extract_claims(output)
        verified = sum(1 for claim in claims if claim_in_kb(claim, kb_docs))
        return verified / len(claims)
    
    elif method == 'web_search':
        claims = extract_claims(output)
        verified = sum(1 for claim in claims if verify_via_web(claim))
        return verified / len(claims)
    
    elif method == 'llm_judge':
        prompt = f"Verify factual accuracy of: {output}"
        score = await stronger_llm.complete(prompt)
        return score
    
    # ... other methods
```

#### Robustness
**When to Enable**: User explicitly opts in (expensive)
**Warning**: High cost due to testing multiple input corruptions
**Cost Example**: 3 corruption types × 3 levels × 5 runs × 4 models = 180 API calls

#### Contextual Relevance
**When to Enable**: RAG systems or multi-turn conversations
**Skip For**: Single-turn tasks with full context

**Complete Metric Recommendation Function**:
```python
async def recommend_metrics(task_description: str) -> dict:
    """LLM analyzes task and recommends optional metrics."""
    prompt = f"""
    Task: {task_description}
    
    Should these optional metrics be enabled?
    - Safety/Toxicity (user-facing content, potential harm)
    - Bias & Fairness (demographics, sensitive attributes)
    - Factuality (verifiable claims required)
    - Latency (real-time requirements) - NOTE: Only if direct APIs
    - Robustness (noisy/imperfect inputs expected)
    - Contextual Relevance (RAG or multi-turn conversations)
    
    For each, return: Yes/No + Reasoning + Severity (High/Medium/Low)
    """
    
    analysis = await llm_analyze(prompt)
    return {
        'safety': {'enabled': analysis.safety_yes, 'reasoning': '...', 'severity': '...'},
        'bias': {'enabled': analysis.bias_yes, 'reasoning': '...', 'severity': '...'},
        'factuality': {'enabled': analysis.factuality_yes, 'reasoning': '...', 'method': 'source_verification'},
        'latency': {'enabled': False, 'reasoning': 'Only enable with direct API keys'},
        'robustness': {'enabled': False, 'reasoning': 'High cost - user opt-in only'},
        'consistency': {'enabled': True, 'runs': 10, 'reasoning': '...', 'cost': '$3.20'}
    }
```

### Cost Estimation

**Formula**:
```python
def estimate_cost(
    task: Task,
    models: list[str],
    consistency_runs: int,
    optional_metrics: dict
) -> float:
    """Estimate total cost before execution."""
    base_cost = sum(
        estimate_model_cost(model, task.avg_input_tokens)
        for model in models
    )
    
    consistency_cost = base_cost * (consistency_runs - 1)
    
    safety_cost = base_cost * 0.2 if optional_metrics.get('safety') else 0
    factuality_cost = base_cost * 0.15 if optional_metrics.get('factuality') else 0
    
    buffer = 0.2  # 20% buffer for overhead
    
    return (base_cost + consistency_cost + safety_cost + factuality_cost) * (1 + buffer)
```

---

## Quick Reference

### Running the Stack

```bash
# Start everything
./RunTaskBench.sh

# Or manually
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Stop
docker-compose down
```

### Running Tests

```bash
# Backend tests
cd backend
pytest --cov
pytest tests/test_quality_gen.py -v

# Frontend tests
cd frontend
npm test
npm run test:coverage
```

### Code Quality

```bash
# Backend
black backend/ && isort backend/
mypy backend/src/

# Frontend
npm run lint
npm run type-check
```

### Database Migrations

```bash
# Create migration
docker-compose exec backend alembic revision --autogenerate -m "Add quality_checks column"

# Run migrations
docker-compose exec backend alembic upgrade head
```

---

## Questions to Ask Before Making Changes

1. **Does this align with task-specific evaluation?**
2. **Will this increase complexity significantly?** (Prefer simple)
3. **Is this testable without real API calls?** (Mock externals)
4. **Does this affect user costs?** (Document clearly)
5. **Should this be MVP or docs/TODO.md?** (Be scope-conscious)
6. **Is there a simpler solution?** (YAGNI principle)
7. **Does this require LLM analysis?** (Use sparingly, costs add up)

---

## When in Doubt

- **Check existing patterns**: Look at similar code in codebase
- **Read ARCHITECTURE.md**: Detailed technical design
- **Check docs/TODO.md**: Maybe it's already documented as post-MVP
- **Run tests**: `pytest --cov` catches most issues
- **Ask the user**: When requirements are ambiguous

---

**Last Updated**: 2025-11-18
**Version**: 0.1.0-alpha (MVP in progress)
**Maintained By**: LLM TaskBench Contributors
**For Questions**: See GitHub Issues