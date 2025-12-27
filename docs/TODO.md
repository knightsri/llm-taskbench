# LLM TaskBench - TODO (Post-MVP Features)

**Last Updated**: 2025-12-27
**Version**: 0.1.0
**Status**: MVP complete - folder-based use cases working

This document tracks all features, enhancements, and improvements planned for **after MVP completion**.

---

## ✅ Recently Completed (2025-12-27)

- [x] **Folder-based Use Case Architecture** - USE-CASE.md format with data/ground-truth folders
- [x] **LLM-powered Model Selection** - Two-phase ModelSelector with task analysis + ranking
- [x] **Programmatic Validation in Judge** - JSON parsing, timestamp range, duration constraint checks
- [x] **Cost Accuracy via OpenRouter** - Inline usage tracking with billed_cost_usd from API
- [x] **Model Catalog Caching** - 24-hour TTL cache in `.cache/openrouter_models.json`
- [x] **Dynamic Orchestrator** - `select_models_dynamically()` integrates with ModelSelector
- [x] **Docker Compose Polish** - Removed obsolete version, added sample-usecases volumes, usage docs
- [x] **Anti-hallucination Guards** - Executor prompts include strict rules about timestamp ranges

---

## 🎯 Post-MVP Roadmap

### Phase 2: Enhanced Core Features (Weeks 7-12)

#### 2.1 Prompt Engineering
- [ ] **Per-Model Prompt Optimization**
  - Support different prompt styles per model (GPT-4 vs Claude vs Llama)
  - A/B test prompt variants
  - Track performance improvement from optimization
  - Clear labeling: "Fair Mode" vs "Optimized Mode" results
  - **Why Deferred**: MVP focuses on fair comparison (same prompt)

#### 2.2 Advanced Parallel Execution
- [ ] **Smart Concurrency Management**
  - Dynamic rate limit detection per provider
  - Adaptive concurrency based on API response times
  - Priority queue for expensive models
  - Cost vs speed optimization toggle
  - **Why Deferred**: MVP uses simple parallel execution (max 5 concurrent)

#### 2.3 Robustness Testing
- [ ] **Input Corruption Testing**
  - Typo injection (5%, 10%, 15% corruption levels)
  - Missing words/tokens
  - Format corruption (JSON → malformed JSON)
  - User-configurable corruption types
  - Cost warning: "Will add 100+ API calls"
  - **Why Deferred**: High cost, only relevant for noisy input scenarios

#### 2.4 Real-Time Progress
- [ ] **WebSocket Progress Updates**
  - Live model-by-model status
  - Real-time cost accumulation
  - Streaming output preview
  - Cancel mid-execution
  - **Why Deferred**: MVP uses polling (simpler, works)

---

### Phase 3: Advanced Analytics (Months 3-4)

#### 3.1 Deep Failure Analysis
- [ ] **Pattern Detection**
  - Cluster similar failures (e.g., "all models struggle with medical abbreviations")
  - Identify systematic errors (e.g., "GPT-4 over-segments concepts")
  - Suggest prompt improvements based on failure patterns
  - Generate failure reports with examples
  - **Why Deferred**: Requires ML clustering, not critical for MVP

#### 3.2 Statistical Deep Dive
- [ ] **Advanced Statistics**
  - Multiple comparison correction (Bonferroni, FDR)
  - Effect size calculations (Cohen's d)
  - Power analysis (sample size recommendations)
  - Regression analysis (cost vs accuracy curves)
  - **Why Deferred**: MVP uses simple bootstrap confidence intervals

#### 3.3 Cost Optimization Engine
- [ ] **Automated Cost Reduction**
  - Suggest cheaper models for acceptable quality loss
  - Identify prompt compression opportunities
  - Recommend batch processing strategies
  - Caching strategy suggestions
  - ROI calculator (spend $X more for Y% accuracy gain)
  - **Why Deferred**: Requires multiple runs to establish baselines

---

### Phase 4: Reproducibility & Versioning (Months 5-6)

#### 4.1 Reproducibility System
- [ ] **Full Reproducibility Tracking**
  - Framework version (Git commit hash)
  - Model versions/checkpoints (e.g., "gpt-4o-2024-11-20")
  - Random seed management
  - Environment snapshot (Python version, dependencies)
  - Reproducibility score calculation
  - **Why Deferred**: MVP focuses on single-run benchmarks

#### 4.2 Experiment Tracking
- [ ] **Version Control for Tasks**
  - Track task definition changes over time
  - Compare results across task versions
  - Rollback to previous configurations
  - Diff viewer for YAML changes
  - **Why Deferred**: Not critical for initial users

#### 4.3 Iteration Workflow
- [ ] **A/B Testing Framework**
  - Test prompt variations side-by-side
  - Track improvement over iterations (v1 → v2 → v3)
  - Suggested experiments ("Try adding few-shot examples")
  - Convergence detection ("Improvements plateaued")
  - **Why Deferred**: Users need baseline results first

---

### Phase 5: Comparative Benchmarking (Months 6-9)

#### 5.1 Cross-Task Comparison
- [ ] **Task-to-Task Analysis**
  - Compare performance across different task types
  - Identify model strengths/weaknesses by domain
  - Meta-analysis: "GPT-4 excels at structured extraction, Claude at reasoning"
  - **Why Deferred**: Requires multiple tasks run by community

#### 5.2 Historical Tracking
- [ ] **Regression Detection**
  - Track model performance over time
  - Alert when performance degrades
  - Compare against previous runs
  - Trend visualization
  - **Why Deferred**: Requires long-term data collection

#### 5.3 Community Benchmarks
- [ ] **Public Benchmark Database**
  - Submit results to public leaderboard
  - Compare against community averages
  - Download reference datasets
  - Participate in shared challenges
  - **Why Deferred**: Needs established user base first

---

### Phase 6: Advanced Features (Months 9-12)

#### 6.1 Batch Evaluation
- [ ] **Multi-Input Processing**
  - Process 100+ inputs in one evaluation
  - Aggregate statistics across inputs
  - Identify input-specific failure modes
  - Stratified sampling for large datasets
  - **Why Deferred**: MVP focuses on single-input tasks

#### 6.2 Custom Judge Models
- [ ] **Bring Your Own Judge**
  - Use models other than Claude Sonnet 4.5 as judge
  - Compare judge agreement (inter-rater reliability)
  - Ensemble judging (multiple judges vote)
  - **Why Deferred**: Claude Sonnet 4.5 sufficient for MVP

#### 6.3 Multi-Modal Support
- [ ] **Image/Audio/Video Inputs**
  - Evaluate vision models (GPT-4V, Claude 3)
  - Audio transcription quality
  - Video understanding tasks
  - **Why Deferred**: MVP focuses on text-only tasks

#### 6.4 Fine-Tuning Guidance
- [ ] **Fine-Tune Decision Support**
  - Analyze when fine-tuning would help
  - Estimate fine-tuning ROI
  - Generate training data suggestions
  - **Why Deferred**: Advanced use case

---

## 🎯 Near-Term Improvements

Items identified during development that would enhance the framework.

### Chunking & Large Input Handling

- [ ] **Auto-chunking for Large Inputs**
  - **Problem**: Chunked mode is CLI opt-in (`--chunked` flag); long inputs run single-shot unless explicitly configured
  - **Solution**: Detect input length vs model context window and auto-enable chunking when input exceeds 80% of available context
  - **Implementation**: Add `auto_chunk` parameter to executor; estimate tokens via chars/4 heuristic; warn user when auto-chunking
  - **Why Deferred**: Current chunking works well with explicit flags; auto-detection needs careful threshold tuning

- [ ] **Token-aware Chunk Sizing**
  - **Problem**: Dynamic chunking uses char-based estimation (4 chars/token) which varies significantly by content type and language
  - **Solution**: Integrate provider-specific tokenizers (tiktoken for OpenAI, Claude tokenizer for Anthropic) for precise sizing
  - **Implementation**: Add optional tokenizer dependency; use tokenizer when available, fall back to char heuristic
  - **Why Deferred**: Char-based works well enough for English text; tokenizer adds dependency complexity

### UI Completeness

- [ ] **Chunking Controls in UI**
  - **Problem**: UI does not expose chunked/dynamic chunk controls that CLI has
  - **Solution**: Add chunking toggle and settings to API endpoints and React UI
  - **Implementation**: Extend `/api/evaluate` endpoint with `chunk_mode`, `chunk_chars` params; add UI controls

- [ ] **Run History with Judge Reruns**
  - **Problem**: No way to view past runs or re-run judge on saved results
  - **Solution**: Add run history view with ability to re-judge results with different models/prompts
  - **Implementation**: Store runs in `/results` with metadata; add history API endpoint; add history component

- [ ] **Cost Rollups and Totals**
  - **Problem**: UI doesn't show cumulative costs across runs
  - **Solution**: Display total cost per session, per use-case, and global
  - **Implementation**: Use existing CostTracker.get_global_costs(); add cost dashboard component

### Orchestrator Enhancements

- [ ] **Use-Case Traits → Model Scoring → Selection Rationale**
  - **Problem**: While ModelSelector does LLM-based analysis, the selection rationale could be richer
  - **Solution**: Emit detailed trace of why each model was selected/rejected; include use-case trait matching
  - **Implementation**: Extend ModelSelector to return `selection_trace` with per-model reasoning
  - **Why Deferred**: Current implementation provides "why" field per model; full trace adds complexity

---

## 🔧 Technical Debt & Improvements

### Code Quality
- [ ] **Improve Test Coverage** (target: 90%+, current target: 80%)
- [ ] **Add Integration Tests** for full end-to-end flows
- [ ] **Performance Profiling** and optimization
- [ ] **Security Audit** (input validation, API key handling)

### Documentation
- [ ] **Video Tutorials** for common workflows
- [ ] **API Client Libraries** (Python, TypeScript, Go)
- [ ] **Example Tasks Repository** (10+ pre-built tasks)
- [ ] **Blog Post Series** on benchmarking best practices

### Infrastructure
- [ ] **Horizontal Scaling** (multiple worker nodes)
- [ ] **CDN Integration** for frontend assets
- [ ] **Monitoring & Alerting** (Prometheus, Grafana)
- [ ] **Backup & Disaster Recovery** strategies

---

## 📊 Community Requests

Track feature requests from users here:

### High Priority (Multiple requests)
- [ ] Export results to PDF with charts
- [ ] Slack/Discord notifications when evaluation completes
- [ ] API-only mode (no UI needed)

### Medium Priority
- [ ] Google Sheets integration for results
- [ ] Zapier integration
- [ ] Template marketplace (share task definitions)

### Low Priority
- [ ] Mobile app
- [ ] Browser extension
- [ ] VS Code extension

---

## ⚠️ Known Limitations (Not Planned to Fix)

These are intentional design decisions, not bugs:

1. **No Streaming for Judge**: Judge needs complete output, can't stream
2. **YAML Configuration Only**: No GUI config builder (keep it simple)
3. **Single Language Only**: English-only UI for MVP
4. **No Model Training**: Evaluation only, not a training platform
5. **Text-Only MVP**: Multi-modal deferred to Phase 6

---

## 🎯 Decision Log

### Why These Features Are Post-MVP

**Prompt Optimization**: 
- MVP needs baseline with fair comparison first
- Per-model optimization adds complexity users don't need initially

**Advanced Statistics**: 
- Simple confidence intervals sufficient for most users
- Power users can export data and analyze elsewhere

**Reproducibility Tracking**: 
- Important for research, less critical for practitioners
- Can be added incrementally without breaking changes

**Cross-Task Comparison**: 
- Requires community adoption and data collection
- Can't implement until we have diverse task library

**Batch Evaluation**: 
- Single-input validation proves framework works
- Batch is optimization, not new capability

---

## 📝 How to Add Items to This TODO

When you identify a new post-MVP feature:

1. **Add to appropriate Phase section**
2. **Include "Why Deferred" rationale**
3. **Estimate effort** (S/M/L/XL)
4. **Link to GitHub issue** if exists
5. **Update Last Updated date** at top

---

## 🔄 Migration Path

When features move from TODO → MVP or MVP → TODO:

**Document the change**:
```
MOVED: Feature X from TODO Phase 2 → MVP
Reason: User demand exceeded expectations, feasibility confirmed
Date: YYYY-MM-DD
```

**Update References**:
- CLAUDE.md (if architecture changes)
- HANDOFF.md (if implementation guide needed)
- INTEGRATION-TASK-LIST.md (add tasks)

---

**Questions about TODO items?** See GitHub Discussions or create an issue with label `feature-request`.