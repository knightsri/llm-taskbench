"""
LLM TaskBench - Streamlit UI

Complete UI workflow for:
1. Managing use cases
2. Selecting models and running evaluations
3. Viewing and comparing results
4. Cost tracking
"""

import json
import os
from pathlib import Path
from datetime import datetime

import httpx
import yaml
import streamlit as st

st.set_page_config(
    page_title="LLM TaskBench",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = st.sidebar.text_input("API URL", value="http://taskbench-api:8000")

# Sidebar - Global Status
st.sidebar.markdown("---")
st.sidebar.subheader("Quick Stats")


@st.cache_data(ttl=60)
def fetch_tasks(api_url: str):
    try:
        resp = httpx.get(f"{api_url}/tasks", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return []


@st.cache_data(ttl=60)
def fetch_usecases(api_url: str):
    try:
        resp = httpx.get(f"{api_url}/usecases", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return []


@st.cache_data(ttl=60)
def fetch_models(api_url: str):
    try:
        resp = httpx.get(f"{api_url}/models", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return []


def start_run(api_url: str, payload: dict):
    resp = httpx.post(f"{api_url}/runs", json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()


def get_run(api_url: str, run_id: str):
    resp = httpx.get(f"{api_url}/runs/{run_id}", timeout=10)
    resp.raise_for_status()
    return resp.json()


def list_runs(api_url: str, usecase: str = None):
    params = {}
    if usecase:
        params["usecase"] = usecase
    resp = httpx.get(f"{api_url}/runs", params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def judge_run(api_url: str, run_id: str):
    resp = httpx.post(f"{api_url}/runs/{run_id}/judge", timeout=300)
    resp.raise_for_status()
    return resp.json()


def usecase_summary(api_url: str, usecase: str):
    resp = httpx.get(f"{api_url}/usecases/{usecase}/summary", timeout=10)
    resp.raise_for_status()
    return resp.json()


def create_usecase(api_url: str, name: str, body: dict):
    resp = httpx.post(f"{api_url}/usecases", params={"name": name}, json=body, timeout=10)
    resp.raise_for_status()
    return resp.json()


def load_local_results():
    """Load results from local results directory."""
    results_dir = Path("results")
    all_results = []

    if not results_dir.exists():
        return all_results

    for json_file in results_dir.rglob("*.json"):
        if json_file.name.startswith("."):
            continue
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            data["_file_path"] = str(json_file)
            data["_file_name"] = json_file.name
            all_results.append(data)
        except Exception:
            continue

    return all_results


def main():
    st.title("🔬 LLM TaskBench")
    st.caption("Task-specific LLM evaluation framework")

    tabs = st.tabs(["📊 Dashboard", "🚀 New Run", "📁 Results", "⚙️ Use Cases", "💰 Costs", "📖 Docs"])

    # Tab 1: Dashboard
    with tabs[0]:
        st.header("Dashboard")

        col1, col2, col3 = st.columns(3)

        # Load local results
        all_results = load_local_results()

        with col1:
            st.metric("Total Runs", len(all_results))

        with col2:
            total_cost = sum(
                r.get("statistics", {}).get("total_cost", 0)
                for r in all_results
            )
            st.metric("Total Cost", f"${total_cost:.2f}")

        with col3:
            judged_runs = sum(1 for r in all_results if r.get("scores"))
            st.metric("Judged Runs", judged_runs)

        st.markdown("---")

        # Recent runs
        st.subheader("Recent Results")
        if all_results:
            for result in all_results[-5:][::-1]:
                file_name = result.get("_file_name", "Unknown")
                task_name = result.get("task", {}).get("name", "Unknown Task")
                num_models = len(result.get("results", []))
                run_cost = result.get("statistics", {}).get("total_cost", 0)

                with st.expander(f"📄 {file_name} - {task_name}"):
                    st.write(f"**Models evaluated:** {num_models}")
                    st.write(f"**Cost:** ${run_cost:.4f}")

                    if result.get("scores"):
                        st.write("**Scores:**")
                        for score in result["scores"]:
                            if score:
                                st.write(f"  - {score.get('model_evaluated')}: {score.get('overall_score')}/100")
        else:
            st.info("No results yet. Run an evaluation to see results here.")

    # Tab 2: New Run
    with tabs[1]:
        st.header("New Evaluation Run")

        # Use case selection
        try:
            usecases = fetch_usecases(API_URL)
            tasks = fetch_tasks(API_URL)
        except Exception as e:
            st.error(f"Failed to connect to API: {e}")
            st.info("Make sure the API server is running at the configured URL.")
            usecases = []
            tasks = []

        col1, col2 = st.columns(2)

        with col1:
            selected_task = st.selectbox(
                "Task Definition",
                tasks if tasks else ["tasks/lecture_analysis.yaml"],
                help="Select the task definition file"
            )

        with col2:
            selected_usecase = st.selectbox(
                "Use Case",
                usecases if usecases else ["usecases/concepts_extraction.yaml"],
                help="Select the use case configuration"
            )

        # Model selection
        st.subheader("Model Selection")

        # Tier-based auto-selection
        auto_models = st.checkbox(
            "🤖 Auto-recommend models based on use case",
            value=False,
            help="Use AI to analyze the task and recommend models from different tiers"
        )

        selected_models = []

        if auto_models:
            st.info("Select which model tiers to include:")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                tier_quality = st.checkbox("💎 Quality", value=True, help="Premium models: Opus, o1, GPT-4-turbo")
            with col2:
                tier_value = st.checkbox("⚖️ Value", value=True, help="Mid-tier: Sonnet, GPT-4o, Gemini Pro")
            with col3:
                tier_budget = st.checkbox("💰 Budget", value=True, help="Low-cost and free models")
            with col4:
                tier_speed = st.checkbox("⚡ Speed", value=False, help="Fast response: Flash, mini models")

            # Build tier list
            tiers = []
            if tier_quality:
                tiers.append("quality")
            if tier_value:
                tiers.append("value")
            if tier_budget:
                tiers.append("budget")
            if tier_speed:
                tiers.append("speed")

            if st.button("🔍 Get Recommendations", disabled=not tiers):
                with st.spinner("Analyzing task and selecting models (~8 seconds)..."):
                    try:
                        import asyncio
                        from taskbench.evaluation.model_selector import select_models_for_task

                        # Get use case description for analysis
                        usecase_path = Path(selected_usecase) if selected_usecase else None
                        task_desc = "General LLM evaluation task"
                        if usecase_path and usecase_path.exists():
                            uc_data = yaml.safe_load(usecase_path.read_text())
                            task_desc = f"Task: {uc_data.get('name', '')}\nGoal: {uc_data.get('goal', '')}\nNotes: {uc_data.get('llm_notes', '')}"

                        result = asyncio.run(select_models_for_task(task_desc, tiers=tiers))
                        st.session_state["recommended_models"] = result

                        st.success(f"Found {len(result.get('models', []))} models across {len(tiers)} tiers (~$0.007 cost)")

                    except Exception as e:
                        st.error(f"Model selection failed: {e}")

            # Display recommendations if available
            if "recommended_models" in st.session_state:
                result = st.session_state["recommended_models"]
                models_by_tier = {}
                for m in result.get("models", []):
                    tier = m.get("best_for", "unknown")
                    if tier not in models_by_tier:
                        models_by_tier[tier] = []
                    models_by_tier[tier].append(m)

                st.markdown("**Recommended Models:**")
                for tier_name in ["quality", "value", "budget", "speed"]:
                    if tier_name in models_by_tier:
                        tier_models = models_by_tier[tier_name]
                        emoji = {"quality": "💎", "value": "⚖️", "budget": "💰", "speed": "⚡"}.get(tier_name, "")
                        st.markdown(f"**{emoji} {tier_name.upper()}:**")
                        for m in tier_models:
                            cost = m.get("input_cost_per_1m", 0) + m.get("output_cost_per_1m", 0)
                            st.markdown(f"- `{m['model_id']}` - ${cost:.2f}/1M tokens")

                # Let user select from recommendations
                all_recommended = [m["model_id"] for m in result.get("models", [])]
                selected_models = st.multiselect(
                    "Select from recommendations",
                    all_recommended,
                    default=result.get("suggested_test_order", all_recommended[:3]),
                    help="Choose which recommended models to evaluate"
                )

        else:
            # Manual model selection
            models_data = fetch_models(API_URL) if API_URL else []
            model_options = [m["id"] for m in models_data] if models_data else [
                "anthropic/claude-sonnet-4.5",
                "openai/gpt-4o",
                "google/gemini-2.5-flash",
                "google/gemini-2.0-flash-001"
            ]

            selected_models = st.multiselect(
                "Select Models",
                model_options,
                default=model_options[:3] if len(model_options) >= 3 else model_options,
                help="Choose which models to evaluate"
            )

        custom_models = st.text_input(
            "Additional model IDs (comma-separated)",
            help="Add models not in the dropdown"
        )

        # Input data
        st.subheader("Input Data")

        input_method = st.radio(
            "Input Method",
            ["Upload File", "Paste Text", "Use Sample"],
            horizontal=True
        )

        input_text = ""

        if input_method == "Upload File":
            uploaded_file = st.file_uploader(
                "Upload text file",
                type=["txt", "md", "csv", "json"]
            )
            if uploaded_file:
                input_text = uploaded_file.read().decode("utf-8")
                st.success(f"Loaded {len(input_text):,} characters")

        elif input_method == "Paste Text":
            input_text = st.text_area(
                "Input Text",
                height=300,
                placeholder="Paste your input text here..."
            )

        else:  # Use Sample
            sample_files = list(Path("tasks").glob("*.txt")) + list(Path("tests/fixtures").glob("*.txt"))
            if sample_files:
                selected_sample = st.selectbox(
                    "Select sample file",
                    [str(f) for f in sample_files]
                )
                if selected_sample and Path(selected_sample).exists():
                    input_text = Path(selected_sample).read_text(encoding="utf-8")
                    st.success(f"Loaded {len(input_text):,} characters from {selected_sample}")
            else:
                st.warning("No sample files found")

        # Options
        st.subheader("Options")

        col1, col2, col3 = st.columns(3)

        with col1:
            run_judge = st.checkbox("Run Judge Evaluation", value=True)

        with col2:
            chunked = st.checkbox("Enable Chunked Processing", value=True)

        with col3:
            dynamic_chunk = st.checkbox("Dynamic Chunk Sizing", value=True)

        # Run button
        st.markdown("---")

        if st.button("🚀 Start Evaluation", type="primary", disabled=not input_text or not selected_models):
            models = selected_models + [m.strip() for m in custom_models.split(",") if m.strip()]

            payload = {
                "task_path": selected_task,
                "usecase_path": selected_usecase,
                "models": models,
                "input_text": input_text,
                "judge": run_judge,
                "auto_models": auto_models,
            }

            try:
                with st.spinner("Starting evaluation..."):
                    run = start_run(API_URL, payload)

                st.success(f"Run started: {run['id']}")
                st.session_state["current_run"] = run["id"]
                st.session_state["current_usecase"] = Path(selected_usecase).stem

                # Poll for completion
                with st.spinner("Running evaluation (this may take a few minutes)..."):
                    import time
                    max_wait = 600  # 10 minutes
                    start_time = time.time()

                    while time.time() - start_time < max_wait:
                        run_data = get_run(API_URL, run["id"])
                        status = run_data.get("status", {}).get("status", "unknown")

                        if status == "completed":
                            st.success("Evaluation completed!")
                            if run_data.get("results"):
                                st.json(run_data["results"])
                            break
                        elif status == "failed":
                            st.error(f"Evaluation failed: {run_data.get('status', {}).get('error')}")
                            break

                        time.sleep(5)
                    else:
                        st.warning("Evaluation is still running. Check the Results tab later.")

            except Exception as e:
                st.error(f"Failed to start run: {e}")

    # Tab 3: Results
    with tabs[2]:
        st.header("Results Browser")

        all_results = load_local_results()

        if not all_results:
            st.info("No results found. Run an evaluation first.")
        else:
            # Filter options
            col1, col2 = st.columns(2)

            with col1:
                result_files = [r.get("_file_name", "Unknown") for r in all_results]
                selected_result = st.selectbox("Select Result File", result_files)

            # Find selected result
            result_data = None
            for r in all_results:
                if r.get("_file_name") == selected_result:
                    result_data = r
                    break

            if result_data:
                st.markdown("---")

                # Summary
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Models", len(result_data.get("results", [])))

                with col2:
                    cost = result_data.get("statistics", {}).get("total_cost", 0)
                    st.metric("Cost", f"${cost:.4f}")

                with col3:
                    tokens = result_data.get("statistics", {}).get("total_tokens", 0)
                    st.metric("Tokens", f"{tokens:,}")

                with col4:
                    has_scores = bool(result_data.get("scores"))
                    st.metric("Judged", "Yes" if has_scores else "No")

                # Scores table
                if result_data.get("scores"):
                    st.subheader("Scores")

                    scores_data = []
                    for i, score in enumerate(result_data["scores"]):
                        if score:
                            result = result_data["results"][i] if i < len(result_data["results"]) else {}
                            scores_data.append({
                                "Model": score.get("model_evaluated", "Unknown"),
                                "Overall": score.get("overall_score", 0),
                                "Accuracy": score.get("accuracy_score", 0),
                                "Format": score.get("format_score", 0),
                                "Compliance": score.get("compliance_score", 0),
                                "Cost": f"${result.get('cost_usd', 0):.4f}",
                                "Violations": len(score.get("violations", []))
                            })

                    if scores_data:
                        import pandas as pd
                        df = pd.DataFrame(scores_data)
                        df = df.sort_values("Overall", ascending=False)
                        st.dataframe(df, use_container_width=True)

                        # Best model highlight
                        best = df.iloc[0]
                        st.success(f"🏆 Best Model: **{best['Model']}** with score **{best['Overall']}/100**")

                # Raw results
                with st.expander("Raw Results JSON"):
                    st.json(result_data)

    # Tab 4: Use Cases
    with tabs[3]:
        st.header("Use Case Management")

        tab_list, tab_create = st.tabs(["📋 List", "➕ Create New"])

        with tab_list:
            usecases = fetch_usecases(API_URL)

            if usecases:
                for uc_path in usecases:
                    try:
                        uc_file = Path(uc_path)
                        if uc_file.exists():
                            uc_data = yaml.safe_load(uc_file.read_text(encoding="utf-8"))

                            with st.expander(f"📁 {uc_data.get('name', uc_file.stem)}"):
                                st.write(f"**Goal:** {uc_data.get('goal', 'No goal specified')}")

                                if uc_data.get("llm_notes"):
                                    st.write(f"**LLM Notes:** {uc_data['llm_notes'][:200]}...")

                                if uc_data.get("default_candidate_models"):
                                    st.write(f"**Default Models:** {', '.join(uc_data['default_candidate_models'])}")

                                st.write(f"**Path:** `{uc_path}`")

                                # Show YAML
                                if st.checkbox(f"Show YAML", key=f"yaml_{uc_path}"):
                                    st.code(yaml.dump(uc_data, sort_keys=False), language="yaml")

                    except Exception as e:
                        st.error(f"Error loading {uc_path}: {e}")
            else:
                st.info("No use cases found. Create one below.")

        with tab_create:
            st.subheader("Create New Use Case")

            new_name = st.text_input("Use Case Name", placeholder="my_use_case")

            new_goal = st.text_area(
                "Goal",
                placeholder="Describe what you want to achieve with this use case..."
            )

            new_llm_notes = st.text_area(
                "LLM Notes (hints for models)",
                placeholder="Add notes about priorities, constraints, etc..."
            )

            col1, col2 = st.columns(2)
            with col1:
                cost_priority = st.selectbox("Cost Priority", ["balanced", "high", "low"])
            with col2:
                quality_priority = st.selectbox("Quality Priority", ["balanced", "high", "low"])

            if st.button("Create Use Case", disabled=not new_name or not new_goal):
                body = {
                    "name": new_name,
                    "goal": new_goal,
                    "llm_notes": new_llm_notes,
                    "cost_priority": cost_priority,
                    "quality_priority": quality_priority,
                    "default_candidate_models": [
                        "anthropic/claude-sonnet-4.5",
                        "openai/gpt-4o",
                        "google/gemini-2.5-flash"
                    ]
                }

                try:
                    result = create_usecase(API_URL, new_name, body)
                    st.success(f"Created: {result.get('path')}")
                    st.cache_data.clear()
                except Exception as e:
                    st.error(f"Failed to create: {e}")

    # Tab 5: Costs
    with tabs[4]:
        st.header("Cost Tracking")

        all_results = load_local_results()

        # Summary metrics
        col1, col2, col3 = st.columns(3)

        total_cost = sum(
            r.get("statistics", {}).get("total_cost", 0)
            for r in all_results
        )
        total_tokens = sum(
            r.get("statistics", {}).get("total_tokens", 0)
            for r in all_results
        )

        with col1:
            st.metric("Total Cost", f"${total_cost:.4f}")

        with col2:
            st.metric("Total Tokens", f"{total_tokens:,}")

        with col3:
            st.metric("Total Runs", len(all_results))

        st.markdown("---")

        # Cost by model
        st.subheader("Cost by Model")

        model_costs = {}
        for result in all_results:
            for r in result.get("results", []):
                model = r.get("model_name", "Unknown")
                cost = r.get("cost_usd", 0)
                if model not in model_costs:
                    model_costs[model] = {"cost": 0, "runs": 0, "tokens": 0}
                model_costs[model]["cost"] += cost
                model_costs[model]["runs"] += 1
                model_costs[model]["tokens"] += r.get("total_tokens", 0)

        if model_costs:
            import pandas as pd
            df = pd.DataFrame([
                {
                    "Model": model,
                    "Total Cost": f"${data['cost']:.4f}",
                    "Runs": data["runs"],
                    "Tokens": f"{data['tokens']:,}",
                    "Avg Cost/Run": f"${data['cost'] / data['runs']:.4f}" if data["runs"] > 0 else "-"
                }
                for model, data in model_costs.items()
            ])
            df = df.sort_values("Total Cost", ascending=False)
            st.dataframe(df, use_container_width=True)

        # Cost trend (if we had timestamps)
        st.subheader("Recent Runs")
        for result in all_results[-10:][::-1]:
            file_name = result.get("_file_name", "Unknown")
            cost = result.get("statistics", {}).get("total_cost", 0)
            tokens = result.get("statistics", {}).get("total_tokens", 0)
            st.write(f"• **{file_name}**: ${cost:.4f} ({tokens:,} tokens)")

    # Tab 6: Docs
    with tabs[5]:
        st.header("Documentation")

        st.subheader("CLI Quick Reference")
        st.code("""
# List all commands
taskbench --help

# Run an evaluation
taskbench evaluate tasks/lecture_analysis.yaml \\
  --usecase usecases/concepts_extraction.yaml \\
  --models anthropic/claude-sonnet-4.5,openai/gpt-4o \\
  --input-file transcript.txt \\
  --output results/my_run.json \\
  --chunked

# Check status of use cases
taskbench status

# Check costs
taskbench costs --openrouter

# Interactive wizard
taskbench wizard

# View recommendations from results
taskbench recommend --results results/my_run.json

# List available models
taskbench models --list

# Create a new use case
taskbench usecases --create my_new_usecase
        """, language="bash")

        st.subheader("Docker Commands")
        st.code("""
# Build containers
docker compose -f docker-compose.cli.yml build
docker compose -f docker-compose.ui.yml build

# Run CLI command
docker compose -f docker-compose.cli.yml run --rm taskbench-cli evaluate ...

# Start UI
docker compose -f docker-compose.ui.yml up

# View logs
docker compose -f docker-compose.ui.yml logs -f
        """, language="bash")

        st.subheader("Use Case YAML Format")
        st.code("""
name: "my_use_case"
goal: "Describe the goal..."

llm_notes: |
  Notes for LLMs about priorities and constraints.

output_format:
  format_type: "json"
  fields:
    - name: "title"
      type: "string"
      description: "Field description"
  example: |
    {"title": "Example"}

cost_priority: "high"  # high/low/balanced
quality_priority: "high"

notes: |
  - Specific constraints
  - What to include/avoid

default_candidate_models:
  - anthropic/claude-sonnet-4.5
  - openai/gpt-4o
        """, language="yaml")


if __name__ == "__main__":
    main()
