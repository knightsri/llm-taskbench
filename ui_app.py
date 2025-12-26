import json
import time
from pathlib import Path

import httpx
import yaml
import streamlit as st


st.set_page_config(page_title="TaskBench UI", layout="wide")

API_URL = st.sidebar.text_input("API URL", value="http://localhost:8000")


@st.cache_data
def fetch_tasks(api_url: str):
    resp = httpx.get(f"{api_url}/tasks", timeout=10)
    resp.raise_for_status()
    return resp.json()


@st.cache_data
def fetch_usecases(api_url: str):
    resp = httpx.get(f"{api_url}/usecases", timeout=10)
    resp.raise_for_status()
    return resp.json()


@st.cache_data
def fetch_models(api_url: str):
    resp = httpx.get(f"{api_url}/models", timeout=10)
    resp.raise_for_status()
    return resp.json()


def start_run(api_url: str, payload: dict):
    resp = httpx.post(f"{api_url}/runs", json=payload, timeout=30)
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
    resp = httpx.post(f"{api_url}/runs/{run_id}/judge", timeout=30)
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


def main():
    st.title("LLM TaskBench")
    tabs = st.tabs(["New Run", "Runs", "Use-Cases", "Docs"])

    with tabs[0]:
        st.header("New Run")
        try:
            tasks = fetch_tasks(API_URL)
            usecases = fetch_usecases(API_URL)
        except Exception as e:  # noqa: BLE001
            st.error(f"Failed to fetch tasks/use-cases: {e}")
            tasks = []
            usecases = []

        selected_task = st.selectbox("Task", tasks)
        selected_usecase = st.selectbox("Use-case", usecases)
        models_data = fetch_models(API_URL) if API_URL else []
        model_options = [m["id"] for m in models_data]
        auto_models = st.checkbox("Auto-recommend models from use-case", value=True)
        selected_models = st.multiselect("Models", model_options, default=model_options[:2])
        custom_models = st.text_input("Additional model IDs (comma-separated)")
        judge = st.checkbox("Run judge", value=True)
        input_text = st.text_area("Input text", height=200)
        uploaded_file = st.file_uploader("...or upload text file", type=["txt"])

        if uploaded_file:
            input_text = uploaded_file.read().decode("utf-8")

        if st.button("Start run", disabled=not input_text or not selected_task):
            models = selected_models + [m.strip() for m in custom_models.split(",") if m.strip()]
            payload = {
                "task_path": selected_task,
                "usecase_path": selected_usecase,
                "models": models,
                "input_text": input_text,
                "judge": judge,
                "auto_models": auto_models,
            }
            try:
                run = start_run(API_URL, payload)
                st.success(f"Run started: {run['id']}")
                st.session_state["current_run"] = run["id"]
                st.session_state["current_usecase"] = Path(selected_usecase).stem
            except Exception as e:  # noqa: BLE001
                st.error(f"Failed to start run: {e}")

    with tabs[1]:
        st.header("Runs")
        usecase_name = st.text_input("Use-case (folder name)", value=st.session_state.get("current_usecase", ""))
        if usecase_name:
            try:
                runs = list_runs(API_URL, usecase=usecase_name)
                st.write(f"Runs for use-case '{usecase_name}':")
                run_ids = [r["id"] for r in runs]
                selected_run = st.selectbox("Run ID", run_ids, index=0 if run_ids else None)
                if selected_run and st.button("Fetch run"):
                    data = get_run(API_URL, selected_run)
                    st.write("Status:", data.get("status"))
                    results = data.get("results")
                    if results:
                        st.subheader("Results JSON")
                        st.json(results)
                    if st.button("Judge this run"):
                        judged = judge_run(API_URL, selected_run)
                        st.json(judged)
                summary = usecase_summary(API_URL, usecase_name)
                st.subheader("Use-case cost summary")
                st.json(summary)
            except Exception as e:  # noqa: BLE001
                st.error(f"Error loading runs/summary: {e}")

    with tabs[2]:
        st.header("Use-Cases")
        st.write("Existing use-cases:")
        try:
            st.json(fetch_usecases(API_URL))
        except Exception as e:  # noqa: BLE001
            st.error(f"Failed to fetch use-cases: {e}")

        st.subheader("Create new use-case")
        new_name = st.text_input("Use-case file name (without .yaml)")
        new_yaml = st.text_area("Use-case YAML", height=200)
        if st.button("Save use-case"):
            try:
                body = yaml.safe_load(new_yaml) if new_yaml else {}
                resp = create_usecase(API_URL, new_name, body)
                st.success(f"Saved: {resp.get('path')}")
                st.cache_data.clear()
            except Exception as e:  # noqa: BLE001
                st.error(f"Failed to save use-case: {e}")

    with tabs[3]:
        st.header("CLI Quick Reference")
        st.markdown(
            """
```bash
taskbench evaluate tasks/lecture_analysis.yaml \
  --models anthropic/claude-sonnet-4.5,openai/gpt-4o \
  --input-file tests/fixtures/sample_transcript.txt \
  --skip-judge

taskbench recommend --results results/run.json
```
            """
        )


if __name__ == "__main__":
    main()
