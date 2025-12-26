import asyncio
import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional
import json
import yaml

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from taskbench.api.client import OpenRouterClient
from taskbench.core.models import EvaluationResult, JudgeScore
from taskbench.core.task import TaskParser
from taskbench.evaluation.comparison import ModelComparison
from taskbench.evaluation.cost import CostTracker
from taskbench.evaluation.executor import ModelExecutor
from taskbench.evaluation.judge import LLMJudge
from taskbench.evaluation.recommender import RecommendationEngine
from taskbench.usecase import UseCase, list_usecases
from taskbench.evaluation.orchestrator import LLMOrchestrator


class RunRequest(BaseModel):
    task_path: str
    models: List[str]
    input_text: str
    judge: bool = True
    max_tokens: int = 4000
    temperature: float = 0.7
    usecase_path: Optional[str] = None
    auto_models: bool = False


class RunStatus(BaseModel):
    id: str
    task_path: str
    usecase_name: str = "default"
    models: List[str]
    judge: bool
    status: str
    output_file: Optional[str] = None
    error: Optional[str] = None


app = FastAPI(title="TaskBench UI API", version="0.1.0")

RUNS: Dict[str, RunStatus] = {}
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/models")
async def list_models():
    tracker = CostTracker()
    models = tracker.list_models()
    return [
        {
            "id": m.model_id,
            "display_name": m.display_name,
            "provider": m.provider,
            "input_price_per_1m": m.input_price_per_1m,
            "output_price_per_1m": m.output_price_per_1m,
            "context_window": m.context_window,
        }
        for m in models
    ]


@app.get("/tasks")
async def list_tasks():
    task_dir = Path("tasks")
    return [str(p) for p in task_dir.glob("*.yaml")]


@app.get("/usecases")
async def list_usecase_files():
    return list_usecases()


@app.post("/usecases")
async def create_usecase(name: str, body: dict):
    path = Path("usecases") / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return {"path": str(path)}


@app.post("/runs", response_model=RunStatus)
async def start_run(request: RunRequest):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="OPENROUTER_API_KEY not set")

    task_path = Path(request.task_path)
    if not task_path.exists():
        raise HTTPException(status_code=404, detail=f"Task not found: {request.task_path}")

    run_id = str(uuid.uuid4())
    usecase = None
    if request.usecase_path and Path(request.usecase_path).exists():
        usecase = UseCase.load(request.usecase_path)
    usecase_name = usecase.name if usecase else "default"

    run_dir = RESULTS_DIR / usecase_name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    input_file = run_dir / "input.txt"
    input_file.write_text(request.input_text, encoding="utf-8")

    run_status = RunStatus(
        id=run_id,
        task_path=str(task_path),
        usecase_name=usecase_name,
        models=request.models,
        judge=request.judge,
        status="running",
    )
    RUNS[run_id] = run_status

    async def run_eval():
        try:
            parser = TaskParser()
            task = parser.load_from_yaml(str(task_path))
            candidate_models = request.models
            if request.auto_models or not candidate_models:
                async with OpenRouterClient(api_key=api_key) as client_tmp:
                    orch = LLMOrchestrator(client_tmp)
                    candidate_models = orch.recommend_for_usecase(
                        usecase_goal=usecase.goal if usecase else task.description,
                        require_large_context=True,
                        prioritize_cost=(usecase.cost_priority == "high") if usecase else False
                    )

            cost_tracker = CostTracker()
            async with OpenRouterClient(api_key=api_key) as client:
                executor = ModelExecutor(client, cost_tracker)
                input_data = request.input_text

                results = await executor.evaluate_multiple(
                    model_ids=candidate_models,
                    task=task,
                    input_data=input_data,
                    usecase=usecase,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                )

                scores: List[Optional[JudgeScore]] = []
                if request.judge:
                    judge_model = usecase.default_judge_model if usecase and usecase.default_judge_model else "anthropic/claude-sonnet-4.5"
                    judge = LLMJudge(client, judge_model=judge_model)
                    for r in results:
                        if r.status == "success":
                            s = await judge.evaluate(task, r, input_data, usecase=usecase)
                            scores.append(s)
                        else:
                            scores.append(None)

                # Save output
                output_data = {
                    "task": task.model_dump(),
                    "usecase": usecase.model_dump() if usecase else None,
                    "results": [r.model_dump() for r in results],
                    "scores": [s.model_dump() if s else None for s in scores] if request.judge else [],
                    "statistics": cost_tracker.get_statistics(),
                    "run_judge": request.judge,
                }
                output_path = run_dir / "results.json"
                import json

                output_path.write_text(
                    json.dumps(output_data, indent=2, default=str),
                    encoding="utf-8"
                )

                run_status.status = "completed"
                run_status.output_file = str(output_path)
                RUNS[run_id] = run_status

        except Exception as exc:  # noqa: BLE001
            run_status.status = "failed"
            run_status.error = str(exc)
            RUNS[run_id] = run_status

    asyncio.create_task(run_eval())
    return run_status


@app.get("/runs", response_model=List[RunStatus])
async def list_runs(usecase: Optional[str] = None):
    if usecase:
        return [r for r in RUNS.values() if r.usecase_name == usecase]
    return list(RUNS.values())


@app.get("/runs/{run_id}")
async def get_run(run_id: str):
    if run_id not in RUNS:
        raise HTTPException(status_code=404, detail="Run not found")
    status = RUNS[run_id]
    data = {"status": status}
    if status.output_file and Path(status.output_file).exists():
        import json

        data["results"] = json.loads(Path(status.output_file).read_text(encoding="utf-8"))
    return data


@app.post("/runs/{run_id}/judge")
async def judge_run(run_id: str):
    if run_id not in RUNS:
        raise HTTPException(status_code=404, detail="Run not found")

    status = RUNS[run_id]
    if not status.output_file or not Path(status.output_file).exists():
        raise HTTPException(status_code=400, detail="Run has no results to judge")

    import json

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="OPENROUTER_API_KEY not set")

    data = json.loads(Path(status.output_file).read_text(encoding="utf-8"))
    if data.get("scores"):
        return {"detail": "Scores already present", "results": data}

    task = TaskParser().load_from_yaml(status.task_path)
    input_data = Path(Path(status.output_file).parent / "input.txt").read_text(encoding="utf-8")
    results = [EvaluationResult(**r) for r in data.get("results", [])]

    async with OpenRouterClient(api_key=api_key) as client:
        judge = LLMJudge(client)
        scores: List[Optional[JudgeScore]] = []
        for r in results:
            if r.status == "success":
                s = await judge.evaluate(task, r, input_data)
                scores.append(s)
            else:
                scores.append(None)

        data["scores"] = [s.model_dump() if s else None for s in scores]
        Path(status.output_file).write_text(
            json.dumps(data, indent=2, default=str),
            encoding="utf-8"
        )

    comp = ModelComparison()
    paired_results = [r for r, s in zip(results, data["scores"]) if s]
    paired_scores = [JudgeScore(**s) for s in data["scores"] if s]
    comparison = comp.compare_results(paired_results, paired_scores)
    engine = RecommendationEngine()
    recs = engine.generate_recommendations(comparison)

    return {"detail": "Judged", "results": data, "comparison": comparison, "recommendations": recs}


def aggregate_usecase_costs(usecase_name: str):
    base = RESULTS_DIR / usecase_name
    total_cost = 0.0
    total_tokens = 0
    runs = []
    if not base.exists():
        return {"total_cost": 0.0, "total_tokens": 0, "runs": []}
    for run_dir in base.iterdir():
        results_file = run_dir / "results.json"
        if not results_file.exists():
            continue
        data = json.loads(results_file.read_text(encoding="utf-8"))
        stats = data.get("statistics", {})
        total_cost += stats.get("total_cost", 0.0)
        total_tokens += stats.get("total_tokens", 0)
        runs.append(
            {
                "id": run_dir.name,
                "cost": stats.get("total_cost", 0.0),
                "tokens": stats.get("total_tokens", 0),
                "judged": bool(data.get("scores")),
            }
        )
    return {"total_cost": total_cost, "total_tokens": total_tokens, "runs": runs}


@app.get("/usecases/{usecase_name}/summary")
async def usecase_summary(usecase_name: str):
    return aggregate_usecase_costs(usecase_name)
