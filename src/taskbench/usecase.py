from typing import List, Optional, Dict, Any
from datetime import datetime

from pydantic import BaseModel, Field
import yaml
from pathlib import Path
import os
import json


class OutputFormatSpec(BaseModel):
    """Specification for expected output format."""
    format_type: str = "json"  # json, csv, markdown
    fields: List[Dict[str, str]] = Field(default_factory=list)  # [{name, type, description}]
    example: Optional[str] = None
    notes: Optional[str] = None


class UseCaseRun(BaseModel):
    """A single evaluation run within a use case."""
    run_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    models: List[str] = Field(default_factory=list)
    input_file: Optional[str] = None
    input_summary: Optional[str] = None  # Brief description of input
    results_file: Optional[str] = None
    status: str = "pending"  # pending, running, completed, failed
    total_cost: float = 0.0
    judged: bool = False
    judge_summary: Optional[str] = None  # LLM-generated summary of results
    best_model: Optional[str] = None
    best_score: Optional[int] = None


class UseCase(BaseModel):
    name: str
    goal: str
    llm_notes: Optional[str] = None  # Notes/hints for LLMs processing this use case
    output_format: Optional[OutputFormatSpec] = None  # Expected output structure
    chunk_min_minutes: Optional[int] = None
    chunk_max_minutes: Optional[int] = None
    coverage_required: bool = False
    allow_shorter_if_justified: bool = True
    cost_priority: str = "balanced"  # high/low/balanced
    quality_priority: str = "balanced"  # high/low/balanced
    notes: List[str] = Field(default_factory=list)
    default_judge_model: Optional[str] = None
    default_candidate_models: List[str] = Field(default_factory=list)
    # Run tracking
    runs: List[UseCaseRun] = Field(default_factory=list)
    total_cost: float = 0.0
    last_run_id: Optional[str] = None
    last_recommendation: Optional[str] = None

    @classmethod
    def load(cls, path: str) -> "UseCase":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        notes = data.get("notes", [])
        if isinstance(notes, str):
            notes = [n.strip() for n in notes.split("\n") if n.strip()]
        data["notes"] = notes
        # Expand env placeholders for judge model
        judge = data.get("default_judge_model")
        if isinstance(judge, str) and judge.startswith("${") and judge.endswith("}"):
            env_spec = judge[2:-1]
            if ":-" in env_spec:
                env_name, fallback = env_spec.split(":-", 1)
                data["default_judge_model"] = os.getenv(env_name, fallback)
            else:
                data["default_judge_model"] = os.getenv(env_spec)

        # Load output_format if present
        if "output_format" in data and isinstance(data["output_format"], dict):
            data["output_format"] = OutputFormatSpec(**data["output_format"])

        # Load runs from state file if exists
        state_path = cls._get_state_path(path)
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            data["runs"] = [UseCaseRun(**r) for r in state.get("runs", [])]
            data["total_cost"] = state.get("total_cost", 0.0)
            data["last_run_id"] = state.get("last_run_id")
            data["last_recommendation"] = state.get("last_recommendation")

        return cls(**data)

    @staticmethod
    def _get_state_path(yaml_path: str) -> Path:
        """Get the state file path for a use case."""
        p = Path(yaml_path)
        return p.parent / f".{p.stem}_state.json"

    def save_state(self, yaml_path: str):
        """Save run state to a separate JSON file."""
        state_path = self._get_state_path(yaml_path)
        state = {
            "runs": [r.model_dump() for r in self.runs],
            "total_cost": self.total_cost,
            "last_run_id": self.last_run_id,
            "last_recommendation": self.last_recommendation
        }
        state_path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")

    def add_run(self, run: UseCaseRun):
        """Add a new run and update totals."""
        self.runs.append(run)
        self.last_run_id = run.run_id
        self.total_cost += run.total_cost

    def get_run(self, run_id: str) -> Optional[UseCaseRun]:
        """Get a specific run by ID."""
        for run in self.runs:
            if run.run_id == run_id:
                return run
        return None

    def get_latest_run(self) -> Optional[UseCaseRun]:
        """Get the most recent run."""
        if self.runs:
            return self.runs[-1]
        return None

    def get_completed_runs(self) -> List[UseCaseRun]:
        """Get all completed runs."""
        return [r for r in self.runs if r.status == "completed"]

    def get_judged_runs(self) -> List[UseCaseRun]:
        """Get all runs that have been judged."""
        return [r for r in self.runs if r.judged]

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the use case status."""
        completed = self.get_completed_runs()
        judged = self.get_judged_runs()

        best_run = None
        if judged:
            best_run = max(judged, key=lambda r: r.best_score or 0)

        return {
            "name": self.name,
            "goal": self.goal,
            "total_runs": len(self.runs),
            "completed_runs": len(completed),
            "judged_runs": len(judged),
            "total_cost": self.total_cost,
            "best_model": best_run.best_model if best_run else None,
            "best_score": best_run.best_score if best_run else None,
            "last_recommendation": self.last_recommendation
        }


def list_usecases(directory: str = "usecases") -> list:
    p = Path(directory)
    return [str(f) for f in p.glob("*.yaml")]


def get_usecase_status(directory: str = "usecases") -> List[Dict[str, Any]]:
    """Get status of all use cases with their run summaries."""
    statuses = []
    for uc_path in list_usecases(directory):
        try:
            uc = UseCase.load(uc_path)
            summary = uc.get_summary()
            summary["path"] = uc_path
            statuses.append(summary)
        except Exception:
            statuses.append({"path": uc_path, "error": "Failed to load"})
    return statuses
