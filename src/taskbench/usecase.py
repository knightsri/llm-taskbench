from typing import List, Optional

from pydantic import BaseModel, Field
import yaml
from pathlib import Path
import os


class UseCase(BaseModel):
    name: str
    goal: str
    chunk_min_minutes: Optional[int] = None
    chunk_max_minutes: Optional[int] = None
    coverage_required: bool = False
    allow_shorter_if_justified: bool = True
    cost_priority: str = "balanced"  # high/low/balanced
    quality_priority: str = "balanced"  # high/low/balanced
    notes: List[str] = Field(default_factory=list)
    default_judge_model: Optional[str] = None
    default_candidate_models: List[str] = Field(default_factory=list)

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
        return cls(**data)


def list_usecases(directory: str = "usecases") -> list:
    p = Path(directory)
    return [str(f) for f in p.glob("*.yaml")]
