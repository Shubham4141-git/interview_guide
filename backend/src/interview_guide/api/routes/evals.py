"""Evaluation endpoint (kept separate from core logic)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from interview_guide.evals.runner import EvalOptions, run_eval

router = APIRouter(prefix="/evals", tags=["evals"])


class EvalRequest(BaseModel):
    dataset_path: Optional[str] = Field(
        None, description="Path to JSONL/JSON eval dataset. Defaults to jd_dataset_50.json in repo root."
    )
    limit: Optional[int] = Field(None, ge=1, description="Limit number of records.")
    include_details: bool = Field(False, description="Include per-record results.")
    judge_questions: bool = Field(False, description="Run LLM judge over questions.")
    judge_model: str = Field("gpt-4o-mini", description="Model for question judge.")


@router.post("/run")
def run_evals(request: EvalRequest):
    dataset_path = Path(request.dataset_path) if request.dataset_path else None
    if dataset_path is None:
        dataset_path = Path(__file__).resolve().parents[5] / "jd_dataset_50.json"
    opts = EvalOptions(
        dataset_path=dataset_path,
        limit=request.limit,
        include_details=request.include_details,
        judge_questions=request.judge_questions,
        judge_model=request.judge_model,
    )
    return run_eval(opts)

