"""Evaluation endpoint (kept separate from core logic)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from interview_guide.evals.runner import EvalOptions, run_eval, AnswerEvalOptions, run_answer_eval

router = APIRouter(prefix="/evals", tags=["evals"])


class EvalRequest(BaseModel):
    dataset_path: Optional[str] = Field(
        None, description="Path to JSONL/JSON eval dataset. Defaults to jd_dataset_50.json in repo root."
    )
    limit: Optional[int] = Field(None, ge=1, description="Limit number of records.")
    include_details: bool = Field(False, description="Include per-record results.")
    judge_questions: bool = Field(False, description="Run LLM judge over questions.")
    judge_model: str = Field("gpt-4o-mini", description="Model for question judge.")
    judge_answers: bool = Field(False, description="Run LLM judge over evaluator scores.")
    answer_model: str = Field("gpt-5-nano", description="Model for synthetic answer generation.")
    answer_judge_model: str = Field("gpt-4o-mini", description="Model for answer judge.")
    max_questions_for_eval: int = Field(3, ge=1, le=10, description="Questions per JD to judge answers.")


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
        judge_answers=request.judge_answers,
        answer_model=request.answer_model,
        answer_judge_model=request.answer_judge_model,
        max_questions_for_eval=request.max_questions_for_eval,
    )
    return run_eval(opts)


class AnswerEvalRequest(BaseModel):
    dataset_path: Optional[str] = Field(
        None, description="Path to JSONL/JSON answer eval dataset."
    )
    limit: Optional[int] = Field(None, ge=1, description="Limit number of records.")
    include_details: bool = Field(False, description="Include per-record results.")


@router.post("/answers")
def run_answer_evals(request: AnswerEvalRequest):
    dataset_path = Path(request.dataset_path) if request.dataset_path else None
    if dataset_path is None:
        dataset_path = Path(__file__).resolve().parents[5] / "interview_dataset.json"
    opts = AnswerEvalOptions(
        dataset_path=dataset_path,
        limit=request.limit,
        include_details=request.include_details,
    )
    return run_answer_eval(opts)
