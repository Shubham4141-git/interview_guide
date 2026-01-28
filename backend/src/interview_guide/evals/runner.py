"""Lightweight evaluation runner (kept separate from core logic).

Evaluates:
- Intent accuracy (router)
- Skill extraction precision/recall/F1 vs expected_skills
- Evidence support rate (skills grounded in JD text)
- Optional question quality via LLM judge
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, conint

from interview_guide.router import classify, _SYSTEM as ROUTER_SYSTEM
from interview_guide.agents import jd_parser as _jd
from interview_guide.agents.qgen import generate_from_topic, _TEMPLATE as QGEN_TEMPLATE
from interview_guide.agents.evaluator import evaluate_from_slots, _SYSTEM as EVAL_SYSTEM, _TEMPLATE as EVAL_TEMPLATE


ROOT = Path(__file__).resolve().parents[4]


def _estimate_tokens(text: str) -> int:
    # Rough heuristic: 1 token ~ 4 chars in English
    return max(1, int(len(text) / 4))


def _load_dataset(path: Path) -> List[Dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    # Try JSON (array)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    # Fallback: JSONL
    items: List[Dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
    return items


def _normalize_skill_set(items: Iterable[str]) -> set[str]:
    normalized = _jd._normalize_skills(list(items))  # reuse project canonicalization
    return {s.strip().lower() for s in normalized if s and s.strip()}


def _extract_expected_skills(rec: Dict[str, Any]) -> List[str]:
    skills = rec.get("expected_skills") or []
    out: List[str] = []
    for s in skills:
        if isinstance(s, dict):
            val = s.get("skill")
            if isinstance(val, str):
                out.append(val.strip())
        elif isinstance(s, str):
            out.append(s.strip())
    return [s for s in out if s]


def _evidence_support_rate(skills: List[str], jd_text: str) -> float:
    if not skills:
        return 0.0
    supported = _jd._evidence_filter(skills, jd_text)
    return round(len(supported) / max(1, len(skills)), 4)


class QuestionJudge(BaseModel):
    relevance: conint(ge=0, le=5) = Field(..., description="How well questions match the JD")
    coverage: conint(ge=0, le=5) = Field(..., description="Coverage of key skills/responsibilities")
    clarity: conint(ge=0, le=5) = Field(..., description="Clarity and precision of questions")
    diversity: conint(ge=0, le=5) = Field(..., description="Non-redundant variety of questions")
    notes: str = Field(..., description="1-2 sentence summary")


class SyntheticAnswers(BaseModel):
    good: str = Field(..., description="High-quality answer")
    bad: str = Field(..., description="Low-quality or incorrect answer")


class AnswerJudge(BaseModel):
    expected_band: str = Field(..., description="poor | ok | good")
    aligned: bool = Field(..., description="Whether evaluator score aligns with expected band")
    notes: str = Field(..., description="Short explanation")


_JUDGE_SYSTEM = (
    "You are a strict evaluator of interview questions.\n"
    "Score each dimension 0–5 (integers only).\n"
    "Use 5 for excellent, 3 for ok, 1 for poor.\n"
    "Return ONLY JSON."
)

_JUDGE_TEMPLATE = (
    "{system}\n\n"
    "Job Description:\n{jd_text}\n\n"
    "Questions:\n{questions}\n\n"
    "Return JSON with keys: relevance, coverage, clarity, diversity, notes."
)


def _judge_questions(jd_text: str, questions: List[str], model: str = "gpt-4o-mini") -> Dict[str, Any]:
    prompt = ChatPromptTemplate.from_template(_JUDGE_TEMPLATE)
    llm = ChatOpenAI(model=model, temperature=0.0)
    chain = prompt | llm.with_structured_output(QuestionJudge, method="function_calling")
    q_block = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
    payload = {"system": _JUDGE_SYSTEM, "jd_text": jd_text[:3000], "questions": q_block[:6000]}
    return chain.invoke(payload).model_dump()


_ANSWER_GEN_SYSTEM = (
    "You generate two answers to the same interview question.\n"
    "Return ONLY JSON.\n"
    "- good: a strong, accurate, concise answer.\n"
    "- bad: a weak or incorrect answer (contains mistakes or missing key points).\n"
)

_ANSWER_GEN_TEMPLATE = (
    "{system}\n\n"
    "Question:\n{question}\n\n"
    "Return JSON: {{\"good\":\"...\",\"bad\":\"...\"}}"
)


def _generate_synthetic_answers(question: str, model: str = "gpt-5-nano") -> Dict[str, str]:
    prompt = ChatPromptTemplate.from_template(_ANSWER_GEN_TEMPLATE)
    llm = ChatOpenAI(model=model, temperature=0.2)
    chain = prompt | llm.with_structured_output(SyntheticAnswers, method="function_calling")
    payload = {"system": _ANSWER_GEN_SYSTEM, "question": question}
    return chain.invoke(payload).model_dump()


_ANSWER_JUDGE_SYSTEM = (
    "You are judging whether a score aligns with an answer's quality.\n"
    "Classify the answer quality as: poor, ok, or good.\n"
    "Alignment rules:\n"
    "- poor: score 0–1\n"
    "- ok: score 2–3\n"
    "- good: score 4–5\n"
    "Return ONLY JSON."
)

_ANSWER_JUDGE_TEMPLATE = (
    "{system}\n\n"
    "Question:\n{question}\n\n"
    "Answer:\n{answer}\n\n"
    "Evaluator output:\nScore: {score}\nFeedback: {feedback}\n\n"
    "Return JSON: {{\"expected_band\":\"poor|ok|good\",\"aligned\":true|false,\"notes\":\"...\"}}"
)


def _judge_answer_alignment(
    question: str,
    answer: str,
    score: int,
    feedback: str,
    model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    prompt = ChatPromptTemplate.from_template(_ANSWER_JUDGE_TEMPLATE)
    llm = ChatOpenAI(model=model, temperature=0.0)
    chain = prompt | llm.with_structured_output(AnswerJudge, method="function_calling")
    payload = {
        "system": _ANSWER_JUDGE_SYSTEM,
        "question": question,
        "answer": answer,
        "score": int(score),
        "feedback": feedback,
    }
    return chain.invoke(payload).model_dump()


@dataclass
class EvalOptions:
    dataset_path: Path
    limit: Optional[int] = None
    include_details: bool = False
    judge_questions: bool = False
    judge_model: str = "gpt-4o-mini"
    judge_answers: bool = False
    answer_model: str = "gpt-5-nano"
    answer_judge_model: str = "gpt-4o-mini"
    max_questions_for_eval: int = 3


def run_eval(opts: EvalOptions) -> Dict[str, Any]:
    data = _load_dataset(opts.dataset_path)
    if opts.limit:
        data = data[: opts.limit]

    totals = {
        "records": 0,
        "intent_correct": 0,
        "precision_sum": 0.0,
        "recall_sum": 0.0,
        "f1_sum": 0.0,
        "support_rate_sum": 0.0,
        "router_ms_sum": 0.0,
        "skills_ms_sum": 0.0,
        "qgen_ms_sum": 0.0,
        "judge_ms_sum": 0.0,
        "answer_gen_ms_sum": 0.0,
        "answer_judge_ms_sum": 0.0,
        "answer_align_sum": 0.0,
        "answer_judged_count": 0,
        "errors": 0,
    }

    details: List[Dict[str, Any]] = []

    for rec in data:
        totals["records"] += 1
        rid = rec.get("id") or f"row_{totals['records']}"
        jd_text = str(rec.get("jd_text") or "")
        expected_intent = str(rec.get("expected_intent") or "INGEST_JD")
        expected_skills = _extract_expected_skills(rec)

        row: Dict[str, Any] = {"id": rid}

        try:
            # --- Router ---
            t0 = time.perf_counter()
            intent_obj = classify(jd_text)
            t1 = time.perf_counter()
            intent = intent_obj.type.value if intent_obj.type else "HELP"
            router_ms = (t1 - t0) * 1000

            row["intent"] = intent
            row["intent_correct"] = intent == expected_intent
            totals["intent_correct"] += int(row["intent_correct"])
            totals["router_ms_sum"] += router_ms

            # --- Skills extraction ---
            t2 = time.perf_counter()
            skills_payload = _jd.extract_skills_payload_from_slots({"jd_text": jd_text})
            t3 = time.perf_counter()
            skills_ms = (t3 - t2) * 1000

            pred_skills = skills_payload.get("skills") or []
            pred_set = _normalize_skill_set(pred_skills)
            exp_set = _normalize_skill_set(expected_skills)

            # precision / recall / f1
            tp = len(pred_set & exp_set)
            precision = tp / max(1, len(pred_set))
            recall = tp / max(1, len(exp_set))
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

            support_rate = _evidence_support_rate(pred_skills, jd_text)

            row.update(
                {
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "f1": round(f1, 4),
                    "support_rate": support_rate,
                    "pred_skills_count": len(pred_set),
                    "expected_skills_count": len(exp_set),
                }
            )
            totals["precision_sum"] += precision
            totals["recall_sum"] += recall
            totals["f1_sum"] += f1
            totals["support_rate_sum"] += support_rate
            totals["skills_ms_sum"] += skills_ms

            # --- Question generation (same as production single-call) ---
            topic = _jd._normalize_skills(list(pred_skills))
            topic = topic if topic else expected_skills
            qgen_input = (
                "Generate interview questions based on these skills:\n"
                + "\n".join(f"- {s}" for s in topic[:30])
            )
            t4 = time.perf_counter()
            qs = generate_from_topic({"topic": qgen_input, "n": "10"})
            t5 = time.perf_counter()
            qgen_ms = (t5 - t4) * 1000
            totals["qgen_ms_sum"] += qgen_ms
            row["questions_count"] = len(qs)

            # Optional question judge
            if opts.judge_questions:
                t6 = time.perf_counter()
                judge = _judge_questions(
                    jd_text,
                    [q.get("text", "") for q in qs],
                    model=opts.judge_model,
                )
                t7 = time.perf_counter()
                totals["judge_ms_sum"] += (t7 - t6) * 1000
                row["question_judge"] = judge

            # Optional answer evaluator judge (synthetic answers)
            if opts.judge_answers:
                q_limit = max(1, int(opts.max_questions_for_eval))
                judged = 0
                aligned = 0
                for q in qs[:q_limit]:
                    q_text = q.get("text", "")
                    if not q_text:
                        continue
                    # generate good/bad answers
                    t8 = time.perf_counter()
                    syn = _generate_synthetic_answers(q_text, model=opts.answer_model)
                    t9 = time.perf_counter()
                    totals["answer_gen_ms_sum"] += (t9 - t8) * 1000

                    for label in ("good", "bad"):
                        ans = syn.get(label, "").strip()
                        if not ans:
                            continue
                        # evaluator scores the answer
                        eval_out = evaluate_from_slots({"answers": [ans], "topic": q_text})
                        if not eval_out:
                            continue
                        score = eval_out[0].get("score", 0)
                        feedback = eval_out[0].get("feedback", "")
                        # judge alignment
                        t10 = time.perf_counter()
                        j = _judge_answer_alignment(
                            q_text,
                            ans,
                            int(score),
                            str(feedback),
                            model=opts.answer_judge_model,
                        )
                        t11 = time.perf_counter()
                        totals["answer_judge_ms_sum"] += (t11 - t10) * 1000
                        judged += 1
                        if j.get("aligned"):
                            aligned += 1
                row["answer_alignment_rate"] = round(aligned / max(1, judged), 4)
                totals["answer_align_sum"] += (aligned / max(1, judged)) if judged else 0.0
                totals["answer_judged_count"] += judged

            # Token estimates (approx)
            router_prompt = ROUTER_SYSTEM + "\n" + jd_text
            skill_prompt = _jd._TEMPLATE_SKILLS.format(system=_jd._SYSTEM_SKILLS, jd_text=jd_text)
            qgen_prompt = QGEN_TEMPLATE.format(topic=qgen_input, n=10)
            row["token_estimate"] = {
                "router": _estimate_tokens(router_prompt),
                "skills": _estimate_tokens(skill_prompt),
                "qgen": _estimate_tokens(qgen_prompt),
            }

        except Exception as exc:
            totals["errors"] += 1
            row["error"] = repr(exc)

        if opts.include_details:
            details.append(row)

    # Aggregates
    n = max(1, totals["records"])
    summary = {
        "records": totals["records"],
        "intent_accuracy": round(totals["intent_correct"] / n, 4),
        "precision_avg": round(totals["precision_sum"] / n, 4),
        "recall_avg": round(totals["recall_sum"] / n, 4),
        "f1_avg": round(totals["f1_sum"] / n, 4),
        "support_rate_avg": round(totals["support_rate_sum"] / n, 4),
        "latency_ms_avg": {
            "router": round(totals["router_ms_sum"] / n, 2),
            "skills": round(totals["skills_ms_sum"] / n, 2),
            "qgen": round(totals["qgen_ms_sum"] / n, 2),
            "judge": round(totals["judge_ms_sum"] / n, 2),
            "answer_gen": round(totals["answer_gen_ms_sum"] / n, 2),
            "answer_judge": round(totals["answer_judge_ms_sum"] / n, 2),
        },
        "answer_alignment_avg": round(
            totals["answer_align_sum"] / max(1, totals["records"]), 4
        )
        if opts.judge_answers
        else None,
        "errors": totals["errors"],
        "token_estimates": "approx (chars/4 heuristic)",
    }

    out: Dict[str, Any] = {"summary": summary}
    if opts.include_details:
        out["details"] = details
    return out


@dataclass
class AnswerEvalOptions:
    dataset_path: Path
    limit: Optional[int] = None
    include_details: bool = False


def _score_band(score: int) -> str:
    if score >= 4:
        return "good"
    if score >= 2:
        return "ok"
    return "bad"


def run_answer_eval(opts: AnswerEvalOptions) -> Dict[str, Any]:
    data = _load_dataset(opts.dataset_path)
    if opts.limit:
        data = data[: opts.limit]

    totals = {
        "records": 0,
        "band_correct": 0,
        "score_exact": 0,
        "mae_sum": 0.0,
        "latency_ms_sum": 0.0,
        "errors": 0,
    }
    details: List[Dict[str, Any]] = []

    for rec in data:
        totals["records"] += 1
        rid = rec.get("id") or f"row_{totals['records']}"
        question = str(rec.get("question") or "")
        answer = str(rec.get("answer") or "")
        expected_score = rec.get("expected_score")
        expected_band = str(rec.get("expected_band") or "").lower()

        row: Dict[str, Any] = {"id": rid}
        try:
            t0 = time.perf_counter()
            eval_out = evaluate_from_slots({"answers": [answer], "topic": question})
            t1 = time.perf_counter()
            totals["latency_ms_sum"] += (t1 - t0) * 1000

            if not eval_out:
                raise ValueError("No evaluation output")
            score = int(eval_out[0].get("score", 0))
            feedback = str(eval_out[0].get("feedback", ""))
            pred_band = _score_band(score)

            band_correct = pred_band == expected_band if expected_band else None
            row.update(
                {
                    "score": score,
                    "feedback": feedback,
                    "pred_band": pred_band,
                    "expected_band": expected_band or None,
                }
            )

            if band_correct is not None:
                row["band_correct"] = band_correct
                totals["band_correct"] += int(band_correct)

            if expected_score is not None:
                try:
                    expected_score_int = int(expected_score)
                except Exception:
                    expected_score_int = None
                if expected_score_int is not None:
                    row["expected_score"] = expected_score_int
                    row["score_exact"] = score == expected_score_int
                    totals["score_exact"] += int(row["score_exact"])
                    totals["mae_sum"] += abs(score - expected_score_int)

            # prompt token estimate (approx)
            prompt = EVAL_TEMPLATE.format(
                system=EVAL_SYSTEM,
                topic=question or "(not specified)",
                answers_block=f"1. {answer.strip()}",
            )
            row["token_estimate"] = _estimate_tokens(prompt)

        except Exception as exc:
            totals["errors"] += 1
            row["error"] = repr(exc)

        if opts.include_details:
            details.append(row)

    n = max(1, totals["records"])
    summary = {
        "records": totals["records"],
        "band_accuracy": round(totals["band_correct"] / n, 4),
        "band_accuracy_note": "Fraction where predicted band (good/ok/bad) matches expected band.",
        "score_accuracy": round(totals["score_exact"] / n, 4),
        "score_accuracy_note": "Strict exact-match accuracy of predicted score vs expected score.",
        "mae_avg": round(totals["mae_sum"] / n, 4),
        "mae_avg_note": "Mean absolute error; average absolute score difference (lower is better).",
        "latency_ms_avg": round(totals["latency_ms_sum"] / n, 2),
        "errors": totals["errors"],
        "token_estimates": "approx (chars/4 heuristic)",
    }

    out: Dict[str, Any] = {"summary": summary}
    if opts.include_details:
        out["details"] = details
    return out


__all__ = ["EvalOptions", "AnswerEvalOptions", "run_eval", "run_answer_eval"]
