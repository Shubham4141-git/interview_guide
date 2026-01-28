# Evaluation Data

This folder stores evaluation datasets and outputs.

Structure:
- `evals_data/`
  - `jd_dataset_50.json` (skills/JD evaluation dataset, JSONL)
  - `interview_dataset.json` (answer evaluation dataset, JSONL)
  - `outputs/` (timestamped evaluation results)

Suggested usage:
- Run evals and redirect output JSON here:
  - `outputs/skills_eval_YYYYMMDD_HHMMSS.json`
  - `outputs/answers_eval_YYYYMMDD_HHMMSS.json`
