# Factorio Green-Science Policy Improvement

Internship evaluation project. Full plan: [`plan.md`](plan.md).

## Setup

```bash
uv sync --extra notebook --extra dev
```

Optional LLM extras (heavy):

```bash
uv sync --extra notebook --extra dev --extra llm
```

## Layout

- `task1_gridworld/` — Sutton & Barto §4.2–4.3 policy iteration demo
- `mini_factorio/` — simplified Factorio simulator (green science)
- `harness/` — LLM prompt/edit/apply pipeline
- `training/` — GRPO training + evaluation
- `translator/` — Mini-Factorio JSON → Factorio blueprint (for FLE validation)
- `notebooks/` — demos, evaluations, plots
- `writeup/report.md` — final writeup

## Run

```bash
# Task 1 tests
uv run pytest task1_gridworld/
# Task 1 notebook
uv run jupyter lab notebooks/task1_policy_iteration.ipynb
```
