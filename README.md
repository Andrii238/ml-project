# Factorio Green-Science Policy Improvement

Internship evaluation project.

Main writeup: [`writeup/github_page_draft.md`](writeup/github_page_draft.md).

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
- `writeup/github_page_draft.md` — main writeup

## Run

Task 1 runs locally on CPU. Task 2 and Task 3 need the `[llm]` extra and a GPU; I ran them on Google Colab.

```bash
# Task 1 — tests + notebook
uv run pytest task1_gridworld/
uv run jupyter lab notebooks/task1_policy_iteration.ipynb

# Task 2 — baseline eval (needs [llm] extra + GPU)
uv run jupyter lab notebooks/task2_baseline_eval.ipynb

# Task 3 — SFT + GRPO training (Colab)
uv run jupyter lab notebooks/task3_grpo_training.ipynb
# Final results
uv run jupyter lab notebooks/final_results.ipynb
```
