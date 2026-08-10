# Follow-ups — bugs, gaps, and improvements

Two lists: **ASAP** (do before or right after Task 3 finishes, may affect the headline claim), and **post-Task-3** (correctness/robustness/polish for a follow-up pass).

Severity: **HIGH** = affects the headline claim or correctness. **MED** = measurement noise or writeup accuracy. **LOW** = polish.

---

## ASAP — before finalizing Task 3 results

### 1. ~~`machine_count()` includes inserters — reward is over-penalized~~ [FIXED 2026-08-09]
`mini_factorio/layout.py:306` now returns `len(machines)` only. Added `entity_count()` for diagnostics. `plan.md` §Reward reporting clarified. Prior baseline eval numbers used the buggy formula; the re-eval after SFT/GRPO uses the fixed one, so all reported checkpoints are internally consistent.

### 2. `parse_edits` returns `-1.0` for parse-fail regardless of the input layout's composite reward [MED]
`training/reward_wrapper.py:32`. Delta semantics say `after − before`. On parse-fail, we return fixed `-1.0`, which is inconsistent: doing nothing on a good starting layout scores `0`, but writing garbage scores `-1`. That's the intended anti-hack, but the magnitude is arbitrary.
**Fix**: keep the penalty, but calibrate the magnitude vs the actual composite range on the training split. If the range is ~[-0.5, +2], a -1 penalty is fine; if it's ~[-0.1, +0.1] it dominates.

### 3. Assistant prefill not applied during GRPO rollouts [MED]
`training/train_grpo.py` — TRL's `GRPOTrainer` generates completions from prompts as-is; there's no injection of `{"edits": [` prefix. SFT teaches the model to emit that prefix naturally, but if SFT drifts, we lose the guarantee that eval-time and train-time prompts match.
**Fix**: verify by inspecting a few rollout logs. If drift observed, patch the prompt template to end with `{"edits": [` and adjust dataset accordingly.

### 4. ~~Green-science rate query is loose~~ [FIXED 2026-08-09]
`mini_factorio/simulator.py:457` now requires `_machine_kind(m.type) == "assembler"` in addition to the recipe match. Defensive against schema drift.

### 5. Eval `max_new_tokens` was 512 for baseline; training targets need 1024–2048 [DONE]
Already bumped default to 2048 in `qwen_policy.py` and `notebooks/run_baseline_eval.py`. Not applicable to `training/evaluate.py` (also bumped). No further action.

### 6. Ship-gate decision if `belt_asm_chain` MAPE stays 32% [DECISION PENDING]
Known simulator gap. Plan gate is MAPE ≤ 20%. Two options at final cross-check time:
- Report as-is, label the outlier in the scatter, note the gap.
- Fix by moving to a tick-based simulator (architectural rewrite).

---

## Post-Task-3 — correctness and robustness

### 7. Belt with no consumers throttles producers to 0 vs implicit-sink assumption [MED]
`mini_factorio/simulator.py:382-383`. Behavior unclear — check whether the code models an implicit chest at belt endpoints (current default seems to allow full-rate production onto a dead-end belt). If not intended, tighten to real Factorio behavior: dead-end belt → producer inserter starves once buffer fills → 0 throughput.

### 8. `_validate_diff()` returns only the first error [LOW]
`harness/edit_applier.py:47`. On a multi-error edit, only the first reason is surfaced. Debugging is harder.
**Fix**: return `"; ".join(after_errs)` or a structured list.

### 9. Prompt formatting for belt speed / inserter throughput [LOW]
`harness/prompt_builder.py:47-48`. Currently emits `f"Belt throughput: {BELT_SPEED} items/sec"` which is a dict for multi-tier; may render awkwardly to the model.
**Fix**: format as `"Yellow belt: 15/s, Red: 30/s, Blue: 45/s"` explicitly if we ever expose belt tiers to the model.

### 10. Recipe→machine hint hardcodes tier-1 only [LOW]
`harness/prompt_builder.py:87-90`. If we later expand to tier-2/3 machines the hint becomes stale.
**Fix**: generate the hint from `RECIPES` at import time.

### 11. Missing full green-science-chain integration test [MED]
`mini_factorio/tests.py`. Tests cover individual pieces (miner→belt, belt FCFS, etc.) but no full miner→furnace→assembler→green-science chain end-to-end.
**Fix**: add a test using a handcrafted layout that produces green science at a known rate, assert within 5%.

### 12. Missing truncated-JSON parser tests [LOW]
`harness/tests.py`. `_repair_truncated_json` and `_drop_mismatched_closers` are exercised only by the sample runs; no unit tests. Adding regression tests protects against parser regressions during future refactoring.

### 13. Policy-iteration convergence oscillation test [LOW]
`task1_gridworld/tests.py`. Exercise 4.4 fix isn't tested with a deliberately crafted oscillating-π env. Add a test that would loop forever without the fix.

### 14. SFT dataset diversity is low [MED]
`training/sft_data.py`. Only 13 source blueprints, 5 augmentations each. All augmentations are `strip N random entities and re-add`. Model may overfit to the "re-add" pattern and not generalize to layouts with resources but no partial builds (like our val set).
**Fix**: augment further — mix in random_layouts train templates with hand-written correct edits (10–20 hand-crafted examples), or include the 5 "partial-error" blueprints for schema diversity.

### 15. FLE spot-check wiring during GRPO [DEFERRED]
`training/train_grpo.py`. Plan calls for a 5-rollout FLE spot-check every 50 steps. Not wired; would need a TRL callback + the FLE driver. Deferred; final cross-check at end of training covers most of the value.

### 16. Grammar-constrained decoding (`outlines`) [DEFERRED]
Would eliminate invalid-JSON class and hallucinated-recipe-name class at once. Adds ~30–60 min plumbing. Skipped for MVP but a strong candidate for follow-up if the 19% invalid_json rate hurts GRPO signal.

### 17. Assembling-machine-2/3 and inserter tiers decoded but not simulated [MED]
`translator/from_fle.py` decodes higher-tier machines by name; `mini_factorio/entities.py` only has tier-1 rates. A blueprint using assembling-machine-2 will translate but its rate will be tier-1's rate.
**Fix**: import all tiers from factoriolab; add tier tables to `entities.py`.

### 18. `plan.md` Verification tests not fully implemented [DOCUMENTATION]
Several items in plan.md §Verification are still "planned in notebook" or "planned in code tests" — see audit table. Rebalance: either write the tests, or drop them from the verification section so the doc doesn't overpromise.

---

## Nice-to-haves (real polish)

- Add a linter for the SFT completion format (round-trip `parse → apply → n_applied == len(edits)`) to catch regressions when the schema evolves.
- Cache the tokenized prompt across the K samples per layout in `evaluate.py` — currently re-tokenizes 4× per layout.
- Add a "before/after" layout renderer (matplotlib grid) to `harness/` and use in `final_results.ipynb` to visualize policy improvement.
- Colab notebook: split the "install deps" cell into base + `[llm]` so users can pick.
- Add a small `contributing.md`-style note that recipe/entity data is *generated* from factoriolab (not hand-edited) — one commented header line in `recipes.py` isn't enough for a new reader.
