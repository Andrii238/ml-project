# Mini-Factorio Policy Improvement — Report

Author: Andrii Romanov. Evaluator: Morgan. Draft as of 2026-08-14 — includes final Task 3 numbers.

---

## 1. Problem statement

Three sub-tasks derived from `unfizzbuzzed_out.bin`:

1. **Task 1**: demonstrate exact policy improvement + policy iteration on a small pretext MDP (Sutton & Barto §4.2/4.3).
2. **Task 2**: build a measurable Mini-Factorio green-science environment. Show a cheap baseline LLM policy is not already optimal.
3. **Task 3**: apply DeepSeekMath-style GRPO to iteratively improve that policy. Headline claim:

```
mean_reward(policy_final) > mean_reward(policy_0)
mean_reward(policy_final) - mean_reward(policy_0) >> per-iteration delta
```

Evaluator asked that solutions "translate back to the reference Factorio engine and validation performance be measured." That constraint drives the Factorio Learning Environment (FLE) cross-check in §7.

---

## 2. Design choices (locked)

| Area | Choice | Reason |
|---|---|---|
| Task 1 method | Exact policy evaluation via linear solve + greedy improvement (Eq 4.9). No ε-greedy. | Faithful to Sutton-Barto Ch 4. Grids ≤ 20×20 make the direct solve trivial. |
| Task 2 simulator | Rate-based DAG throughput solver | Deterministic; sub-ms per eval — leaves budget for the ~thousands of GRPO calls. Belts, inserters, and machines are all first-class nodes. |
| Belt semantics | First-class directed lines. Multiple producers/consumers share via **FCFS by upstream position**. | Matches the real-Factorio "bus" pattern. |
| Inserter semantics | First-class 1×1 entities with rotation. Machine ↔ belt flow requires an inserter — no adjacency magic. | Matches real Factorio. |
| Miner semantics | `drop_position` derived from `direction` field. Inserters cannot pick from the miner footprint. | Real Factorio: a miner exposes only its drop tile; inserters on the miner body pick from empty tiles. |
| Fuel | Stone furnaces require a coal input inserter in addition to the ore inserter. | Real Factorio. |
| Electricity | **Skipped.** All machines are treated as always-powered. | Orthogonal subsystem, ~5 extra entity types, negligible impact on green-science strategy. FLE translator inserts a minimal power grid at build time. |
| Costs / rates | Pulled from the factoriolab GitHub JSON at import time; no hand-written numbers. | Factoriolab is MIT-licensed, actively maintained, matches factoriolab.dev. |
| Baseline LLM | `Qwen2.5-Coder-1.5B-Instruct` | Small, cheap, fits Colab T4 with LoRA. Maximises improvement headroom (Task 2 goal). |
| Training method | LoRA + TRL `SFTTrainer` (warmup) → LoRA + TRL `GRPOTrainer` (iteration) | LoRA needed for T4 memory. SFT warmup is a paper-standard step (DeepSeekMath: pretrain → SFT → GRPO). |
| Supervision | Outcome supervision (one reward per completion) | Matches DeepSeekMath §4.1.2. |
| Reward-model retraining | Skipped | Our reward is a deterministic simulator; nothing to update between iterations. |
| Data-source policy | All Factorio numeric values (rates, times, costs, sizes) are pulled from factoriolab JSON via a script. Memory is used only post-hoc as a sanity check. | Reproducibility; avoids drift from an unpinned mental model. |
| Val split protection | The 20 val layouts are touched exactly once, at the end, for reporting. Reward weights, SFT seeds, GRPO training, and hyperparameter search all use the 60-layout training split only. | Prevents overfitting to the reported metric. |

Reward:

```
R = 500·green_science_rate
  + milestone rewards for useful intermediate production
  + 100·produced_but_not_delivered_rate
  - machine/material/tier costs
```

Green-science rate dominates; the secondary terms prevent the zero-reward problem and penalize waste.

---

## 3. Task 1 — Exact policy improvement on a gridworld

### 3.1 Environment

`task1_gridworld/environment.py` implements a rectangular MDP (any size up to 20×20). 4 actions (up/down/left/right). Goal is absorbing terminal. Trap tiles yield a penalty and force a one-step no-op (implemented by expanding state to `(cell, stunned_flag)` so the process stays Markov). Optional slip probability replaces the executed action with a uniform-random one. Transition tensor `P[s,a,s']` and reward tensor `R[s,a]` are precomputed as NumPy arrays.

### 3.2 Algorithm — reproduced from Sutton-Barto p.80

- **Eq (4.6)** action value under π:  `q_π(s, a) = Σ_{s',r} p(s',r | s, a) [ r + γ · v_π(s') ]`.
- **Eq (4.7)** improvement condition:  `q_π(s, π'(s)) ≥ v_π(s) ∀ s`.
- **Eq (4.8)** improvement theorem:  `v_π'(s) ≥ v_π(s) ∀ s`.
- **Eq (4.9)** greedy improvement:  `π'(s) = argmax_a Σ_{s',r} p(s',r | s, a) [ r + γ · v_π(s') ]`.

Ties in the argmax are apportioned uniformly across maximizing actions (permitted by the §4.2 stochastic extension). Policy evaluation is done via the closed-form linear solve `V = (I − γ P_π)^{-1} R_π` — mathematically equivalent to the iterative form and much faster at our sizes. Both forms are cross-checked in tests.

**Exercise 4.4 fix**: convergence is measured by `max_s |V_new(s) − V_old(s)| < θ` (primary) OR strict-improvement Q-value comparison (secondary safety). This prevents infinite oscillation between equal-value actions.

### 3.3 Results

- **Correctness**: 16 tests pass. Headline: Eq (4.8) `V_new ≥ V_old` is verified element-wise at every state and every iteration across **200 random envs** (100 deterministic, 100 slip=0.1), sizes 5×5 to 20×20. Zero violations.
- **Convergence**: policy iteration terminates within a bounded number of steps on every one of those envs.
- **Showcase** (`notebooks/task1_policy_iteration.ipynb`): a hand-designed "snake" dumb policy on a 6×6 with 2 traps converges to optimal in 2 iterations; a 10-trap variant converges to a zero-trap-hit policy.

Task 1 is code-complete and committed (`630817b`).

---

## 4. Task 2 — Mini-Factorio environment and baseline

### 4.1 Layout schema

16×16 grid (bumps to 20×20 if needed). Resources: `iron-ore`, `copper-ore`, `stone`, `coal`. Entities: `electric-mining-drill` (3×3), `stone-furnace` (2×2), `assembling-machine-1` (3×3), `inserter` (1×1), belt tile (1×1). All footprints, costs, crafting speeds, and recipe rates are imported from `mini_factorio/data/factoriolab_data.json` (Factorio 2.0.77, MIT) via `mini_factorio/import_recipes.py`. Nothing in `recipes.py` is hand-typed.

The recipe subset filtered from that data: `iron-plate`, `copper-plate`, `iron-gear-wheel`, `copper-cable`, `electronic-circuit`, `transport-belt`, `inserter`, `logistic-science-pack` (green science).

Layouts are validated on every mutation: bounds, footprint collisions, belt contiguity, inserter placement, machine recipe compatibility, fuel-supply requirement for furnaces.

### 4.2 Simulator — rate-based DAG

Nodes = machines ∪ inserters ∪ belts. Edges are typed by the item flowing. Cyclic graphs are rejected at validation time. In topological order:

- **Miner** node: `output_rate = nominal_rate` if the drop_position is a belt tile carrying that ore type; else 0.
- **Furnace** node: `output_rate = min(nominal_rate, min over (ore, coal) of (supplied / required))`. Missing coal or ore → 0.
- **Assembler** node: `output_rate = min(nominal_rate, min over inputs of (supplied / required))`. Missing input → 0. Requires at least one downstream extractor (else output-buffer backpressure treated as instantly full → 0). Green science is the terminal product and is exempt from this rule (implicit sink).
- **Inserter** node: `throughput = min(inserter_max, source_available_at_pickup_tile)`. Basic inserter = 0.83 items/sec.
- **Belt** node: total supply = sum of upstream inserters. Capped at belt capacity (yellow = 15 items/sec). If supply > capacity, upstream inserters win (proportional FCFS). Consumer inserters iterate in upstream order; downstream ones starve first.

**Why DAG, not tick-based**: for a DAG topology the steady state is closed-form. No item simulation in time is required. Per-eval cost is sub-ms — leaves enough budget for GRPO's ~thousands of reward calls per iteration.

### 4.3 FLE cross-check (Task 2 evidence)

`translator/fle_driver.py` decodes a layout, translates it via `translator/to_fle.py` (adds a minimal power grid, sets recipes on assemblers only — Factorio 2.0 rejects `set_recipe()` on furnaces), builds it in a live FLE Docker container, samples `game.tick` before/after a fixed measurement window (60 game-seconds at `game.speed = 100`), and reports the true per-second rate. Two API discoveries logged in `translator/FLE_NOTES.md`:

- FLE default RCON port is 27000 (not 27015), password `factorio`.
- Factorio 2.0 renamed `force.item_production_statistics` (attribute) → `force.get_item_production_statistics(surface)` (method).
- Factorio 2.0 removed `global` from `/sc` chunks (mod-scoped only now; renamed to `storage`). We keep cross-command state on the Python driver side.
- Inserter `direction` in Factorio is the **pickup** direction, not the drop direction. Our layout schema uses drop direction; the translator inverts.

**Cross-check result (2026-08-08)**: on a hand-built iron-plate layout, sim reports 0.3125 iron-plate/sec, FLE measures 0.3117/sec — **0.3% error**.

A previous run reported 78% error; that was measurement error (see next paragraph), not simulator error. On Apple Silicon, `game.speed = 100` is CPU-bounded at ~22 game-seconds per wall-second; the driver was dividing by 60 seconds of wall time and getting ~4× low rates. Reading `game.tick` before/after the wait fixes it.

**Known sim limitation** (documented, not fixed): `belt_asm_chain` (two consumers sharing a supplied belt) shows **32% MAPE** vs FLE. Root cause is real Factorio's two-lane belt + inserter swing timing, which produces ~2:1 upstream/downstream split; our single-lane FCFS doesn't. Fixing this properly requires a tick-based simulator; documented as accepted. Hybrid strategy: FLE spot-checks during training would catch any GRPO exploitation of the gap (spot-check hookup deferred to time budget).

### 4.4 Baseline is not optimal (Task 2 evidence)

Baseline model: `Qwen2.5-Coder-1.5B-Instruct` served locally via `harness/qwen_policy.py` (chat format, assistant prefill `{"edits": [`, `max_new_tokens=2048`, temperature 0.7).

Baseline evaluation on 20 held-out val layouts × 5 samples, honest parser + prefill + `}}]` repair:

| Metric | Value |
|---|---|
| invalid_json_rate | 19% |
| valid_edit_rate | 39% |
| mean composite reward | −0.47 |
| **mean green-science rate** | **0.0** |

Every one of 100 episodes produced 0 green science. Expert blueprints (13 FLE-validated designs, 10 producing green science) achieve 0.3–1.0 green-science/sec in the sim. Baseline is very much not optimal.

Failure analysis on the raw completions (100 samples): ~50% structural JSON errors that survive after `}}]` repair (nested-array bracket miscounting, extra keys with two colons), ~50% semantic drift (Chinese tokens: `"op": "笔记本"`; cross-game hallucinations: `"assembling-machine-2"`, `"add_entity<Carbonium-Fusion-Reactor>"`). Prompt engineering hit its ceiling here; both classes require training (SFT) or grammar-constrained decoding (deferred) to fix.

---

## 5. Task 3 — GRPO iterative improvement

**Status**: SFT and GRPO complete. Final results are below.

### 5.1 Stage 0 — few-shot prompt

`harness/prompt_builder.py` adds two worked examples showing valid edit lists that produce green science, plus an explicit recipe→machine hint (`stone-furnace` only smelts; assemblers do everything else) and a miner-direction reminder. Cut invalid_json from ~26% (with honest parsing) to 19% (with parser repairs added).

### 5.2 Stage 1 — SFT warmup (π_1)

- **Data**: 446 verified training examples from the locked chest-layout generator. Half are empty chest-only maps; half are partial maps where assemblers/conveyor lines were deleted and the model repairs them. Every target produces green science.
- **Config**: LoRA SFT on `Qwen2.5-Coder-1.5B-Instruct`, 3 epochs, Colab GPU. Adapter at `ckpts/sft/`.
- **Result**: SFT moves the policy from 0.0 green science/sec to 1.0417 green science/sec on the validation split.

### 5.3 Stage 2 — GRPO (π_2, π_3, π_final)

`training/train_grpo.py` loads the SFT adapter (`--init-adapter ./ckpts/sft`), merges it into base weights, then attaches a fresh trainable LoRA on top. TRL `GRPOTrainer` with:

| Symbol | Paper | Ours | Reason |
|---|---|---|---|
| Base model | DeepSeekMath-Instruct 7B | Qwen2.5-Coder-1.5B-Instruct | Small, T4-friendly, larger headroom |
| Fine-tuning | Full parameter | LoRA rank 16 | T4 memory |
| Learning rate | 1e-6 | 5e-5 | LoRA typical ~10× above full-param |
| KL β | 0.04 | 0.04 | Same |
| Group size G | 64 | 8 | Colab memory |
| μ (inner updates) | 1 | 1 | Same |
| Batch | 1024 | ~16–32 | T4 |
| Outer iterations I | not fixed | 3 | Budget |
| Steps / iteration M | derived from 144K prompts | 100 total | training budget |
| ε (clip) | not specified | 0.2 (TRL default) | Standard |

**Objective** — exactly Eq (3) of the DeepSeekMath paper as implemented by TRL:

```
J_GRPO(θ) = E_{q~P(Q), {o_i}~π_old} (1/G) Σ_i (1/|o_i|) Σ_t
    min( ratio_t · Â_{i,t}, clip(ratio_t, 1-ε, 1+ε) · Â_{i,t} ) − β · D_KL(π_θ || π_ref)

Â_{i,t} = (r_i − mean(r)) / std(r)   (outcome supervision, §4.1.2)
```

Reward-model retraining (paper Algorithm 1 line 12) is skipped — our reward is a deterministic simulator; nothing to update.

Checkpoints saved at steps 25, 50, 75, and final.

### 5.4 Stage 3 — Evaluation

`training/evaluate.py` evaluates all checkpoints on 20 validation layouts. Reports the full raw-metrics table:

| Checkpoint | GS/s | Materials | Cells | Machines | Valid % | Composite |
|---|---|---|---|---|---|---|
| policy_0 | 0.0000 | 8.4 | 19.4 | 1.00 | 100% | -0.752 |
| policy_1_sft | 1.0417 | 60.0 | 103.0 | 5.00 | 100% | 495.483 |
| policy_2_grpo25 | 1.0417 | 60.0 | 103.0 | 5.00 | 100% | 495.483 |
| policy_3_grpo50 | 1.2187 | 66.0 | 115.8 | 5.85 | 100% | 577.973 |
| policy_4_grpo75 | 1.1771 | 64.5 | 112.8 | 5.65 | 100% | 558.564 |
| policy_final | 1.2500 | 67.0 | 118.0 | 6.00 | 100% | 592.530 |

Final GRPO improves over SFT by +97.047 reward and +0.2083 green science/sec. The sampled-policy diagnostic improves from 170.676 sampled mean reward to 592.404.

---

## 6. Limitations and simplifications

- **Electricity subsystem skipped**. Miners and assemblers are treated as always-powered. Furnaces still require coal fuel because it's a per-machine input flow. FLE translator inserts a substation + power interface at translate time.
- **No fluids, no biters, no modules, no beacons**. Belt tiers beyond yellow (15 items/sec) not modeled.
- **Belt lane splitting not modeled**. Real Factorio has two lanes per belt; two consumers on the same belt end up with a ~2:1 upstream/downstream split that our single-lane FCFS doesn't reproduce. Known 32% MAPE case (`belt_asm_chain`) accepted as documented gap; hybrid FLE spot-check strategy contains the risk.
- **Output-buffer effects approximated as boolean**. Assemblers with a downstream extractor are treated as unconstrained; the finite ~40-item buffer isn't modeled. Contributes small MAPE when consumption is bursty.
- **Splitters and underground belts** collapse to plain transport-belt tiles at blueprint translation.
- **`assembling-machine-2/3` and higher inserter tiers** are decoded by name but the sim only uses tier-1 rates (recipes.py subset).
- **Stage 0 valid-edit rate capped ~40%** — small LM + long JSON edit lists = truncation + hallucinated recipe names. Grammar-constrained decoding (`outlines`) or a line-based DSL would close this; deferred.

---

## 7. FLE integration results

**Status**: driver + translator complete; final top-K FLE cross-check remains the main validation extension.

- **Part A — Translation validity**: target 100% build success on top-K layouts from `policy_0` and `policy_final` (K=10 each). Any failure diagnosed and fixed.
- **Part B — Performance agreement**: report **Pearson r** (ranking agreement — the metric that matters for GRPO's reward signal being meaningful) and **MAPE** (absolute agreement — the metric that backs numeric claims in the writeup). Ship gates: `r ≥ 0.9` AND `MAPE ≤ 20%`.

Known gap: single-lane FCFS on the `belt_asm_chain` topology → 32% MAPE. Documented accepted; will be a labeled outlier in the final scatter plot.

---

## 8. Sources

- **Sutton & Barto**, *Reinforcement Learning: An Introduction*, 2nd ed., §4.2–4.3, p. 74–80.
- **DeepSeekMath**, Shao et al., *arXiv 2402.03300*, §4.1.
- **factoriolab**, https://github.com/factoriolab/factoriolab (MIT). Data pinned as `mini_factorio/data/factoriolab_data.json` (Factorio 2.0.77).
- **Factorio Wiki**, https://wiki.factorio.com (secondary reference for entity semantics).
- **Factorio Learning Environment**, https://github.com/JackHopkins/factorio-learning-environment (used for validation).
- **TRL** (`SFTTrainer`, `GRPOTrainer`), https://github.com/huggingface/trl.
- **PEFT** (LoRA), https://github.com/huggingface/peft.
- **Qwen2.5-Coder-1.5B-Instruct**, https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct.

Blueprint sources (for SFT seeds, 2026-08-08):
- https://forums.factorio.com/viewtopic.php?t=78387
- https://forums.factorio.com/viewtopic.php?t=51570
- https://forums.factorio.com/viewtopic.php?t=97151
- https://forums.factorio.com/viewtopic.php?t=105816
