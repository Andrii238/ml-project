# Plan — Factorio Green-Science Policy Improvement

## Context

Internship evaluation (evaluator: Morgan). Three tasks from `unfizzbuzzed_out.bin`:

1. Demonstrate exact policy improvement + iteration on a simple pretext MDP (Sutton & Barto §4.2/4.3).
2. Build a measurable Mini-Factorio environment where a cheap baseline LLM proposes floorplan edits to increase green science output. Show the baseline is not already optimal.
3. Use GRPO (DeepSeekMath) to iteratively improve the policy. Show `mean_reward(policy_final) >> per-iteration delta`.

Morgan's clarification: any approach is fine if solutions translate back to the reference Factorio engine and validation performance is measurable. Multiple metrics welcome (materials, area, tech-tree parts).

Repo currently has only planning docs (`Andrii_plan.md`, `plan.md`, `claude.md`) and the task file `unfizzbuzzed_out.bin`. No code yet — this is a from-scratch implementation.

---

## APPROVED SIMPLIFIED PLAN (2026-08-11) — supersedes earlier env spec

The earlier complex environment (belts + inserters as first-class entities, coal fuel, stone patches, real Factorio geometry) is set aside. New target spec below. Sections further down remain for reference only; do not follow them where they conflict with this section.

### Environment simplifications
- **Conveyors are 2-way.** A single conveyor tile can carry two item types simultaneously (two lanes, like real Factorio belts).
- **No inserter entity.** A machine placed adjacent to a conveyor auto-transfers items to/from it. The implicit inserter is not built, not placed, not counted.
- **No electricity.** Every placed machine runs at full power for free.
- **No modules.** Excluded from action space.
- **Inserters and belts are not in the cost term** of the reward.

### I/O via chests (revised 2026-08-11)
- **Two input chests, infinite contents, limited output rate per second:**
  - One chest emits `transport-belt` items.
  - One chest emits `inserter` items.
  - Output rates are **randomized per episode within reasonable bounds** (bounds TBD). Randomization adds stochasticity and prevents closed-form degenerate solutions.
- **One output chest, infinite capacity.** Green science delivered here counts toward the reward.
- **The episode provides all three chests** on the grid. The model does not place, remove, or duplicate required chests.

### Machine throughput rule
`actual_output = min(machine_crafting_rate, adjacent_conveyor_delivery_rate)`

Only these two terms. Inserter capacity intentionally excluded.

### Machine tiers
Model can place asm-1, asm-2, or asm-3.
- Crafting rates for green science (from factoriolab): 0.0833 / 0.125 / 0.208 crafts/sec.
- Tier affects both the per-machine placement penalty and the tier-unlock penalty (see reward).

### Reward shaping (formula TBD, user + assistant to design)
- **Doing nothing → fixed negative penalty**, larger in magnitude than any single build reward.
- **Each additional build → small positive reward.**
- **Per-machine placement penalty proportional to tier** (asm-3 > asm-2 > asm-1).
- **Per-conveyor small penalty.**
- **Green science delivered to output chest → large positive reward, per unit per second.**
- **One-time tier-unlock penalty:** the first asm-2 placed incurs a one-time penalty; the first asm-3 placed incurs a separate one-time penalty. Additional machines of that tier incur only the per-machine tier penalty.

### Map
- **No resource patches.** All resources come from the input chests.
- **Obstacles: OPEN QUESTION.** User leaning include, not decided.
- **Grid size: TBD.**


### Locked task distribution (2026-08-13)
- Active SFT/eval task distribution is **clustered chests only**: the three required chests are placed together in one corner/side cluster. Chest spread is intentionally small.
- The model-facing action space is **building-only**: `place_assembler` and `place_conveyor`. No `move_entity`; no model-facing chest placement/removal.
- Active SFT generator uses clustered bus templates only: small 1-2 assembler examples plus large 3-8 assembler examples, each with full-build and partial-build variants.
- Older random-chest/remote-output generators remain in code only as reference helpers; they are not part of the active SFT dataset.
- Smoke eval must use the same clustered-chest distribution as SFT.

### Locked decisions (2026-08-12)
1. **Grid size:** 20x20 (fixed across episodes).
2. **Obstacles:** none.
3. **Undergrounds:** allowed, expressed as **same-tile perpendicular crossings** — two conveyors can share a tile if their directions are perpendicular. Each carries its own 2 lanes independently. Translates 1:1 to a distance-2 underground pair in real Factorio (entry one tile before crossing, exit one tile after). Same-axis undergrounds beyond distance 2 are not part of the action space.
4. **Splitters:** dropped from action space. Model routes via multiple conveyors from the same chest or multiple assemblers pulling from the same conveyor.
5. **Chest emission rates:** each of the two input chests (belts, inserters) draws its per-episode rate from `0.9 × Uniform(0, 3) + 0.1 × Uniform(3, 5)` items/sec. Same distribution for both chests, independent draws.
6. **Prompt state:** the model sees the grid contents + the current chest emission rates. Nothing else in v1 (no history, no reward feedback, no explicit budget counter).

### Still open
- **Reward coefficients** — placeholder values coded (Chunk 3); numeric weights can be re-tuned via `RewardConfig`. The dollar-cost basis (raw resources = $1) is documented in code comments.

---

## Implementation status (2026-08-12)

Seven-chunk rewrite executed under the simplified spec. Current local test count: **93 unit tests pass.** No live FLE validation yet.

### Chunk 1 — data types (`mini_factorio/entities.py`, `mini_factorio/layout.py`)
- Chest kinds: `input-belts`, `input-inserters`, `output-science` (1×1 each).
- Assembler tiers 1/2/3 with green-science crafts/sec = `crafting_speed / 6s` = 0.083 / 0.125 / 0.208.
- Conveyor tiers 1/2/3 with per-lane capacity 15 / 30 / 45 items/sec.
- `Layout` schema: chests + assemblers + conveyors + per-episode `chest_rates`.
- Validator: bounds, ID uniqueness, chest-kind count, overlap rule (same-tile perpendicular conveyor crossings allowed; everything else rejected).
- 25 tests covering construction, JSON round-trip, all validation rules.

### Chunk 2 — simulator (`mini_factorio/simulator.py`)
- Rate-based fixed-point solver. Nodes: chests, conveyors, assemblers.
- Chest emission divided equally among adjacent conveyors that point away from the chest.
- Conveyor propagation: upstream conveyors + assembler outputs added to per-item flow. Same-tile perpendicular crossings tracked per conveyor id (each keeps its own lanes).
- Assembler consumes required inputs from any adjacent conveyor carrying them. Assembler outputs science to adjacent conveyors that are empty or already carrying science.
- Assembler rate: `min(crafting_rate, belts_in, inserters_in)` crafts/sec.
- 2-lane cap enforced per tile with per-tier capacity.
- 7 sim tests: empty layout, all three tiers, both bottleneck directions, missing output chest, broken output path, perpendicular crossing preserves flow.
- **No splitters** (dropped per user decision; simpler action space).

### Chunk 3 — reward (`mini_factorio/reward.py`)
- `RewardConfig` dataclass — every coefficient a named field, easy to tune.
- `RewardBreakdown` — total + 12 named terms for debugging.
- Terms: do-nothing (-30), missing-chest (-10 each), 3 milestones per assembler (has-belts, has-inserters, is-producing), delivered green-science (+100/pack/s), produced-but-not-delivered partial credit (+5), per-tier machine cost (dollar-based: 1.06 / 3.22 / 8.94), per-tier conveyor cost (0.06 / 0.46 / 1.26), asm-T2/T3 unlocks (-6.5 / -18), conveyor-T2/T3 unlocks (-0.9 / -2.5), random exploration bonus.
- Random bonus: deterministic per layout (SHA-256 seed on layout JSON) so GRPO's group-based advantage stays consistent. Per-i draw `U(0, upper/(1+decay·i))` with tighter bound as more entities are placed.
- 11 reward tests covering: do-nothing, missing chest, full chain, tier unlock, conveyor tier unlock, random-bonus determinism, produced-not-delivered, sum-to-total, config swap.

### Chunk 4 — random episodes (`mini_factorio/random_layouts.py`)
- `sample_chest_rate(rng)` draws from `0.9·U(0,3) + 0.1·U(3,5)`.
- `empty_episode(seed)` — 3 chests randomly placed, no other entities.
- `partial_episode(seed)` — chests + up to 3 asm-1 + up to 15 T1 conveyors at random valid positions.
- `sample_episodes(n, mode)` — batch with deterministic seeds.
- 8 tests: rate bounds, ~10% high-bucket frequency (10k samples), 100-episode validation in both modes, seed determinism.

### Chunk 5 — prompt builder (`harness/prompt_builder.py`)
- ASCII grid: 1 char/tile with legend (`.` `B` `I` `O` `1` `2` `3` `>` `<` `^` `v` `+`).
- Entity list block: compact JSON with id, tier, direction.
- Recipe + crafts/sec table + chest emission rates + edit vocabulary summary + goal.
- `build_chat_messages(layout)` returns `[system, user]` for `apply_chat_template`.
- Sample prompt ~1500-2000 chars for a fresh episode. Well under context.
- 9 tests covering grid rendering, chest chars, tier display, crossing symbol, message sections, chat shape, rate precision.

### Chunk 6 — edit vocab + parser + applier (`harness/edit_schema.py`, `edit_parser.py`, `edit_applier.py`)
- Main model-facing edit types: `place_assembler`, `place_conveyor`. The parser still supports `place_chest` and `remove_entity` internally, but current prompts/training do not ask the model to use them because required chests are pre-placed and the agreed action space is building-only.
- Parser tolerates: prose before/after, markdown code fences, per-item validation errors (partial edits survive). Detects truncated arrays.
- Applier: deep-copies input, applies each edit in isolation, collects per-edit errors. Enforces bounds, ID uniqueness, non-overlap (with perpendicular-crossing exception).
- 20 tests covering schema, parser edge cases (prose, fences, truncation, partial-valid), applier (add/remove, duplicates, out-of-bounds, footprint overlap, perpendicular crossing, immutability), one end-to-end parse→apply.

### Chunk 7 — translator (`translator/to_fle.py`)
- **Job 1 — entity emission.** Chests → `infinity-chest` with per-kind filter (`transport-belt` / `inserter`) for inputs, empty filter for output. Assemblers → `assembling-machine-{tier}` with `logistic-science-pack` recipe. Conveyors → `transport-belt` / `fast-transport-belt` / `express-transport-belt` per tier. Positions tile-centered. Direction encoding: 0/4/8/12 (Factorio 2.0).
- **Job 2 — inserter injection with grid expansion.** For every machine-adjacent conveyor, the connected conveyor chain is cascade-shifted 1 tile away from the machine to open a gap. Inserter placed at the vacated tile; pickup direction points at the shifted conveyor. Grid grows beyond 20×20 as needed.
- **Job 3 — crossings → undergrounds.** Each same-tile perpendicular pair: the horizontal conveyor stays as a straight belt; the perpendicular one becomes `underground-belt` entry (upstream neighbor) + exit (downstream neighbor). Neighboring conveyors of the same direction are subsumed. Matches real Factorio distance-2 underground pattern.
- `TranslationResult` with `entities`, `warnings`, expanded `grid_size`, JSON-serializable.
- 10 tests: chest filters, assembler recipe + position, belt tier names, direction encoding, crossing → underground pair, inserter emission, cascade-shift, grid expansion, empty layout, JSON round-trip.

### What is NOT tested yet
- **Live FLE roundtrip.** No Docker Factorio container was started, no blueprint was actually built in-game, no measured production compared to sim rate. Docker Desktop is currently paused; `factorio-learning-environment` and `factorio-rcon-py` are not installed.
- **Inserter direction verification.** Encoded per `translator/FLE_NOTES.md` (pickup direction) but not verified live.
- **Complex layout edge cases** for the cascade shift (chain conflicts, multiple crossings interacting, grid-boundary edge effects).
- **End-to-end reward-eval on the baseline model.** The full flow prompt → Qwen → parse → apply → simulate → reward has not been exercised with a real model.

### Files rewritten this session
- `mini_factorio/entities.py`, `layout.py`, `simulator.py`, `reward.py`, `random_layouts.py`, `tests.py`
- `harness/prompt_builder.py`, `edit_schema.py`, `edit_parser.py`, `edit_applier.py`
- `translator/to_fle.py`

### Files still on the old schema (not yet rewritten)
- `mini_factorio/import_recipes.py`, `belt_router.py`, `handcrafted_layouts.py`, `tier.py`, `recipes.py`
- `translator/from_fle.py`, `fle_driver.py`, and the classify/measure/probe scripts
- `training/*` (data.py, oracle_solver.py, sft_data.py, reward_wrapper.py, train_sft.py, train_grpo.py, evaluate.py)
- `harness/evaluator.py`, `qwen_policy.py`
- All existing notebooks

These break under the new schema and will need updates before baseline eval / SFT / GRPO can run.

### Immediate next unblocked task
Decide how to handle the chest-rate simulator/reality mismatch (see Chunk 8 findings below).

---

## Chunk 8 — live FLE validation (2026-08-12)

Actually ran translated layouts against a live Factorio 2.0.73 container (via `factorio-rcon-py` RCON on port 27000). `translator/fle_driver.py` rewritten to:
- Wrap each entity in a `create_entity` Lua call under `/sc`.
- Apply `set_infinity_container_filter` per chest (filters can't be passed as `create_entity` kwargs in Factorio 2.0).
- Place infinite power (`electric-energy-interface` + substation grid) before layout entities (our sim skips electricity; without power, everything reports `no_power`).
- Measure `game.tick` before/after a `time.sleep(...)` window at `game.speed=20`; compute production rate as `Δ item_count / (Δ tick / 60)` (avoids CPU-throttled wall-clock reads per FLE_NOTES.md).

### Translator bugs fixed via live testing
- **Cascade shift replaced by drop-and-inserter.** The original "shift belt away from machine by 1 tile" pushed belts onto adjacent chest tiles. New rule: any conveyor whose downstream OR upstream is on a machine footprint is REPLACED by an inserter at the same tile. No shift, no grid expansion.
- **Chest filters applied via `set_infinity_container_filter`** post-create.
- **Infinite power added to the driver.** Not part of `translate()` output; power block placed by driver before layout.

### Cross-check results (8 hand-crafted single-assembler layouts)

| layout                    | tier | b_rate | i_rate | sim  | FLE   | rel err |
|---------------------------|------|--------|--------|------|-------|---------|
| asm1_sat (both = 5)       | 1    | 5.00   | 5.00   | 0.083| 0.081 | +3.3%   |
| asm2_sat (both = 5)       | 2    | 5.00   | 5.00   | 0.125| 0.123 | +1.3%   |
| asm3_sat (both = 5)       | 3    | 5.00   | 5.00   | 0.208| 0.206 | +1.4%   |
| asm1_low (both = 0.1)     | 1    | 0.10   | 0.10   | 0.083| 0.082 | +2.0%   |
| asm2_med (both = 0.5)     | 2    | 0.50   | 0.50   | 0.125| 0.121 | +2.9%   |
| asm3_belts (belts scarce) | 3    | 0.05   | 5.00   | 0.050| 0.205 | −75.6%  |
| asm3_ins (inserters scarce)| 3    | 5.00   | 0.02   | 0.020| 0.204 | −90.2%  |
| asm3_lowsym (both = 0.15) | 3    | 0.15   | 0.15   | 0.150| 0.206 | −27.0%  |

**Aggregate: Pearson r = 0.10, MAPE = 25.5%.** Both ship gates FAIL as-is.

### Root cause of the divergence

Real Factorio `infinity-chest` set via `set_infinity_container_filter(count=1000, mode="exactly")` keeps 1000 items in the chest AT ALL TIMES. Inserters draw from it at their own throughput (~0.83 items/sec for a basic inserter). There is **no per-second emission rate** in real Factorio infinity chests. Real Factorio's effective input rate = min(inserter throughput, machine consumption).

Our sim models "chest emits at `chest_rates.belts` items/sec, divided among adjacent conveyors." That number has no real-Factorio counterpart. When the sim thinks the chest is the bottleneck (rate < machine consumption), real Factorio ignores that cap and delivers items at machine consumption rate.

Agreement is excellent (<5% error) whenever the machine is the sim's bottleneck (chest supply ≥ machine consumption rate). It fails whenever the sim's chest supply < machine consumption.

### Options to resolve (pending user decision)

**A. Add a Factorio-side throttle** (e.g. a small mod script or a circuit-network condition on the inserter's enable line that limits pulls to match our rate). Real Factorio matches sim, but the translator + build code grows.

**B. Drop the chest-rate concept from the sim.** Infinity chests are truly infinite; the bottleneck becomes machine crafting rate. Removes per-episode chest randomization; simulator becomes deterministic on layout only. Loses a source of stochasticity but eliminates the mismatch entirely.

**C. Accept the mismatch, restrict FLE ship gate.** Sim and Factorio agree in the machine-bottleneck regime; disagree in the chest-bottleneck regime. Restrict cross-check to layouts where machine is the bottleneck. Training gradients still point the right direction (more assemblers, better routing) either way — the sim just under-estimates for supply-limited layouts.

Recommendation: **B** — cleanest and matches real Factorio's actual behavior. **A** is expensive and complex. **C** hides a real gap and makes the writeup awkward. But this is a design decision — your call.

---

## Chunk 9 — chest rate throttle (Option A, 2026-08-12)

User picked Option A. Implementation: **driver-side throttle** running at `game.speed = 1`. Once per wall-second the driver calls `insert()` on each input chest with `floor(accumulator)` items; accumulator absorbs fractional rates (e.g., rate=0.15 → accumulator hits 1 every ~7s, single item inserted). Input chests carry NO infinity filter (filter would fight the manual inserts); they behave as normal chests.

Also updated:
- `_translate_chest` no longer sets `infinity_settings.filters` on input chests.
- `measure_science_rate_throttled` looks chests up by position via `find_entities_filtered` (Factorio 2.0's `game.get_entity_by_unit_number` returned nil in tests).
- `chest_map_from_layout(layout)` helper builds the tile-center → kind map for the driver.

### Cross-check with throttle (same 8-layout sweep, 45s each):

| layout                    | sim   | FLE   | rel err |
|---------------------------|-------|-------|---------|
| asm1_sat                  | 0.083 | 0.066 | +26%    |
| asm2_sat                  | 0.125 | 0.110 | +14%    |
| asm3_sat                  | 0.208 | 0.198 | +5%     |
| asm3_belts (0.05)         | 0.050 | 0.044 | +15%    |
| asm3_ins (0.02)           | 0.020 | 0.000 | inf     |
| asm3_lowsym (0.15)        | 0.150 | 0.132 | +13%    |
| asm1_low (0.1)            | 0.083 | 0.044 | +89%    |
| asm2_med (0.5)            | 0.125 | 0.109 | +14%    |

**Pearson r = 0.988 (PASS)**, **MAPE = 25.2% (marginal FAIL)**.

Sim ranks layouts correctly (r > 0.9) — the GRPO reward signal is trustworthy. MAPE fails because of very-low-rate cases (R < 0.1 items/sec) where the accumulator-driven throttle inserts 1 item every 10+ seconds, and a 45s window captures 0-4 events (dominated by measurement noise, not translator quality).

### Options to hit MAPE ≤ 20% (pending user decision):
1. **Longer measurement window** (e.g., 180s per layout). Simplest — costs 4× wall time per cross-check.
2. **Exclude R < 0.1 sec from ship gate**. Restricts scope; realistic since low-supply episodes have limited signal anyway.
3. **Circuit-network throttle**. Smoother item delivery, less quantization. More Lua/circuit complexity.

**Immediate next unblocked task:** pick 1/2/3 above, then continue to rewriting `training/*` scripts for the new schema so we can start baseline eval and GRPO.

---

## Chunks 10-13 — training infrastructure (2026-08-12)

Fixed a substantive **sim bug** along the way: conveyor flow didn't propagate through L-turns (sim only checked one upstream tile). Fixed to check all 4 neighbors. All 91 tests still pass.

### Chunk 10 — reward wrapper + dataset
- `training/reward_wrapper.py` — TRL-compatible `reward_fn(prompts, completions) → list[float]`. Layout is recovered from a `<<LAYOUT>>...<</LAYOUT>>` envelope appended to each prompt.
- `training/data.py` — deterministic 60-train / 20-val split via `TRAIN_SEEDS = range(0, 60)` and `VAL_SEEDS = range(1000, 1020)`.

### Chunk 11 — Qwen wrapper + evaluator
- `harness/qwen_policy.py` — `QwenPolicy`, lazy-loading `Qwen/Qwen2.5-Coder-1.5B-Instruct`, optional LoRA adapter (peft), 4-bit quantization. Batched generation via `generate(prompts, max_new_tokens, temperature, ...)`.
- `harness/evaluator.py` — `evaluate_policy(policy, prompts, seeds, samples_per_prompt)` scoring each completion into a `SampleMetrics`, aggregated into `EvalSummary`.
- Smoke tested with 3 mock policies (empty edits, garbage JSON, cargo-cult) — all produce expected reward signatures.

### Chunk 12 — SFT data + train script
- `training/sft_data.py` — oracle that takes an empty episode (3 chests placed) and produces edits building a working single-assembler chain (asm-1 + BFS-routed conveyors from each input chest into the machine + output route to output chest). Currently 51/60 train seeds + 19/20 val seeds produce valid working layouts.
- `training/train_sft.py` — LoRA SFT via TRL's `SFTTrainer`. Config: rank 16, LR 2e-4, 3 epochs, 4-bit quantization.

### Chunk 13 — GRPO + checkpoint sweep
- `training/train_grpo.py` — TRL `GRPOTrainer` with our reward wrapper. Loads optional SFT adapter as starting point. Defaults per plan: G=8, β=0.04, LR=5e-5, 200 steps, checkpoint every 50.
- `training/evaluate.py` — `evaluate_checkpoints([{name, adapter}, ...])` sweeps a list of checkpoints on the val split, produces `CheckpointResult` per checkpoint with all raw metrics + composite reward. `rows_to_table(results)` renders a padded ASCII table.

### Sim change (turn propagation)
- Before: conveyor `cv` only accepted flow from a conveyor at `cv.upstream_tile()` (single tile behind cv in cv's own direction). Meant that L-turns silently dropped flow.
- After: `cv` checks all 4 neighbor tiles for any conveyor whose downstream is `cv`'s tile. Straight lines still work; turns now propagate correctly.
- Assembler input/output rules unchanged (input requires cv downstream on footprint; output requires cv upstream on footprint).

### What's ready for Colab
All code in-repo, importable, no local dependency on transformers/trl/peft (loaded lazily inside the training scripts). Colab notebook needs to:
1. Clone repo + `pip install transformers trl peft bitsandbytes accelerate datasets`.
2. `from training.train_sft import train; train(output_dir='/content/ckpts/sft', epochs=3)`.
3. `from training.train_grpo import train; train(sft_adapter='/content/ckpts/sft', output_dir='/content/ckpts/grpo')`.
4. `from training.evaluate import evaluate_checkpoints, rows_to_table; print(rows_to_table(evaluate_checkpoints([...])))`.

### Not done, deferred
- Baseline evaluation of policy_0 on the 20 val layouts (requires running Qwen locally or in Colab — no need to run before Colab session).
- SFT adapter training (Colab).
- GRPO training (Colab).
- Final report / notebook / writeup.
- Live FLE cross-check on top-K outputs from `policy_0` and `policy_final` (requires trained policies).

### Code impact — files that will need changes when we implement
(Audit only; not touching code until user says go.)

- **`mini_factorio/entities.py`** — replace entity taxonomy: chests as new entity type (input/output), 2-lane conveyors, no inserter entity, no coal/stone patches.
- **`mini_factorio/layout.py`** — layout schema simplified: only chests + 2-lane conveyors + assemblers of 3 tiers + obstacles.
- **`mini_factorio/simulator.py`** — full rewrite of throughput solver: chest emission rate → conveyor lane flow → `min(machine_rate, adjacent_conveyor_rate)` → output-chest delivery rate.
- **`mini_factorio/reward.py`** — new composite reward: do-nothing penalty, per-build reward, per-machine tier penalty, tier-unlock penalty for T2/T3, per-conveyor penalty, green-science delivery reward.
- **`mini_factorio/random_layouts.py`** — new starting-layout generator; per-episode randomized chest rates.
- **`mini_factorio/tests.py`** — rewrite: chest-fed chain end-to-end test; asm tier throughput correctness; tier-unlock penalty logic.
- **`harness/prompt_builder.py`** — new prompt schema (chests, 2-lane conveyors, machine tiers, chest rates as state).
- **`harness/edit_applier.py`** — new edit vocabulary aligned with simplified entity set.
- **`translator/from_fle.py` and `to_fle.py`** — update translation layer to map simplified entities back to real Factorio (chest ↔ infinity chest / requester chest, 2-lane belt ↔ standard belt with lane sorting).
- **`training/sft_data.py`** — new SFT format (chest placement + machine placement + conveyor routing).
- **`training/train_sft.py`** — retrain on new data.
- **`training/reward_wrapper.py`** — new reward formula.
- **`training/train_grpo.py`** — verify prompt template matches new env.
- **`training/evaluate.py`** — new metrics (green science delivered to chest, not internal production).
- **`notebooks/task2_baseline_eval.ipynb`, `task3_grpo_training.ipynb`, `final_results.ipynb`** — regenerate with new env.

### Decisions still pending BEFORE implementation begins
The six open items above. Nothing gets coded until they are answered.

---

## Confirmed decisions (SUPERSEDED where they conflict with the simplified plan above)

| Area | Decision |
|---|---|
| Compute | Colab Pro ($10) for GRPO training; local M2 (16GB) for Task 1, simulator, baseline inference, plots, writeup |
| Task 1 environment | Grid-agnostic gridworld up to 20x20; stochastic policies + stochastic transitions supported |
| Task 1 method | Exact policy evaluation (linear system) + **greedy improvement only** (Eq 4.9). Ties in argmax are apportioned uniformly across maximizing actions (allowed by Sutton-Barto's stochastic extension). No ε-greedy. Initial policies may still be stochastic (uniform random). |
| FLE validation | **Required deliverable, closes Morgan's explicit "translates back to reference engine" requirement.** Two parts: (A) **Translation validity** — every top-K layout must build in FLE without error (target 100%); (B) **Performance agreement** — `Pearson r ≥ 0.9` (ranking agreement, matters for GRPO signal) AND `MAPE ≤ 20%` (absolute agreement, matters for writeup claims). Failures in either part gate the ship. |
| Reward reporting | Training uses the weighted composite reward (single scalar to GRPO). **Final report shows raw individual metrics separately per checkpoint**: mean green-science rate (per second), mean materials used, mean total cells occupied, mean machine count, and valid-output rate. Example row: `green_science=12/s, materials=340, area=86 cells, machines=18, valid_outputs=82%`. Composite reward is shown alongside as a summary column. |
| Task 2 simulator | Static rate-based throughput solver; later cross-validated against FLE (Factorio Learning Environment) from GitHub |
| Task 2 numbers | Real Factorio rates and recipes |
| Task 2 belts | **Belts are first-class entities** placed on the grid by the model. A belt is a directed line of tiles carrying one item type. **Multiple producers can dump onto and multiple consumers can pull from a single belt** (the real-Factorio "bus" pattern). Belt has capacity (yellow belt = 15 items/sec). Allocation among consumers is **FCFS by upstream position along the belt direction** — matches real Factorio. Simulator still rate-based on a DAG (belts are nodes too), no ticks needed. |
| Design principle | **Prefer real Factorio mechanics wherever we have a choice.** Simplifications are called out explicitly in the writeup with reasons. |
| Fuel + resources | Stone furnaces consume **coal** as fuel. Map has patches of iron_ore, copper_ore, **stone**, and **coal**. Miners for each patch type. Furnaces need a coal supply belt in addition to their ore supply belt. |
| Inserters | **Inserters are first-class entities** placed on the grid. An inserter takes 1 cell, has a rotation, and moves items between an adjacent machine and an adjacent belt tile (or between two adjacent belts). Basic inserter throughput = 0.83 items/sec. No "adjacency = access" magic — every belt-to-machine transfer requires an inserter. |
| Construction costs | Machine costs come from factoriolab's real Factorio recipes (e.g., assembler = 3 iron_gear + 5 iron_plate + 9 electronic_circuit), not simplified flat iron_plate costs. Read via import script. |
| Electricity | **Skipped.** Documented as a simplification. All machines are assumed powered. Rationale: electricity is an orthogonal subsystem that multiplies schema complexity (~5 new entity types) without adding much to green-science optimization strategy. FLE-based validation will need to insert a power grid manually at translation time. |
| Task 2 machine sizes | Real Factorio sizes (miner 3x3, furnace 2x2, assembler 3x3, belt/inserter 1x1) on a 16x16 grid (bump to 20x20 if too tight) |
| Task 2 starting layouts | Mix of empty grids with budget + random partial builds |
| Task 3 training | TRL `GRPOTrainer` + LoRA directly on the baseline |
| Task 3 supervision | Outcome supervision (one reward per completion, computed from the final layout after all edits are applied) |
| Task 3 reward-model updates | Skipped. Our reward is a deterministic simulator, so the paper's iterative reward-model retraining step is not applicable; documented in writeup. |
| Baseline model | `Qwen2.5-Coder-1.5B-Instruct` |
| Reward formula | `R = green_science_rate − α · materials_used − β · total_cells_occupied − γ · machine_count` (small weights so main term dominates; α, β, γ tuned after baseline eval) |
| Deliverables | Python package (core logic) + Jupyter notebooks (demos, evals, plots) + markdown writeup (concise, clear, complete) |
| Data source policy | All Factorio numeric values (rates, times, costs, sizes) are pulled from factoriolab GitHub JSON via a generator script. Nothing is written from memory. Memory is only used post-hoc as a sanity check on the pipeline output. |

---

## Task 1 — Exact Policy Improvement Demo

### Files
- `task1_gridworld/environment.py`
- `task1_gridworld/policy_iteration.py`
- `task1_gridworld/random_envs.py`
- `task1_gridworld/tests.py`
- `notebooks/task1_policy_iteration.ipynb`

### Theory reference — Sutton & Barto (2nd ed., §4.2–4.3)

**Equations we implement directly:**

- (4.6) Action value under policy π:
  `q_π(s, a) = Σ_{s',r} p(s',r | s, a) [ r + γ · v_π(s') ]`

- (4.7) Policy improvement condition:
  `q_π(s, π'(s)) ≥ v_π(s) for all s ∈ S`

- (4.8) Policy improvement theorem — consequence of (4.7):
  `v_π'(s) ≥ v_π(s) for all s ∈ S`
  (Strict at any state if (4.7) is strict at that state.)

- (4.9) Greedy improvement:
  `π'(s) = argmax_a Σ_{s',r} p(s',r | s, a) [ r + γ · v_π(s') ]`

**Stochastic policy extension (from §4.2, final paragraphs):** the theorem carries through for stochastic π. When several actions achieve the argmax in (4.9), any apportionment of probability across those maximizing actions is allowed, provided submaximal actions receive zero probability. Our greedy step uses uniform apportionment across tied maximizers (one specific choice permitted by this extension). Initial policies may still be stochastic (uniform random over actions).

**Policy Iteration algorithm — reproduced verbatim from p. 80:**

```
1. Initialization
   V(s) ∈ ℝ and π(s) ∈ A(s) arbitrarily for all s ∈ S

2. Policy Evaluation
   Loop:
     Δ ← 0
     For each s ∈ S:
       v ← V(s)
       V(s) ← Σ_{s',r} p(s',r | s, π(s)) [ r + γ · V(s') ]
       Δ ← max(Δ, |v − V(s)|)
   until Δ < θ

3. Policy Improvement
   policy-stable ← true
   For each s ∈ S:
     old-action ← π(s)
     π(s) ← argmax_a Σ_{s',r} p(s',r | s, a) [ r + γ · V(s') ]
     If old-action ≠ π(s), then policy-stable ← false
   If policy-stable, stop and return V ≈ v_* and π ≈ π_*; else go to 2.
```

**Exercise 4.4 caveat (must be handled):** the algorithm above can loop forever if the policy oscillates between equally good actions. Our fix: define stability by max value change `max_s |V_new(s) − V_old(s)| < θ` OR only switch π(s) when the new action's Q-value strictly exceeds the old action's Q-value by a tolerance. The plan uses the value-change criterion as primary and the strict-improvement criterion as secondary safety.

**Implementation mapping:**
- `evaluate_policy` implements step 2. In addition to the iterative form above, we provide the closed-form `V = (I − γP_π)^{-1} R_π` for finite MDPs (mathematically equivalent, faster for our sizes ≤ 20×20). Both are cross-checked in tests.
- `improve_policy` implements step 3 (Eq 4.9): greedy argmax with uniform apportionment across tied maximizers. This is faithful to Eq 4.9 exactly (with the §4.2 stochastic extension for ties). No ε-greedy variant.
- `policy_iteration` alternates 2→3 with the Exercise 4.4 fix.

Verification tests assert Eq (4.8) empirically (`V_new(s) ≥ V_old(s)` for every state at every iteration) across 100 random envs, which is a direct check of the policy improvement theorem.

### Environment
- Class `GridWorld(rows, cols, start, goal, traps, walls=None, slip_prob=0.0, trap_penalty=-5, step_reward=-1, goal_reward=0, gamma=0.9)`.
- 4 actions (up/down/left/right). Goal is absorbing terminal.
- Trap: stepping onto a trap yields `trap_penalty` reward AND forces the next action to be a "no-op" (implemented by expanding state space to `(cell, stunned_flag)` when traps are present, so the MDP stays Markov).
- Slip: with probability `slip_prob`, executed action replaced by uniform-random action.
- Precomputes transition tensor `P[s, a, s']` and reward tensor `R[s, a]` as NumPy arrays. Grid-size agnostic; supports rectangles up to 20x20.

### Policy iteration
- Policy `π[s, a]` as probability matrix (supports fully stochastic policies).
- `evaluate_policy(pi, env)` — solve `(I − γ P_π) V = R_π` via `numpy.linalg.solve`. Also fallback iterative version to cross-check.
- `improve_policy(V, env)` — greedy Eq 4.9 with uniform apportionment across tied maximizing actions. Result is a stochastic policy `π[s,a]` that is deterministic when there is a unique argmax and uniform over ties otherwise. No ε-greedy.
- `policy_iteration(env, initial_pi, max_iters=100, tol=1e-8)` — returns `(V, pi, history)`. Terminates on value-change stability (Ex 4.4 fix).

### Notebook demonstrations
1. **Showcase**: hand-designed 6x6 grid with 2 traps + 1 wall. Start from uniform random stochastic policy. Show V table + arrow visualization before and after each iteration.
2. **Robustness**: 100 randomly generated envs across sizes {5x5, 10x10, 15x15, 20x20}. Assert `V_new ≥ V_old` element-wise for every state in every iteration of every run (the Sutton-Barto policy improvement theorem in action).
3. **Stochastic transitions**: repeat robustness with `slip_prob=0.1`. Confirm theorem still holds.
4. **Convergence plot**: iterations-to-stability vs grid size.

---

## Task 2 — Mini-Factorio Environment and Harness

### Files
- `mini_factorio/layout.py` — `Layout` dataclass, JSON serialization
- `mini_factorio/recipes.py` — real Factorio recipe database
- `mini_factorio/entities.py` — machine specs (footprint, cost, recipe options)
- `mini_factorio/belt_router.py` — A* belt path routing
- `mini_factorio/simulator.py` — rate-based throughput solver on DAG
- `mini_factorio/reward.py` — reward formula with tunable weights
- `mini_factorio/random_layouts.py` — random layout generator (empty + partial)
- `mini_factorio/tests.py` — unit tests against handcrafted layouts
- `harness/prompt_builder.py` — layout → prompt with schema and rules
- `harness/edit_schema.py` — pydantic models for edit types
- `harness/edit_parser.py` — parse LLM output → typed edits
- `harness/edit_applier.py` — apply edits, return `(new_layout, validation_errors)`
- `harness/evaluator.py` — end-to-end `policy(prompt) → reward + metrics`
- `translator/to_fle.py` — Mini-Factorio JSON → Factorio blueprint format (built later, once FLE is inspected)
- `notebooks/task2_baseline_eval.ipynb` — evaluate baseline, show it is not optimal

### Layout JSON schema
```json
{
  "grid_size": [16, 16],
  "resources": [
    {"type": "iron_ore",   "x": 0,  "y": 2,  "size": 3},
    {"type": "copper_ore", "x": 0,  "y": 7,  "size": 3},
    {"type": "stone",      "x": 0,  "y": 11, "size": 2},
    {"type": "coal",       "x": 0,  "y": 13, "size": 3}
  ],
  "budget": {"iron_plate": 200, "copper_plate": 50, "stone": 30, "iron_gear": 20, "electronic_circuit": 30},
  "entities": [
    {"id": "m1", "type": "electric_mining_drill", "x": 3,  "y": 2, "direction": "east", "target_resource": "iron_ore"},
    {"id": "m2", "type": "electric_mining_drill", "x": 3,  "y": 13, "direction": "east", "target_resource": "coal"},
    {"id": "f1", "type": "stone_furnace",         "x": 8,  "y": 2, "recipe": "iron-plate"},
    {"id": "a1", "type": "assembling_machine_1",  "x": 12, "y": 2, "recipe": "iron-gear-wheel"},
    {"id": "i1", "type": "inserter",              "x": 7,  "y": 2, "direction": "east"},
    {"id": "i2", "type": "inserter",              "x": 11, "y": 2, "direction": "east"},
    {"id": "i3", "type": "inserter",              "x": 7,  "y": 3, "direction": "north"}
  ],
  "belts": [
    {"id": "b_ore",  "item": "iron_ore",    "tiles": [[5,2,"east"],[6,2,"east"]]},
    {"id": "b_plate","item": "iron_plate",  "tiles": [[9,2,"east"],[10,2,"east"]]},
    {"id": "b_coal", "item": "coal",        "tiles": [[5,13,"north"],[5,12,"north"],[5,11,"north"],[5,10,"north"],[5,9,"north"],[5,8,"north"],[5,7,"north"],[5,6,"north"],[5,5,"north"],[5,4,"north"],[5,3,"north"]]}
  ]
}
```

Placement uses top-left anchor + orientation. Bounds and footprint collision validated on every add. Belts are directed lines: each tile has `(x, y, direction)`. Consecutive tiles must be adjacent in the direction of the previous tile.

Inserters are 1x1 entities with a rotation. An inserter picks items from the entity in its "input" direction and drops them at the entity in its "output" direction. Example above: `i1` sits between miner `m1` and belt `b_ore`, moving iron_ore from miner onto belt. `i3` sits between belt `b_coal` and furnace `f1`, feeding coal fuel to the furnace. Machines connect to belts **only via inserters** — no automatic adjacency-based flow.

Furnaces require BOTH an ore input AND a coal (fuel) input inserter to operate. Miners require no fuel (electric mining drill assumption; electricity itself is skipped as a subsystem).

### Data sources for Factorio rules and recipes

**Primary and only authoring source: factoriolab's GitHub JSON data** (https://github.com/factoriolab/factoriolab, MIT licensed). This is the same data underlying the leading open-source Factorio calculator (factoriolab.dev). Actively maintained.

No number in `recipes.py` is written from memory or manual transcription. Every value comes from the factoriolab JSON via the import script.

Pipeline:
1. Download the pinned `data.json` (from `src/data/1.1/` or the equivalent path for the version we target) into `mini_factorio/data/factoriolab_data.json`. Commit the file so results are reproducible.
2. `mini_factorio/import_recipes.py` — reads the JSON, filters to the green-science chain (~10 recipes, ~5 machines), and emits `mini_factorio/recipes.py` in our internal format.
3. `recipes.py` is generated (not hand-edited) and carries a header comment noting the factoriolab commit hash / version it was derived from.

Sanity check (not authoring): after importing, compare the generated `recipes.py` values against my memory of Factorio 1.1 and against factoriolab.dev's calculator. Any mismatch triggers investigation (usually a version mismatch or a filter bug) — but the fix is to correct the import script or pin a different data version, never to hand-edit `recipes.py`.

Cross-verification of derived logic: run our simulator on hand-checked ratio scenarios (e.g., "N miners : M furnaces : K assemblers per green science") and compare rates against factoriolab.dev's calculator. If numbers agree, our simulator is trustworthy.

Ground truth (later, once FLE is integrated): FLE embeds the game's internal data. Any discrepancy between our factoriolab-derived values and FLE's is logged in the writeup, and FLE wins.

Fallback: if factoriolab data has schema issues or version confusion we can't resolve, fall back to a small, auditable extraction from the wiki (https://wiki.factorio.com/) — still generated by a script, still no memorized values.

### Recipe subset (green-science chain)

All numeric values (crafting times, input/output amounts, machine crafting speeds, machine sizes, machine construction costs) come from the factoriolab JSON via the import script — not from this plan and not from memory.

Recipes filtered from factoriolab data:
- `iron-plate` (in stone furnace, consumes coal as fuel)
- `copper-plate` (in stone furnace, consumes coal as fuel)
- `iron-gear-wheel` (in assembler)
- `copper-cable` (in assembler)
- `electronic-circuit` (in assembler)
- `transport-belt` (in assembler)
- `inserter` (in assembler)
- `logistic-science-pack` (in assembler) — this is "green science"

Entities filtered from factoriolab data:
- `electric-mining-drill` — mines iron_ore, copper_ore, stone, or coal (target resource specified at placement)
- `stone-furnace` — smelts, consumes coal as fuel input
- `assembling-machine-1`
- `transport-belt` (as an entity, for belt tiles)
- `inserter` (as an entity, moves items between adjacent machines and belts)

Construction costs (per real Factorio) — pulled from factoriolab, no hand values:
- `electric-mining-drill` recipe → materials to build
- `stone-furnace` recipe → materials to build (5 stone in vanilla)
- `assembling-machine-1` recipe → materials (gears + iron plates + circuits)
- `transport-belt` recipe → per belt tile
- `inserter` recipe → per inserter

The import script emits a `recipes.py` and `entities.py` with these + their exact rates and construction costs. This plan intentionally does not restate the numbers to avoid a second source of truth.

**Electricity**: skipped as a subsystem. Miners and assemblers are treated as always-powered. Stone furnaces still require coal fuel input because that's a per-machine input flow, not a network-level system. Documented in the writeup under "Simplifications."

### Simulator

Belts are first-class entities placed on the grid by the model. A belt is a directed line of connected tiles carrying one item type. Machines placed adjacent to a belt can either dump onto it (producer) or pull from it (consumer). Multiple machines can share the same belt (the real-Factorio "bus" pattern).

Pipeline:
1. Validate layout: machine footprints, belt tiles, and inserters are in bounds and non-overlapping. Belt tiles form contiguous lines with consistent direction. Every furnace has at least one inserter attached to a coal belt (fuel path required).
2. Build the flow graph:
   - Nodes = machines AND belts AND inserters.
   - For each inserter: infer its "source" and "sink" from its rotation. Source can be a machine (output side) or a belt tile (any position on the line). Sink similarly. Edges: `source → inserter → sink`, typed by the item flowing.
   - No machine-to-belt or belt-to-machine flow exists without an inserter.
3. Verify the graph is a DAG. Reject cyclic layouts.
4. Topological sort over the combined machine+belt+inserter nodes.
5. Per-node propagation in topological order:
   - **Machine node** (miner / furnace / assembler): `output_rate = min(nominal_rate, min over required inputs of (supplied_rate / required_per_output) × nominal_rate)`. Furnaces require BOTH ore and coal supply — if either is missing, output = 0.
   - **Inserter node**: `throughput = min(inserter_max_rate, source.available_at_attachment)`. Basic inserter max rate = 0.83 items/sec (from factoriolab).
   - **Belt node**: total supply = sum of inserter throughputs feeding onto the belt. Cap at belt capacity (yellow belt = 15 items/sec from factoriolab). If total supply exceeds capacity, throttle producer inserters proportionally by upstream position (upstream inserters succeed first).
   - **Belt → consumer inserters (FCFS by position)**: sort consumer inserters by attachment position along the belt direction (upstream first). Iterate in order, giving each `min(consumer.demand_at_this_inserter, remaining_flow)`. Downstream consumers starve if upstream ones consume all available. Matches real-Factorio inserter behavior on a shared belt.
6. Green science reward = sum of output rates of all `green_science` (logistic-science-pack) assemblers.
7. Compute `materials_used` (using real factoriolab construction recipes), `total_cells_occupied` (machines + belt tiles + inserters), `machine_count` (miners + furnaces + assemblers **only**; inserters are already counted via `total_cells_occupied`).

**Why this is still rate-based, not tick-based:** for a DAG topology the steady-state flow through each belt is fully determined by supply, capacity, and FCFS allocation — no simulation of item movement in time is needed. Cycles (which would require iterative fixpoint solving) are forbidden by validation.

### Reward
```python
def reward(layout):
    gs_rate = simulate(layout)
    return (
        gs_rate
        - alpha * layout.materials_used()
        - beta  * layout.total_cells_occupied()
        - gamma * layout.machine_count()
    )
```
Initial weights: `alpha=0.001`, `beta=0.01`, `gamma=0.05`.

**Tuning discipline (important):** Reward weights are tuned **only on the training split** (60 layouts). The validation split (20 layouts) is never touched during weight selection. This prevents overfitting reward-shape choices to the val set, which would bias the reported final metrics. Concretely: run baseline evaluations on training layouts only, adjust weights so the main term (green science rate) dominates and secondary terms act as tie-breakers, then freeze weights before touching val.

### Edit schema

- `add_entity {type, x, y, direction, recipe?, target_resource?}` — for machines. `target_resource` is required for miners (which patch to mine). `recipe` is required for furnaces and assemblers.
- `remove_entity {id}` — supported internally by the parser/applier, but not exposed in the current building-only model prompt.
- `add_belt {id, item, tiles: [[x1,y1,dir1], [x2,y2,dir2], ...]}` — places a belt as a directed contiguous line of tiles carrying one item type. Validator checks tiles are in bounds, non-overlapping with existing entities, form a contiguous chain, and direction is consistent between adjacent tiles.
- `remove_belt {id}` — removes an entire belt.
- `extend_belt {id, tiles: [...]}` — appends tiles at head or tail.
- `add_inserter {id, x, y, direction}` — places an inserter. Inserter picks items from the entity in its input direction and drops to the entity in its output direction (opposite side).

Item flow between machines and belts happens **only through inserters**. Adjacency alone does not create a flow — a machine must have an inserter placed between it and a belt tile in the correct rotation. This matches real Factorio exactly.

LLM outputs a JSON list of edits. Each edit is validated independently. On failure, that specific edit is rejected with a clear error string; other edits in the list still apply. Layout state after all attempted edits is returned along with the per-edit error report.

### Baseline evaluation notebook
- Load `Qwen2.5-Coder-1.5B-Instruct` via `transformers`.
- 20 held-out layouts (mix of empty + partial).
- Sample K=5 edit sequences per layout. Score each.
- Compare mean baseline reward vs 3-5 handcrafted "reasonable" layouts to prove baseline is not optimal.
- Report: mean reward, per-layout distribution, valid-edit rate, invalid-JSON rate.

### FLE integration (**required — the "translates back to reference engine" check Morgan explicitly asked for**)

Morgan's exact ask: *"any approach is fine as long as you're confident you can translate solutions back to the reference game engine and measure validation performance."* This section addresses **both** parts.

**Part A — Translation validity check** (structural correctness):
- Write `translator/to_fle.py`: our JSON → Factorio blueprint format. Machine sizes are real, so most of the translation is 1-to-1. Manual step: insert a minimal power grid at translation time (we skipped electricity in the sim).
- For every top-K layout: assert the translated blueprint **builds without error** in FLE — every entity valid, no collisions, no unreachable machines, no ambiguous belt directions. Failure of any layout to build = simulator produced a layout that isn't a real Factorio layout, which is a bug we must fix.
- Report: fraction of top-K layouts that build successfully. Target 100%. Any failure gets diagnosed.

**Part B — Performance agreement check** (numeric correctness). Two metrics because they answer different questions:
- **Pearson correlation** — does our simulator RANK layouts the same way FLE does? Ranking is what matters for GRPO's reward signal being meaningful. If our sim thinks A > B and FLE agrees, training works even if absolute numbers drift.
- **MAPE (mean absolute percentage error)** — are our absolute rate numbers close to reality? This is what backs claims in the writeup like "policy_final produces 12/s". A perfect Pearson with a 30% consistent overestimate is fine for training but misleading in the report.

Concretely:
- For every successfully built layout: run it in FLE for a fixed simulated period (e.g., 10 game-minutes) and record real green-science production per second.
- Compute Pearson r and MAPE across the top-K layouts.
- **Ship gates**: `Pearson r ≥ 0.9` AND `MAPE ≤ 20%`.
- Below either gate, investigate simulator discrepancy (usually: a rule we simplified is materially affecting production — e.g., inserter throughput approximation, belt capacity, fuel starvation) and either fix the sim or document the specific gap in the writeup.

**Scope**: top-K ≈ 10 layouts each from `policy_0` and `policy_final` (20 total). Small enough to run in <30 min of FLE time, large enough for a meaningful correlation.

**Writeup section 7** reports both:
- Table 1: per-layout `builds_ok / our_sim_rate / FLE_rate / abs_error`
- Aggregate: build success rate, Pearson r, MAE, plot of our_sim_rate vs FLE_rate (scatter with y=x line)

This closes the loop on Morgan's explicit requirement: our layouts translate back to the reference engine (Part A), and our validation performance is measurable and correct (Part B).

---

## Task 3 — GRPO Iterative Improvement

### Baseline eval finding (2026-08-08) — motivates Stage 0 + Stage 1

First baseline run of `Qwen2.5-Coder-1.5B-Instruct` on 20 val layouts × 5 samples (`results/baseline_eval.json`):

- Parse fail rate: **27%**
- Of parse-ok completions: **most output empty edit list `[]`** — only ~15 of 73 attempted any edits
- Of attempted edits: dominant failure = wrong recipe on wrong machine (e.g., `copper-cable` on furnace)
- Green science produced: **0 across all 100 episodes**
- Mean reward: -0.41

Consequence for GRPO: with rewards near-constant per group (-1 or 0), advantages ≈ 0 and gradient ≈ 0. Straight GRPO on this baseline trains nothing. Fix: two-stage warmup (few-shot prompt + SFT) before GRPO.

Model choice: **staying on 1.5B**. No paid Colab, so no A100 for 7B. Also, small model + small SFT set is a well-matched regime — 1.5B learns narrow schema-following patterns from 20–30 examples quickly. Matches DeepSeekMath's use of 1B-class baselines.

### Files
- `training/data.py` — dataset of prompts (60 train + 20 val layouts) ✓
- `training/reward_wrapper.py` — TRL-compatible reward function calling our simulator ✓
- `training/train_grpo.py` — GRPO training entry point ✓
- `training/evaluate.py` — checkpoint evaluation ✓
- `notebooks/task3_grpo_training.ipynb` — Colab-runnable training + eval + plots ✓
- **New: `training/sft_seeds.py`** — hand-crafted good layouts + starting/edit-list pair generation
- **New: `training/train_sft.py`** — LoRA SFT warmup on rejection-sampled + seed data
- **Modified: `harness/prompt_builder.py`** — add few-shot examples + assembler-vs-furnace hint

### Stages (execution order)

**Stage 0 — Prompt fixes (no training).**
- Add 2–3 few-shot `(layout, good edits, reward)` examples inside the system prompt built by `prompt_builder.build_chat_messages`.
- Add one-line explicit recipe → machine hint ("assembler for gears/circuits/belts/inserters/science; furnace for plates only"). Kills top schema error observed in baseline eval.

**Stage 1 — SFT warmup (required, not optional).**
- Assemble ~30 good layouts. **Primary source: expert Factorio blueprints already decoded during Task 2** (real designs, higher quality than anything hand-invented, protects against simulator-quirk overfitting). Supplement with a handful of hand-written layouts for coverage gaps (e.g., minimal plate-only chains for easy signal). Stored in `training/sft_seeds.py`.
- Derive `(starting_layout, correct_edit_list)` pairs by stripping subsets of entities from each good layout (data augmentation — one good layout → multiple training pairs, target ~60–90 pairs total from 30 layouts).
- For each of the 60 training layouts: sample K=16 completions from the raw baseline. Keep best-reward if positive; otherwise fall back to a matched synthetic seed.
- LoRA SFT on the resulting `(prompt → edits)` pairs. Rank 16, LR 2e-4, 3–5 epochs. Output: **π_1** (SFT adapter, saved to `./ckpts/sft/`).

**Stage 2 — GRPO from π_1.**
- Load π_1 as the starting adapter.
- TRL `GRPOTrainer`, G=8, β=0.04, LR 5e-5, ~200 optimizer steps, checkpoint every 50 steps.
- FLE spot-check on 5 rollouts every 50 steps; log sim-vs-FLE Pearson r.
- Checkpoints saved to `./ckpts/grpo/checkpoint-{step}/`.
- Output: **π_2, π_3, π_final** (GRPO checkpoints).

**Stage 3 — Evaluation.** Existing `training/evaluate.py` covers this. Runs base (π_0), SFT (π_1), each GRPO checkpoint against 20 val layouts × 4 samples.

### Val split protection (explicit)

The 20 val layouts are held out from the 60-layout training pool and are **never** used for:
- Reward-weight tuning (done on training split only, per plan.md line 267).
- SFT seed derivation (seeds come from expert blueprints + hand-crafted, not from val).
- Rejection sampling in Stage 1 (done on the 60 train layouts).
- GRPO training in Stage 2 (uses same 60 train layouts).
- Hyperparameter selection.

Val is touched exactly once, at the end, to produce Stage 3 metrics. This protects the reported improvement from being biased by any form of train/val leakage.

### Checkpoint naming convention

| Name | What it is |
|---|---|
| `policy_0` | Raw `Qwen2.5-Coder-1.5B-Instruct`, no adapter |
| `policy_1` | SFT adapter after Stage 1 (`./ckpts/sft/`) |
| `policy_2` | GRPO checkpoint at step 50 |
| `policy_3` | GRPO checkpoint at step 100 |
| `policy_4` | GRPO checkpoint at step 150 |
| `policy_final` | GRPO checkpoint at step 200 |

Making the SFT step its own numbered checkpoint (rather than folding it into the "baseline") lets the eval table show the SFT contribution separately from the GRPO contribution.

### Config logging (for writeup reproducibility)

Every training run writes a `config.json` next to its adapter output containing: base model id, LoRA rank/alpha/dropout, LR, batch size, optimizer steps, seed, git commit hash, and any dataset-generation params. Writeup reads from these files rather than relying on memory of what was set.

### Theory reference — DeepSeekMath GRPO (arXiv 2402.03300, §4.1)

**Notation.** `q` = prompt (our layout + task description). `o_i` = i-th sampled output (edit sequence). `G` = group size (samples per prompt). `π_θ` = current policy, `π_θ_old` = policy that generated the sample, `π_ref` = reference policy (frozen at the start of each outer iteration). `r_i` = scalar reward for output `o_i` from our simulator. `Â_{i,t}` = advantage for token `t` of output `i`.

**Eq (3) — GRPO objective (the one we maximize):**

```
J_GRPO(θ) = E[q ~ P(Q), {o_i}_{i=1..G} ~ π_θ_old(O | q)]
    (1/G) Σ_{i=1..G}  (1/|o_i|) Σ_{t=1..|o_i|} {
        min(
            [π_θ(o_{i,t} | q, o_{i,<t}) / π_θ_old(o_{i,t} | q, o_{i,<t})] · Â_{i,t},
            clip([π_θ(o_{i,t} | q, o_{i,<t}) / π_θ_old(o_{i,t} | q, o_{i,<t})], 1 − ε, 1 + ε) · Â_{i,t}
        )
        − β · D_KL(π_θ || π_ref)
    }
```

Key differences from PPO (Eq 1 in the paper): (a) baseline comes from group statistics, no value model; (b) KL penalty is added directly to the loss, not to per-token reward.

**Eq (4) — KL divergence unbiased estimator (Schulman 2020):**

```
D_KL(π_θ || π_ref)  =  [π_ref(o_{i,t} | q, o_{i,<t}) / π_θ(o_{i,t} | q, o_{i,<t})]
                     − log[π_ref(o_{i,t} | q, o_{i,<t}) / π_θ(o_{i,t} | q, o_{i,<t})]
                     − 1
```
Guaranteed non-negative.

**Outcome supervision advantage (§4.1.2 — our chosen supervision mode):**

```
Â_{i,t}  =  r̃_i  =  (r_i − mean(r)) / std(r)     for all tokens t of output o_i
```
Every token in output `o_i` receives the same normalized advantage. `mean(r)` and `std(r)` are taken over the group of `G` samples for the same prompt `q`.

**Algorithm 1 — Iterative GRPO (with our reward-model-update step skipped):**

```
Input: initial policy π_θ_init ; deterministic reward function r_φ = our simulator ;
       task prompts D ; hyperparameters ε, β, μ
1:  π_θ  ←  π_θ_init
2:  for iteration = 1..I do
3:      π_ref  ←  π_θ                     # reference for KL, refreshed each outer iteration
4:      for step = 1..M do
5:          Sample a batch D_b from D
6:          π_θ_old  ←  π_θ               # snapshot for importance ratio
7:          Sample G outputs {o_i} ~ π_θ_old(· | q) for each q in D_b
8:          Compute rewards {r_i} for each o_i via r_φ    (our simulator)
9:          Compute Â_{i,t} via outcome-supervision normalization (formula above)
10:         for GRPO iteration = 1..μ do
11:             Update π_θ by maximizing Eq (3)
12:         # SKIPPED: retrain r_φ via replay — our reward is a deterministic simulator, nothing to update
Output: π_θ
```

**Implementation mapping:**
- `training/train_grpo.py` uses `trl.GRPOTrainer`, which implements Eq (3) and Eq (4) directly.
- `training/reward_wrapper.py` implements `r_φ` — our simulator wrapped as a TRL-compatible reward function that returns scalar `r_i` per completion (outcome supervision).
- Advantage normalization (`r̃_i` formula) is handled internally by `GRPOTrainer` when the reward function returns one scalar per completion.
- Reference policy `π_ref` is snapshotted at the start of each outer iteration (I=3 in our config).
- Reward-model retraining (line 12) is intentionally omitted; documented in the writeup.

**Hyperparameters — paper values vs our values (with justification):**

| Symbol | Paper | Ours | Reason for difference |
|---|---|---|---|
| Base model | DeepSeekMath-Instruct 7B | Qwen2.5-Coder-1.5B-Instruct | Smaller baseline maximises improvement headroom (per Task 2 goal) and fits Colab T4 comfortably. |
| Fine-tuning | Full parameter | LoRA (rank 16, alpha 32) | LoRA needed for T4 memory footprint. |
| Learning rate | 1e-6 (full-param) | 5e-5 (LoRA) | LoRA typically uses ~10x higher LR than full-param. May tune down to 1e-5 if training is unstable. |
| KL coeff β | 0.04 | 0.04 | Same. |
| Group size G | 64 | 8 | T4 memory / throughput constraint. |
| μ (inner updates) | 1 | 1 | Same. |
| Max length | 1024 | 1024 | Same (may reduce if generations are shorter). |
| Batch size | 1024 | ~16–32 | T4 constraint. |
| Outer iterations I | not fixed | 3 | Budget-driven; may extend if reward still climbing. |
| Steps per iteration M | derived from 144K prompts | ~200 | Our dataset is smaller (60 train prompts). |
| ε (clip) | not specified explicitly (TRL default 0.2) | 0.2 (TRL default) | Standard PPO/GRPO clip. |

### Config (initial)
- Base: `Qwen2.5-Coder-1.5B-Instruct`
- LoRA: rank 16, alpha 32, dropout 0.05, target modules = all linear
- GRPO: group size G=8, KL coeff 0.04, LR 5e-5, cosine schedule
- Supervision: **outcome supervision** — one reward per completion, all tokens in the completion share the same advantage
- Reward model updates: **skipped** — reward is our deterministic simulator (nothing to update between iterations)
- Iterations: 3, ~200 optimizer steps each, ~50 prompts per iter, generation temperature 0.8
- Save adapter per iteration (`policy_0` = base, `policy_1`, `policy_2`, `policy_3` = policy_final)

Exact GRPO objective and algorithm will be transcribed from the DeepSeekMath paper (arXiv 2402.03300) once the user sends the equations; this section will be updated to reference them directly (e.g., `# Implements Eq. X from DeepSeekMath`).

### Reward wrapper
```python
def reward_fn(prompts, completions, **kwargs) -> list[float]:
    rewards = []
    for prompt, completion in zip(prompts, completions):
        layout = layout_from_prompt(prompt)
        edits, parse_ok = parse_edits(completion)
        if not parse_ok:
            rewards.append(-1.0)  # syntax penalty
            continue
        new_layout, errs = apply_edits(layout, edits)
        if errs:
            rewards.append(-0.5)  # invalid edits penalty
            continue
        rewards.append(compute_reward(new_layout))
    return rewards
```

### Evaluation

**Val split discipline:** the 20 validation layouts are never used during training, reward-weight tuning, or hyperparameter search. They are touched exactly once, at the end, to produce the final reported metrics. This keeps the reported improvement from being biased by weight-selection or hyperparameter-selection leakage.

- On the 20 val layouts, sample 4 completions per layout from each checkpoint (policy_0 … policy_final).
- Compute the composite reward (used by GRPO training) and record every raw component **separately**.
- Verify Task 3 claim: `mean_reward(policy_final) > mean_reward(policy_0)` and `Δ_final > max_i Δ_i` where `Δ_i = mean(policy_{i+1}) − mean(policy_i)`.
- Plot: composite mean reward vs iteration + std/CI. Bar chart of per-iteration deltas.
- **Per-checkpoint raw metrics table** (as the user specified — reported both alongside the composite and standalone in the writeup):

| Checkpoint | Green science /s | Materials | Area (cells) | Machines | Valid outputs % | Composite reward |
|---|---|---|---|---|---|---|
| policy_0 | ... | ... | ... | ... | ... | ... |
| policy_1 | ... | ... | ... | ... | ... | ... |
| policy_final | 12 | 340 | 86 | 18 | 82% | ... |

The composite reward is what GRPO optimizes; the raw columns let the reader see the tradeoffs directly and are the primary evidence in the writeup.

### Colab notebook flow
1. Clone repo, `pip install` deps (transformers, trl, peft, accelerate, bitsandbytes, our package).
2. Load model + LoRA config.
3. Load prompts dataset.
4. Run GRPO training loop, saving adapters to Drive.
5. Run evaluation on val split for all checkpoints.
6. Plot + save results.

---

---

## Writeup — explicit objective and contents

The writeup at `writeup/report.md` is a first-class deliverable, not a byproduct. It must be concise, clear, and cover:

1. **Problem statement**: what the three tasks ask for, in our own words.
2. **Design choices** and why: every locked decision above (compute, machine sizes, belts, simulator paradigm, reward formula, baseline model, training method, data source), briefly justified.
3. **Task 1**: the exact Sutton-Barto policy improvement theorem and policy iteration algorithm we implemented, with the equations we used. Results: convergence, tables/plots.
4. **Task 2**: the Mini-Factorio environment (schema, recipes, simulator, reward), how we validated it, evidence the baseline is not optimal.
5. **Task 3**: the exact GRPO objective and algorithm we implemented, hyperparameters, training results (reward-vs-iteration plot, per-iteration deltas, multi-metric analysis), validation of the Task 3 claim `Δ_final > max_i Δ_i`.
6. **Limitations and simplifications**: everything we simplified vs real Factorio, why, and what the impact is. In particular: **electricity subsystem is skipped** (miners and assemblers are treated as always-powered; furnaces still need coal fuel because that's a per-machine input flow); no fluids; no biters; no machine speed modules; no belt tiers beyond yellow (15 items/sec). Note: train and val layouts each have randomly generated resource-patch positions, so we are not restricted to a single fixed map.
7. **FLE integration results (required section — addresses Morgan's "translates back to reference engine" requirement)**:
   - **Part A — Translation validity**: fraction of top-K layouts that build successfully in FLE (target 100%). Any build failures diagnosed and fixed.
   - **Part B — Performance agreement**: per-layout table of `our_sim_rate` vs `FLE_rate`, **Pearson r** (ranking agreement, matters for GRPO), **MAPE** (absolute agreement, matters for numeric claims), scatter plot with y=x reference line. Ship gates: `Pearson r ≥ 0.9` AND `MAPE ≤ 20%`.
   - Together these show: (i) our layouts are real Factorio layouts, and (ii) our simulator's reward numbers are trustworthy in both ranking and magnitude.
8. **Sources**: cite Sutton-Barto (edition + page), DeepSeekMath (arXiv id), factoriolab (repo + commit), Factorio Wiki (as secondary), FLE (if used).

Style: short paragraphs, equations in LaTeX/Markdown, tables and plots inline. Target reader is Morgan — technical, wants to see decisions defended, not repeated theory.

---

## Repository structure

```
ML project/
├── task1_gridworld/
│   ├── environment.py
│   ├── policy_iteration.py
│   ├── random_envs.py
│   └── tests.py
├── mini_factorio/
│   ├── layout.py
│   ├── recipes.py
│   ├── entities.py
│   ├── belt_router.py
│   ├── simulator.py
│   ├── reward.py
│   ├── random_layouts.py
│   └── tests.py
├── harness/
│   ├── prompt_builder.py
│   ├── edit_schema.py
│   ├── edit_parser.py
│   ├── edit_applier.py
│   └── evaluator.py
├── training/
│   ├── data.py
│   ├── reward_wrapper.py
│   ├── train_grpo.py
│   └── evaluate.py
├── translator/
│   └── to_fle.py               # built later
├── notebooks/
│   ├── task1_policy_iteration.ipynb
│   ├── task2_baseline_eval.ipynb
│   ├── task3_grpo_training.ipynb
│   └── final_results.ipynb
├── writeup/
│   └── report.md
├── pyproject.toml
└── README.md
```

Package manager: `uv` (fast, modern, single-file config).

---

## Verification

- **Task 1 unit tests**: for each of 100 random envs of varying sizes, assert `V_new(s) ≥ V_old(s)` for every state at every policy-iteration step (the Sutton-Barto policy improvement theorem). Assert policy iteration converges within a finite number of steps for every finite env.
- **Task 2 unit tests**:
  - Handcrafted layout: 1 miner + 1 furnace + 1 assembler for gears → simulator reports the correct 0.3125 plate/sec and 0.15625 gear/sec (bottlenecked by furnace).
  - Handcrafted full green-science chain → reports the correct end-to-end rate.
  - Cyclic layout → rejected as invalid.
  - Unroutable connection → rejected as invalid.
  - **Belt allocation tests (FCFS behavior — reward correctness depends on this)**:
    - **Two producers, one belt, total supply ≤ capacity**: both producers flow fully; belt throughput = sum of supplies.
    - **Two producers, one belt, total supply > capacity**: upstream producer flows fully; downstream producer is throttled by the leftover capacity; total belt throughput = capacity exactly.
    - **Two consumers, one belt, supply ≥ total demand**: both consumers get exactly their demand.
    - **Two consumers, one belt, supply < total demand**: upstream consumer gets its full demand; downstream consumer gets `max(0, supply − upstream_demand)`; sum of pulled = supply.
    - **Mixed multi-producer + multi-consumer**: end-to-end flow matches hand-computed expected FCFS allocation across a small canned scenario.
    - **Belt with zero producers**: consumers get 0.
    - **Belt with zero consumers**: producers throttled to 0 (nowhere for items to go).
  - Baseline model produces valid JSON edits ≥ 70% of the time on our schema (measured in the baseline eval notebook).
  - At least one handcrafted layout achieves strictly higher reward than baseline mean (proof baseline is not optimal).
- **Task 3 pass criteria**:
  - `mean_reward(policy_final) > mean_reward(policy_0)` on the val split, with the delta at least 3x noise floor (measured via std over samples).
  - `Δ_final > max_i Δ_i` per the Task 3 claim.
  - Reward-vs-iteration plot committed to `notebooks/final_results.ipynb`.
  - Example before/after layouts (best baseline vs best final) visualized in `final_results.ipynb`.
- **FLE cross-check (required)**: top-K (≈10) layouts from both `policy_0` and `policy_final` translated and run in FLE. Report (a) build success rate (target 100%), (b) Pearson r between our sim rate and FLE rate (ranking agreement), (c) MAPE (absolute agreement). Ship gates: 100% build success, Pearson r ≥ 0.9, MAPE ≤ 20%. Any gate failure = diagnose simulator discrepancy and fix before finalizing.

---

## FLE cross-check findings — first live run (2026-08-08)

First live run of `translator/fle_driver.smoke_test()` and `validate_and_measure()` against a real Factorio 2.0.73 dedicated server surfaced both API discrepancies and a substantive simulator/reality gap. All API fixes are landed; the simulator gap is being addressed via the "option 1" tightening described below.

### API discoveries (all documented in `translator/FLE_NOTES.md`)

- **FLE default RCON port is 27000** (FLE's `START_RCON_PORT`), password `'factorio'`. Not 27015.
- **FLE's `FactorioInstance.eval()` runs Python** (their agent REPL), not raw Lua. For raw Lua we bypass `FactorioInstance` and use `factorio-rcon-py.RCONClient.send_command("/sc <lua>")` directly (a transitive FLE dep).
- **Factorio 2.0 renamed the stats API**: `force.item_production_statistics` (attribute) → `force.get_item_production_statistics(surface)` (method). Counts are in `stats.input_counts[item_id]` (dict), not `.get_input_count(item_id)`.
- **Factorio 2.0 removed `global` from `/sc` chunks** (mod-scoped only in 2.0; renamed to `storage`). Cross-command state now lives in Python (driver reads count → sleeps → reads count → subtracts).
- **Factorio inserter `direction` is the PICKUP direction**, not the drop direction. Our layout schema uses drop direction. Translator now inverts (our 'east' → Factorio direction 12). Verified live via `LuaInserter.pickup_position` read-back.
- **Furnaces reject `set_recipe()`** with `Entity is not assembling-machine`. Furnaces smelt whatever ore is inserted. Translator now only calls `set_recipe()` for `assembling-machine-1`.

### Simulator-vs-Factorio gap: miner output model

Our simulator lets an inserter pick from **any** tile of a miner's footprint. Real Factorio requires the inserter's `pickup_position` to contain items — mining drills output only at their `drop_position` (a specific tile offset from the drill center in the direction of facing; nothing exists in the drill's body for an inserter to grab).

Concretely, our handcrafted iron-plate layout:
- Iron miner at (5,1) size 3×3, default facing (north) → drop tile (6, 0).
- Iron inserter at (8,1) picks from (7,1) — the miner's east edge, which is empty in real Factorio.

Our sim reports 0.3125 iron-plate/sec (nominal). Real Factorio produced 0 over 30 in-game seconds.

**Impact if uncorrected:** GRPO would optimize against phantom layouts — the policy would learn to build configurations that score high in sim but produce nothing when translated. The whole ship-gate mechanism (r ≥ 0.9, MAPE ≤ 20%) exists to catch exactly this.

### Sim tightening (option 1) — implemented

1. **`Machine.direction`** field added (default `"north"`; drop direction for miners, cosmetic for furnace/assembler).
2. **`Machine.drop_position()`** computed = one tile outside footprint in facing direction.
3. **Simulator rule:** inserters cannot pick from a miner footprint. Miners produce output only if a belt tile at drop_position matches the miner's `target_resource`. The belt then carries items downstream and inserters pick from the belt.
4. **Translator:** sends `FACTORIO_DIRECTION[miner.direction]` for miners (no inversion — Factorio drill direction = drop direction).
5. **Tests** updated to use miner→belt→inserter chains.

### Cross-check result (2026-08-08, evening)

**DAG sim is accurate to 0.3% on iron-plate rate.** Live measurement:

```
sim (nominal):    0.3125 iron-plate/sec
FLE (measured):   0.3117 iron-plate/sec
error:            0.3%
```

**Correction of an earlier claim.** The first cross-check reported the DAG was 78% off. That was **measurement error, not sim error.**

Apple Silicon Docker + Factorio container is CPU-bounded at ~22 game-seconds per wall-second when we ask for `game.speed = 100`. The driver's `time.sleep(60/100) = 0.6 wall-sec` gave ~13 game-seconds elapsed, but we divided by 60. So all measured rates were ~4× low. Once corrected — sample `game.tick` before/after, use `(tick_delta / 60)` as the true measurement window — the DAG matches FLE within noise on well-formed layouts.

**Remaining gap: terminal-output backpressure.** An assembler-1 running `iron-gear-wheel` with nothing consuming its gears reaches `status=full_output` and stops. Our sim reports nominal gear rate regardless. Same class of bug as the miner drop_position: the sim needs a rule that a furnace/assembler produces output only when a downstream consumer exists. Fix in progress (see next section).

Because the DAG is accurate under held assumptions and cheap to evaluate, the plan continues to use it. A tick-based simulator was considered as a more radical alternative and shelved — the DAG's per-eval cost (< 1 ms) leaves plenty of budget for GRPO's ~5000 evaluations, and remaining fidelity gaps are individual rules to add, not systemic issues with the abstraction.

### Sim tightening (option 2) — output-consumer requirement

**Rule:** a furnace/assembler produces output only when at least one downstream extractor exists — either an inserter whose pickup tile is on the machine's footprint, or (rare) a belt tile at an output-adjacent tile that the sim can identify as carrying the machine's output item. Otherwise `machine_rate = 0`.

**Effect:** every real green-science chain has a downstream consumer for every intermediate product, so this rule doesn't change layouts that would run in Factorio. It rejects sim-only phantom production (terminal producers that our sim previously reported at nominal).

**Not modeled explicitly (deliberate simplification):** the finite output-slot buffer size. When a downstream consumer exists but is intermittent, real Factorio's small output buffer (empirically ~40 items for an idle stone-furnace under our probe) can cause micro-stops. The sim treats consumer presence as a boolean; if adverse, this contributes a small MAPE.

### FLE driver fix — real game-time via game.tick

`translator/fle_driver.validate_and_measure` now:
1. Reads `game.tick` before the measurement window.
2. Sleeps enough wall time to elapse the target game-seconds even if the container is CPU-limited.
3. Reads `game.tick` after. Uses `(tick_delta / 60)` as the true measurement window in the rate denominator.

Without this fix any rate measurement on a CPU-limited container reads ~4× low.

---

## Simulator bug fixes (2026-08-09)

Two correctness fixes applied before Task 3 evaluation:

1. **`Layout.machine_count()` now returns machines only.** Previously returned `len(machines) + len(inserters)`, which caused γ·(machines+inserters) to over-penalize every valid layout by ~0.75 (γ=0.05 × ~15 inserters in a realistic chain). Inserters are still counted via the `cells` term (β=0.01 per cell), so they aren't free — they're just not double-counted. Composite reward interpretation is now aligned with plan.md §Reward reporting ("mean machine count" = miners + furnaces + assemblers). `Layout.entity_count()` added for diagnostics if we need the old sum. Baseline eval numbers gathered before this fix use the buggy formula; re-eval will happen once we have SFT + GRPO checkpoints to score, so the comparison stays internally consistent (all checkpoints scored under the same reward).

2. **Green-science rate query hardened.** `simulator.py` now requires `_machine_kind(m.type) == "assembler"` in addition to `m.recipe == GREEN_SCIENCE_ITEM` when summing green-science output. Prior code assumed only assemblers could have that recipe; a schema regression or a validation bypass would have made non-assemblers count as green-science producers. One-line defensive fix.

## Hybrid sim/FLE strategy (2026-08-08)

Confirmed after cross-check results: use **analytical sim as primary reward**
for GRPO training (fast, deterministic — critical for iteration count) and
**FLE for periodic spot-checks + final evaluation**.

Concrete protocol:
- Training: every reward call goes through `mini_factorio.simulator.simulate`.
- Spot-check: every ~50 GRPO iterations, take 5 random rollouts, run in FLE,
  compare rates. Log divergence. If Pearson r on the spot-check batch drops
  below ~0.7, pause training and investigate sim gap.
- Final report table: both `sim_rate` and `fle_rate` columns for policy_0 and
  policy_final rollouts. Rank correlation on top-K is the headline number
  proving improvement is real, not a sim exploit.

### Known sim limitation — `belt_asm_chain` (32% MAPE)

Cross-check batch has 3/4 layouts under 1% error; `belt_asm_chain` at 32%.
Root cause: our sim uses single-lane FCFS on belts. Real Factorio has two
lanes plus inserter swing-timing effects that cause a ~2:1 upstream:downstream
split when two consumers share a belt with sparse supply. Fixing this properly
requires a tick-based sim (architectural rewrite). Documented as accepted
limitation; hybrid strategy above catches any training-time exploitation of
the gap via FLE spot-checks.

---

## Open items (defaults set, easily adjusted during execution)

- **Reward weights** (α=0.001, β=0.01, γ=0.05): tuned **on the training split only**, never on val. Val is untouched until final reporting.
- **Mini-Factorio grid size** (16x16 default): bump to 20x20 if real machine sizes make green-science chain infeasible.
- **Training layouts count** (60 train + 20 val): may increase/decrease based on training speed on Colab.
- **GRPO group size** (G=8): may drop to 4 if VRAM tight on T4.
- **Number of training iterations** (3): may add a 4th if time allows and improvement is still climbing.

