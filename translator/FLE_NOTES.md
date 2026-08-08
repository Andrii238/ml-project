# FLE (Factorio Learning Environment) — setup and usage

**Purpose:** cross-validate our Mini-Factorio simulator against real Factorio via
[factorio-learning-environment](https://github.com/JackHopkins/factorio-learning-environment).
Plan.md §FLE integration ship gates: build success 100%, Pearson r ≥ 0.9, MAPE ≤ 20%.

## What you need (no Factorio license required)

- **Docker Desktop** — free for personal/open-source use.
- **Factorio dedicated server** — free (headless server distributed by factorio.com; FLE's Docker image downloads it automatically).
- **Our `[fle]` extra**: `uv sync --extra fle` (this pulls FLE, which pulls `factorio-rcon-py` as a transitive dep).

## First-time setup

### 1. Docker + Python deps

```bash
# Verify Docker is running
docker info

# From this repo, install FLE (also gives us factorio-rcon-py)
uv sync --extra fle
```

### 2. Start Factorio via FLE's CLI

FLE ships a `fle` command that manages the Docker container for you.

```bash
# Activate the venv (uv put it at .venv/)
source .venv/bin/activate

# Start the server cluster (downloads the Factorio headless image on first run)
fle cluster start
```

Expected: one Factorio container listening on RCON port **27000** (FLE's
`START_RCON_PORT`) with password `factorio`. Check with:

```bash
docker ps           # should show a factorio-* container
```

### 3. Smoke test

```bash
python -c "from translator.fle_driver import smoke_test; print(smoke_test())"
```

Expected:

```
FLE reachable on localhost:27000; test furnace placed and cleared
```

Common failures:

- `ImportError: factorio-rcon-py not installed` → `uv sync --extra fle` didn't run or venv not active.
- Connection refused → container not running (`docker ps`) or wrong port. Try `fle cluster start` again.
- `RuntimeError: expected 1 furnace after placement, got 0` → RCON reached the server but `/sc` didn't create the entity. Usually a Factorio-version mismatch; FLE's image should be Factorio 2.0.x which matches our factoriolab data.

## Cross-checking a single layout

```python
from mini_factorio.random_layouts import empty_layout
from mini_factorio.simulator import simulate
from translator.fle_driver import validate_and_measure

lay = empty_layout(seed=42)
sim_rate = simulate(lay).green_science_rate
result = validate_and_measure(lay, layout_id="seed42", sim_rate=sim_rate)
print(result.to_dict())
```

## Batch cross-check with ship gates

```python
from translator.fle_driver import cross_check

entries = [(f"L{i}", lay, simulate(lay).green_science_rate) for i, lay in enumerate(layouts)]
report = cross_check(entries, measurement_seconds=60.0)

print(f"build success:  {report.build_success_rate:.1%}")
print(f"Pearson r:      {report.pearson_r}")
print(f"MAPE:           {report.mape}")
print(f"ship gates:     {report.ship_gates}")
```

Any `False` in `ship_gates`:

- `build_success_100pct` fails → our translator emits invalid Factorio for at least one layout; inspect that layout's `build_errors`.
- `pearson_r_ge_0_9` fails → our simulator ranks layouts differently from real Factorio; GRPO's reward signal may be misleading.
- `mape_le_0_20` fails → absolute rates drift too far; writeup claims like "policy_final produces 12/sec" would need FLE's number instead.

## How our translator maps to real Factorio

- **Directions:** N=0, E=4, S=8, W=12 (Factorio 2.0 16-way encoding).
- **Positions:** tile-centered floats. Layout top-left `(x, y)` with footprint `(w, h)` → Factorio position `(x + w/2, y + h/2)`.
- **Power:** our sim skips electricity; the translator inserts a **substation** + **electric-energy-interface** (creative-mode infinite source) before placing other entities.
- **Resources:** each tile of an ore patch is placed as a separate resource entity with `amount=100000`.
- **Recipes:** assemblers/furnaces get their recipe set via `set_recipe()` immediately after `create_entity`.
- **Technologies:** the driver runs a one-time `research_all` call so 'locked' recipes (like `logistic-science-pack`) can be crafted.

## Wire protocol

We use `factorio-rcon-py.RCONClient` directly rather than FLE's high-level
`FactorioInstance`. Reason: `FactorioInstance.eval()` runs *Python* (their agent
REPL), not raw Lua — and our translator emits Lua. `factorio-rcon-py` is a
transitive FLE dep, so `uv sync --extra fle` installs both.

Every Lua command is wrapped with Factorio's `/sc` (silent-command) prefix.
Read-back values use `rcon.print(expr)` inside the Lua chunk; the printed
string comes back as the RCON reply.

## When you're done

```bash
fle cluster stop     # stops the Factorio container
```
