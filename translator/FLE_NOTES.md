# Factorio Learning Environment (FLE) Notes

Repo: https://github.com/JackHopkins/factorio-learning-environment

## Setup

- `pip install factorio-learning-environment`
- **Docker Desktop** required (free for personal/open-source use). Runs the Factorio dedicated server as a container. No Factorio game license needed — dedicated server is free and runs headlessly.
- Factorio 2.0.73+ (matches our recipe data at 2.0.77).

## How it works (for our validation use case)

FLE connects to the Factorio server via RCON (TCP). The core API is `FactorioInstance`:

```python
from fle.env import FactorioInstance

instance = FactorioInstance(address="localhost", tcp_port=27015)
instance.initialise(all_technologies_researched=True)

# Place entities via Lua through RCON
instance.eval("game.player.surface.create_entity{name='assembling-machine-1', position={0,0}}")

# Advance time
instance.set_speed(100)
# ... wait / run ticks

# Read production stats
result = instance.eval("game.player.force.get_item_production_statistics(...)")
```

For our validation: `translator/to_fle.py` converts our Mini-Factorio JSON layout to Lua placement commands, runs them via `instance.eval()`, advances the game clock, and reads green science production.

## Not relevant to us

The REPL / LLM agent interaction model (agents write Python/Lua code) is for research benchmarks. We don't use it — we call the API directly.

## Validation plan

1. Start Factorio server via Docker.
2. Load our top-K layouts from `policy_0` and `policy_final`.
3. For each layout: translate JSON → Lua placement commands → `instance.eval()` → run 10 game-minutes → read production.
4. Compare to our simulator's predicted rate.
5. Report Pearson r and MAPE. Ship gates: r ≥ 0.9, MAPE ≤ 20%.
