# Factorio Learning Environment (FLE) Notes

Repo: https://github.com/JackHopkins/factorio-learning-environment

## Key facts
- Targets Factorio 2.0.73+ (NOT 1.1 — recipe differences possible)
- Requires Docker + a Factorio game license (for actual gameplay)
- Interaction model: REPL (agents write Python code to manipulate the game)
- NOT blueprint import-based — cannot simply import a JSON layout and measure throughput

## Implications for our validation plan
Our plan was: translate Mini-Factorio JSON → Factorio blueprint → run in FLE → measure production.
FLE's REPL model makes this non-trivial. We would need to:
1. Write Python code to place entities programmatically via FLE's API
2. Advance the game clock
3. Read production statistics

This is viable but more complex than "import blueprint and read output."

## Resolved
- License: only needed for graphics/GUI. FLE runs the dedicated server headlessly — free.
- Docker: standard infra, no issue.
- Recipe data updated to 2.0.77 (factoriolab public/data/2.0/). Green science recipe unchanged from 1.1.
- FLE is confirmed as validation tool.
