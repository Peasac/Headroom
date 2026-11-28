# Headroom

Goal: test whether an agent that adapts its configuration to live laptop conditions beats the best fixed configuration. Read docs/PROJECT.md first.

## Rules
- Python 3.11+, uv, pydantic, pytest. No paid APIs; local models only.
- Every experiment is paired: same task, seed and condition across controllers.
- Graders check final environment state, never the model's wording.
- Mark unavailable sensors as unavailable. Never estimate energy from tokens.
- A controller must never disable validation on state-changing tool calls.
- Raw run logs in results/ are append-only and committed in release snapshots. Thresholds live in experiments/.
- Keep the config space small: 2 models, 4 modes, 2 context caps, 2 retry limits.

## Current stage
Week 6 release (2026-11-28): Gate A failed (emulated and real workloads), Gate C passed; held-out tasks are spent. Raw logs are committed with results/MANIFEST.json (regenerate with `uv run headroom manifest` before each release commit). BFCL slice done (86.2% Ollama, 85.5% llama.cpp). Next: the power-saving-mode condition (queued, needs its own pre-registered rule).

## Done means
A test passes, a command reproduces the result, and failures are inspected.