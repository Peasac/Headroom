# Headroom

Goal: test whether an agent that adapts its configuration to live laptop conditions beats the best fixed configuration. Read docs/PROJECT.md first.

## Rules
- Python 3.11+, uv, pydantic, pytest. No paid APIs; local models only.
- Every experiment is paired: same task, seed and condition across controllers.
- Graders check final environment state, never the model's wording.
- Mark unavailable sensors as unavailable. Never estimate energy from tokens.
- A controller must never disable validation on state-changing tool calls.
- Raw run logs in results/ are append-only. Thresholds live in experiments/.
- Keep the config space small: 2 models, 4 modes, 2 context caps, 2 retry limits.

## Current stage
Week 1: one backend, one model, 20 tasks, telemetry, baseline condition.

## Commands
- `uv run pytest` — offline tests (mocked HTTP, fake backend).
- `uv run pytest -m live` — against the local Ollama server; checks pinned hashes and a warm call.
- `uv run headroom smoke [--model qwen3.5-4b]` — verify pinned model, warm up, one measured call, print the record. Exit 2 on cold start.
- `uv run headroom run-task --task <id> [--arm direct]` — one run with quiet check + telemetry; writes nothing.
- `uv run headroom run-experiment [--repeats N] [--arm ID ...] [--order-seed N]` — paired runner: all tasks x repeats x conditions x arms, appended to results/<experiment>/runs.jsonl, then the report.
- `uv run headroom report [--session ID | --all]` — report regenerated from the run log.

## Done means
A test passes, a command reproduces the result, and failures are inspected.