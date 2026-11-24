# AdaptiveEdge: Do laptop agents need live adaptation? Project analysis

Nov 24, 2026 · @Ayaan Tabrez

## Summary

The project tests one claim: on a laptop, an agent that changes its configuration as device conditions change completes more tasks per second and per GB of memory than the best fixed configuration. The benchmark, harness and report exist only to test that claim.

- **Novelty verdict:** a plain laptop agent benchmark is not novel. LocalScore, MLPerf Client, Bench360, AA-AgentPerf-Local, LegitOnEdge and AgBench already exist. The open part is the adaptation question, which I did not find tested in the abstracts I read.
- **Decisive first experiment (week 2):** measure the oracle gap, meaning how much better the best configuration per condition is than the single best fixed configuration. If the gap is under 10% (proposed threshold), adaptation has little to win and the project becomes a benchmark and tool paper.
- **Deliverables:** an open benchmark with condition injection, a small controller, a results report and a pip-installable CLI.
- **Cost:** one laptop, one free local inference backend, no paid APIs. About 6 weeks for one person.
- **To confirm:** your laptop's OS and whether it has a discrete GPU, since that decides which telemetry is available.

## Prior art

Existing tools measure how fast or how reliably a fixed setup runs on given hardware. None that I found tests whether changing the setup mid-run helps. This table rests on abstracts and search snippets, not full papers; the check below should confirm it.

| Work | What it measures | What it leaves open |
| --- | --- | --- |
| [LocalScore](https://github.com/cjpais/LocalScore) | Prompt speed, generation speed and time to first token on your hardware | No agent tasks, no task success |
| [MLPerf Client](https://mlcommons.org/benchmarks/client) | LLM performance on PCs, laptops to workstations | Single-turn LLM tests, not agents |
| [Bench360](https://arxiv.org/html/2511.16682v1) | Latency, throughput, energy per query, cold start, plus task quality | Fixed deployment configurations |
| [AA-AgentPerf-Local](https://artificialanalysis.ai/es/articles/aa-agentperf-local) | Replays recorded agent trajectories on laptops for inference speed | Skips tool execution by default, so no task success |
| [LegitOnEdge](https://neurips.cc/virtual/2026/171187) | On-device reliability of small agent models; one model scored 0.54 on one device and 0.04 on another | Jetson and DGX Spark, not laptops; no adaptation |
| [AgBench](https://arxiv.org/pdf/2609.38652) | 60 agent tasks in eight categories on personal AI devices | Characterization; I did not see adaptation in the snippet |
| [Telemetry study of on-device computer-using agents](https://neurips.cc/virtual/2026/171317) | Per-step latency, token growth and VRAM across hardware tiers | Builds one system (AgOS) rather than testing adaptation in general |
| [Agent.xpu](https://arxiv.org/pdf/2506.24045v2) | Scheduling of agent flows on shared on-device hardware to cut latency | Targets throughput and latency, not task success under changing load |

**Check before day 1 (about 1 hour):** search for work that changes an agent's model, mode or context size while it runs, based on device state. If something close exists, narrow the claim to laptops, or to one control such as when to escalate to a larger model.

The benchmark side is a measurement tool. The contribution has to be the adaptation result.

## Hypotheses and gates

Four questions, asked in order, each with a stop condition. Thresholds are proposed project targets, not predicted results; fix them before the final evaluation.

| Question | Test | Proposed threshold |
| --- | --- | --- |
| RQ1: Does the best configuration change with conditions? | Compare the best fixed configuration against the best configuration per condition (the oracle) | Oracle gap of at least 10% in success per second or in p95 latency, in at least 2 of the conditions |
| RQ2: Does a simple rule capture the gap? | Threshold rules on free memory, CPU load and power state | Rule recovers at least 50% of the oracle gap |
| RQ3: Does a learned policy beat the rule? | Decision tree or lookup trained on logged runs, evaluated on held-out task templates | Gain with a confidence interval that excludes zero |
| RQ4: Does adapting mid-task beat choosing once at the start? | Re-select on timeout, memory spike or load change | Positive gain on the tasks that run longest |

**Gate A (end of week 2):** if RQ1 fails, the best fixed setting is the same everywhere. Stop the adaptation claim and ship the benchmark as a tool with a measured finding that fixed settings suffice.

**Gate B (week 4):** if a rule matches a learned policy, report the rule as the result. A simple answer is still a result.

**Gate C (week 5):** repeat the winning comparison on held-out tasks, and on a second device if you can borrow one.

## System design

The harness runs the same tasks under controlled conditions, records what happens, and lets a controller choose the configuration. Start with one backend and one laptop.

| Component | Job | Notes |
| --- | --- | --- |
| Task suite | About 40 tool-use tasks in a mock environment (files, notes, calendar, calculator) in three difficulty tiers, plus a small BFCL subset | Deterministic graders that check final state, not the model's wording |
| Backend adapter | Talks to a local model server over an OpenAI-compatible API | Ollama first; llama.cpp server as a second adapter. Pin `keep_alive` (model stays loaded, so cold start never leaks into latency) and `num_ctx` (set explicitly, never Ollama's default) |
| Condition injector | Applies a repeatable load before and during a run | See conditions below |
| Telemetry | Samples memory, CPU, battery state and, if available, GPU | Mark unavailable sensors as unavailable; never estimate joules from token counts |
| Config space | Model size, harness mode, context cap, retry limit | Keep it small: 2 models, 4 modes, 2 context caps, 2 retry limits |
| Controllers | Pick a configuration | C0 fixed best, C1 rule, C2 learned policy, C3 mid-task re-selection |
| Grader and reporter | Scores runs, writes tables and plots | One command regenerates every result |

**Harness modes:** direct tool call; tool call with schema validation; plan then act; plan, act and verify.

**Conditions to inject:**

| Condition | How | Real or emulated |
| --- | --- | --- |
| Baseline | Plugged in, idle machine | Real |
| Memory pressure | A ballast process holds a fixed amount of RAM | Emulated |
| CPU contention | N busy worker processes | Emulated |
| Reduced compute | Fewer threads, or the OS low-power mode | Emulated, then real |
| Battery and thermal | Unplugged runs; a 10-minute sustained load before the test | Real, a small manual set only |

**Safety rule:** a controller may never turn off validation for a tool call that changes state. Speed is optimized only inside that constraint.

## Experimental design

Every comparison is paired: the same task, seed and condition run under each controller, so differences come from the controller alone.

**Models:** one 3B-class and one 7B to 8B instruct model with tool-calling support, both 4-bit quantized, pinned by file hash. Add a second family only after the first results are in.

**Current working model (2026-11-24):** qwen3.5:4b, Q4_K_M, thinking off. llama3.2:3b is dropped from the config space; see the decision log.

**Baselines**

| ID | Baseline | Purpose |
| --- | --- | --- |
| B0 | Best single fixed configuration | The number adaptation must beat |
| B1 | Always-largest, always-full-verification | Quality ceiling, usually slow |
| B2 | Always-smallest, no verification | Speed floor |
| B3 | Oracle: best configuration per condition | Upper bound; defines the oracle gap |
| B4 | Rule controller | Interpretable adaptive baseline |
| B5 | Learned policy | The candidate result |

**Metrics**

| Family | Metrics |
| --- | --- |
| Quality | Task success rate; tool-call accuracy; policy violations |
| Speed | Time to success p50 and p95; time to first token |
| Resources | Peak RAM and VRAM; CPU use; battery percent per successful task where measurable |
| Stability | Timeouts; out-of-memory kills; controller switches per task |
| Efficiency | Successes per minute; successes per GB of peak memory |

**Statistics**

- At least 5 seeds per task and condition, fixed sampling settings.
- Bootstrap confidence intervals on paired differences.
- Split tasks by template into a tuning half and a held-out half; freeze the held-out half until the final run.
- Report failed runs and unsupported conditions, not only wins.

**Reproducibility:** pin model hashes, backend version, quantization, OS power mode and background apps. Log all of it with every run.

## Implementation plan

**Stack:** Python 3.11+ managed with uv; pydantic for configs and results; httpx with asyncio for the backend; psutil for telemetry, pynvml only if you have an NVIDIA GPU; Typer for the CLI; pytest; pandas and numpy for analysis; Optuna optional for the learned policy. Everything runs offline once the models are downloaded.

**Repository layout**

```
adaptiveedge/
  tasks/        task definitions, mock tools, graders
  backends/     Ollama and llama.cpp adapters
  conditions/   ballast memory, CPU load, thread limits
  telemetry/    samplers with explicit unavailable values
  controllers/  fixed, rule, learned, mid-task
  runner/       paired experiment runner and run logs
  analysis/     tables, plots, bootstrap statistics
  cli.py
experiments/    configs and pre-registered thresholds
results/        raw run logs (never edited by hand)
```

**Six-week plan**

1. **Week 1:** one backend, one model, 20 tasks with graders, telemetry logging. Run the baseline condition end to end.
2. **Week 2:** add the three emulated conditions and the fixed configurations. Compute the oracle gap and decide at Gate A.
3. **Week 3:** rule controller, then run it against B0 and B3.
4. **Week 4:** learned policy and mid-task re-selection; ablations for each telemetry signal. Gate B.
5. **Week 5:** held-out run, small battery and thermal set, second device if available. Gate C.
6. **Week 6:** write-up, plots, CLI packaging, public release of tasks, configs and raw logs.

**First two weeks, day by day**

1. Day 1: repository, config schema, backend adapter returning a model reply and its latency.
2. Day 2: three mock tools and five tasks with state-based graders.
3. Day 3: telemetry sampler writing peak memory and CPU per run.
4. Day 4: grow to 20 tasks; run each 5 times on the baseline.
5. Day 5: review every failed run. Separate model errors from grader bugs.
6. Days 6 to 7: memory-pressure and CPU-contention injectors, checked against telemetry.
7. Days 8 to 9: the fixed configurations (modes, context caps) across all conditions.
8. Day 10: compute best fixed versus per-condition oracle. Write down the gap and the Gate A decision.

## Risks and what a negative result gives you

The most likely outcome is that a simple rule is enough, or that the best fixed setting never changes. Both are publishable findings, and the benchmark stays useful either way.

| Risk | Why it matters | Mitigation |
| --- | --- | --- |
| Best fixed setting never changes | Adaptation has nothing to win | Gate A in week 2; pivot to a benchmark and tool paper |
| Emulated load differs from real use | Ballast and busy threads are not a real browser and video call | Add a small set of real background workloads; label which results are emulated |
| Laptop noise | Thermal state and background apps change timings | Fix power mode, close apps, randomize run order, repeat runs, log temperature if available |
| Small tool-calling models fail too often | Success near zero hides any effect | Pick models by a quick tool-call pre-test; keep tasks within their reach |
| Task graders are wrong | Bad grades look like model errors | Review every failure in week 1; keep graders deterministic |
| Related work appears | Someone may publish the adaptation test first | Do the 1-hour search now and again before writing up; release the benchmark early |
| Energy claims | Joules cannot be inferred from token counts | Report battery percent only where measured; otherwise omit energy |

**If the result is negative:** the finding is that fixed settings suffice on laptops, with numbers per condition. That tells builders not to invest in complex controllers.

## Decision log

**2026-11-24: Ollama is the first backend, llama.cpp server the second.** Ollama was already installed and its native API pins `keep_alive`, `num_ctx` and `think` per call.

**2026-11-24: llama3.2:3b dropped from the config space; qwen3.5:4b (thinking off) is the working model.**

- *What failed:* after a tool result, llama3.2:3b wrote tool calls as plain text, naming tools that do not exist (`get_event`, `find_next_available_time_slot`). Ollama did not parse them, so they were never executed. This happened in 35 of 35 such calls in week 1, every one right after a tool result.
- *Cause:* the Ollama chat template for llama3.2 adds the tool list and the call-format instructions only when the last message is a user message. After a tool result the prompt contains no tools at all. We confirmed this by rebuilding the prompt from the template and getting the same output from raw generation.
- *Not a harness bug:* our messages match Ollama's documented format: the assistant's tool-call message stays in history, tool results go back as `{"role": "tool", "tool_name", "content"}`, and tools are sent on every call. Tool-call `id` and `index` are now sent back too; the llama template ignores them.
- *Comparison:* on the same harness, tasks, prompts and seed, qwen3.5:4b passed 5/5 tasks cleanly in both direct and validated modes, against 1–2/5 for llama3.2:3b.
- *Consequence:* the 3B-class slot is empty. A replacement must keep tools visible after tool results; check this with a tool-call pre-test.

## Claude Code kickoff

Export this doc as Markdown into an empty repo as `docs/PROJECT.md`, then add the `CLAUDE.md` below. It keeps Claude Code on the claim and away from scope creep.

```markdown
# AdaptiveEdge

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

## Done means
A test passes, a command reproduces the result, and failures are inspected.
```

**First prompts, in order**

1. Read docs/PROJECT.md and CLAUDE.md. Propose the repository skeleton and the pydantic schemas for task, run, condition and result. Do not write code yet.
2. Implement the Ollama backend adapter and a fake backend for tests. Add a test that a run records reply, latency and token counts.
3. Implement three mock tools and five tasks with state-based graders, each with a passing and a failing test.
4. Add the telemetry sampler with explicit unavailable values, and attach peak memory and CPU to every run record.
5. Write the paired runner and a command that runs all tasks 5 times and prints success rate and p95 latency per condition.
6. Add the memory-ballast and CPU-load injectors, and a test that telemetry shows the pressure they apply.
7. Write the analysis script that computes the best fixed configuration, the per-condition oracle and the oracle gap with a bootstrap interval.

Ask Claude Code to stop after prompt 7 and show you the oracle gap before it builds any controller.
