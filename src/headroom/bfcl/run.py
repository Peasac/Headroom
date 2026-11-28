"""BFCL slice: run the pinned model on BFCL v3 single-call categories through the harness's own
backend adapter (same pinned weights, sampling, seed and thinking setting as the experiments), score
each reply with the AST checker, and append one record per item to results/<bfcl_id>/runs.jsonl.

The data is downloaded from a pinned Hugging Face revision and every file is checked by sha256
before use. One model call per item, no agent loop, no tool execution: BFCL's AST categories judge
the call the model would make.
"""

import hashlib
import json
import time
import tomllib
import uuid
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import Field

from headroom.backends import Backend, is_warm, warm_up
from headroom.bfcl.checker import check_call
from headroom.bfcl.convert import to_tool
from headroom.experiments import load_experiment
from headroom.runner.awake import unbiased_s
from headroom.runner.log import AppendOnlyLog, read_jsonl
from headroom.schemas.chat import ChatMessage, ToolCall
from headroom.schemas.common import Frozen, Reading, unavailable
from headroom.schemas.config import ExperimentConfig
from headroom.schemas.run import Provenance, QuietCheck
from headroom.stats import median, nearest_rank


class BfclCategory(Frozen):
    name: str
    questions: str
    questions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    answers: str
    answers_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class BfclConfig(Frozen):
    bfcl_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    base_experiment: str
    model_id: str
    context_cap: int
    data_repo: str
    data_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    data_license: str
    data_dir: str
    categories: list[BfclCategory] = Field(min_length=1)


class BfclSession(Frozen):
    schema_version: int = 1
    bfcl_id: str
    session_id: str
    started_at: datetime
    config: BfclConfig
    experiment: ExperimentConfig
    backend_version: str
    provenance: Provenance
    quiet: QuietCheck
    items: int


class BfclRecord(Frozen):
    schema_version: int = 1
    bfcl_id: str
    session_id: str
    item_id: str
    category: str
    model_id: str
    started_at: datetime
    latency_s: float
    load_s: Reading
    cold_start: bool | None
    prompt_tokens: Reading
    completion_tokens: Reading
    done_reason: str | None
    content: str
    tool_calls: tuple[ToolCall, ...]
    error: str | None = Field(default=None, description="Backend error; the item counts as failed.")
    suspended_s: float | None = Field(
        default=None, description="Time the machine was asleep during the call (None: not measured, "
        "records before 2026-11-28 morning). Latency of an item with suspended_s > 1 is not used.",
    )
    valid: bool
    error_type: str | None
    error_detail: str | None


def load_bfcl(path: str | Path) -> BfclConfig:
    with open(path, "rb") as f:
        return BfclConfig.model_validate(tomllib.load(f))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(cfg: BfclConfig, root: Path = Path(".")) -> list[Path]:
    """Download the pinned files (skipping ones already present with the right hash). Raises on a
    hash mismatch, so a changed upstream file is never used."""
    out = []
    base = f"https://huggingface.co/datasets/{cfg.data_repo}/resolve/{cfg.data_revision}/"
    with httpx.Client(follow_redirects=True, timeout=60) as client:
        for c in cfg.categories:
            for rel, digest in ((c.questions, c.questions_sha256), (c.answers, c.answers_sha256)):
                dest = root / cfg.data_dir / rel
                if not (dest.exists() and _sha256(dest) == digest):
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    r = client.get(base + rel)
                    r.raise_for_status()
                    if hashlib.sha256(r.content).hexdigest() != digest:
                        raise ValueError(f"{rel}: sha256 {hashlib.sha256(r.content).hexdigest()} != pinned {digest}")
                    dest.write_bytes(r.content)
                out.append(dest)
    return out


def load_items(cfg: BfclConfig, root: Path = Path(".")) -> list[tuple[str, dict[str, Any], list[dict[str, Any]]]]:
    """(category, item, ground truth) for every item, after checking each file's sha256."""
    items = []
    for c in cfg.categories:
        q, a = root / cfg.data_dir / c.questions, root / cfg.data_dir / c.answers
        for path, digest in ((q, c.questions_sha256), (a, c.answers_sha256)):
            if not path.exists():
                raise FileNotFoundError(f"{path} missing; run `headroom bfcl-fetch` first")
            if _sha256(path) != digest:
                raise ValueError(f"{path}: sha256 does not match the pinned {digest}")
        answers = {d["id"]: d["ground_truth"] for d in map(json.loads, a.read_text(encoding="utf-8").splitlines())}
        for line in q.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            items.append((c.name, item, answers[item["id"]]))
    return items


def messages_of(item: dict[str, Any]) -> list[ChatMessage]:
    turns = item["question"]
    if len(turns) != 1:
        raise ValueError(f"{item['id']}: single-turn categories only")
    return [ChatMessage(role=m["role"], content=m["content"]) for m in turns[0]]


async def run_items(
    backend: Backend,
    cfg: BfclConfig,
    exp: ExperimentConfig,
    items: list[tuple[str, dict[str, Any], list[dict[str, Any]]]],
    *,
    session_id: str,
    log: AppendOnlyLog,
    progress=None,
) -> None:
    spec = exp.model(cfg.model_id)
    await warm_up(backend, spec, exp.sampling, num_ctx=cfg.context_cap)
    for n, (category, item, truth) in enumerate(items, 1):
        tools = [to_tool(f) for f in item["function"]]
        started = datetime.now(UTC)
        t0, u0 = time.perf_counter(), unbiased_s()
        try:
            reply = await backend.chat(
                spec, messages_of(item), sampling=exp.sampling, seed=exp.sampling.seed, tools=tools,
                num_ctx=cfg.context_cap,
            )
        except Exception as e:  # noqa: BLE001 - a backend error is recorded and scored as a failure
            rec = BfclRecord(
                bfcl_id=cfg.bfcl_id, session_id=session_id, item_id=item["id"], category=category,
                model_id=spec.model_id, started_at=started, latency_s=0.0,
                load_s=_unavail(), cold_start=None, prompt_tokens=_unavail(), completion_tokens=_unavail(),
                done_reason=None, content="", tool_calls=(), error=f"{type(e).__name__}: {e}"[:500],
                valid=False, error_type="backend_error", error_detail=str(e)[:500],
            )
        else:
            v = check_call(reply.message.tool_calls, item["function"], truth)
            rec = BfclRecord(
                bfcl_id=cfg.bfcl_id, session_id=session_id, item_id=item["id"], category=category,
                model_id=spec.model_id, started_at=started, latency_s=reply.latency_s, load_s=reply.load_s,
                cold_start=None if (w := is_warm(reply, exp.backend.max_warm_load_s)) is None else not w,
                prompt_tokens=reply.prompt_tokens, completion_tokens=reply.completion_tokens,
                done_reason=reply.done_reason, content=reply.message.content, tool_calls=reply.message.tool_calls,
                valid=v.valid, error_type=v.error_type, error_detail=v.error,
            )
        u1 = unbiased_s()
        if u0 is not None and u1 is not None:
            rec = rec.model_copy(update={"suspended_s": max(0.0, (time.perf_counter() - t0) - (u1 - u0))})
        log.append(rec)
        if progress:
            progress(n, len(items), rec)


def _unavail() -> Reading:
    return unavailable("backend error")


def new_session_id() -> str:
    return f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"


def rescore(records: list[BfclRecord], items) -> tuple[list[BfclRecord], int]:
    """Re-apply the current checker to the logged tool calls (the log itself is never edited).
    Returns the re-scored records and how many verdicts changed from the logged ones."""
    by_id = {item["id"]: (item, truth) for _, item, truth in items}
    out, changed = [], 0
    for r in records:
        if r.error is not None:
            out.append(r)
            continue
        item, truth = by_id[r.item_id]
        v = check_call(r.tool_calls, item["function"], truth)
        changed += v.valid != r.valid
        out.append(r.model_copy(update={"valid": v.valid, "error_type": v.error_type, "error_detail": v.error}))
    return out, changed


SUSPEND_TOLERANCE_S = 1.0
UNMEASURED_MAX_LATENCY_S = 60.0


def timing_ok(r: BfclRecord) -> bool:
    """Latency is usable unless the machine slept during the call. Records from before sleep was
    measured fall back to a latency ceiling (the 2026-11-27 llama.cpp session slept overnight)."""
    if r.suspended_s is not None:
        return r.suspended_s <= SUSPEND_TOLERANCE_S
    return r.latency_s <= UNMEASURED_MAX_LATENCY_S


def report(records: list[BfclRecord]) -> str:
    by_cat: dict[str, list[BfclRecord]] = defaultdict(list)
    for r in records:
        by_cat[r.category].append(r)
    lines = [f"{'category':<14}{'items':>6}{'valid':>7}{'accuracy':>10}{'median s':>10}{'p95 s':>8}{'cold':>6}"
             f"  error types"]
    for cat, rs in sorted(by_cat.items()):
        lat = [r.latency_s for r in rs if r.error is None and timing_ok(r)]
        errs = Counter(r.error_type for r in rs if not r.valid)
        lines.append(
            f"{cat:<14}{len(rs):>6}{sum(r.valid for r in rs):>7}{sum(r.valid for r in rs) / len(rs):>10.1%}"
            f"{median(lat) if lat else float('nan'):>10.2f}{nearest_rank(lat, 0.95) if lat else float('nan'):>8.2f}"
            f"{sum(r.cold_start is True for r in rs):>6}  "
            + ", ".join(f"{k} {v}" for k, v in errs.most_common())
        )
    total = len(records)
    lines.append(f"{'all':<14}{total:>6}{sum(r.valid for r in records):>7}"
                 f"{sum(r.valid for r in records) / total:>10.1%}" if total else "no records")
    dropped = sum(not timing_ok(r) for r in records if r.error is None)
    lines.append("accuracy: share of items whose single function call passes the BFCL AST check "
                 "(name, required and allowed parameters, types, possible values).")
    if dropped:
        lines.append(f"timing: {dropped} item(s) left out of the latency columns because the machine slept "
                     "during the call (their accuracy counts).")
    return "\n".join(lines)


def read_records(results: Path, bfcl_id: str, session_id: str | None = None) -> tuple[list[BfclRecord], str | None]:
    path = results / bfcl_id / "runs.jsonl"
    recs = read_jsonl(path, BfclRecord)[0] if path.exists() else []
    if not recs:
        return [], None
    sid = session_id or recs[-1].session_id
    return [r for r in recs if r.session_id == sid], sid


def base_experiment(cfg: BfclConfig, root: Path = Path(".")) -> ExperimentConfig:
    exp = load_experiment(root / cfg.base_experiment)
    exp.model(cfg.model_id)  # raises if unknown
    if cfg.context_cap > exp.backend.num_ctx:
        raise ValueError(f"context_cap {cfg.context_cap} exceeds the backend num_ctx {exp.backend.num_ctx}")
    return exp

