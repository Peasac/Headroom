"""Results manifest for a release snapshot: what every raw log holds, with a sha256 per file, so a
reader can check that the logs in a release are the ones the write-ups were computed from."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from headroom.runner.log import read_jsonl, read_sessions
from headroom.schemas.run import Run

MANIFEST = "MANIFEST.json"


def build_manifest(results: Path) -> dict:
    files, experiments = [], {}
    for path in sorted(p for p in results.rglob("*") if p.is_file() and p.name != MANIFEST):
        data = path.read_bytes()
        files.append({
            "path": path.relative_to(results).as_posix(),
            "bytes": len(data),
            "lines": data.count(b"\n"),
            "sha256": hashlib.sha256(data).hexdigest(),
        })
    for exp_dir in sorted(p for p in results.iterdir() if p.is_dir()):
        runs_path = exp_dir / "runs.jsonl"
        runs, bad = read_jsonl(runs_path, Run) if runs_path.exists() else ([], [])
        sessions = []
        for s in read_sessions(results, exp_dir.name):
            mine = [r for r in runs if r.session_id == s.session_id]
            sessions.append({
                "session_id": s.session_id,
                "started_at": s.started_at.isoformat(),
                "order_seed": s.order_seed,
                "planned_runs": len(s.plan),
                "logged_runs": len(mine),
                "git_commit": c if mine and isinstance(c := mine[0].provenance.git_commit, str) else None,
                "arms": sorted({p.arm_id for p in s.plan}),
                "conditions": sorted({p.condition_id for p in s.plan}),
                "tasks": len({p.task_id for p in s.plan}),
            })
        experiments[exp_dir.name] = {"runs": len(runs), "unreadable_lines": len(bad), "sessions": sessions}
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "note": "Raw logs are append-only; a file's sha256 identifies this snapshot of it.",
        "experiments": experiments,
        "files": files,
    }


def write_manifest(results: Path) -> Path:
    path = results / MANIFEST
    path.write_text(json.dumps(build_manifest(results), indent=1) + "\n", encoding="utf-8")
    return path
