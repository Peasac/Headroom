import hashlib
import json

from headroom.runner.manifest import MANIFEST, build_manifest, write_manifest


def test_manifest_hashes_every_log_and_skips_itself(tmp_path):
    exp = tmp_path / "exp1"
    exp.mkdir()
    (exp / "notes.txt").write_bytes(b"a\nb\n")
    first = write_manifest(tmp_path)
    m = build_manifest(tmp_path)  # the manifest written above is not listed
    assert [f["path"] for f in m["files"]] == ["exp1/notes.txt"]
    f = m["files"][0]
    assert f["lines"] == 2 and f["bytes"] == 4 and f["sha256"] == hashlib.sha256(b"a\nb\n").hexdigest()
    assert m["experiments"]["exp1"] == {"runs": 0, "unreadable_lines": 0, "sessions": []}
    assert first.name == MANIFEST and json.loads(first.read_text())["files"] == m["files"]
