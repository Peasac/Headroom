"""File tasks.

Tiers are set by task structure, not by model results:
  1: one call does it, everything needed is in the prompt.
  2: read a named file, then write something that depends on its content.
  3: find the target first (list/read several), or a tempting first call is valid but wrong.
"""

from headroom.schemas.env import EnvState, Event
from headroom.schemas.task import Count, FileAbsent, FileEquals, Preserved, Task, Unchanged

FILE_TOOLS = ("read_file", "write_file", "rename_file", "list_files")
NO_SIDE_EFFECTS = (Unchanged(scope="notes"), Unchanged(scope="events"))

_PLAN = "Q1 plan\n- ship v2 importer\n- hire one designer\n"
_BASE_YAML = "port: 8080\nlog_level: info\n"
_SETTINGS = "[ui]\ntheme=light\nfont_size=14\nlanguage=en\n"
_DRAFTS = {
    "drafts/a.md": "Team offsite ideas: hiking, cooking class.\n",
    "drafts/b.md": "Q4 budget: hold travel spend flat, add one contractor.\n",
    "drafts/c.md": "Q3 budget review notes: travel was 8% over.\n",
}
_LOGS = {
    "logs/2026-11-28.log": "boot ok\n",
    "logs/2026-11-30.log": "boot ok\ndisk 91%\n",
    "logs/2026-11-29.log": "boot ok\nbackup done\n",
}
_REPORT = "Q3 report v1 (signed off by finance)\n"

TASKS = [
    # ---- tier 1 ----
    Task(
        task_id="files.rename_01",
        template_id="files.rename",
        split="tune",
        tier=1,
        prompt="Move drafts/plan.md to archive/plan.md.",
        tools=FILE_TOOLS,
        initial_state=EnvState(files={"drafts/plan.md": _PLAN, "readme.txt": "Team folder."}),
        checks=(
            FileAbsent(path="drafts/plan.md"),
            FileEquals(path="archive/plan.md", content=_PLAN),
            Count(scope="files", n=2),
            *NO_SIDE_EFFECTS,
        ),
        max_steps=4,
    ),
    Task(
        task_id="files.create_01",
        template_id="files.create",
        split="tune",
        tier=1,
        prompt="Create a file named todo/today.txt containing exactly this line: call the bank",
        tools=FILE_TOOLS,
        initial_state=EnvState(files={"todo/yesterday.txt": "buy stamps\n"}),
        checks=(
            FileEquals(path="todo/today.txt", content="call the bank"),
            Preserved(scope="files"),
            Count(scope="files", n=2),
            *NO_SIDE_EFFECTS,
        ),
        max_steps=4,
    ),
    Task(
        task_id="files.overwrite_01",
        template_id="files.overwrite",
        split="tune",
        tier=1,
        prompt="Replace the entire contents of status.txt with exactly: deployed",
        tools=FILE_TOOLS,
        initial_state=EnvState(files={"status.txt": "pending\nchecked by Ana\n", "owners.txt": "Ana, Tom\n"}),
        checks=(
            FileEquals(path="status.txt", content="deployed"),
            FileEquals(path="owners.txt", content="Ana, Tom\n"),
            Count(scope="files", n=2),
            *NO_SIDE_EFFECTS,
        ),
        max_steps=4,
    ),
    # ---- tier 2 ----
    Task(
        task_id="files.append_01",
        template_id="files.append",
        split="tune",
        tier=2,
        prompt="Add 'milk' as a new last line of shopping.txt. Keep the existing items.",
        tools=FILE_TOOLS,
        initial_state=EnvState(
            files={"shopping.txt": "eggs\nbread\n"},
            events=(Event.model_validate({"title": "Market", "start": "2026-12-05T10:00", "end": "2026-12-05T11:00"}),),
        ),
        checks=(
            FileEquals(path="shopping.txt", content="eggs\nbread\nmilk\n"),
            Count(scope="files", n=1),
            *NO_SIDE_EFFECTS,
        ),
        max_steps=6,
    ),
    Task(
        task_id="files.copy_01",
        template_id="files.copy",
        split="tune",
        tier=2,
        prompt="Copy config/base.yaml to config/prod.yaml. Keep the original file.",
        tools=FILE_TOOLS,
        initial_state=EnvState(files={"config/base.yaml": _BASE_YAML, "config/dev.yaml": "port: 3000\n"}),
        checks=(
            FileEquals(path="config/prod.yaml", content=_BASE_YAML),
            Preserved(scope="files"),
            Count(scope="files", n=3),
            *NO_SIDE_EFFECTS,
        ),
        max_steps=6,
    ),
    Task(
        task_id="files.edit_01",
        template_id="files.edit",
        split="tune",
        tier=2,
        prompt="In settings.ini, change the line 'theme=light' to 'theme=dark'. Leave every other line unchanged.",
        tools=FILE_TOOLS,
        initial_state=EnvState(files={"settings.ini": _SETTINGS}),
        checks=(
            FileEquals(path="settings.ini", content=_SETTINGS.replace("theme=light", "theme=dark")),
            Count(scope="files", n=1),
            *NO_SIDE_EFFECTS,
        ),
        max_steps=6,
    ),
    Task(
        task_id="files.merge_01",
        template_id="files.merge",
        split="tune",
        tier=2,
        prompt="Create all.txt containing the lines of a.txt followed by the lines of b.txt. Keep a.txt and b.txt.",
        tools=FILE_TOOLS,
        initial_state=EnvState(files={"a.txt": "alpha\nbeta\n", "b.txt": "gamma\n"}),
        checks=(
            FileEquals(path="all.txt", content="alpha\nbeta\ngamma\n"),
            Preserved(scope="files"),
            Count(scope="files", n=3),
            *NO_SIDE_EFFECTS,
        ),
        max_steps=6,
    ),
    # ---- tier 3 ----
    Task(
        task_id="files.find_and_move_01",
        template_id="files.find_and_move",
        split="tune",
        tier=3,
        prompt="One of the files in drafts/ mentions the Q4 budget. Move that file into final/, keeping its file name.",
        tools=FILE_TOOLS,
        initial_state=EnvState(files={**_DRAFTS, "readme.txt": "Drafts go in drafts/, finished work in final/.\n"}),
        checks=(
            FileEquals(path="final/b.md", content=_DRAFTS["drafts/b.md"]),
            FileAbsent(path="drafts/b.md"),
            FileEquals(path="drafts/a.md", content=_DRAFTS["drafts/a.md"]),
            FileEquals(path="drafts/c.md", content=_DRAFTS["drafts/c.md"]),
            Count(scope="files", n=4),
            *NO_SIDE_EFFECTS,
        ),
        max_steps=10,
    ),
    Task(
        task_id="files.latest_log_01",
        template_id="files.latest_log",
        split="tune",
        tier=3,
        prompt="Append the line 'reviewed' to the end of the most recent log in logs/. Log files are named by date.",
        tools=FILE_TOOLS,
        initial_state=EnvState(files=dict(_LOGS)),
        checks=(
            FileEquals(path="logs/2026-11-30.log", content=_LOGS["logs/2026-11-30.log"] + "reviewed\n"),
            FileEquals(path="logs/2026-11-28.log", content=_LOGS["logs/2026-11-28.log"]),
            FileEquals(path="logs/2026-11-29.log", content=_LOGS["logs/2026-11-29.log"]),
            Count(scope="files", n=3),
            *NO_SIDE_EFFECTS,
        ),
        max_steps=10,
    ),
    Task(
        task_id="files.safe_write_01",
        template_id="files.safe_write",
        split="tune",
        tier=3,
        prompt=(
            "Save the text 'Final numbers attached.' as archive/report.md. If that file already exists, "
            "do not overwrite it: save the text as archive/report-2.md instead."
        ),
        tools=FILE_TOOLS,
        initial_state=EnvState(files={"archive/report.md": _REPORT, "archive/index.txt": "report.md\n"}),
        checks=(
            FileEquals(path="archive/report.md", content=_REPORT),
            FileEquals(path="archive/report-2.md", content="Final numbers attached."),
            Count(scope="files", n=3),
            *NO_SIDE_EFFECTS,
        ),
        max_steps=10,
    ),
]
