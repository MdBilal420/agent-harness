"Planning Support"

from datetime import datetime, timezone
from pathlib import Path


from harness.config import (
    PLAN_FILENAME,
    PLAN_REMINDER_INTERVAL,
    PLAN_INJECTION_MAX_CHARS,
    WORKSPACE_PATH,
)

from harness.tools.registry import tool


VALID_STATUSES = {"todo", "done", "blocked"}

_CHECKBOX = {
    "todo": "[ ]",
    "done": "[x]",
    "blocked": "[ ]",
}


def _plan_path() -> Path:
    return WORKSPACE_PATH / PLAN_FILENAME

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

def _empty_plan(goal: str = "New Task") -> str:
    return (
        f"# Plan: {goal}\n\n"
        f"## Status: in progress\n\n"
        f"## Tasks\n\n"
        f"## Log\n"
        f"- {_now_iso()} — Plan created\n"
    )

def _parse_tasks(plan_text: str) -> list[dict]:

    tasks = []
    in_tasks_section = False
    for line in plan_text.splitlines():
        stripped = line.strip()

        if stripped.startswith("## Tasks"):
            in_tasks_section = True
            continue

        if stripped.startswith("##") and in_tasks_section:
            in_tasks_section = False
            continue

        if not in_tasks_section:
            continue

        if stripped.startswith("- [x] "):
            tasks.append({"text": stripped[6:].strip(), "status": "done", "notes": None})
        elif stripped.startswith("- [ ] "):
            text = stripped[6:]
            status = "blocked" if text.endswith(" (blocked)") else "todo"
            if status == "blocked":
                text = text[:-len(" (blocked)")].strip()
            tasks.append({"text": text.strip(), "status": status, "notes": None})
        elif stripped.startswith("- Notes: ") and tasks:
            tasks[-1]["notes"] = stripped[9:]

    return tasks

def _render_plan(goal: str, tasks: list[dict], log_entries: list[str]) -> str:
    lines = [f"# Plan: {goal}","","## Status: in progress","","## Tasks"]

    for task in tasks:
        marker = _CHECKBOX[task["status"]]
        suffix = f" (blocked)" if task["status"] == "blocked" else ""
        lines.append(f"- {marker} {task['text']}{suffix}")

        if task.get("notes"):
            lines.append(f"- Notes: {task['notes']}")

    lines.extend(["", "## Log"])
    for log_entry in log_entries:
        lines.append(f"- {log_entry}")

    return "\n".join(lines)

def _extract_goal(plan_text: str) -> str:
    for line in plan_text.splitlines():
        if line.startswith("# Plan: "):
            return line[len("# Plan: "):].strip()
    return "New Task"

def _extract_log(plan_text: str) -> list[str]:
    """Extract the log entries from the plan text."""
    log_entries = []
    in_log_section = False
    for line in plan_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## Log"):
            in_log_section = True
            continue
        if stripped.startswith("- ") and in_log_section:
            log_entries.append(stripped[1:].strip())
    return log_entries

def plan_exists() -> bool:
    """Return True if workspace/plan.md exists."""
    return _plan_path().exists()


def has_open_items() -> bool:
    """Check if the plan has any open items."""
    path = _plan_path()
    if not path.exists():
        return False
    existing_text = path.read_text()
    tasks = _parse_tasks(existing_text)
    return any(task["status"] != "done" for task in tasks)

def load_plan_for_injection() -> str | None:
    """Load the plan for injection."""
    path = _plan_path()
    if not path.exists():
        return None
    content = path.read_text()
    if len(content) > PLAN_INJECTION_MAX_CHARS:
        truncated_content = content[:PLAN_INJECTION_MAX_CHARS]
        path.write_text(truncated_content)
        return f"Plan truncated to {PLAN_INJECTION_MAX_CHARS} characters"
    print(f"Plan: {content}")
    return content

def build_reminder_message() -> str:
    """Build the reminder message for an existing plan with open items."""
    return (
        "[Planning reminder] You have open items in plan.md. "
        "If you've made progress on any of them, call `update_plan` now "
        "before continuing."
    )


def build_create_plan_reminder() -> str:
    """Nudge the model to create a plan when none exists yet."""
    return (
        "[Planning reminder] There is no plan.md yet. "
        "If this is a multi-step task, call `update_plan(task, status=\"todo\")` "
        "once per subtask BEFORE writing code or creating files. "
        "Do not skip planning on long-horizon work."
    )


@tool
def update_plan(task: str, status: str, notes: str | None = None) -> str:
    """Create or update tasks in workspace/plan.md for multi-step work.

    Call this on the first turn of any multi-step task to add each subtask
    with status=\"todo\", then again whenever a subtask becomes done or blocked.
    Do not use write_file on plan.md — this tool is the only way to change it.
    status must be one of: todo, done, blocked.
    """
    
    if status not in VALID_STATUSES:
        return (
            f"Invalid status: {status}. Valid statuses are: {VALID_STATUSES}"
        )
    
    path = _plan_path()

    # load the plan
    if path.exists():
        existing_text = path.read_text()
        goal = _extract_goal(existing_text)
        tasks = _parse_tasks(existing_text)
        log_entries = _extract_log(existing_text)
    else:
        goal = task
        tasks = []
        log_entries = [f"{_now_iso()} — Plan created"]
    
    # find or create the task
    matching = next((t for t in tasks if t["text"] == task), None)
    if matching is not None:
        prev_status = matching["status"]
        matching["status"] = status
        if notes is not None:
            matching["notes"] = notes
        log_entries.insert(0, f"{_now_iso()} — Task updated: {task} from {prev_status} to {status}")
        action = f"Updated task: {task} from {prev_status} to {status}"
    else:
        tasks.append({"text": task, "status": status, "notes": notes})
        log_entries.insert(0, f"{_now_iso()} — Task: {task} added as {status}")
        action = f"Created task: {task} as {status}"
    
    # render and write
    path.write_text(_render_plan(goal, tasks, log_entries))
    return f"[Updated Plan] {action}"