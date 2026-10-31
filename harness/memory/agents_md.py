from harness.tools.filesystem import WORKSPACE_PATH
from harness.tools.registry import tool


# The valid category names for remember(). Anything else is rejected.
VALID_CATEGORIES = {
    "project_identity",
    "conventions",
    "decisions",
    "gotchas",
    "active_tasks",
}

# Map from category identifier to the section header string in AGENTS.md.
CATEGORY_HEADERS = {
    "project_identity": "## Project Identity",
    "conventions": "## Conventions",
    "decisions": "## Decisions",
    "gotchas": "## Gotchas",
    "active_tasks": "## Active Tasks",
}

AGENTS_MD_PATH = WORKSPACE_PATH / "AGENTS.md"


AGENTS_MD_TEMPLATE = """\
# Project Memory

> This file is the agent's durable memory across sessions. The harness
> loads it at the start of every session. Updates happen two ways:
> the `remember(category, entry)` tool for in-session writes, and an
> end-of-session consolidation step that reviews the whole conversation
> and updates this file before the sandbox tears down.

## Project Identity
(What is this project? What does it do? Who is it for?)

## Conventions
(Code style, naming patterns, libraries used, tool preferences.)

## Decisions
(Choices that have been made and the reasoning behind them.)

## Gotchas
(Things to remember about this codebase that might trip up future
sessions — quirks, non-obvious dependencies, common mistakes.)

## Active Tasks
(What's currently being worked on. Clear entries when work completes.)
"""


def load_agents_md() -> str:
    if not AGENTS_MD_PATH.exists():
        AGENTS_MD_PATH.write_text(AGENTS_MD_TEMPLATE)
    return AGENTS_MD_PATH.read_text()

def validate_agents_md_structure(agents_md: str) -> bool:
    """
    Validate the structure of the AGENTS.md file.
    """
    return all(header in agents_md for header in CATEGORY_HEADERS.values())

def save_agents_md(agents_md: str) -> None:
    """
    Save the AGENTS.md file.
    """
    tmp_path = AGENTS_MD_PATH.with_suffix(".md.tmp")
    tmp_path.write_text(agents_md)
    tmp_path.replace(AGENTS_MD_PATH)

@tool
def remember(category: str, entry: str) -> str:
    """
    Remember a piece of information.
    """
    if category not in VALID_CATEGORIES:
        return f"Invalid category: {category}"
    
    current_agents_md = load_agents_md()
    header = CATEGORY_HEADERS[category]
    if header not in current_agents_md:
        return f"Header {header} not found in AGENTS.md"
    
    lines = current_agents_md.split("\n")
    new_lines = []
    inserted = False
    for line in lines:
        new_lines.append(line)
        if line == header and not inserted:
            new_lines.append(f"- {entry}")
            inserted = True
    new_agents_md = "\n".join(new_lines)
    save_agents_md(new_agents_md)
    return f"[remember] Entry added to {category}"