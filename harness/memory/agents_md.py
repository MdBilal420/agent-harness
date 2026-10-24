from harness.tools.filesystem import WORKSPACE_PATH

AGENTS_MD_PATH = WORKSPACE_PATH / "AGENTS.md"


AGENTS_MD_TEMPLATE = """\
# Project Memory

> This file is the agent's durable memory across sessions. The harness
> loads it at the start of every session. Update it whenever you learn
> something worth remembering for future sessions.

## Project context
(What is this project? What does it do? Who is it for?)

## Conventions
(Code style, naming patterns, libraries used, tool preferences.)

## Decisions
(Choices that have been made and the reasoning behind them.)

## Gotchas
(Things to remember about this codebase that might trip up future
sessions — quirks, non-obvious dependencies, common mistakes.)

## Active tasks
(What's currently being worked on. Clear entries when work completes.)
"""


def load_agents_md() -> str:
    if not AGENTS_MD_PATH.exists():
        AGENTS_MD_PATH.write_text(AGENTS_MD_TEMPLATE)
    return AGENTS_MD_PATH.read_text()