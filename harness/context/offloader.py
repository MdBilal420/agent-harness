"Tool Offloader"

from datetime import datetime, timezone
from pathlib import Path

from harness.config import (
    OFFLOAD_THRESHOLD_TOKENS,
    OFFLOAD_HEAD_TOKENS,
    OFFLOAD_TAIL_TOKENS,
    WORKSPACE_PATH,
)


CHARS_PER_TOKEN = 4

OFFLOAD_SUBDIR = ".tool_outputs"

NEVER_OFFLOAD_TOOLS = {"read"}


def _offload_dir() -> Path:
    return WORKSPACE_PATH / OFFLOAD_SUBDIR

def _tokens_of(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def _make_offload_path(tool_name: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return _offload_dir() / f"{tool_name}_{timestamp}.txt"

def _workspace_relative(path: Path) -> Path:
    return str(path.relative_to(WORKSPACE_PATH))


def maybe_offload(tool_name: str, result: str) -> str:
    if tool_name in NEVER_OFFLOAD_TOOLS:
        return result

    total_tokens = _tokens_of(result)
    if total_tokens < OFFLOAD_THRESHOLD_TOKENS:
        return result

    head_chars = OFFLOAD_HEAD_TOKENS * CHARS_PER_TOKEN
    tail_chars = OFFLOAD_TAIL_TOKENS * CHARS_PER_TOKEN
    head = result[:head_chars]
    tail = result[-tail_chars:]

    try:
        offload_path = _make_offload_path(tool_name)
        offload_path.parent.mkdir(parents=True, exist_ok=True)
        offload_path.write_text(result)
    except Exception as e:
        print(f"Error offloading tool {tool_name}: {e}")
        return result

    relative_path = _workspace_relative(offload_path)
    offloaded_tokens = total_tokens - _tokens_of(head) - _tokens_of(tail)
    print(f"Offloaded {offloaded_tokens} tokens from {tool_name} to {relative_path}")

    marker = (
        f"\n... [TRUNCATED TO {total_tokens} TOKENS] ..."
        f"\n... [OFFLOADED TO {relative_path}] ..."
        f"Call read_file() to retrieve the full output."
    )
    return head + marker + tail
