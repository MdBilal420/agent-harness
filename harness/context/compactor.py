
import json
from datetime import datetime, timezone
from pathlib import Path

from anthropic import Anthropic

from harness.config import (
    MODEL,
    COMPACTION_THRESHOLD,
    COMPACTION_KEEP_RECENT,
)

from harness.spend import (
    get_session_spend,
    record_session_spend,
    SpendLimitExceeded,
)


SUMMARIZATION_MAX_TOKENS = 4096


# Path to the on-disk offload.
def _log_path() -> Path:
    """Return the absolute path to the context log."""
    return (
        Path(__file__).parent.parent.parent
        / ".harness"
        / "context_log.jsonl"
    )


# Load the summarization prompt once at import.
SUMMARIZATION_PROMPT = (
    Path(__file__).parent.parent / "prompts" / "compaction.txt"
).read_text()


# --- Private helpers ---


def _msg_get(msg, field: str, default=None):
    """Read a field from a dict or Pydantic object."""
    if isinstance(msg, dict):
        return msg.get(field, default)
    return getattr(msg, field, default)


def _msg_role(msg) -> str:
    """Return the message role."""
    return _msg_get(msg, "role", "") or ""


def _msg_content(msg):
    """Return message content."""
    return _msg_get(msg, "content", "") or ""


def _block_type(block) -> str:
    """Return the type of an Anthropic content block."""
    return _msg_get(block, "type", "") or ""


def _is_tool_result_message(msg) -> bool:
    """Identify Anthropic user messages containing tool results only.

    Tool results use role='user' in Anthropic's API, but they are not
    actual user turns and should not count toward recent exchanges.
    """
    if _msg_role(msg) != "user":
        return False

    content = _msg_content(msg)

    return (
        isinstance(content, list)
        and len(content) > 0
        and all(_block_type(block) == "tool_result" for block in content)
    )


def _is_user_turn(msg) -> bool:
    """Identify actual user messages, excluding tool-result messages."""
    return (
        _msg_role(msg) == "user"
        and not _is_tool_result_message(msg)
    )


def _content_to_text(content) -> str:
    """Convert Anthropic content blocks into readable text."""
    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        return str(content) if content else ""

    lines = []

    for block in content:
        block_type = _block_type(block)

        if block_type == "text":
            lines.append(_msg_get(block, "text", ""))

        elif block_type == "tool_use":
            name = _msg_get(block, "name", "")
            tool_input = _msg_get(block, "input", {})
            lines.append(
                f"TOOL CALL: {name}({json.dumps(tool_input, default=str)})"
            )

        elif block_type == "tool_result":
            tool_id = _msg_get(block, "tool_use_id", "")
            result = _msg_get(block, "content", "")
            is_error = _msg_get(block, "is_error", False)

            result_text = _content_to_text(result)

            prefix = "TOOL ERROR" if is_error else "TOOL RESULT"

            lines.append(
                f"{prefix} [{tool_id}]: {result_text}"
            )

        else:
            # Preserve other block types in a readable form.
            lines.append(
                f"[{block_type}]: {str(block)}"
            )

    return "\n".join(lines)


def _split_recent(
    conversation: list,
    keep_recent_exchanges: int,
) -> tuple[list, list]:
    """Split conversation into (recent, older).

    Count actual user turns backward, excluding Anthropic tool-result
    messages. Preserve all messages belonging to the selected turns.
    """
    if keep_recent_exchanges < 1:
        raise ValueError("keep_recent_exchanges must be at least 1")

    user_indices = [
        i
        for i, msg in enumerate(conversation)
        if _is_user_turn(msg)
    ]

    if len(user_indices) <= keep_recent_exchanges:
        return conversation, []

    split_index = user_indices[-keep_recent_exchanges]

    return conversation[split_index:], conversation[:split_index]


def _format_messages_for_summary(messages: list) -> str:
    """Format conversation history for the summarization model.

    Handles Anthropic text, tool_use, and tool_result blocks.
    """
    lines = []

    for msg in messages:
        role = _msg_role(msg).upper()
        content = _msg_content(msg)

        if role == "USER" and _is_tool_result_message(msg):
            lines.append(
                f"USER (tool results):\n{_content_to_text(content)}"
            )

        elif role == "ASSISTANT":
            formatted = _content_to_text(content)

            if formatted:
                lines.append(f"ASSISTANT:\n{formatted}")

        elif role == "USER":
            lines.append(
                f"USER:\n{_content_to_text(content)}"
            )

        else:
            lines.append(
                f"{role}:\n{_content_to_text(content)}"
            )

    return "\n\n".join(lines)


def _to_jsonable(value):
    """Convert Anthropic SDK objects into JSON-serializable values."""
    if isinstance(value, dict):
        return {
            key: _to_jsonable(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]

    if hasattr(value, "model_dump"):
        return _to_jsonable(
            value.model_dump(exclude_none=True)
        )

    return value


def _offload(messages: list, summary: str) -> None:
    """Append pre-compaction messages and summary to the context log."""
    log_path = _log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "message_count": len(messages),
        "summary": summary,
        "messages": [
            _to_jsonable(msg)
            for msg in messages
        ],
    }

    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


# --- The Compactor class ---


class Compactor:
    """Owns the compaction lifecycle for an Anthropic agent."""

    def __init__(self, client: Anthropic):
        self._client = client

        self._last_token_count: int = 0

    def record_token_usage(self, input_tokens: int) -> None:
        """Record the latest Anthropic input token usage."""
        self._last_token_count = input_tokens

    def record_response_usage(self, response) -> None:
        """Record total input usage, including cached input tokens."""
        usage = response.usage

        total = (
            usage.input_tokens
            + getattr(usage, "cache_creation_input_tokens", 0)
            + getattr(usage, "cache_read_input_tokens", 0)
        )

        self._last_token_count = total

    def get_last_token_count(self) -> int:
        """Return the latest recorded input token count."""
        return self._last_token_count

    def should_compact(self) -> bool:
        """Check whether the compaction threshold has been reached."""
        return self._last_token_count >= COMPACTION_THRESHOLD

    def approximate_char_count(self, messages: list) -> int:
        """Estimate token count using character length divided by four."""
        return sum(
            len(_content_to_text(_msg_content(msg)))
            for msg in messages
        ) // 4

    def compact(self, messages: list) -> list | None:
        """Run one compaction pass.

        Returns:
            Rebuilt conversation history if compaction succeeds.
            None if no compaction was performed.
        """

        # Step 1: Split conversation into older and recent turns.
        recent, older = _split_recent(
            messages,
            COMPACTION_KEEP_RECENT,
        )

        if not older:
            return None

        # Step 2: Format older messages for summarization.
        older_as_text = _format_messages_for_summary(older)

        summarization_messages = [
            {
                "role": "user",
                "content": older_as_text,
            }
        ]

        # Step 3: summarize older conversation history.
        tracker = get_session_spend()
        if tracker is not None:
            tracker.raise_if_exceeded()

        response = self._client.messages.create(
            model=MODEL,
            max_tokens=SUMMARIZATION_MAX_TOKENS,
            system=SUMMARIZATION_PROMPT,
            messages=summarization_messages,
        )
        record_session_spend(response, model=MODEL)
        if tracker is not None and tracker.exceeded:
            raise SpendLimitExceeded(tracker)

        summary_text = "\n".join(
            block.text
            for block in response.content
            if block.type == "text"
        ).strip()

        if not summary_text:
            print(
                "[compactor] Summarization returned empty text. "
                "Skipping compaction."
            )
            return None

        # Step 4: Offload original conversation
        try:
            _offload(messages, summary_text)
        except Exception as e:
            print(
                f"[compactor] Offload failed: "
                f"{type(e).__name__}: {e}. Continuing anyway."
            )

        # Step 5: Rebuild conversation.
        summary_message = {
            "role": "assistant",
            "content": (
                "## Compacted Conversation Summary\n\n"
                + summary_text
            ),
        }

        rebuilt = [summary_message] + recent

        # Step 6: Reset token count.
        self._last_token_count = 0

        return rebuilt