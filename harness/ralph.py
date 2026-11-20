"""
Ralph is an outer loop that drives the agent to completion of a goal,
carrying forward a compact summary when the agent stops early.
"""

from harness.config import (
    RALPH_EVAL_MODEL,
    RALPH_EVAL_PROMPT,
    RALPH_CARRY_FORWARD_CHARS,
)

from harness.planning import (
    get_open_tasks,
    get_all_task_texts,
    load_plan_for_injection,
)

from harness.spend import (
    get_session_spend,
    record_session_spend,
    SpendLimitExceeded,
)


def _content_to_text(content) -> str:
    """Extract plain text from Anthropic message content."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", "") or "")
            else:
                if getattr(block, "type", None) == "text":
                    parts.append(getattr(block, "text", "") or "")
        return "\n".join(p for p in parts if p)

    return str(content) if content else ""


def _summarize_recent(messages: list, max_chars: int = 2000) -> str:
    """Summarize the recent assistant messages for goal evaluation."""
    assistant_texts = []

    for msg in reversed(messages):
        role = (
            msg.get("role")
            if isinstance(msg, dict)
            else getattr(msg, "role", None)
        )
        content = (
            msg.get("content")
            if isinstance(msg, dict)
            else getattr(msg, "content", None)
        )
        if role == "assistant":
            text = _content_to_text(content)
            if text:
                assistant_texts.append(text)
            if len(assistant_texts) >= 3:
                break

    combined_text = "\n".join(reversed(assistant_texts))
    if len(combined_text) > max_chars:
        return combined_text[:max_chars] + "..."
    return combined_text or "No recent messages."


def _semantic_check(client, goal: str, messages: list) -> tuple[bool, str]:
    """Ask the model to check if the goal has been met."""
    plan = load_plan_for_injection() or "(no plan)"
    recent_summary = _summarize_recent(messages)

    prompt = RALPH_EVAL_PROMPT.format(
        goal=goal,
        plan=plan,
        recent_summary=recent_summary,
    )

    tracker = get_session_spend()
    if tracker is not None:
        tracker.raise_if_exceeded()

    response = client.messages.create(
        model=RALPH_EVAL_MODEL,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    record_session_spend(response, model=RALPH_EVAL_MODEL)
    if tracker is not None and tracker.exceeded:
        raise SpendLimitExceeded(tracker)

    result = "\n".join(
        block.text
        for block in response.content
        if block.type == "text"
    ).strip()

    if result.startswith("DONE:"):
        return True, result[len("DONE:"):].strip()
    if result.startswith("NOT_DONE:"):
        return False, result[len("NOT_DONE:"):].strip()

    print(
        f"[Ralph] Unexpected eval response format: {result[:200]!r}. "
        "Treating as NOT_DONE."
    )
    return False, "eval failed to provide a clear result."


def snapshot_plan_tasks() -> set[str]:
    """Snapshot every task text currently in plan.md (for Ralph baseline)."""
    return get_all_task_texts()


def open_tasks_fingerprint() -> frozenset[str]:
    """Stable fingerprint of currently open plan tasks."""
    return frozenset(get_open_tasks())


def is_goal_met(
    client,
    goal: str,
    messages: list,
    baseline_tasks: set[str] | None = None,
) -> tuple[bool, str]:
    """Check if the goal has been met.

    ``baseline_tasks`` is the set of task texts that existed when Ralph
    started. Open items added mid-run do not block completion of the
    original goal — if originals are done, we semantic-check the goal and
    may still return DONE while noting follow-ups.
    """
    current_open = set(get_open_tasks())

    if baseline_tasks is not None:
        unfinished_original = current_open & baseline_tasks
        new_open = current_open - baseline_tasks

        if unfinished_original:
            preview = "; ".join(sorted(unfinished_original)[:5])
            return False, f"Original open items remain: {preview}"

        done, reason = _semantic_check(client, goal, messages)
        if done and new_open:
            follow = "; ".join(sorted(new_open)[:5])
            return True, (
                f"{reason} "
                f"(mid-run follow-ups still open, not blocking: {follow})"
            )
        return done, reason

    if current_open:
        preview = "; ".join(sorted(current_open)[:5])
        return False, f"Open items remain in the plan: {preview}"

    return _semantic_check(client, goal, messages)


def build_carry_forward(
    goal: str,
    reason: str,
    assistant_text: str,
) -> str:
    """Compact context carried into the next Ralph iteration."""
    open_tasks = get_open_tasks()
    open_block = (
        "\n".join(f"- [ ] {t}" for t in open_tasks)
        if open_tasks
        else "(none)"
    )
    recent = (assistant_text or "").strip()
    if len(recent) > RALPH_CARRY_FORWARD_CHARS:
        recent = recent[:RALPH_CARRY_FORWARD_CHARS] + "\n...[truncated]..."

    return (
        "## Ralph carry-forward\n"
        f"Original goal:\n{goal}\n\n"
        f"Why we're continuing: {reason}\n\n"
        f"Open plan items:\n{open_block}\n\n"
        f"Last agent summary:\n{recent or '(empty)'}\n\n"
        "Do not re-explore the whole repo from scratch. "
        "Resume from the open items and the summary above."
    )


def should_fire(goal: str | None) -> bool:
    """Return True if Ralph has a prior substantive user goal to chase."""
    return goal is not None


def build_no_plan_refusal() -> str:
    """Message shown when /ralph is invoked but plan.md doesn't exist."""
    return (
        "[Ralph] No plan.md exists — invoke /ralph only after a plan has "
        "been created. Ask the agent to draw up a detailed plan first, "
        "then re-invoke."
    )


def build_no_goal_refusal() -> str:
    """Message shown when /ralph is invoked with no prior substantive input."""
    return (
        "[Ralph] No task in flight — send the agent a real task first, "
        "then invoke /ralph to have it drive the task to completion."
    )


def build_continuation_notice(iteration: int, reason: str) -> str:
    """The user-visible message printed when Ralph forces a continuation."""
    return (
        f"[Ralph continuation #{iteration}] Agent stopped but work remains. "
        f"Reason: {reason}. Resetting history but carrying forward a "
        f"compact summary so the agent does not re-explore from scratch."
    )


def build_cap_hit_notice(max_continuations: int) -> str:
    """The user-visible message printed when Ralph hits the continuation cap."""
    return (
        f"[Ralph] Continuation cap ({max_continuations}) reached. Work still "
        f"appears to remain — please investigate before continuing. The agent "
        f"may be stuck in a loop it can't resolve, or the goal may be "
        f"underspecified. Check plan.md and recent output."
    )


def build_stuck_notice(threshold: int, open_tasks: list[str]) -> str:
    """Printed when open tasks stop changing across continuations."""
    preview = "; ".join(open_tasks[:5]) if open_tasks else "(none)"
    return (
        f"[Ralph] Stuck: open plan items unchanged for {threshold} "
        f"continuation(s). Stopping to avoid burning more credits. "
        f"Open items: {preview}"
    )
