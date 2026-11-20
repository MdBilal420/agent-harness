"""Session spend tracking and hard USD limit for Anthropic API calls."""

from __future__ import annotations

from dataclasses import dataclass


# USD per million tokens. Keep in sync with https://platform.claude.com/docs/en/about-claude/pricing
# Values are (input, output, cache_write_5m, cache_read).
_MODEL_PRICES_PER_MTOK: dict[str, tuple[float, float, float, float]] = {
    "claude-haiku-4-5": (1.0, 5.0, 1.25, 0.10),
    "claude-haiku-4": (0.80, 4.0, 1.0, 0.08),
    "claude-sonnet-4-5": (3.0, 15.0, 3.75, 0.30),
    "claude-sonnet-4": (3.0, 15.0, 3.75, 0.30),
    "claude-opus-4": (15.0, 75.0, 18.75, 1.50),
    "claude-opus-4-1": (15.0, 75.0, 18.75, 1.50),
    "claude-opus-4-5": (5.0, 25.0, 6.25, 0.50),
}

# Fallback if the model id is unknown — prefer over-estimating slightly.
_DEFAULT_PRICES = (3.0, 15.0, 3.75, 0.30)


class SpendLimitExceeded(Exception):
    """Raised when estimated session spend reaches the configured USD limit."""

    def __init__(self, tracker: "SpendTracker"):
        self.tracker = tracker
        super().__init__(tracker.format_limit_notice())


def _prices_for(model: str) -> tuple[float, float, float, float]:
    key = (model or "").lower().strip()
    for prefix, prices in _MODEL_PRICES_PER_MTOK.items():
        if key.startswith(prefix):
            return prices
    return _DEFAULT_PRICES


def estimate_response_cost_usd(response, model: str) -> float:
    """Estimate USD cost of one Anthropic Messages response."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0.0

    input_rate, output_rate, cache_write_rate, cache_read_rate = _prices_for(
        model
    )

    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    cache_write = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
    cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)

    return (
        input_tokens * input_rate
        + output_tokens * output_rate
        + cache_write * cache_write_rate
        + cache_read * cache_read_rate
    ) / 1_000_000.0


@dataclass
class SpendStats:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    total_usd: float = 0.0


class SpendTracker:
    """Accumulate estimated API spend for one agent session."""

    def __init__(self, limit_usd: float, default_model: str):
        self.limit_usd = float(limit_usd)
        self.default_model = default_model
        self.stats = SpendStats()
        self._exceeded = False

    @property
    def total_usd(self) -> float:
        return self.stats.total_usd

    @property
    def exceeded(self) -> bool:
        return self._exceeded or self.stats.total_usd >= self.limit_usd

    def remaining_usd(self) -> float:
        return max(0.0, self.limit_usd - self.stats.total_usd)

    def record(self, response, model: str | None = None) -> float:
        """Add a response's estimated cost. Marks exceeded when limit is hit."""
        model_id = model or self.default_model
        cost = estimate_response_cost_usd(response, model_id)

        usage = getattr(response, "usage", None)
        self.stats.calls += 1
        self.stats.total_usd += cost

        if usage is not None:
            self.stats.input_tokens += int(
                getattr(usage, "input_tokens", 0) or 0
            )
            self.stats.output_tokens += int(
                getattr(usage, "output_tokens", 0) or 0
            )
            self.stats.cache_write_tokens += int(
                getattr(usage, "cache_creation_input_tokens", 0) or 0
            )
            self.stats.cache_read_tokens += int(
                getattr(usage, "cache_read_input_tokens", 0) or 0
            )

        if self.stats.total_usd >= self.limit_usd:
            self._exceeded = True

        return cost

    def raise_if_exceeded(self) -> None:
        if self.exceeded:
            raise SpendLimitExceeded(self)

    def format_status(self) -> str:
        return f"${self.stats.total_usd:.4f} / ${self.limit_usd:.2f}"

    def format_limit_notice(self) -> str:
        s = self.stats
        return (
            f"[Spend limit] Estimated session cost reached "
            f"${s.total_usd:.4f} (limit ${self.limit_usd:.2f}). "
            f"API calls={s.calls}, "
            f"in={s.input_tokens:,}, out={s.output_tokens:,}, "
            f"cache_write={s.cache_write_tokens:,}, "
            f"cache_read={s.cache_read_tokens:,}. "
            "Stopping the agent to avoid further charges."
        )


# Session singleton — set once in run().
_SESSION: SpendTracker | None = None


def init_session_spend(limit_usd: float, model: str) -> SpendTracker:
    """Create and install the session spend tracker."""
    global _SESSION
    _SESSION = SpendTracker(limit_usd=limit_usd, default_model=model)
    return _SESSION


def get_session_spend() -> SpendTracker | None:
    return _SESSION


def record_session_spend(response, model: str | None = None) -> float:
    """Record against the session tracker if one is installed."""
    tracker = _SESSION
    if tracker is None:
        return 0.0
    return tracker.record(response, model=model)
