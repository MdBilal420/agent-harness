"""LLM client factory for Anthropic cloud, local Ollama, Amazon Bedrock, and OpenRouter."""

from __future__ import annotations

import os
from typing import Literal

import anthropic
from langsmith.wrappers import wrap_anthropic

from harness.config import (
    MODEL as DEFAULT_MODEL,
    MAX_TOKENS as DEFAULT_MAX_TOKENS,
    OLLAMA_BASE_URL,
    BEDROCK_DEFAULT_MODEL,
    BEDROCK_DEFAULT_REGION,
)

Provider = Literal["ollama", "bedrock", "anthropic", "openrouter"]

OPENROUTER_BASE_URL = "https://openrouter.ai/api"
OPENROUTER_DEFAULT_MODEL = "nvidia/nemotron-3.5-super-vl:free"


def resolve_provider() -> Provider:
    """Pick the LLM backend from env.

    Explicit ``LLM_PROVIDER`` wins. Otherwise:
    - OpenRouter if ``OPENROUTER_API_KEY`` is set
    - Bedrock if a Bedrock bearer token / key is set
    - Anthropic cloud if ``ANTHROPIC_API_KEY`` looks like a real key and
      no Ollama base URL is configured
    - Ollama otherwise (local default)
    """
    explicit = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if explicit in {"ollama", "bedrock", "anthropic", "openrouter"}:
        return explicit  # type: ignore[return-value]
    if explicit:
        raise ValueError(
            f"Unknown LLM_PROVIDER={explicit!r}. "
            "Use ollama, bedrock, anthropic, or openrouter."
        )

    if os.getenv("OPENROUTER_API_KEY"):
        return "openrouter"

    if (
        os.getenv("AWS_ACCESS_KEY_ID")
        or os.getenv("AWS_PROFILE")
        or os.getenv("AWS_BEARER_TOKEN_BEDROCK")
        or os.getenv("BEDROCK_API_KEY")
    ):
        return "bedrock"

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    base_url = os.getenv("ANTHROPIC_BASE_URL")
    if api_key.startswith("sk-ant") and not base_url:
        return "anthropic"

    return "ollama"


def resolve_model(provider: Provider | None = None) -> str:
    """Return the model id for the active provider."""
    provider = provider or resolve_provider()
    override = os.getenv("ANTHROPIC_MODEL") or os.getenv("BEDROCK_MODEL")
    if override:
        return override
    if provider == "bedrock":
        return BEDROCK_DEFAULT_MODEL
    if provider == "anthropic":
        return "claude-haiku-4-5"
    if provider == "openrouter":
        return OPENROUTER_DEFAULT_MODEL
    return DEFAULT_MODEL


def resolve_max_tokens() -> int:
    return int(os.getenv("ANTHROPIC_MAX_TOKENS", str(DEFAULT_MAX_TOKENS)))


def build_client(provider: Provider | None = None):
    """Build an Anthropic-compatible Messages API client.

    All clients are wrapped with LangSmith ``wrap_anthropic`` so model
    calls nest under ``@traceable`` spans when ``LANGSMITH_TRACING`` is on.
    """
    provider = provider or resolve_provider()

    if provider == "bedrock":
        region = (
            os.getenv("AWS_REGION")
            or os.getenv("AWS_DEFAULT_REGION")
            or BEDROCK_DEFAULT_REGION
        )

        # Prefer IAM / profile credentials. Many orgs SCP-deny
        # bedrock:CallWithBearerToken, which blocks Bedrock API keys.
        access_key = os.getenv("AWS_ACCESS_KEY_ID")
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        session_token = os.getenv("AWS_SESSION_TOKEN")
        profile = os.getenv("AWS_PROFILE")

        kwargs: dict = {"aws_region": region}

        if access_key or secret_key or session_token or profile:
            if access_key:
                kwargs["aws_access_key"] = access_key
            if secret_key:
                kwargs["aws_secret_key"] = secret_key
            if session_token:
                kwargs["aws_session_token"] = session_token
            if profile:
                kwargs["aws_profile"] = profile
            return wrap_anthropic(anthropic.AnthropicBedrock(**kwargs))

        api_key = (
            os.getenv("AWS_BEARER_TOKEN_BEDROCK")
            or os.getenv("BEDROCK_API_KEY")
        )
        if api_key:
            kwargs["api_key"] = api_key
            return wrap_anthropic(anthropic.AnthropicBedrock(**kwargs))

        # Fall through to default credential chain (~/.aws/credentials).
        return wrap_anthropic(anthropic.AnthropicBedrock(**kwargs))

    if provider == "anthropic":
        return wrap_anthropic(anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        ))

    if provider == "openrouter":
        headers = {
            "X-Title": os.getenv("OPENROUTER_APP_TITLE", "agent-harness"),
        }
        referer = os.getenv("OPENROUTER_HTTP_REFERER", "")
        if referer:
            headers["HTTP-Referer"] = referer
        return wrap_anthropic(anthropic.Anthropic(
            api_key=(
                os.getenv("OPENROUTER_API_KEY")
                or os.getenv("ANTHROPIC_API_KEY")
            ),
            base_url=OPENROUTER_BASE_URL,
            default_headers=headers,
        ))

    # ollama — Anthropic Messages API compatibility
    base_url = os.getenv("ANTHROPIC_BASE_URL", OLLAMA_BASE_URL)
    return wrap_anthropic(anthropic.Anthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY", "ollama"),
        base_url=base_url or None,
    ))
