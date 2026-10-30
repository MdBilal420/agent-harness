"""Runtime configuration for the harness.

"""

# -- Model configuration --
MODEL: str = "claude-sonnet-5-5"
MAX_TOKENS = 8192



# -- ReAct loop bounds --
# Maximum number of tool-call rounds per user turn. When hit, the harness
# forces the model to summarize what it did and hand control back to the
# user.
STEP_BUDGET: int = 25


# -- Tool timeouts (seconds) --
# Per-command execution timeouts. Longer for bash (covers real work like
# installs) than for git (should always be fast on a local repo).
BASH_TIMEOUT: int = 60
GIT_TIMEOUT: int = 10


# -- Bash command policy --
# Allow-list of permitted commands (matched against the first token of
# each chain segment). Empty = permissive.
ALLOW_LIST: set[str] = set({
    "python",
    "python3",
    "pip",
    "pip3",

    "ls",
    "cat",
    "wc",
    "grep",
    "find",
    "head",
    "tail",

    "curl",

    "which",
    "echo",
    "pwd",

    "git"
})

# Deny-list of forbidden commands. Empty = no denials.
# Precedence: deny wins on conflict.
DENY_LIST: set[str] = set({
    "rm",
    "sudo",
    "dd",
    "mkfs",
    "mv"
})

SANDBOX_IMAGE: str = "agent-harness:0.1"

SANDBOX_WORKSPACE_PATH: str = "/workspace"

SANDBOX_STARTUP_TIMEOUT: int = 30

SANDBOX_CONTAINER_PREFIX: str = "agent-harness-"