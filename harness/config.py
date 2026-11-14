"""Runtime configuration for the harness.

"""

from pathlib import Path


# Workspace path 
WORKSPACE_PATH = Path("./workspace").resolve()

# Create workspace directory if it doesn't exist
WORKSPACE_PATH.mkdir(parents=True, exist_ok=True)


# -- Model configuration --
MODEL: str = "claude-haiku-5-5"
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

WEB_SEARCH_MAX_RESULTS: int = 5

# RAG configuration

RAG_CORPUS_PATH: str = "knowledge"

RAG_INDEX_PATH: str = ".rag_index"

RAG_EMBEDDING_MODEL: str = "voyage-4-lite"

RAG_CHUNK_SIZE: int = 500

RAG_CHUNK_OVERLAP: int = 50

RAG_DEFAULT_RESULTS: int = 5

# Compaction

# COMPACTION_THRESHOLD: int = 30000
COMPACTION_THRESHOLD: int = 10000000


COMPACTION_KEEP_RECENT: int = 3

# Tool Offloading

# OFFLOAD_THRESHOLD_TOKENS: int = 1000
OFFLOAD_THRESHOLD_TOKENS: int = 1000000


OFFLOAD_HEAD_TOKENS: int = 300

OFFLOAD_TAIL_TOKENS: int = 300