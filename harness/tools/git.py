import subprocess
from harness.tools.filesystem import WORKSPACE_PATH
from harness.tools.registry import tool

GIT_TIMEOUT = 10


def _run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=WORKSPACE_PATH,
        capture_output=True,
        text=True,
        timeout=GIT_TIMEOUT,
        check=False,
    )

    output = result.stdout.strip() or result.stderr.strip()

    if result.returncode != 0:
        return f"[git exit {result.returncode}] {output}"

    return output or "(no output)"

if not (WORKSPACE_PATH / ".git").exists():
    print("Initializing git repository...")
    _run_git("init")
    _run_git("config", "user.name", "agent")
    _run_git("config", "user.email", "agent@harness.local")
    _run_git("symbolic-ref", "HEAD", "refs/heads/main")

@tool
def git_status() -> str:
    """Show the current working-tree status — modified, staged, and untracked files."""
    return _run_git("status", "--short")

@tool
def git_commit(message: str) -> str:
    """Commit the current changes."""
    stage_result = _run_git("add", "-A")

    return _run_git("commit", "-m", message)


@tool
def git_branch(name: str) -> str:
    """Show the current branch."""
    if not name:
        return _run_git("branch")
    return _run_git("branch", name)