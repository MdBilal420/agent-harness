import subprocess
from harness.tools.filesystem import WORKSPACE_PATH
from harness.tools.registry import tool
from harness.sandbox import Sandbox

from harness.config import BASH_TIMEOUT, ALLOW_LIST, DENY_LIST


_sandbox: Sandbox | None = None

def set_sandbox(sandbox: Sandbox) -> None:
    global _sandbox
    _sandbox = sandbox

_CHAIN_SEPARATORS = ("&&", "|", ";", "||")


def _first_token(command: str) -> str:
    """Return the first token of a command."""
    command = command.strip()
    if not command:
        return ""
    return command.split()[0]

def _segments(command: str) -> list[str]:
    """Return the segments of a command."""
    segments = [command]
    for separator in _CHAIN_SEPARATORS:
        segments = [
            piece
            for seg in segments
            for piece in seg.split(separator)
        ]
    return segments


def _check_policy(command: str) -> str | None:
    """Check if a command is allowed by the policy."""
    tokens = [_first_token(seg) for seg in _segments(command)]
    tokens = [token for token in tokens if token]

    if DENY_LIST:
        for token in tokens:
            if token in DENY_LIST:
                return f"Command '{token}' is in the deny list and is not allowed"
    if ALLOW_LIST:
        for token in tokens:
            if token not in ALLOW_LIST:
                return f"Command '{token}' is not in the allow list and is not allowed"
    return None

def _combine_output(stdout: str, stderr: str) -> str:
    """Combine the stdout and stderr of a command."""
    stdout = stdout.strip()
    stderr = stderr.strip()

    if stdout and stderr:
        return f"stdout: {stdout}\n-----\nstderr: {stderr}"
    return stdout or stderr


@tool 
def bash(command: str) -> str:
    """Execute a bash command and return the output."""

    policy_error = _check_policy(command)
    if policy_error:
        return policy_error

    if _sandbox is None:
        print("Sandbox is not initialized")
        raise ValueError("Sandbox is not initialized")

    print(f"Executing command: {command}")
    exit_code, stdout, stderr = _sandbox.execute(command)

    output = _combine_output(stdout, stderr) or "(No output)"
    if exit_code != 0:
        return f"Command failed with exit code {exit_code}: {output}"

    return output