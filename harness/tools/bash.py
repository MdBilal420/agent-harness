import subprocess
from harness.tools.filesystem import WORKSPACE_PATH
from harness.tools.registry import tool


BASH_TIMEOUT = 60
ALLOW_LIST: set[str] = {
    "ls",
    "cd",
    "pwd",
    "mkdir",
    "rm",
    "cp",
    "mv",
    "pip",
    "cat",
    "python",
    "python3",
    "which",
}
DENY_LIST: set[str] = {
    "dd"
}

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

    print(f"Tokens: {tokens}")

    print(f"DENY_LIST: {DENY_LIST}")
    print(f"ALLOW_LIST: {ALLOW_LIST}")
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

    try:
        result = subprocess.run(
            command,
            shell=True,
            timeout=BASH_TIMEOUT,
            cwd=WORKSPACE_PATH,
            capture_output=True,
            text=True,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "Command timed out"
    except subprocess.CalledProcessError as e:
        return f"Command failed with exit code {e.returncode}: {e.stderr}"
    
    output = _combine_output(result.stdout, result.stderr) or "(No output)"

    if result.returncode != 0:
        return f"Command failed with exit code {result.returncode}: {output}"
    
    return output