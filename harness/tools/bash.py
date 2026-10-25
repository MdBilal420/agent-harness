import subprocess
from harness.tools.filesystem import WORKSPACE_PATH
from harness.tools.registry import tool


BASH_TIMEOUT = 60


def _command_output(stdout: str, stderr: str) -> str:
    """Format the output of a command."""
    stdout = stdout.strip()
    stderr = stderr.strip()

    if stdout and stderr:
        return f"stdout: {stdout}\n-----\nstderr: {stderr}"
    return stdout or stderr


@tool 
def bash(command: str) -> str:
    """Execute a bash command and return the output."""
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
    
    output = _command_output(result.stdout, result.stderr) or "(No output)"

    if result.returncode != 0:
        return f"Command failed with exit code {result.returncode}: {output}"
    
    return output