"""
File system tools : read, write, delete, list - bounded to workspace
"""


from harness.tools.registry import tool
from pathlib import Path

# Workspace path 
WORKSPACE_PATH = Path("./workspace").resolve()

# Create workspace directory if it doesn't exist
WORKSPACE_PATH.mkdir(parents=True, exist_ok=True)


def _resolve_path(path: str) -> Path:
    """
    Resolve a path to a file or directory in the workspace.
    """
    
    target_path = (WORKSPACE_PATH / path).resolve()

    if not target_path.is_relative_to(WORKSPACE_PATH):
        raise ValueError(f"Path {path} is not within the workspace")

    return target_path

@tool
def read_file(path: str) -> str:
    """
    Read a file and return its contents.
    """
    target_path = _resolve_path(path)
    with open(target_path, "r") as f:
        return f.read()

@tool
def write_file(path: str, content: str) -> None:
    """
    Write to a file.
    """
    target_path = _resolve_path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(content)

    return f"File {path} written successfully"

@tool
def delete_file(path: str) -> None:
    """
    Delete a file.
    """
    target_path = _resolve_path(path)

    if not target_path.exists():
        raise FileNotFoundError(f"File {path} does not exist")
    if target_path.is_dir():
        raise IsADirectoryError(f"Path {path} is a directory")
    target_path.unlink()

    return f"File {path} deleted successfully"

@tool
def list_files(path: str) -> list[str]:
    """
    List files in a directory.
    """
    target_path = _resolve_path(path)
    if not target_path.exists():
        raise FileNotFoundError(f"Path {path} does not exist")
    if not target_path.is_dir():
        raise NotADirectoryError(f"Path {path} is not a directory")

    entries = sorted(target_path.iterdir())
    return "\n".join(
        f"{entry.name} (dir)" if entry.is_dir() else f"{entry.name} (file)"
        for entry in entries
    )

@tool
def create_directory(path: str) -> str:
    """
    Create a directory.
    """
    target_path = _resolve_path(path)
    target_path.mkdir(parents=True, exist_ok=True)
    return f"Directory {path} created successfully"
