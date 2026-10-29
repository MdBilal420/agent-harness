

import uuid
import docker
from docker.errors import DockerException, NotFound

from harness.tools.filesystem import WORKSPACE_PATH
from harness.config import (
    SANDBOX_IMAGE,
    SANDBOX_WORKSPACE_PATH,
    SANDBOX_STARTUP_TIMEOUT,
    SANDBOX_CONTAINER_PREFIX,
    BASH_TIMEOUT
)


class Sandbox:

    def __init__(self) -> None:
        self._client = docker.from_env()

        self._container_name = f"{SANDBOX_CONTAINER_PREFIX}{uuid.uuid4().hex[:8]}"

        self._container = None

    def start(self) -> None:

        self._cleanup_orphans()

        self._container = self._client.containers.run(
            image=SANDBOX_IMAGE,
            name=self._container_name,
            command="sleep infinity",
            detach=True,
            working_dir=SANDBOX_WORKSPACE_PATH,
            volumes={WORKSPACE_PATH: {"bind": SANDBOX_WORKSPACE_PATH, "mode": "rw"}},
        )

        self._wait_for_startup()

    def execute(self, command:str) -> tuple[int, str, str]:
        "Execute a command in the sandbox and return the output"

        if self._container is None:
            raise RuntimeError("Sandbox execute called but container not started")

        wrapped = f'timeout {BASH_TIMEOUT} bash -c {_shell_quote(command)}'

        result = self._container.exec_run(wrapped, demux=True, workdir=SANDBOX_WORKSPACE_PATH)
        exit_code = result.exit_code
        stdout_bytes, stderr_bytes = result.output
        stdout = (stdout_bytes or b"").decode("utf-8", errors="replace")
        stderr = (stderr_bytes or b"").decode("utf-8", errors="replace")
        return exit_code, stdout, stderr

    def _cleanup_orphans(self) -> None:

        "Remove all containers with the prefix SANDBOX_CONTAINER_PREFIX"

        orphans = self._client.containers.list(
            all=True,
            filters={"name":SANDBOX_CONTAINER_PREFIX},
        )

        for orphan in orphans:
            try:
                orphan.remove(force=True)
            except DockerException:
                pass

    def _wait_for_startup(self) -> None:

        "Wait for the container to start"

        import time
        deadline = time.time() + SANDBOX_STARTUP_TIMEOUT

        while time.time() < deadline:
            self._container.reload()
            if self._container.status == "running":
                return
            time.sleep(0.1)

        raise TimeoutError(f"Container {self._container_name} failed to start")

    def stop(self) -> None:

        "Stop the container"

        self._container.stop()
        self._container.remove(force=True)
        self._container = None

    def __enter__(self) -> "Sandbox":
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()

def _shell_quote(s:str) -> str:
    "Quote a string for shell use"
    return "'" + s.replace("'", "'\\''") + "'"
