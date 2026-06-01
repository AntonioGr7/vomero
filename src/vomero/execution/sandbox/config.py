"""Knobs for the gVisor sandbox backend.

Defaults are conservative and safe: gVisor (`runsc`) for syscall isolation, no
network, a hard memory cap, a bounded CPU share, and a capped process count
(fork-bomb guard). All of it is tunable so a deployment can size containers to
its workload — `memory` / `cpus` are the two most users will touch.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SandboxConfig:
    """Configuration for one sandboxed execution backend.

    The two resource knobs the task asked for are `memory` and `cpus`; they map
    straight onto Docker's `--memory` / `--cpus` (and thus onto the gVisor
    sandbox's cgroup limits)."""

    # Base image. Stock Python works out of the box; point this at an image with
    # your own deps (pandas, numpy, ...) if the model's code needs them. Vomero's
    # own source is bind-mounted in at runtime, so the image needs only Python.
    image: str = "python:3.11-slim"

    # The OCI runtime. "runsc" is gVisor; must be registered with the Docker
    # daemon (see docs/adr/0004). Set to "runc" to test the plumbing without
    # gVisor isolation (NOT recommended for untrusted code).
    runtime: str = "runsc"

    # --- resource limits (the headline feature) ---
    memory: str = "512m"          # hard cap; Docker --memory syntax (e.g. "1g")
    cpus: float = 1.0             # fractional vCPUs; Docker --cpus
    pids_limit: int = 256         # max processes/threads — fork-bomb guard
    tmpfs_size: str = "64m"       # size of the writable /tmp tmpfs

    # --- isolation ---
    network: str = "none"         # Docker --network; "none" = no network at all
    # Container user. None => the host's uid:gid, so the read-only corpus mount
    # is readable with exactly the host user's permissions and the model's code
    # runs non-root. Override with "uid:gid" or a name the image knows.
    user: str | None = None

    # --- plumbing ---
    docker_path: str = "docker"   # CLI to invoke (e.g. "podman")
    startup_timeout: float = 60.0  # seconds to wait for the container to connect
                                    # (first run may pull the image)
    # Escape hatch: extra args spliced into `docker run` before the image.
    extra_run_args: list[str] = field(default_factory=list)

    # Test-only seam. "docker" launches a real (gVisor) container; "local" runs
    # the agent as a plain host subprocess with NO isolation, used by the test
    # suite to exercise the host<->agent protocol without Docker. Never use
    # "local" for untrusted code — it defeats the entire point of the sandbox.
    runner: str = "docker"
