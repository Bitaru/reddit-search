"""Explicit, offline-only text generation through a local CLI."""

from __future__ import annotations

import os
import selectors
import signal
import subprocess
import tempfile
import time
from pathlib import Path


class LocalModelError(RuntimeError):
    """A local generator could not safely produce a completion."""


def _run_bounded(
    command: list[str],
    prompt: bytes,
    *,
    timeout_seconds: float,
    max_output_bytes: int,
) -> tuple[int, bytes, bytes]:
    """Stream bounded stdout/stderr and terminate the child on any limit."""
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "LANG": "C.UTF-8"},
        start_new_session=True,
    )
    assert process.stdin is not None and process.stdout is not None and process.stderr is not None
    stdout = bytearray()
    stderr = bytearray()
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, (stdout, max_output_bytes))
    selector.register(process.stderr, selectors.EVENT_READ, (stderr, 2048))
    os.set_blocking(process.stdin.fileno(), False)
    selector.register(process.stdin, selectors.EVENT_WRITE, None)
    sent = 0
    deadline = time.monotonic() + timeout_seconds
    try:
        while selector.get_map() or process.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise LocalModelError(
                    f"local generation exceeded timeout of {timeout_seconds:g} seconds"
                )
            for key, _ in selector.select(min(remaining, 0.1)):
                if key.fileobj is process.stdin:
                    try:
                        sent += os.write(process.stdin.fileno(), prompt[sent : sent + 8192])
                    except BrokenPipeError:
                        sent = len(prompt)
                    if sent >= len(prompt):
                        selector.unregister(process.stdin)
                        process.stdin.close()
                    continue
                target, limit = key.data
                chunk = os.read(key.fileobj.fileno(), min(8192, limit + 1 - len(target)))
                if not chunk:
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
                    continue
                target.extend(chunk)
                if len(target) > limit:
                    label = "stdout" if target is stdout else "stderr"
                    raise LocalModelError(
                        f"local generator {label} exceeded its {limit}-byte limit"
                    )
        return process.wait(), bytes(stdout), bytes(stderr)
    except BaseException:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait()
        raise
    finally:
        selector.close()
        for stream in (process.stdin, process.stdout, process.stderr):
            if not stream.closed:
                stream.close()


def generate_local_completion(
    prompt: str,
    *,
    system_prompt: str,
    executable: Path,
    model_path: Path,
    timeout_seconds: float = 120.0,
    max_output_bytes: int = 65_536,
    max_prompt_bytes: int = 1_048_576,
) -> str:
    """Run one bounded local completion with distinct system and user messages.

    The trusted system prompt is read from a private temporary file; the user
    prompt is sent unchanged as binary stdin. No downloads or network clients
    are used.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise LocalModelError("prompt must be a non-empty string")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise LocalModelError("system_prompt must be a non-empty string")
    if not 0 < timeout_seconds <= 600:
        raise LocalModelError("timeout_seconds must be greater than 0 and at most 600")
    if not 0 < max_output_bytes <= 1_048_576:
        raise LocalModelError("max_output_bytes must be between 1 and 1048576")
    if not 0 < max_prompt_bytes <= 8_388_608:
        raise LocalModelError("max_prompt_bytes must be between 1 and 8388608")
    system_prompt_bytes = system_prompt.encode("utf-8")
    prompt_bytes = prompt.encode("utf-8")
    if len(system_prompt_bytes) + len(prompt_bytes) > max_prompt_bytes:
        raise LocalModelError("system and user prompts exceed max_prompt_bytes")

    executable = Path(executable)
    model_path = Path(model_path)
    if not executable.is_absolute():
        raise LocalModelError(
            "executable must be an absolute path to an installed local llama-completion tool"
        )
    if not model_path.is_absolute():
        raise LocalModelError(
            "model_path must be an absolute path to an installed local model file"
        )
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise LocalModelError(
            f"local generator executable is missing or not executable: {executable}"
        )
    if not model_path.is_file() or not os.access(model_path, os.R_OK):
        raise LocalModelError(f"local model file is missing or unreadable: {model_path}")

    try:
        with tempfile.TemporaryDirectory(prefix="reddit-search-llama-") as temp_dir:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix="system-prompt-",
                dir=temp_dir,
                delete=False,
            ) as system_file:
                system_file.write(system_prompt_bytes + b"\n")
            command = [
                str(executable),
                "-m",
                str(model_path),
                "-sysf",
                system_file.name,
                "-bf",
                "/dev/stdin",
                "--no-escape",
                "--single-turn",
                "-c",
                "8192",
                "-n",
                "512",
                "--no-context-shift",
                "--no-display-prompt",
                "--offline",
                "--no-perf",
                "--log-verbosity",
                "2",
                "--log-colors",
                "off",
                "--temp",
                "0",
                "--seed",
                "0",
            ]
            returncode, stdout, stderr = _run_bounded(
                command,
                prompt_bytes,
                timeout_seconds=timeout_seconds,
                max_output_bytes=max_output_bytes,
            )
    except LocalModelError:
        raise
    except OSError as exc:
        raise LocalModelError(f"could not start local generator: {exc}") from exc
    if returncode != 0:
        detail = stderr.decode("utf-8", errors="replace").strip()
        suffix = f": {detail}" if detail else ""
        raise LocalModelError(f"local generator exited with status {returncode}{suffix}")
    native_eos_frame = b" [end of text]\n\n\n"
    if stdout.endswith(native_eos_frame):
        stdout = stdout[: -len(native_eos_frame)]
    completion = stdout.decode("utf-8", errors="replace").strip()
    if not completion:
        raise LocalModelError("local generator returned an empty completion")
    return completion
