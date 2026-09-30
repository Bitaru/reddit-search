from __future__ import annotations

from pathlib import Path

import pytest

from reddit_search.engagement.local_model import LocalModelError, generate_local_completion

_SYSTEM_PROMPT = "trusted instructions"


def make_local_executable(tmp_path: Path, script: str) -> Path:
    executable = tmp_path / "llama-completion"
    executable.write_text(f"#!/bin/sh\n{script}\n", encoding="utf-8")
    executable.chmod(0o700)
    return executable


def make_model(tmp_path: Path) -> Path:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"local test model placeholder")
    return model


def test_missing_model_fails_actionably_before_process_launch(tmp_path: Path) -> None:
    executable = make_local_executable(tmp_path, "exit 99")

    with pytest.raises(LocalModelError, match="model file is missing or unreadable"):
        generate_local_completion(
            "prompt",
            system_prompt=_SYSTEM_PROMPT,
            executable=executable,
            model_path=tmp_path / "missing.gguf",
        )


def test_rejects_relative_paths_and_invalid_prompt(tmp_path: Path) -> None:
    model = make_model(tmp_path)
    executable = make_local_executable(tmp_path, "cat")

    with pytest.raises(LocalModelError, match="absolute path"):
        generate_local_completion(
            "prompt",
            system_prompt=_SYSTEM_PROMPT,
            executable=Path("llama-completion"),
            model_path=model,
        )
    with pytest.raises(LocalModelError, match="non-empty"):
        generate_local_completion(
            "  ",
            system_prompt=_SYSTEM_PROMPT,
            executable=executable,
            model_path=model,
        )


def test_nonzero_exit_is_an_actionable_error(tmp_path: Path) -> None:
    executable = make_local_executable(tmp_path, "echo backend-failure >&2; exit 7")

    with pytest.raises(LocalModelError, match="status 7: backend-failure"):
        generate_local_completion(
            "prompt",
            system_prompt=_SYSTEM_PROMPT,
            executable=executable,
            model_path=make_model(tmp_path),
        )


def test_stdout_overflow_fails_and_terminates_child(tmp_path: Path) -> None:
    executable = make_local_executable(tmp_path, "while :; do printf 1234567890; done")

    with pytest.raises(LocalModelError, match="stdout exceeded its 8-byte limit"):
        generate_local_completion(
            "prompt",
            system_prompt=_SYSTEM_PROMPT,
            executable=executable,
            model_path=make_model(tmp_path),
            max_output_bytes=8,
        )


def test_stderr_overflow_fails_and_terminates_child(tmp_path: Path) -> None:
    executable = make_local_executable(tmp_path, "while :; do printf 1234567890 >&2; done")

    with pytest.raises(LocalModelError, match="stderr exceeded its 2048-byte limit"):
        generate_local_completion(
            "prompt",
            system_prompt=_SYSTEM_PROMPT,
            executable=executable,
            model_path=make_model(tmp_path),
        )


def test_timeout_is_bounded(tmp_path: Path) -> None:
    executable = make_local_executable(tmp_path, "sleep 2")

    with pytest.raises(LocalModelError, match="exceeded timeout"):
        generate_local_completion(
            "prompt",
            system_prompt=_SYSTEM_PROMPT,
            executable=executable,
            model_path=make_model(tmp_path),
            timeout_seconds=0.01,
        )


def test_timeout_covers_large_prompt_when_child_does_not_read_stdin(tmp_path: Path) -> None:
    executable = make_local_executable(tmp_path, "sleep 2")

    with pytest.raises(LocalModelError, match="exceeded timeout"):
        generate_local_completion(
            "x" * 1_048_576,
            system_prompt=_SYSTEM_PROMPT,
            executable=executable,
            model_path=make_model(tmp_path),
            timeout_seconds=0.05,
            max_prompt_bytes=2_000_000,
        )


def test_empty_completion_fails_closed(tmp_path: Path) -> None:
    executable = make_local_executable(tmp_path, "cat >/dev/null")

    with pytest.raises(LocalModelError, match="empty completion"):
        generate_local_completion(
            "prompt",
            system_prompt=_SYSTEM_PROMPT,
            executable=executable,
            model_path=make_model(tmp_path),
        )
