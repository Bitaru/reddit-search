"""Shared test environment.

Typer forces Rich terminal rendering (ANSI escapes, width padding) whenever
``GITHUB_ACTIONS`` is set, which is always true on CI runners. CLI acceptance
tests assert on plain help text, so disable the forced terminal for every
``subprocess.run`` invocation, which inherits this environment.
"""

import os

os.environ.setdefault("_TYPER_FORCE_DISABLE_TERMINAL", "1")
