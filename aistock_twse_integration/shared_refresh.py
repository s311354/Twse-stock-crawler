"""Package entrypoint for the once-daily shared TWSE and MOPS refresh."""

from __future__ import annotations

import os
from pathlib import Path


def cli() -> int:
    runtime_root = Path.cwd().resolve()
    os.environ.setdefault("TWSE_PROJECT_ROOT", str(runtime_root))
    os.environ.setdefault("TWSE_DATA_DIR", str(runtime_root / "data"))
    from interface.cli import main as crawler_main

    return crawler_main()


if __name__ == "__main__":
    raise SystemExit(cli())
