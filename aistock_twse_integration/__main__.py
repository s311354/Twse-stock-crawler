"""Print the installed integration contract for deployment diagnostics."""

from __future__ import annotations

import json

from .contract import package_contract


def main() -> int:
    print(json.dumps(package_contract(), separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
