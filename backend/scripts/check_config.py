"""Guard against config clobbering (cloud-sync conflicts have silently reverted
app/core/config.py before, breaking startup with ImportErrors).

Verifies that EVERY name any module imports from app.core.config actually
exists, plus that the full app imports. Run after pulling/syncing:

    python scripts/check_config.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import app.core.config as config  # noqa: E402

IMPORT_BLOCK = re.compile(r"from app\.core\.config import \(([^)]*)\)", re.S)
IMPORT_LINE = re.compile(r"^from app\.core\.config import ([A-Za-z0-9_, ]+)$", re.M)


def main() -> int:
    missing: list[str] = []
    for py in list((BACKEND / "app").rglob("*.py")) + list((BACKEND / "scripts").glob("*.py")):
        source = py.read_text(encoding="utf-8", errors="ignore")
        tokens: list[str] = []
        for match in IMPORT_BLOCK.finditer(source):
            tokens += [t.split("#")[0].strip() for t in match.group(1).split(",")]
        for match in IMPORT_LINE.finditer(source):
            tokens += [t.strip() for t in match.group(1).split(",")]
        for token in tokens:
            if token and not hasattr(config, token):
                missing.append(f"{py.relative_to(BACKEND)}: {token}")

    if missing:
        print("MISSING CONFIG NAMES (config.py was likely clobbered by a sync conflict):")
        for line in missing:
            print(f"  {line}")
        return 1

    try:
        import app.main  # noqa: F401
    except Exception as error:
        print(f"App import failed: {error}")
        return 1

    print("Config OK — all imported names exist and the app imports cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
