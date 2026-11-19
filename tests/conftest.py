import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)

# tests/test_visualization_meta.py leaves a stub named `markdown` in
# sys.modules when the real package has not been imported yet. The web UI
# renders meta notes with the real one, so load it before any test module.
import markdown  # type: ignore[import-untyped]  # noqa: E402, F401
