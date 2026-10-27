"""Ensure the repo root (containing the src/ package) is importable.

pytest's rootdir-insertion only adds the first ancestor directory *without*
an __init__.py to sys.path. Since tests/ has no __init__.py, that would be
tests/ itself, not the repo root where src/ lives — so this is added
explicitly rather than relying on invocation directory.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
