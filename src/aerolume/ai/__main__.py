"""``python -m aerolume.ai --project P "goal"`` (see :mod:`aerolume.ai.cli`)."""
from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
