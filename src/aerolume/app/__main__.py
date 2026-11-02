"""Allow ``python -m aerolume.app`` to launch the desktop application."""
from .main import main

raise SystemExit(main())
