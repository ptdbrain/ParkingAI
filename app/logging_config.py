from __future__ import annotations

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    """Configure process-wide structured-ish console logging."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
