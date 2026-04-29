from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn


def main() -> None:
    """Start the FastAPI server with the demo AI frame feeder."""

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
    )


if __name__ == "__main__":
    main()
