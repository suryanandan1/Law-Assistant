"""Single place to configure Loguru. Call `configure_logging()` once per process.

Every module used to define its own `setup_logging()` writing to a different
file, and `app.py` added a sink on every Streamlit rerun (a handler leak). This
replaces all of that: the first call wins, later calls are no-ops.
"""

import sys
from pathlib import Path

from loguru import logger

_CONFIGURED = False

_CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan> - {message}"
)


def configure_logging(logfile: str | Path = "logs/app.log", level: str = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    logger.remove()
    logger.add(sys.stderr, level=level, format=_CONSOLE_FORMAT)

    path = Path(logfile)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(path, rotation="10 MB", retention=5, level=level, encoding="utf-8")

    _CONFIGURED = True
