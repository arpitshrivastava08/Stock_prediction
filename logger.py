import logging
import os
import sys
from datetime import datetime


def get_logger(name: str, log_level: str = "INFO", log_file: str = "logs/trading_system.log") -> logging.Logger:
    """
    Returns a configured logger that writes to both console and file.

    Args:
        name: Logger name (typically __name__ of calling module)
        log_level: Logging level string
        log_file: Path to log output file

    Returns:
        Configured Logger instance
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    if not logger.handlers:
        fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        # File handler
        fh = logging.FileHandler(log_file, mode="a")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger