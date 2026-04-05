from datetime import date, datetime
import logging
from utils.path_tool import get_abs_path
import os

# The root directory of the log file
LOG_ROOT = get_abs_path("logs")

# Ensure the directory of the log file exists
os.makedirs(LOG_ROOT, exist_ok=True)

# The format of the log message
DEFAULT_LOG_FORMAT = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)


def get_logger(
    name: str = "agent",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    log_file=None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Set the logger to the lowest level

    if logger.handlers:
        return logger  # Return the existing logger if it already has handlers

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(DEFAULT_LOG_FORMAT)
    logger.addHandler(console_handler)

    # Create file handler if log_file is provided
    if not log_file:
        log_file = os.path.join(
            LOG_ROOT, f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)
    logger.addHandler(file_handler)

    return logger


# get the default logger
logger = get_logger()

if __name__ == "__main__":
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")
