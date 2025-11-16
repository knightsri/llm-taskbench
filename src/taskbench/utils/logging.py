"""
Logging configuration for LLM TaskBench.

This module provides centralized logging setup with support for different
log levels and nice formatting.
"""

import logging
import sys
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds colors to log levels.

    This formatter uses ANSI color codes to make log messages more readable
    in the terminal.
    """

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record with colors.

        Args:
            record: LogRecord to format

        Returns:
            Formatted log message string
        """
        # Add color to level name
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
            )

        # Format the message
        return super().format(record)


def setup_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    use_colors: bool = True,
    log_file: Optional[str] = None
) -> None:
    """
    Setup logging configuration for LLM TaskBench.

    This function configures the root logger with:
    - Specified log level (INFO, DEBUG, ERROR, etc.)
    - Nice formatting with timestamps and module names
    - Optional colored output for terminal
    - Optional file logging

    Args:
        level: Logging level (default: logging.INFO)
        format_string: Custom format string (optional)
        use_colors: Use colored output for terminal (default: True)
        log_file: Optional file path to write logs to

    Example:
        >>> # Basic setup
        >>> setup_logging()
        >>>
        >>> # Verbose debugging
        >>> setup_logging(level=logging.DEBUG)
        >>>
        >>> # With file logging
        >>> setup_logging(level=logging.INFO, log_file="taskbench.log")
    """
    # Default format string
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    if use_colors and sys.stdout.isatty():
        # Use colored formatter for terminal
        console_formatter = ColoredFormatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    else:
        # Use plain formatter for non-terminal or if colors disabled
        console_formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        # Always use plain formatter for file
        file_formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        root_logger.info(f"Logging to file: {log_file}")

    # Set logging level for noisy third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    root_logger.debug(f"Logging configured: level={logging.getLevelName(level)}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: Logger name (usually __name__ of the module)

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Hello, world!")
    """
    return logging.getLogger(name)


def set_level(level: int) -> None:
    """
    Change the logging level at runtime.

    Args:
        level: New logging level (e.g., logging.DEBUG, logging.INFO)

    Example:
        >>> # Enable debug logging
        >>> set_level(logging.DEBUG)
        >>>
        >>> # Back to info
        >>> set_level(logging.INFO)
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Update all handlers
    for handler in root_logger.handlers:
        handler.setLevel(level)

    root_logger.debug(f"Logging level changed to {logging.getLevelName(level)}")


def disable_colors() -> None:
    """
    Disable colored logging output.

    This is useful when piping output to files or when colors cause issues.

    Example:
        >>> setup_logging()
        >>> disable_colors()  # Switch to plain formatting
    """
    root_logger = logging.getLogger()

    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            # Replace with plain formatter
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            plain_formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
            handler.setFormatter(plain_formatter)


def add_file_handler(log_file: str, level: Optional[int] = None) -> None:
    """
    Add a file handler to the root logger.

    Args:
        log_file: Path to log file
        level: Optional logging level for this handler (defaults to root level)

    Example:
        >>> setup_logging()
        >>> add_file_handler("taskbench.log")
    """
    root_logger = logging.getLogger()

    if level is None:
        level = root_logger.level

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)

    format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_formatter)

    root_logger.addHandler(file_handler)
    root_logger.info(f"Added file handler: {log_file}")
