"""Compatibility imports for CALVIN structured event logging."""

from backends.calvin.logging.event_logger import JsonlEventLogger, NullEventLogger

__all__ = ["JsonlEventLogger", "NullEventLogger"]
