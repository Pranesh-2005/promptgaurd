"""Exceptions raised by promptguard."""

from typing import Optional


class GuardError(Exception):
    """Raised when the guard itself fails in fail-closed mode."""

    def __init__(self, message: str, original_error: Optional[Exception] = None) -> None:
        super().__init__(message)
        self.original_error = original_error


class GuardBlocked(Exception):
    """Raised when a prompt is blocked by the guard."""

    def __init__(self, message: str, decision: "Decision") -> None:
        super().__init__(message)
        self.decision = decision
