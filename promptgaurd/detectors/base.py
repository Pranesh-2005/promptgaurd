"""Base detector interface."""

from abc import ABC, abstractmethod


class BaseDetector(ABC):
    """Abstract base class for all prompt injection detectors."""

    name: str = "base"

    @abstractmethod
    def detect(self, prompt: str) -> float:
        """Return a confidence score between 0.0 and 1.0."""
        ...
