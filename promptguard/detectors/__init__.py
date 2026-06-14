"""Detector exports — single BERT-mini model replaces all rule-based detectors."""

from .base import BaseDetector
from .bert_detector import BertDetector

__all__ = ["BaseDetector", "BertDetector"]
