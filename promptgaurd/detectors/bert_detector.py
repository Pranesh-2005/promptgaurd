"""Single BERT-mini PyTorch detector — replaces all 16 old rule detectors."""

import re
import threading
from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .base import BaseDetector


MODEL_ID = "PraneshJs/PromptGaurd"

# Process-wide cache so every Gaudrial/BertDetector instance shares one
# loaded model instead of re-downloading and re-loading per instance.
_MODEL_CACHE: Dict[str, Tuple] = {}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?\n])\s+")


def _load_model(model_id: str) -> Tuple:
    if model_id not in _MODEL_CACHE:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device)
        model.eval()
        # Fast tokenizers are not thread-safe ("Already borrowed"); the
        # shared cache means concurrent callers must serialize inference.
        _MODEL_CACHE[model_id] = (tokenizer, model, device, threading.Lock())
    return _MODEL_CACHE[model_id]


class BertDetector(BaseDetector):
    """Binary detector using fine-tuned BERT-mini (safe/attack).

    Returns 0.0–1.0 attack probability. Also provides class-level prediction.
    Long prompts cannot bypass detection via truncation: the prompt is scored
    as overlapping 128-token sliding windows AND as individual sentences (so a
    short injection buried in benign text gets an undiluted look), all in one
    batched forward pass. The worst (most attack-like) segment wins.
    """

    name = "bert_mini"

    def __init__(self, model_id: str = MODEL_ID):
        self.tokenizer, self.model, self.device, self._lock = _load_model(model_id)
        self.max_len = 128
        self.stride = 64

    def _segments(self, prompt: str) -> List[str]:
        """Full prompt plus per-sentence segments for undiluted scoring.

        Sentence segments are only added when the prompt exceeds the model's
        max length: a short prompt cannot hide an injection via truncation,
        and scoring its sentences in isolation causes false positives
        (e.g. a bare "You are a helpful assistant." reads as role hijacking).
        """
        segments = [prompt]
        n_tokens = len(self.tokenizer(prompt, truncation=False)["input_ids"])
        if n_tokens > self.max_len:
            sentences = [s.strip() for s in _SENTENCE_SPLIT.split(prompt) if len(s.strip()) >= 8]
            if len(sentences) > 1:
                segments.extend(sentences)
        return segments

    def _predict(self, prompt: str) -> Tuple[float, int, float]:
        """Run one batched inference over all sliding windows and sentences.

        Returns (attack_prob, pred_id, confidence) taken from the segment
        with the highest attack probability.
        """
        with self._lock:
            inputs = self.tokenizer(
                self._segments(prompt),
                truncation=True,
                padding=True,
                max_length=self.max_len,
                stride=self.stride,
                return_overflowing_tokens=True,
                return_tensors="pt",
            )
            inputs.pop("overflow_to_sample_mapping", None)
            inputs = inputs.to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)  # (num_segments, 2)
        worst = int(torch.argmax(probs[:, 1]))
        segment_probs = probs[worst]
        pred_id = int(torch.argmax(segment_probs))
        return float(segment_probs[1]), pred_id, float(segment_probs[pred_id])

    def detect(self, prompt: str) -> float:
        """Return 0.0-1.0 attack probability."""
        return self.detect_and_classify(prompt)[0]

    def classify(self, prompt: str) -> tuple:
        """Return (class_name, confidence)."""
        _, class_name, confidence = self.detect_and_classify(prompt)
        return (class_name, confidence)

    def detect_and_classify(self, prompt: str) -> Tuple[float, str, float]:
        """Return (attack_prob, class_name, confidence) from a single inference."""
        if not prompt or not prompt.strip():
            return (0.0, "safe", 1.0)
        attack_prob, pred_id, confidence = self._predict(prompt)
        labels = self.model.config.id2label
        return (attack_prob, labels[pred_id], confidence)
