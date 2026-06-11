"""Provider-shaped mock responses returned when a prompt is blocked.

Instead of raising and breaking the caller's pipeline, adapters return one
of these objects. Each mimics the real provider response shape, so code
like ``response.choices[0].message.content`` or ``response.content[0].text``
keeps working. The blocked Decision's prompt_id is embedded in the response
``id`` so it can be traced back to the structured log entry.
"""

import time
from typing import Any, Dict, List, Union

DEFAULT_BLOCK_MESSAGE = (
    "This request was blocked by PromptGaurd: potential prompt injection "
    "detected (score={score:.2f}). Reference ID: {prompt_id}."
)


class MockObject:
    """Dict-backed object with both attribute and item access.

    Mimics provider SDK response objects closely enough for typical
    pipelines: ``r.choices[0].message.content``, ``r["choices"]``,
    ``r.model_dump()`` / ``r.to_dict()`` all work.
    """

    def __init__(self, data: Dict[str, Any]) -> None:
        self._data = data
        for key, value in data.items():
            setattr(self, key, _wrap(value))

    def __getitem__(self, key: str) -> Any:
        return _wrap(self._data[key])

    def get(self, key: str, default: Any = None) -> Any:
        return _wrap(self._data.get(key, default))

    def to_dict(self) -> Dict[str, Any]:
        return self._data

    # pydantic-style aliases so downstream .model_dump()/.dict() don't crash
    def model_dump(self) -> Dict[str, Any]:
        return self._data

    def dict(self) -> Dict[str, Any]:
        return self._data

    def __repr__(self) -> str:
        return f"MockObject({self._data!r})"


def _wrap(value: Any) -> Any:
    if isinstance(value, dict):
        return MockObject(value)
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def render_block_message(decision: Any, template: Union[str, None] = None) -> str:
    """Render the user-facing blocked-message text from a Decision."""
    template = template or DEFAULT_BLOCK_MESSAGE
    score = max(decision.scores.values()) if decision.scores else 0.0
    return template.format(
        score=score,
        prompt_id=decision.prompt_id,
        reason=decision.reason,
    )


def openai_blocked_response(decision: Any, model: str = "unknown", message: Union[str, None] = None) -> MockObject:
    """chat.completion-shaped mock. Also fits Groq, OpenRouter, Azure OpenAI,
    Together, and any other OpenAI-compatible provider."""
    text = render_block_message(decision, message)
    return MockObject({
        "id": f"promptgaurd-blocked-{decision.prompt_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text, "refusal": None},
                "finish_reason": "content_filter",
                "logprobs": None,
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "promptgaurd": {"blocked": True, "prompt_id": decision.prompt_id, "reason": decision.reason},
    })


def anthropic_blocked_response(decision: Any, model: str = "unknown", message: Union[str, None] = None) -> MockObject:
    """Anthropic Messages API-shaped mock."""
    text = render_block_message(decision, message)
    return MockObject({
        "id": f"promptgaurd-blocked-{decision.prompt_id}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0},
        "promptgaurd": {"blocked": True, "prompt_id": decision.prompt_id, "reason": decision.reason},
    })


def gemini_blocked_response(decision: Any, message: Union[str, None] = None) -> MockObject:
    """google-genai GenerateContentResponse-shaped mock (`.text` works)."""
    text = render_block_message(decision, message)
    return MockObject({
        "text": text,
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": text}]},
                "finish_reason": "SAFETY",
                "index": 0,
            }
        ],
        "usage_metadata": {"prompt_token_count": 0, "candidates_token_count": 0, "total_token_count": 0},
        "promptgaurd": {"blocked": True, "prompt_id": decision.prompt_id, "reason": decision.reason},
    })


def is_blocked_response(response: Any) -> bool:
    """True if a response object is a promptgaurd mock for a blocked prompt."""
    meta = getattr(response, "promptgaurd", None)
    return bool(meta is not None and getattr(meta, "blocked", False))
