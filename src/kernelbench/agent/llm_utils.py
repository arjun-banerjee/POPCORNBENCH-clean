"""LLM retry backoff, request diagnostics, and conversation compaction for KernelAgent."""

from __future__ import annotations

import json
import random
import re
from typing import Any


def extract_retry_after_seconds(exc: BaseException) -> float | None:
    """Parse Retry-After from OpenAI-compatible errors (header or message)."""
    try:
        resp = getattr(exc, "response", None)
        if resp is not None:
            h = getattr(resp, "headers", None)
            if h is not None:
                raw = h.get("retry-after") or h.get("Retry-After")
                if raw is not None:
                    return float(raw)
    except (TypeError, ValueError):
        pass
    m = re.search(r"retry[_\s-]*after[:\s]+(\d+)", str(exc), re.I)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def is_requested_tokens_zero(msg: str) -> bool:
    return "Requested tokens: 0" in msg or "requested tokens: 0" in msg.lower()


def llm_usage_to_dict(usage: Any) -> dict[str, Any] | None:
    """Convert an OpenAI SDK(or compatible) *usage* object to a JSON-safe dict.

    Chat Completions typically expose ``prompt_tokens`` / ``completion_tokens`` /
    ``total_tokens``. The Responses API uses ``input_tokens`` / ``output_tokens``.
    Extra fields (reasoning breakdowns, cache details) are preserved when present.
    """
    if usage is None:
        return None
    raw: dict[str, Any]
    if hasattr(usage, "model_dump"):
        try:
            raw = usage.model_dump(mode="python")
        except TypeError:
            raw = usage.model_dump()
    elif isinstance(usage, dict):
        raw = dict(usage)
    else:
        return None

    def _coerce(v: Any) -> Any:
        if v is None or isinstance(v, (bool, str, int, float)):
            return v
        if isinstance(v, dict):
            return {str(k): _coerce(x) for k, x in v.items()}
        if isinstance(v, list):
            return [_coerce(x) for x in v]
        if hasattr(v, "model_dump"):
            try:
                return _coerce(v.model_dump(mode="python"))
            except TypeError:
                return _coerce(v.model_dump())
        return str(v)

    out = _coerce(raw)
    return out if isinstance(out, dict) else None


def llm_retry_delay_s(
    attempt_idx: int,
    exc: BaseException,
    *,
    base: float = 2.0,
    max_s: float = 180.0,
) -> float:
    """Exponential backoff with jitter; honors Retry-After when present."""
    ra = extract_retry_after_seconds(exc)
    if ra is not None and ra > 0:
        return min(ra + random.uniform(0.0, 1.5), max_s)
    exp = min(base ** float(attempt_idx), max_s)
    jitter = random.uniform(0.0, min(3.0, 0.25 * exp))
    return min(exp + jitter, max_s)


def log_trimmed_create_kwargs_diagnostics(tag: str, payload: dict[str, Any]) -> None:
    """Log JSON size and a shallow preview; redacts/truncates huge string values."""

    def _trim_obj(obj: Any, depth: int = 0, max_str: int = 4000) -> Any:
        if depth > 6:
            return "<max depth>"
        if isinstance(obj, str):
            if len(obj) <= max_str:
                return obj
            return (
                f"{obj[: max_str // 2]}\n... [{len(obj) - max_str} chars omitted] ...\n"
                f"{obj[-max_str // 2 :]}"
            )
        if isinstance(obj, dict):
            return {k: _trim_obj(v, depth + 1, max_str) for k, v in obj.items()}
        if isinstance(obj, list):
            if len(obj) > 64:
                return [_trim_obj(x, depth + 1, max_str) for x in obj[:32]] + [
                    f"... [{len(obj) - 32} more items] ..."
                ]
            return [_trim_obj(x, depth + 1, max_str) for x in obj]
        return obj

    try:
        trimmed = _trim_obj(payload)
        raw = json.dumps(trimmed, default=str)
        print(
            f"{tag} LLM request diagnostics: json_len={len(raw)} chars "
            f"(trimmed preview below)\n{raw[:8000]}"
            + ("..." if len(raw) > 8000 else ""),
            flush=True,
        )
    except Exception as e:
        print(f"{tag} LLM request diagnostics: could not serialize payload: {e}", flush=True)


def truncate_context_str(s: str, max_chars: int, label: str) -> str:
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    omit = len(s) - max_chars + 120
    half = max_chars // 2
    return (
        f"{label} (truncated {omit} chars): "
        f"{s[:half]}\n...[ omitted {omit} characters ]...\n{s[-half:]}"
    )


def compact_responses_input_items(
    items: list[dict[str, Any]],
    *,
    tool_output_max_chars: int,
) -> None:
    """Truncate tool outputs in Responses-API `input` items.

    Reasoning items are left unchanged so reasoning-capable models still receive
    verbatim chains required by the API.
    """
    for it in items:
        if it.get("type") != "function_call_output":
            continue
        out = it.get("output")
        if isinstance(out, str):
            it["output"] = truncate_context_str(
                out, tool_output_max_chars, "tool_output"
            )


def compact_chat_tool_messages(
    messages: list[dict[str, Any]],
    *,
    tool_output_max_chars: int,
) -> None:
    for m in messages:
        if m.get("role") != "tool":
            continue
        c = m.get("content")
        if isinstance(c, str):
            m["content"] = truncate_context_str(c, tool_output_max_chars, "tool_output")


def maybe_sliding_window_chat(
    messages: list[dict[str, Any]],
    *,
    keep_tail: int,
) -> None:
    """Keep system + first user message + placeholder + last `keep_tail` messages.

    May break API invariants if the cut splits an assistant/tool batch; callers
    should use a large enough `keep_tail` that recent complete turns remain.
    """
    if keep_tail <= 0 or len(messages) <= 2 + keep_tail:
        return
    head = messages[:2]
    tail = messages[-keep_tail:]
    placeholder = {
        "role": "user",
        "content": (
            "[Context compressed: older turns were removed. "
            "The full problem and reference are in the first user message.]"
        ),
    }
    messages.clear()
    messages.extend(head + [placeholder] + tail)


_CODE_FENCE_RE = re.compile(
    r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE
)


def extract_modelnew_kernel_from_text(text: str) -> str | None:
    """Return last fenced Python block if it plausibly contains ModelNew."""
    if not text or not text.strip():
        return None
    matches = list(_CODE_FENCE_RE.finditer(text))
    if not matches:
        return None
    code = matches[-1].group(1).strip()
    if "ModelNew" not in code:
        return None
    return code


def extract_modelnew_from_response_items(response_items: list[dict[str, Any]]) -> str | None:
    """Collect assistant output_text from Responses-shaped `response_items`."""
    chunks: list[str] = []
    for it in response_items:
        if it.get("type") != "message":
            continue
        for part in it.get("content") or []:
            if isinstance(part, dict) and part.get("type") == "output_text":
                t = part.get("text")
                if isinstance(t, str):
                    chunks.append(t)
    return extract_modelnew_kernel_from_text("\n".join(chunks))
