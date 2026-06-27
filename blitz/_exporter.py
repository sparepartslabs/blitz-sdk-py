"""Span exporter that converts OTel GenAI spans into blitz's wire format and
POSTs them to the blitz backend.

We use our own JSON shape (not raw OTLP) because we own both ends — it keeps the
FastAPI ingest endpoint trivial and lets redaction happen cleanly during the
conversion step rather than mutating immutable OTel spans. The instrumentors
still emit standard OTel spans, so a customer can additionally attach a vanilla
OTLP exporter to fan telemetry out to Datadog/Phoenix/etc.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from typing import Callable, Optional, Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import StatusCode

logger = logging.getLogger("blitz")

_PROMPT_PREFIX = "gen_ai.prompt."
_COMPLETION_PREFIX = "gen_ai.completion."


class BlitzSpanExporter(SpanExporter):
    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        project_id: str,
        capture_content: bool = True,
        redact: Optional[Callable[[str], str]] = None,
        max_content_chars: int = 24_000,
        timeout: float = 10.0,
    ) -> None:
        self._url = endpoint.rstrip("/") + "/blitz/v1/traces"
        self._headers = {"content-type": "application/json", "x-api-key": api_key}
        self._project_id = project_id
        self._capture_content = capture_content
        self._redact = redact
        self._max = max_content_chars
        self._timeout = timeout

    # -- SpanExporter interface ---------------------------------------------

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            payload = {
                "project_id": self._project_id,
                "spans": [self._convert(s) for s in spans],
            }
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self._url, data=data, headers=self._headers, method="POST"
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                if resp.status >= 300:
                    logger.warning("blitz export got HTTP %s", resp.status)
                    return SpanExportResult.FAILURE
            return SpanExportResult.SUCCESS
        except Exception:  # noqa: BLE001 - exporting must never raise into the app
            logger.warning("blitz export failed", exc_info=True)
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:  # pragma: no cover - nothing to clean up
        pass

    # -- conversion ----------------------------------------------------------

    def _convert(self, span: ReadableSpan) -> dict:
        attrs = dict(span.attributes or {})
        prompt, completion, other = self._split_content(attrs)

        content = None
        if self._capture_content:
            content = self._redact_content(
                {"prompt": prompt, "completion": completion}
            )

        ctx = span.get_span_context()
        status = (
            "error"
            if span.status is not None and span.status.status_code == StatusCode.ERROR
            else "ok"
        )
        resource_attrs = dict(span.resource.attributes or {})

        return {
            "trace_id": format(ctx.trace_id, "032x"),
            "span_id": format(ctx.span_id, "016x"),
            "parent_span_id": (
                format(span.parent.span_id, "016x") if span.parent else None
            ),
            "name": span.name,
            "service_name": resource_attrs.get("service.name"),
            "provider": attrs.get("gen_ai.system"),
            "model": attrs.get("gen_ai.response.model")
            or attrs.get("gen_ai.request.model"),
            "input_tokens": _first_int(
                attrs,
                "gen_ai.usage.input_tokens",
                "gen_ai.usage.prompt_tokens",
                "llm.usage.prompt_tokens",
            ),
            "output_tokens": _first_int(
                attrs,
                "gen_ai.usage.output_tokens",
                "gen_ai.usage.completion_tokens",
                "llm.usage.completion_tokens",
            ),
            "start_unix_ns": span.start_time,
            "end_unix_ns": span.end_time,
            "status": status,
            "attributes": _jsonable(other),
            "content": content,
        }

    def _split_content(self, attrs: dict):
        """Pull the indexed prompt/completion attributes
        (gen_ai.prompt.0.role, gen_ai.prompt.0.content, ...) into ordered lists,
        leaving everything else in `other`."""
        prompts: dict[str, dict] = {}
        completions: dict[str, dict] = {}
        other: dict = {}

        for key, value in attrs.items():
            if key.startswith(_PROMPT_PREFIX):
                idx, _, field = key[len(_PROMPT_PREFIX) :].partition(".")
                prompts.setdefault(idx, {})[field or "value"] = value
            elif key.startswith(_COMPLETION_PREFIX):
                idx, _, field = key[len(_COMPLETION_PREFIX) :].partition(".")
                completions.setdefault(idx, {})[field or "value"] = value
            else:
                other[key] = value

        return _ordered(prompts), _ordered(completions), other

    def _redact_content(self, content: dict) -> dict:
        for bucket in ("prompt", "completion"):
            for msg in content.get(bucket, []):
                if "content" in msg and isinstance(msg["content"], str):
                    msg["content"] = self._scrub(msg["content"])
        return content

    def _scrub(self, text: str) -> str:
        if self._redact:
            try:
                text = self._redact(text)
            except Exception:  # noqa: BLE001
                logger.warning("blitz redact callable raised", exc_info=True)
        if self._max and len(text) > self._max:
            text = text[: self._max] + "…[truncated]"
        return text


def _ordered(indexed: dict[str, dict]) -> list[dict]:
    return [
        indexed[i]
        for i in sorted(indexed, key=lambda x: int(x) if x.isdigit() else 0)
    ]


def _first_int(attrs: dict, *keys: str):
    for key in keys:
        if key in attrs and attrs[key] is not None:
            try:
                return int(attrs[key])
            except (TypeError, ValueError):
                continue
    return None


def _jsonable(obj):
    """OTel attribute values are already JSON-safe scalars/sequences, but coerce
    tuples to lists so json.dumps is happy."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj
