"""blitz — drop-in distributed tracing for LLM calls.

Usage:

    import blitz

    blitz.init(
        project_id="proj_abc",
        api_key="sk_...",
        endpoint="https://api.sparepartslabs.com",
        sample_rate=0.1,
    )

After init(), any OpenAI / Anthropic / Gemini call made in this process is traced
and shipped to the blitz backend. Nothing else in your code changes.
"""

from __future__ import annotations

import asyncio
import atexit
import functools
import logging
import os
from typing import Any, Callable, Optional

from opentelemetry import trace
from opentelemetry import trace as _otel_trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

from ._exporter import BlitzSpanExporter
from ._instrument import instrument_providers

__all__ = ["init", "workflow"]

logger = logging.getLogger("blitz")

_initialized = False


def init(
    *,
    project_id: str,
    api_key: str,
    endpoint: str,
    sample_rate: float = 1.0,
    capture_content: bool = True,
    redact: Optional[Callable[[str], str]] = None,
    max_content_chars: int = 24_000,
    service_name: str = "llm-app",
    environment: str = "production",
) -> list[str]:
    """Initialize blitz tracing.

    Args:
        project_id: Your blitz project id (multitenancy key).
        api_key: Project API key, sent as the ``x-api-key`` header.
        endpoint: Base URL of the blitz backend, e.g.
            ``https://api.sparepartslabs.com``. The SDK posts to
            ``{endpoint}/blitz/v1/traces``.
        sample_rate: Head sampling ratio in [0.0, 1.0]. 0.1 = trace 10% of
            requests. Sampling is parent-based so a sampled trace keeps all its
            child spans.
        capture_content: When False, prompts/completions are stripped before
            export — only metadata (model, tokens, latency, cost) is sent.
        redact: Optional callable applied to every prompt/completion string
            before export (PII scrubbing).
        max_content_chars: Hard cap per content field; longer values are
            truncated with a ``…[truncated]`` marker.
        service_name: Logical service name attached to every span.
        environment: Deployment environment the traces originate from, e.g.
            ``"production"``, ``"staging"``, ``"dev"``. Attached to every span
            (via the OTel ``deployment.environment`` resource attribute) so the
            dashboard can tell where traffic came from and scope observability
            to one environment. Defaults to ``"production"``.

    Returns:
        The list of providers that were successfully instrumented
        (e.g. ``["openai", "anthropic"]``).
    """
    global _initialized
    if _initialized:
        logger.warning("blitz.init() called more than once; ignoring")
        return []

    if not 0.0 <= sample_rate <= 1.0:
        raise ValueError("sample_rate must be between 0.0 and 1.0")

    # Tell the underlying instrumentors not to capture prompt content at the
    # source when the caller opted out — cheaper and avoids the content ever
    # entering a span. The exporter enforces this again as a backstop.
    if not capture_content:
        os.environ.setdefault("TRACELOOP_TRACE_CONTENT", "false")

    resource = Resource.create(
        {
            "service.name": service_name,
            "deployment.environment": environment,
            "blitz.project_id": project_id,
        }
    )
    provider = TracerProvider(
        resource=resource,
        sampler=ParentBased(TraceIdRatioBased(sample_rate)),
    )

    exporter = BlitzSpanExporter(
        endpoint=endpoint,
        api_key=api_key,
        project_id=project_id,
        capture_content=capture_content,
        redact=redact,
        max_content_chars=max_content_chars,
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    instrumented = instrument_providers(provider)
    atexit.register(provider.shutdown)

    _initialized = True
    logger.info(
        "blitz initialized — project=%s env=%s providers=[%s] sample_rate=%s",
        project_id,
        environment,
        ", ".join(instrumented) or "none",
        sample_rate,
    )
    return instrumented


class _Workflow:
    """Returned by :func:`workflow` — usable as a context manager or decorator."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._ctx: Any = None

    # -- context manager -------------------------------------------------------

    def __enter__(self) -> None:
        self._ctx = _otel_trace.get_tracer("blitz").start_as_current_span(self._name)
        self._ctx.__enter__()

    def __exit__(self, *args: Any) -> Optional[bool]:
        return self._ctx.__exit__(*args)

    # -- decorator -------------------------------------------------------------

    def __call__(self, fn: Callable) -> Callable:
        if asyncio.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrap(*args: Any, **kwargs: Any) -> Any:
                with _Workflow(self._name):
                    return await fn(*args, **kwargs)
            return async_wrap

        @functools.wraps(fn)
        def sync_wrap(*args: Any, **kwargs: Any) -> Any:
            with _Workflow(self._name):
                return fn(*args, **kwargs)
        return sync_wrap


def workflow(name: str) -> _Workflow:
    """Wrap LLM calls in a named parent span.

    Works as a context manager or a decorator (sync and async)::

        # context manager
        with blitz.workflow("mechanic-assistant"):
            response = client.messages.create(...)

        # decorator
        @blitz.workflow("mechanic-assistant")
        async def _ai_reply(...):
            response = client.messages.create(...)

    The span name becomes ``root_name`` on the blitz trace, enabling
    per-feature cost grouping in the dashboard.
    """
    return _Workflow(name)
