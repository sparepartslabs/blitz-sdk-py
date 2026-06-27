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

import atexit
import contextlib
import logging
import os
from typing import Callable, Generator, Optional

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
        "blitz initialized — project=%s providers=[%s] sample_rate=%s",
        project_id,
        ", ".join(instrumented) or "none",
        sample_rate,
    )
    return instrumented


@contextlib.contextmanager
def workflow(name: str) -> Generator[None, None, None]:
    """Wrap LLM calls in a named parent span.

    The span name becomes the ``root_name`` on the blitz trace, enabling
    per-feature cost grouping in the dashboard::

        with blitz.workflow("mechanic-assistant"):
            response = client.messages.create(...)
    """
    tracer = _otel_trace.get_tracer("blitz")
    with tracer.start_as_current_span(name):
        yield
