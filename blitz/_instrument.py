"""Wire up the provider instrumentors.

We wrap the maintained openllmetry (Traceloop) instrumentors. Each is optional:
if the provider SDK (and its instrumentor extra) isn't installed, we skip it
silently so a customer who only uses Anthropic doesn't need OpenAI installed.
"""

from __future__ import annotations

import logging

from opentelemetry.sdk.trace import TracerProvider

logger = logging.getLogger("blitz")


def instrument_providers(tracer_provider: TracerProvider) -> list[str]:
    """Instrument every supported provider that is importable. Returns the
    list of provider names that were successfully instrumented."""
    instrumented: list[str] = []

    def _try(name: str, importer) -> None:
        try:
            instrumentor = importer()
            instrumentor.instrument(tracer_provider=tracer_provider)
            instrumented.append(name)
        except ImportError as exc:
            logger.debug("blitz: %s instrumentation unavailable (%s)", name, exc)
        except Exception:  # noqa: BLE001 - never let instrumentation crash the app
            logger.warning("blitz: failed to instrument %s", name, exc_info=True)

    def _openai():
        from opentelemetry.instrumentation.openai import OpenAIInstrumentor

        return OpenAIInstrumentor()

    def _anthropic():
        from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor

        return AnthropicInstrumentor()

    def _gemini():
        from opentelemetry.instrumentation.google_generativeai import (
            GoogleGenerativeAiInstrumentor,
        )

        return GoogleGenerativeAiInstrumentor()

    _try("openai", _openai)
    _try("anthropic", _anthropic)
    _try("gemini", _gemini)
    return instrumented
